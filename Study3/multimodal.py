
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from .models import FusionLayer, FusionOp


# ----------------------------------------------------------------------------
# Image extraction from PDFs
# ----------------------------------------------------------------------------

def extract_images_from_pdfs(
    pdf_paths: List[str],
    cache_dir: str,
    max_images_per_doc: int = 4,
    page_render_dpi: int = 100,
    min_image_size: int = 64,
) -> Dict[str, List[str]]:
    """Extract images from each PDF.

    Strategy:
      1. Extract embedded images (figures, tables, equations)
      2. If a document has fewer than `max_images_per_doc` embedded images,
         fall back to rendering the first 1-2 pages as images
      3. Save each image to `cache_dir/<doc_id>/img_<i>.png`
      4. Return a dict: pdf_path -> list of saved image paths

    Filters out tiny images (icons, separators) below `min_image_size` px.
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    result = {}
    for pdf_path in pdf_paths:
        doc_id = Path(pdf_path).stem
        out_dir = cache_root / doc_id
        out_dir.mkdir(exist_ok=True)

        # Skip if already cached
        existing = sorted(out_dir.glob("img_*.png"))
        if existing:
            result[pdf_path] = [str(p) for p in existing[:max_images_per_doc]]
            continue

        saved_paths = []
        try:
            doc = fitz.open(pdf_path)

            # 1. Try embedded images first
            for page_idx in range(len(doc)):
                if len(saved_paths) >= max_images_per_doc:
                    break
                page = doc.load_page(page_idx)
                for img_info in page.get_images(full=True):
                    if len(saved_paths) >= max_images_per_doc:
                        break
                    xref = img_info[0]
                    try:
                        base = doc.extract_image(xref)
                        img_bytes = base["image"]
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        if min(img.size) < min_image_size:
                            continue
                        save_path = out_dir / f"img_{len(saved_paths):02d}.png"
                        img.save(save_path)
                        saved_paths.append(str(save_path))
                    except Exception:
                        continue

            # 2. Fallback: render first pages if not enough embedded images
            page_idx = 0
            while len(saved_paths) < max_images_per_doc and page_idx < min(2, len(doc)):
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(dpi=page_render_dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                save_path = out_dir / f"img_{len(saved_paths):02d}.png"
                img.save(save_path)
                saved_paths.append(str(save_path))
                page_idx += 1

            doc.close()
        except Exception as e:
            print(f"[warn] could not extract images from {pdf_path}: {e}")

        result[pdf_path] = saved_paths

    n_with_images = sum(1 for v in result.values() if v)
    print(f"Extracted images from {n_with_images}/{len(pdf_paths)} PDFs "
          f"into {cache_dir}")
    return result


# ----------------------------------------------------------------------------
# Multimodal dataset
# ----------------------------------------------------------------------------

# Standard ImageNet preprocessing for ResNet
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Blank tensor used when a doc has no images
BLANK_IMAGE = torch.zeros(3, 224, 224)


class MultimodalDataset(Dataset):
    """Yields BERT tokens + GPT-2 tokens + a stack of image tensors per doc.

    Image tensor shape: (n_images, 3, 224, 224). Documents with fewer than
    `n_images` actual images are padded with blank tensors and a mask.
    """

    def __init__(self, texts, labels, pdf_paths, image_index,
                 bert_tok, gpt_tok,
                 max_len: int = 128, n_images: int = 4):
        assert len(texts) == len(labels) == len(pdf_paths)
        self.texts = list(texts)
        self.labels = list(labels)
        self.pdf_paths = list(pdf_paths)
        self.image_index = image_index    # dict: pdf_path -> [image_path, ...]
        self.bert_tok = bert_tok
        self.gpt_tok = gpt_tok
        self.max_len = max_len
        self.n_images = n_images

    def __len__(self):
        return len(self.texts)

    def _load_images(self, pdf_path):
        paths = self.image_index.get(pdf_path, [])[: self.n_images]
        tensors, mask = [], []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(IMAGE_TRANSFORM(img))
                mask.append(1)
            except Exception:
                tensors.append(BLANK_IMAGE.clone())
                mask.append(0)
        # Pad
        while len(tensors) < self.n_images:
            tensors.append(BLANK_IMAGE.clone())
            mask.append(0)
        return torch.stack(tensors, dim=0), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx):
        text = self.texts[idx]
        b = self.bert_tok(text, max_length=self.max_len, padding="max_length",
                          truncation=True, return_tensors="pt")
        g = self.gpt_tok(text, max_length=self.max_len, padding="max_length",
                         truncation=True, return_tensors="pt")
        imgs, img_mask = self._load_images(self.pdf_paths[idx])

        return {
            "bert_ids":   b["input_ids"].squeeze(0),
            "bert_mask":  b["attention_mask"].squeeze(0),
            "gpt_ids":    g["input_ids"].squeeze(0),
            "gpt_mask":   g["attention_mask"].squeeze(0),
            "images":     imgs,            # (n_images, 3, 224, 224)
            "image_mask": img_mask,        # (n_images,)
            "label":      torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ----------------------------------------------------------------------------
# True multimodal model: BERT + GPT-2 + Vision CNN
# ----------------------------------------------------------------------------

class MultimodalFusion(nn.Module):
    """Genuine multimodal architecture for scientific PDF classification.

    Streams:
      - Text: BERT [CLS]            (768-d)
      - Text: GPT-2 last hidden     (768-d)
      - Vision: ResNet-18 over      (512-d each, mean-pooled across the
        extracted PDF images         document's images via masked mean)

    All three projected to a shared fusion_dim and combined via FusionLayer.
    """

    def __init__(self, n_classes: int, fusion_dim: int = 256,
                 dropout: float = 0.5,
                 fusion_op: FusionOp = FusionOp.WEIGHTED_SUM,
                 freeze_backbones: bool = False,
                 freeze_vision: bool = True,
                 bert_name: str = "bert-base-uncased",
                 gpt_name: str = "gpt2"):
        super().__init__()
        from transformers import BertModel, GPT2Model
        self.bert = BertModel.from_pretrained(bert_name)
        self.gpt2 = GPT2Model.from_pretrained(gpt_name)

        # ResNet-18 with ImageNet weights, drop the final FC, keep 512-d features
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.vision_features = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        vision_dim = 512

        if freeze_backbones:
            for p in self.bert.parameters():
                p.requires_grad = False
            for p in self.gpt2.parameters():
                p.requires_grad = False
        if freeze_vision:
            # ResNet on 211 docs is overkill if fully fine-tuned; freeze by default
            for p in self.vision_features.parameters():
                p.requires_grad = False

        h_b = self.bert.config.hidden_size
        h_g = self.gpt2.config.hidden_size

        self.fusion = FusionLayer(
            dims=[h_b, h_g, vision_dim], fusion_dim=fusion_dim,
            op=fusion_op, dropout=0.1,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def _encode_images(self, images: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        """images: (B, n_imgs, 3, 224, 224); mask: (B, n_imgs).

        Returns (B, 512) — masked mean pool over each document's images.
        """
        B, N, C, H, W = images.shape
        flat = images.reshape(B * N, C, H, W)
        feat = self.vision_features(flat).reshape(B * N, -1)  # (B*N, 512)
        feat = feat.reshape(B, N, -1)                          # (B, N, 512)

        # Masked mean: ignore blank-padded image slots
        m = image_mask.unsqueeze(-1).float()                   # (B, N, 1)
        denom = m.sum(dim=1).clamp(min=1.0)                    # (B, 1)
        pooled = (feat * m).sum(dim=1) / denom                 # (B, 512)
        return pooled

    def forward(self, bert_ids, bert_mask, gpt_ids, gpt_mask,
                images, image_mask, **_):
        h_bert = self.bert(input_ids=bert_ids, attention_mask=bert_mask).last_hidden_state[:, 0, :]
        h_gpt  = self.gpt2(input_ids=gpt_ids,  attention_mask=gpt_mask ).last_hidden_state[:, -1, :]
        h_img  = self._encode_images(images, image_mask)

        fused = self.fusion(h_bert, h_gpt, h_img)
        return self.classifier(self.dropout(fused))


# ----------------------------------------------------------------------------
# Convenience: end-to-end run
# ----------------------------------------------------------------------------

def run_multimodal_experiment(
    pdf_directory: str,
    image_cache_dir: str = "/content/pdf_images",
    label_strategy: str = "sbert_kmeans",
    n_clusters: int = 5,
    seeds=(42, 7, 123),
    epochs: int = 5,
    batch_size: int = 8,           # smaller because images take memory
    max_len: int = 128,
    n_images_per_doc: int = 4,
    fusion_op: FusionOp = FusionOp.WEIGHTED_SUM,
    freeze_backbones: bool = True,
):
    """One-call multimodal experiment.

    Steps:
      1. Load corpus and extract images
      2. Build labels
      3. For each seed: split, train MultimodalFusion, evaluate on test
      4. Report mean ± std
    """
    from transformers import BertTokenizer, GPT2Tokenizer
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    from .data import (
        load_corpus,
        labels_tfidf_kmeans, labels_sbert_kmeans, labels_from_nus_categories,
    )
    from .training import set_seed, evaluate, train_one_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load text and extract images
    texts, paths = load_corpus(pdf_directory)
    print(f"\nExtracting images (cached after first run)...")
    image_index = extract_images_from_pdfs(
        paths, cache_dir=image_cache_dir,
        max_images_per_doc=n_images_per_doc,
    )

    # Labels
    if label_strategy == "tfidf_kmeans":
        labels, _ = labels_tfidf_kmeans(texts, n_clusters=n_clusters, seed=seeds[0])
    elif label_strategy == "sbert_kmeans":
        labels, _ = labels_sbert_kmeans(texts, n_clusters=n_clusters, seed=seeds[0])
    elif label_strategy == "nus_categories":
        labels, _ = labels_from_nus_categories(paths)
    else:
        raise ValueError(label_strategy)
    n_classes = int(np.max(labels)) + 1
    print(f"n_docs={len(texts)}, n_classes={n_classes}\n")

    # Tokenisers
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tok.pad_token = gpt_tok.eos_token
    gpt_tok.padding_side = "left"

    # Multi-seed train + test
    test_results = []
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        set_seed(seed)

        idx = np.arange(len(texts))
        idx_tr, idx_te = train_test_split(
            idx, test_size=0.2, stratify=labels, random_state=seed,
        )

        ds_tr = MultimodalDataset(
            [texts[i] for i in idx_tr], [labels[i] for i in idx_tr],
            [paths[i] for i in idx_tr], image_index,
            bert_tok, gpt_tok, max_len=max_len, n_images=n_images_per_doc,
        )
        ds_te = MultimodalDataset(
            [texts[i] for i in idx_te], [labels[i] for i in idx_te],
            [paths[i] for i in idx_te], image_index,
            bert_tok, gpt_tok, max_len=max_len, n_images=n_images_per_doc,
        )
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

        model = MultimodalFusion(
            n_classes=n_classes, fusion_op=fusion_op,
            freeze_backbones=freeze_backbones,
        ).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        for ep in range(epochs):
            tl = train_one_epoch(model, dl_tr, opt, loss_fn, device)
            print(f"  epoch {ep+1}: loss={tl:.4f}")

        m = evaluate(model, dl_te, device)
        test_results.append(m)
        print(f"  test_acc={m['accuracy']:.4f}  test_f1={m['f1']:.4f}")

        del model
        torch.cuda.empty_cache()

    accs = [r["accuracy"] for r in test_results]
    f1s = [r["f1"] for r in test_results]
    print(f"\n{'=' * 50}")
    print(f"MULTIMODAL FUSION (BERT + GPT-2 + ResNet-18)")
    print(f"{'=' * 50}")
    print(f"Test accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Test F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    return {
        "per_seed": test_results,
        "test_acc_mean": float(np.mean(accs)),
        "test_acc_std": float(np.std(accs)),
        "test_f1_mean": float(np.mean(f1s)),
        "test_f1_std": float(np.std(f1s)),
    }
