"""
Study 3: Multimodal Scientific PDF Classification
Proposed Method: BERT + GPT2 + ResNet18 with Ablation Baselines and Wilcoxon Statistical Testing

This module implements the complete pipeline for evaluating the proposed
multimodal fusion architecture on scientific PDF classification, alongside
single-modality ablation baselines (BERT-only, GPT2-only, ResNet18-only).
"""

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore")

import random
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import fitz  # PyMuPDF
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer


# ===========================================================================
# Configuration
# ===========================================================================

SEEDS = (42, 7, 123, 999, 2023, 8888, 7777)

PDF_DIR         = "/kaggle/input/nuswide-scientific-pdfs"
IMAGE_CACHE_DIR = "/kaggle/working/pdf_images"
OUTPUT_DIR      = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

RESULTS_CSV  = f"{OUTPUT_DIR}/study3_per_seed_results.csv"
SUMMARY_CSV  = f"{OUTPUT_DIR}/study3_summary.csv"
WILCOXON_CSV = f"{OUTPUT_DIR}/study3_wilcoxon.csv"

EPOCHS        = 5
BATCH_SIZE    = 8
MAX_LENGTH    = 128
LEARNING_RATE = 5e-5
WEIGHT_DECAY  = 0.01

N_IMAGES_PER_PDF = 4
FUSION_DIM       = 256
DROPOUT          = 0.5

IMAGE_SIZE           = 224
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD  = [0.229, 0.224, 0.225]

TEST_SIZE  = 0.2
N_CLUSTERS = 5

MODELS_TO_RUN = ("BERT-only", "GPT2-only", "ResNet18-only", "Multimodal")
PROPOSED      = "Multimodal"


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Image Transform
# ===========================================================================

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD),
])

BLANK_IMAGE = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)


# ===========================================================================
# Image Extraction from PDFs
# ===========================================================================

def extract_images_from_pdfs(pdf_paths, cache_dir, max_images_per_doc=4,
                              page_render_dpi=100, min_image_size=64):
    """
    Extract images from PDFs and cache them.

    Strategy:
    1. Extract embedded images (figures, tables).
    2. If fewer than max_images_per_doc, render first pages.
    3. Save to cache_dir/<doc_id>/img_<i>.png.
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    result = {}
    for pdf_path in pdf_paths:
        doc_id  = Path(pdf_path).stem
        out_dir = cache_root / doc_id
        out_dir.mkdir(exist_ok=True)

        existing = sorted(out_dir.glob("img_*.png"))
        if existing:
            result[pdf_path] = [str(p) for p in existing[:max_images_per_doc]]
            continue

        saved_paths = []
        try:
            doc = fitz.open(pdf_path)

            for page_idx in range(len(doc)):
                if len(saved_paths) >= max_images_per_doc:
                    break
                page = doc.load_page(page_idx)
                for img_info in page.get_images(full=True):
                    if len(saved_paths) >= max_images_per_doc:
                        break
                    xref = img_info[0]
                    try:
                        base      = doc.extract_image(xref)
                        img_bytes = base["image"]
                        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        if min(img.size) < min_image_size:
                            continue
                        save_path = out_dir / f"img_{len(saved_paths):02d}.png"
                        img.save(save_path)
                        saved_paths.append(str(save_path))
                    except Exception:
                        continue

            page_idx = 0
            while len(saved_paths) < max_images_per_doc and page_idx < min(2, len(doc)):
                page = doc.load_page(page_idx)
                pix  = page.get_pixmap(dpi=page_render_dpi)
                img  = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                save_path = out_dir / f"img_{len(saved_paths):02d}.png"
                img.save(save_path)
                saved_paths.append(str(save_path))
                page_idx += 1

            doc.close()
        except Exception as e:
            print(f"[warn] Could not extract images from {pdf_path}: {e}")

        result[pdf_path] = saved_paths

    n_with_images = sum(1 for v in result.values() if v)
    print(f"Extracted images from {n_with_images}/{len(pdf_paths)} PDFs into {cache_dir}\n")
    return result


# ===========================================================================
# Datasets
# ===========================================================================

class TextOnlyDataset(Dataset):
    """For BERT-only and GPT2-only baselines (no image)."""

    def __init__(self, texts, labels, bert_tok, gpt_tok, max_len=128):
        self.texts    = list(texts)
        self.labels   = list(labels)
        self.bert_tok = bert_tok
        self.gpt_tok  = gpt_tok
        self.max_len  = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        b = self.bert_tok(text, max_length=self.max_len, padding="max_length",
                          truncation=True, return_tensors="pt")
        g = self.gpt_tok(text, max_length=self.max_len, padding="max_length",
                         truncation=True, return_tensors="pt")

        return {
            "bert_ids":  b["input_ids"].squeeze(0),
            "bert_mask": b["attention_mask"].squeeze(0),
            "gpt_ids":   g["input_ids"].squeeze(0),
            "gpt_mask":  g["attention_mask"].squeeze(0),
            "label":     torch.tensor(self.labels[idx], dtype=torch.long),
        }


class ImageOnlyDataset(Dataset):
    """For ResNet18-only baseline (no text)."""

    def __init__(self, pdf_paths, labels, image_index, n_images=4):
        self.pdf_paths   = list(pdf_paths)
        self.labels      = list(labels)
        self.image_index = image_index
        self.n_images    = n_images

    def __len__(self):
        return len(self.pdf_paths)

    def _load_images(self, pdf_path):
        paths   = self.image_index.get(pdf_path, [])[:self.n_images]
        tensors, mask = [], []

        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(image_transform(img))
                mask.append(1)
            except Exception:
                tensors.append(BLANK_IMAGE.clone())
                mask.append(0)

        while len(tensors) < self.n_images:
            tensors.append(BLANK_IMAGE.clone())
            mask.append(0)

        return torch.stack(tensors, dim=0), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx):
        imgs, img_mask = self._load_images(self.pdf_paths[idx])
        return {
            "images":     imgs,
            "image_mask": img_mask,
            "label":      torch.tensor(self.labels[idx], dtype=torch.long),
        }


class MultimodalDataset(Dataset):
    """For the proposed Multimodal model: BERT + GPT2 + ResNet18."""

    def __init__(self, texts, labels, pdf_paths, image_index, bert_tok, gpt_tok,
                 max_len=128, n_images=4):
        assert len(texts) == len(labels) == len(pdf_paths)
        self.texts       = list(texts)
        self.labels      = list(labels)
        self.pdf_paths   = list(pdf_paths)
        self.image_index = image_index
        self.bert_tok    = bert_tok
        self.gpt_tok     = gpt_tok
        self.max_len     = max_len
        self.n_images    = n_images

    def __len__(self):
        return len(self.texts)

    def _load_images(self, pdf_path):
        paths   = self.image_index.get(pdf_path, [])[:self.n_images]
        tensors, mask = [], []

        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(image_transform(img))
                mask.append(1)
            except Exception:
                tensors.append(BLANK_IMAGE.clone())
                mask.append(0)

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
            "images":     imgs,
            "image_mask": img_mask,
            "label":      torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ===========================================================================
# Models
# ===========================================================================

class BertOnlyClassifier(nn.Module):
    """Ablation: text-only baseline using BERT [CLS] token."""

    def __init__(self, n_classes=5):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(self.bert.config.hidden_size, n_classes),
        )

    def forward(self, bert_ids, bert_mask, **_):
        out = self.bert(input_ids=bert_ids, attention_mask=bert_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


class GPT2OnlyClassifier(nn.Module):
    """Ablation: text-only baseline using GPT2 last token."""

    def __init__(self, n_classes=5):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(self.gpt2.config.hidden_size, n_classes),
        )

    def forward(self, gpt_ids, gpt_mask, **_):
        out = self.gpt2(input_ids=gpt_ids, attention_mask=gpt_mask)
        return self.classifier(out.last_hidden_state[:, -1, :])


class ResNet18OnlyClassifier(nn.Module):
    """Ablation: image-only baseline using ResNet18 with masked mean-pooling."""

    def __init__(self, n_classes=5):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vision_features = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.vision_features.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(512, n_classes),
        )

    def _encode_images(self, images, image_mask):
        batch_size, n_images = images.shape[:2]
        flat = images.reshape(batch_size * n_images, 3, IMAGE_SIZE, IMAGE_SIZE)
        feat = self.vision_features(flat).reshape(batch_size * n_images, -1)
        feat = feat.reshape(batch_size, n_images, -1)
        mask  = image_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feat * mask).sum(dim=1) / denom

    def forward(self, images, image_mask, **_):
        pooled = self._encode_images(images, image_mask)
        return self.classifier(pooled)


class FusionLayer(nn.Module):
    """Weighted sum fusion of multiple modality streams."""

    def __init__(self, input_dims, fusion_dim=256, dropout=0.1):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(d, fusion_dim) for d in input_dims
        ])
        self.weights = nn.Parameter(torch.ones(len(input_dims)) / len(input_dims))
        self.dropout = nn.Dropout(dropout)

    def forward(self, *inputs):
        projected = [proj(x) for proj, x in zip(self.projections, inputs)]
        weights   = torch.softmax(self.weights, dim=0)
        fused     = sum(w * p for w, p in zip(weights, projected))
        return self.dropout(fused)


class MultimodalFusion(nn.Module):
    """
    Proposed multimodal architecture for scientific PDF classification.

    Fuses three modality streams:
    - BERT [CLS] token (768 dims)
    - GPT2 last token (768 dims)
    - ResNet18 masked mean-pooled image features (512 dims)

    Projects all to fusion_dim and combines via weighted sum.
    """

    def __init__(self, n_classes=5):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.gpt2 = GPT2Model.from_pretrained("gpt2")

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vision_features = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.vision_features.parameters():
            p.requires_grad = False

        self.fusion = FusionLayer(
            input_dims=[self.bert.config.hidden_size,
                        self.gpt2.config.hidden_size, 512],
            fusion_dim=FUSION_DIM,
            dropout=0.1,
        )

        # Modality gate for the visual stream: initialised to -4 so
        # sigmoid(gate) ≈ 0.018 at the start (defaults closed).
        # The gate is learned during training and opens as visual
        # features prove useful, preventing the vision backbone from
        # dominating early in training when weights are random.
        self.vision_gate = nn.Parameter(torch.tensor(-4.0))

        self.dropout    = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(FUSION_DIM, n_classes)

    def _encode_images(self, images, image_mask):
        batch_size, n_images = images.shape[:2]
        flat = images.reshape(batch_size * n_images, 3, IMAGE_SIZE, IMAGE_SIZE)
        feat = self.vision_features(flat).reshape(batch_size * n_images, -1)
        feat = feat.reshape(batch_size, n_images, -1)
        mask  = image_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (feat * mask).sum(dim=1) / denom
        # Apply the modality gate: scales the visual stream by sigmoid(gate)
        return torch.sigmoid(self.vision_gate) * pooled

    def forward(self, bert_ids, bert_mask, gpt_ids, gpt_mask, images, image_mask, **_):
        h_bert   = self.bert(input_ids=bert_ids, attention_mask=bert_mask).last_hidden_state[:, 0, :]
        h_gpt2   = self.gpt2(input_ids=gpt_ids, attention_mask=gpt_mask).last_hidden_state[:, -1, :]
        h_vision = self._encode_images(images, image_mask)
        fused    = self.fusion(h_bert, h_gpt2, h_vision)
        return self.classifier(self.dropout(fused))


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ===========================================================================
# Train and evaluate one model on one seed
# ===========================================================================

def train_and_evaluate(model_name, train_loader, val_loader, n_classes, device):
    """Build the named model, train for EPOCHS, then evaluate on val_loader."""

    if model_name == "BERT-only":
        model       = BertOnlyClassifier(n_classes=n_classes)
        needs_text  = True
        needs_image = False
    elif model_name == "GPT2-only":
        model       = GPT2OnlyClassifier(n_classes=n_classes)
        needs_text  = True
        needs_image = False
    elif model_name == "ResNet18-only":
        model       = ResNet18OnlyClassifier(n_classes=n_classes)
        needs_text  = False
        needs_image = True
    elif model_name == "Multimodal":
        model       = MultimodalFusion(n_classes=n_classes)
        needs_text  = True
        needs_image = True
    else:
        raise ValueError(model_name)

    model     = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    loss_fn   = nn.CrossEntropyLoss()

    # Train
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            kwargs = {}
            if needs_text:
                kwargs["bert_ids"]  = batch["bert_ids"].to(device)
                kwargs["bert_mask"] = batch["bert_mask"].to(device)
                kwargs["gpt_ids"]   = batch["gpt_ids"].to(device)
                kwargs["gpt_mask"]  = batch["gpt_mask"].to(device)
            if needs_image:
                kwargs["images"]     = batch["images"].to(device)
                kwargs["image_mask"] = batch["image_mask"].to(device)
            logits = model(**kwargs)
            loss   = loss_fn(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            kwargs = {}
            if needs_text:
                kwargs["bert_ids"]  = batch["bert_ids"].to(device)
                kwargs["bert_mask"] = batch["bert_mask"].to(device)
                kwargs["gpt_ids"]   = batch["gpt_ids"].to(device)
                kwargs["gpt_mask"]  = batch["gpt_mask"].to(device)
            if needs_image:
                kwargs["images"]     = batch["images"].to(device)
                kwargs["image_mask"] = batch["image_mask"].to(device)
            preds  = model(**kwargs).argmax(dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    metrics = compute_metrics(np.array(y_true), np.array(y_pred))

    del model
    torch.cuda.empty_cache()
    return metrics


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Study 3: Multimodal Scientific PDF Classification")
    print("Models: BERT-only | GPT2-only | ResNet18-only | Multimodal (proposed)")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Get PDF paths
    pdf_paths = sorted([str(p) for p in Path(PDF_DIR).glob("*.pdf")])
    if not pdf_paths:
        print(f"ERROR: No PDFs found in {PDF_DIR}")
        return
    print(f"Found {len(pdf_paths)} PDFs\n")

    # Load texts from PDFs
    print("Loading text from PDFs...")
    texts = []
    kept_paths = []
    for pdf_path in pdf_paths:
        try:
            doc  = fitz.open(pdf_path)
            text = " ".join("".join(p.get_text("text") for p in doc).lower().split())
            doc.close()
        except Exception as e:
            print(f"[warn] Could not read {pdf_path}: {e}")
            text = ""
        if text.strip():
            texts.append(text)
            kept_paths.append(pdf_path)
    pdf_paths = kept_paths
    print(f"Loaded {len(texts)} readable documents\n")

    # Cluster with Sentence-BERT + KMeans (breaks TF-IDF circularity)
    print("Generating labels via Sentence-BERT + KMeans...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install sentence-transformers")
    from sklearn.cluster import KMeans

    encoder  = SentenceTransformer("all-MiniLM-L6-v2")
    snippets = [t[:2000] for t in texts]
    embeddings = encoder.encode(snippets, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=SEEDS[0], n_init=10)
    labels = km.fit_predict(embeddings).tolist()

    n_classes = len(set(labels))
    print(f"Documents: {len(texts)}, Classes: {n_classes}\n")

    # Extract / load cached images
    print("Extracting/loading cached images...")
    image_index = extract_images_from_pdfs(
        pdf_paths, IMAGE_CACHE_DIR, max_images_per_doc=N_IMAGES_PER_PDF
    )

    # Tokenisers
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    gpt_tok  = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tok.pad_token    = gpt_tok.eos_token
    gpt_tok.padding_side = "left"

    # Resume if possible
    if os.path.exists(RESULTS_CSV):
        existing_df = pd.read_csv(RESULTS_CSV)
        completed   = set(zip(existing_df["seed"], existing_df["model"]))
        print(f"Resuming: {len(completed)} (seed, model) combos already done.\n")
        rows_so_far = existing_df.to_dict(orient="records")
    else:
        completed   = set()
        rows_so_far = []

    # Multi-seed, multi-model evaluation
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)

        indices = np.arange(len(texts))
        idx_train, idx_test = train_test_split(
            indices, test_size=TEST_SIZE, random_state=seed, stratify=labels
        )

        # Build all three dataset variants once per seed
        train_text_ds = TextOnlyDataset(
            [texts[i] for i in idx_train], [labels[i] for i in idx_train],
            bert_tok, gpt_tok, MAX_LENGTH)
        val_text_ds = TextOnlyDataset(
            [texts[i] for i in idx_test], [labels[i] for i in idx_test],
            bert_tok, gpt_tok, MAX_LENGTH)

        train_img_ds = ImageOnlyDataset(
            [pdf_paths[i] for i in idx_train], [labels[i] for i in idx_train],
            image_index, N_IMAGES_PER_PDF)
        val_img_ds = ImageOnlyDataset(
            [pdf_paths[i] for i in idx_test], [labels[i] for i in idx_test],
            image_index, N_IMAGES_PER_PDF)

        train_mm_ds = MultimodalDataset(
            [texts[i] for i in idx_train], [labels[i] for i in idx_train],
            [pdf_paths[i] for i in idx_train],
            image_index, bert_tok, gpt_tok, MAX_LENGTH, N_IMAGES_PER_PDF)
        val_mm_ds = MultimodalDataset(
            [texts[i] for i in idx_test], [labels[i] for i in idx_test],
            [pdf_paths[i] for i in idx_test],
            image_index, bert_tok, gpt_tok, MAX_LENGTH, N_IMAGES_PER_PDF)

        train_text_loader = DataLoader(train_text_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_text_loader   = DataLoader(val_text_ds,   batch_size=BATCH_SIZE, shuffle=False)
        train_img_loader  = DataLoader(train_img_ds,  batch_size=BATCH_SIZE, shuffle=True)
        val_img_loader    = DataLoader(val_img_ds,    batch_size=BATCH_SIZE, shuffle=False)
        train_mm_loader   = DataLoader(train_mm_ds,   batch_size=BATCH_SIZE, shuffle=True)
        val_mm_loader     = DataLoader(val_mm_ds,     batch_size=BATCH_SIZE, shuffle=False)

        for model_name in MODELS_TO_RUN:
            if (seed, model_name) in completed:
                print(f"  [skip] {model_name} (already done)")
                continue

            if model_name in ("BERT-only", "GPT2-only"):
                train_loader, val_loader = train_text_loader, val_text_loader
            elif model_name == "ResNet18-only":
                train_loader, val_loader = train_img_loader, val_img_loader
            else:  # Multimodal
                train_loader, val_loader = train_mm_loader, val_mm_loader

            print(f"  Training {model_name}...", end=" ", flush=True)
            try:
                metrics = train_and_evaluate(
                    model_name, train_loader, val_loader, n_classes, device
                )
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            print(f"acc={metrics['accuracy']:.4f}  "
                  f"prec={metrics['precision']:.4f}  "
                  f"rec={metrics['recall']:.4f}  "
                  f"f1={metrics['f1']:.4f}")

            row = {"seed": seed, "model": model_name, **metrics}
            rows_so_far.append(row)
            pd.DataFrame(rows_so_far).to_csv(RESULTS_CSV, index=False)

    # ======================================================================
    # Summary
    # ======================================================================
    df_results  = pd.DataFrame(rows_so_far)
    summary_rows = []
    for name in df_results["model"].unique():
        sub = df_results[df_results["model"] == name]
        summary_rows.append({
            "model":      name,
            "acc_mean":   sub["accuracy"].mean(),
            "acc_std":    sub["accuracy"].std(),
            "prec_mean":  sub["precision"].mean(),
            "prec_std":   sub["precision"].std(),
            "rec_mean":   sub["recall"].mean(),
            "rec_std":    sub["recall"].std(),
            "f1_mean":    sub["f1"].mean(),
            "f1_std":     sub["f1"].std(),
        })

    df_summary = (pd.DataFrame(summary_rows)
                  .sort_values("acc_mean", ascending=False)
                  .reset_index(drop=True))

    print("\n" + "=" * 70)
    print(f"Summary across {len(SEEDS)} seeds")
    print("=" * 70)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200):
        print(df_summary.to_string(index=False))
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved {SUMMARY_CSV}")

    # ======================================================================
    # Wilcoxon: Multimodal vs each baseline (paired by seed)
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"Wilcoxon signed-rank tests ({PROPOSED} vs each baseline)")
    print("alternative='greater' tests whether Multimodal > baseline")
    print("=" * 70)

    proposed_accs = (df_results[df_results["model"] == PROPOSED]
                     .sort_values("seed")["accuracy"].values)

    wilcoxon_rows = []
    for name in MODELS_TO_RUN:
        if name == PROPOSED:
            continue
        baseline_accs = (df_results[df_results["model"] == name]
                         .sort_values("seed")["accuracy"].values)
        if len(baseline_accs) != len(proposed_accs):
            print(f"  [skip] {name}: incomplete seed coverage")
            continue
        diffs = proposed_accs - baseline_accs
        if np.allclose(diffs, 0):
            stat, p, note = np.nan, 1.0, "all paired differences are zero"
        else:
            try:
                stat, p = wilcoxon(proposed_accs, baseline_accs,
                                   alternative="greater", zero_method="wilcox")
                note = ""
            except ValueError as e:
                stat, p, note = np.nan, np.nan, f"wilcoxon error: {e}"

        wilcoxon_rows.append({
            "baseline":            name,
            "multimodal_mean_acc": proposed_accs.mean(),
            "baseline_mean_acc":   baseline_accs.mean(),
            "diff":                proposed_accs.mean() - baseline_accs.mean(),
            "wilcoxon_statistic":  stat,
            "p_value":             p,
            "significant_at_005":  (not np.isnan(p)) and (p < 0.05),
            "note":                note,
        })

    df_wilcoxon = pd.DataFrame(wilcoxon_rows)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200):
        print(df_wilcoxon.to_string(index=False))
    df_wilcoxon.to_csv(WILCOXON_CSV, index=False)
    print(f"\nSaved {WILCOXON_CSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
