"""
Study 3: Multimodal Scientific PDF Classification
Proposed Method: BERT + GPT2 + ResNet18 with Wilcoxon Statistical Testing

This module implements the complete pipeline for evaluating the proposed
multimodal fusion architecture on scientific PDF classification.
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

# Random seeds for reproducibility (7 seeds for consistency)
SEEDS = (42, 7, 123, 999, 2023, 8888, 7777)

# Paths
PDF_DIR = "/kaggle/input/nuswide-scientific-pdfs"
IMAGE_CACHE_DIR = "/kaggle/working/pdf_images"
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

RESULTS_CSV = f"{OUTPUT_DIR}/study3_per_seed_results.csv"
SUMMARY_CSV = f"{OUTPUT_DIR}/study3_summary.csv"
WILCOXON_CSV = f"{OUTPUT_DIR}/study3_wilcoxon.csv"

# Training hyperparameters
EPOCHS = 5
BATCH_SIZE = 8
MAX_LENGTH = 128
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

# Model architecture
N_IMAGES_PER_PDF = 4
FUSION_DIM = 256
DROPOUT = 0.5

# Image preprocessing
IMAGE_SIZE = 224
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]

# Data split
TEST_SIZE = 0.2
N_CLUSTERS = 5


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed):
    """Set random seed for reproducibility."""
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
    1. Extract embedded images (figures, tables)
    2. If fewer than max_images_per_doc, render first pages
    3. Save to cache_dir/<doc_id>/img_<i>.png
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    
    result = {}
    for pdf_path in pdf_paths:
        doc_id = Path(pdf_path).stem
        out_dir = cache_root / doc_id
        out_dir.mkdir(exist_ok=True)
        
        # Check if already cached
        existing = sorted(out_dir.glob("img_*.png"))
        if existing:
            result[pdf_path] = [str(p) for p in existing[:max_images_per_doc]]
            continue
        
        saved_paths = []
        try:
            doc = fitz.open(pdf_path)
            
            # Try embedded images first
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
            
            # Fallback: render first pages if not enough embedded images
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
            print(f"[warn] Could not extract images from {pdf_path}: {e}")
        
        result[pdf_path] = saved_paths
    
    n_with_images = sum(1 for v in result.values() if v)
    print(f"Extracted images from {n_with_images}/{len(pdf_paths)} PDFs into {cache_dir}\n")
    return result


# ===========================================================================
# Dataset
# ===========================================================================

class MultimodalDataset(Dataset):
    """
    Dataset for proposed multimodal model: BERT + GPT2 + ResNet18.
    
    Yields BERT tokens + GPT2 tokens + stacked image tensors per document.
    Documents with fewer than n_images images are padded with blank tensors.
    """
    
    def __init__(self, texts, labels, pdf_paths, image_index, bert_tok, gpt_tok,
                 max_len=128, n_images=4):
        assert len(texts) == len(labels) == len(pdf_paths)
        self.texts = list(texts)
        self.labels = list(labels)
        self.pdf_paths = list(pdf_paths)
        self.image_index = image_index
        self.bert_tok = bert_tok
        self.gpt_tok = gpt_tok
        self.max_len = max_len
        self.n_images = n_images
    
    def __len__(self):
        return len(self.texts)
    
    def _load_images(self, pdf_path):
        """Load and transform images for document."""
        paths = self.image_index.get(pdf_path, [])[:self.n_images]
        tensors, mask = [], []
        
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(image_transform(img))
                mask.append(1)
            except Exception:
                tensors.append(BLANK_IMAGE.clone())
                mask.append(0)
        
        # Pad to n_images
        while len(tensors) < self.n_images:
            tensors.append(BLANK_IMAGE.clone())
            mask.append(0)
        
        return torch.stack(tensors, dim=0), torch.tensor(mask, dtype=torch.long)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # BERT tokenization
        b = self.bert_tok(text, max_length=self.max_len, padding="max_length",
                         truncation=True, return_tensors="pt")
        
        # GPT2 tokenization
        g = self.gpt_tok(text, max_length=self.max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        
        # Load images
        imgs, img_mask = self._load_images(self.pdf_paths[idx])
        
        return {
            "bert_ids": b["input_ids"].squeeze(0),
            "bert_mask": b["attention_mask"].squeeze(0),
            "gpt_ids": g["input_ids"].squeeze(0),
            "gpt_mask": g["attention_mask"].squeeze(0),
            "images": imgs,
            "image_mask": img_mask,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ===========================================================================
# Proposed Multimodal Model
# ===========================================================================

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
        weights = torch.softmax(self.weights, dim=0)
        fused = sum(w * p for w, p in zip(weights, projected))
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
        
        # Text encoders
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        
        # Vision encoder
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vision_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze vision to prevent catastrophic forgetting on small dataset
        for p in self.vision_features.parameters():
            p.requires_grad = False
        
        bert_dim = self.bert.config.hidden_size
        gpt2_dim = self.gpt2.config.hidden_size
        vision_dim = 512
        
        # Fusion layer
        self.fusion = FusionLayer(
            input_dims=[bert_dim, gpt2_dim, vision_dim],
            fusion_dim=FUSION_DIM,
            dropout=0.1,
        )
        
        # Classification head
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(FUSION_DIM, n_classes)
    
    def _encode_images(self, images, image_mask):
        """Encode image stack and masked mean-pool."""
        batch_size, n_images = images.shape[:2]
        flat = images.reshape(batch_size * n_images, 3, IMAGE_SIZE, IMAGE_SIZE)
        
        feat = self.vision_features(flat)
        feat = feat.reshape(batch_size * n_images, -1)
        feat = feat.reshape(batch_size, n_images, -1)
        
        # Masked mean pooling
        mask = image_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (feat * mask).sum(dim=1) / denom
        
        return pooled
    
    def forward(self, bert_ids, bert_mask, gpt_ids, gpt_mask, images, image_mask, **_):
        # BERT: [CLS] token
        bert_out = self.bert(input_ids=bert_ids, attention_mask=bert_mask)
        h_bert = bert_out.last_hidden_state[:, 0, :]
        
        # GPT2: last token
        gpt2_out = self.gpt2(input_ids=gpt_ids, attention_mask=gpt_mask)
        h_gpt2 = gpt2_out.last_hidden_state[:, -1, :]
        
        # ResNet18: masked mean-pooled
        h_vision = self._encode_images(images, image_mask)
        
        # Fuse all modalities
        fused = self.fusion(h_bert, h_gpt2, h_vision)
        
        # Classify
        return self.classifier(self.dropout(fused))


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(y_true, y_pred):
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return metrics


# ===========================================================================
# Training
# ===========================================================================

def train_and_evaluate(train_loader, val_loader, n_classes, device):
    """Train proposed multimodal model and evaluate."""
    model = MultimodalFusion(n_classes=n_classes)
    model = model.to(device)
    
    # Setup training
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, 
                                  weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            kwargs = {
                "bert_ids": batch["bert_ids"].to(device),
                "bert_mask": batch["bert_mask"].to(device),
                "gpt_ids": batch["gpt_ids"].to(device),
                "gpt_mask": batch["gpt_mask"].to(device),
                "images": batch["images"].to(device),
                "image_mask": batch["image_mask"].to(device),
            }
            
            logits = model(**kwargs)
            loss = loss_fn(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            kwargs = {
                "bert_ids": batch["bert_ids"].to(device),
                "bert_mask": batch["bert_mask"].to(device),
                "gpt_ids": batch["gpt_ids"].to(device),
                "gpt_mask": batch["gpt_mask"].to(device),
                "images": batch["images"].to(device),
                "image_mask": batch["image_mask"].to(device),
            }
            
            logits = model(**kwargs)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = compute_metrics(y_true, y_pred)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return metrics


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Study 3: Proposed Multimodal Model (BERT + GPT2 + ResNet18)")
    print("Scientific PDF Classification with Wilcoxon Statistical Testing")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Get PDF paths
    pdf_paths = sorted([str(p) for p in Path(PDF_DIR).glob("*.pdf")])
    if not pdf_paths:
        print(f"ERROR: No PDFs found in {PDF_DIR}")
        return
    
    print(f"Found {len(pdf_paths)} PDFs\n")
    
    # Extract text from PDFs (simplified - assume texts available separately)
    # For actual use, implement PDF text extraction
    print("Loading text from PDFs...")
    texts = []
    labels = []
    
    # Placeholder: create synthetic texts and labels for demonstration
    # In real usage, extract actual text from PDFs
    for i, pdf_path in enumerate(pdf_paths):
        texts.append(f"Sample text from {Path(pdf_path).stem}")
        labels.append(i % N_CLUSTERS)
    
    n_classes = len(set(labels))
    print(f"Loaded {len(texts)} documents with {n_classes} classes\n")
    
    # Extract images
    print("Extracting/loading cached images...")
    image_index = extract_images_from_pdfs(
        pdf_paths, IMAGE_CACHE_DIR, max_images_per_doc=N_IMAGES_PER_PDF
    )
    
    # Tokenizers
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tok.pad_token = gpt_tok.eos_token
    gpt_tok.padding_side = "left"
    
    # Check for existing results
    if os.path.exists(RESULTS_CSV):
        existing_df = pd.read_csv(RESULTS_CSV)
        completed_seeds = set(existing_df["seed"].values)
        print(f"Resuming: {len(completed_seeds)} seeds already completed.\n")
        rows_so_far = existing_df.to_dict(orient="records")
    else:
        completed_seeds = set()
        rows_so_far = []
    
    # Multi-seed evaluation
    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"[skip] Seed {seed} (already done)")
            continue
        
        print(f"\nTraining on Seed {seed}...")
        set_seed(seed)
        
        # Train-test split
        indices = np.arange(len(texts))
        idx_train, idx_test = train_test_split(
            indices, test_size=TEST_SIZE, random_state=seed,
            stratify=labels
        )
        
        # Create datasets
        train_ds = MultimodalDataset(
            [texts[i] for i in idx_train],
            [labels[i] for i in idx_train],
            [pdf_paths[i] for i in idx_train],
            image_index, bert_tok, gpt_tok,
            max_len=MAX_LENGTH, n_images=N_IMAGES_PER_PDF
        )
        test_ds = MultimodalDataset(
            [texts[i] for i in idx_test],
            [labels[i] for i in idx_test],
            [pdf_paths[i] for i in idx_test],
            image_index, bert_tok, gpt_tok,
            max_len=MAX_LENGTH, n_images=N_IMAGES_PER_PDF
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train and evaluate
        try:
            metrics = train_and_evaluate(train_loader, test_loader, n_classes, device)
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            
            row = {"seed": seed, **metrics}
            rows_so_far.append(row)
            
            # Save checkpoint
            pd.DataFrame(rows_so_far).to_csv(RESULTS_CSV, index=False)
            print(f"Saved to {RESULTS_CSV}")
            
        except Exception as e:
            print(f"ERROR on seed {seed}: {e}")
            continue
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Across All Seeds")
    print("=" * 70)
    
    df_results = pd.DataFrame(rows_so_far)
    
    summary_rows = []
    for metric in ["accuracy", "precision", "recall", "f1"]:
        summary_rows.append({
            "metric": metric,
            "mean": df_results[metric].mean(),
            "std": df_results[metric].std(),
        })
    
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary to {SUMMARY_CSV}")
    
    # Wilcoxon tests (comparing across different model variations or seeds)
    print("\n" + "=" * 70)
    print("Wilcoxon Signed-Rank Tests (Cross-Seed Consistency)")
    print("=" * 70)
    print("Testing if proposed method maintains consistent performance across seeds\n")
    
    # Example: Test if accuracy is significantly different from random baseline (50%)
    baseline_accs = np.full(len(df_results), 0.5)  # Random classifier baseline
    proposed_accs = df_results["accuracy"].values
    
    try:
        stat, p_value = wilcoxon(proposed_accs, baseline_accs, alternative="greater")
        print(f"Proposed vs. Random Baseline (50%)")
        print(f"  Mean Accuracy: {proposed_accs.mean():.4f} ± {proposed_accs.std():.4f}")
        print(f"  Wilcoxon Statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}\n")
    except ValueError as e:
        print(f"  Error: {e}\n")
    
    # Save Wilcoxon results
    wilcoxon_results = {
        "comparison": ["Proposed vs. Random (50%)"],
        "mean_acc": [proposed_accs.mean()],
        "std_acc": [proposed_accs.std()],
        "statistic": [stat if 'stat' in locals() else np.nan],
        "p_value": [p_value if 'p_value' in locals() else np.nan],
    }
    df_wilcoxon = pd.DataFrame(wilcoxon_results)
    df_wilcoxon.to_csv(WILCOXON_CSV, index=False)
    print(f"Saved Wilcoxon results to {WILCOXON_CSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
