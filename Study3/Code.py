#!/usr/bin/env python3
import os
import sys
import subprocess
import warnings

for pkg in ["PyMuPDF", "transformers", "scikit-learn", "pandas", "seaborn", "matplotlib", "scipy"]:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"], check=False)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModel, GPT2Model,
                          get_linear_schedule_with_warmup)
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, brier_score_loss, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Path Resolution ────────────────────────────────────────────────────────
project_root = pdf_dir = None
for root, _, files in os.walk("/kaggle/input"):
    if "multimodal.py" in files:
        project_root = os.path.dirname(root)
    if root.rstrip("/").endswith("extracted_pdfs/data"):
        pdf_dir = root

if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

from pdf_hybrid.multimodal import extract_images_from_pdfs, MultimodalDataset
from pdf_hybrid.data import load_corpus, labels_sbert_kmeans
from pdf_hybrid.training import set_seed

# ── Global Config ──────────────────────────────────────────────────────────
OUT_DIR    = "/kaggle/working/thesis_outputs"
IMG_CACHE  = "/kaggle/working/pdf_images"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_CACHE, exist_ok=True)

SEEDS      = (42, 7, 123, 999, 2023, 8888, 7777)
N_CLUSTERS = 5
MAX_LEN    = 512
BATCH_SIZE = 16
MM_EPOCHS  = 5
N_IMAGES   = 4
FUSION_DIM = 128
DROPOUT    = 0.5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Metric Helpers ─────────────────────────────────────────────────────────
def compute_ece(y_true, y_prob, n_bins=10):
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    acc  = (pred == np.asarray(y_true)).astype(float)
    ece  = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = (conf >= lo) & (conf < hi)
        if m.sum() > 0:
            ece += m.sum() * abs(acc[m].mean() - conf[m].mean())
    return ece / len(y_true)

def compute_brier(y_true, y_prob, n_classes):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return float(np.mean([brier_score_loss(y_bin[:, c], y_prob[:, c])
                          for c in range(n_classes)]))

def compute_auc(y_true, y_prob, n_classes):
    try:
        return roc_auc_score(
            label_binarize(y_true, classes=list(range(n_classes))),
            y_prob, average="macro", multi_class="ovr")
    except Exception:
        return float("nan")

def metrics_from_pairs(pairs, n_classes):
    rows = []
    for y_t, y_p, y_s in pairs:
        acc = accuracy_score(y_t, y_p)
        _, _, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average="weighted", zero_division=0)
        rows.append({
            "acc":   acc,
            "f1":    f1,
            "auc":   compute_auc(y_t, y_s, n_classes)  if y_s is not None else np.nan,
            "brier": compute_brier(y_t, y_s, n_classes) if y_s is not None else np.nan,
            "ece":   compute_ece(y_t, y_s)               if y_s is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        out[f"{col}_mean"] = df[col].mean()
        out[f"{col}_std"]  = df[col].std()
    return out, df

# ── Neural-Model Architectural Components ──────────────────────────────────
class VisualAttentionPool(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, seq, mask):
        scores = self.attention(seq).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), float('-inf'))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (seq * weights).sum(dim=1)

def _make_head(in_dim, n_classes, dropout=0.5):
    return nn.Sequential(
        nn.LayerNorm(in_dim), nn.Dropout(dropout),
        nn.Linear(in_dim, in_dim // 2), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(in_dim // 2, n_classes))

# ─── Full proposed model ──────────────────────────────────────────────────────
class MultimodalBertGptResNet(nn.Module):
    def __init__(self, n_classes, fusion_dim=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        self.bert  = AutoModel.from_pretrained("bert-base-uncased")
        self.gpt2  = GPT2Model.from_pretrained("gpt2")
        h_bert = self.bert.config.hidden_size
        h_gpt = self.gpt2.config.hidden_size

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.vision_features = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.vision_features.parameters():
            p.requires_grad = False
        
        self.vis_pool = VisualAttentionPool(512)

        self.bert_proj = nn.Sequential(nn.Linear(h_bert, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.gpt_proj  = nn.Sequential(nn.Linear(h_gpt, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.vis_proj  = nn.Sequential(nn.Linear(512, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())

        self.mm_gate = nn.Parameter(torch.tensor([-4.0]))

        # Concatenation of 3 streams (128-d each) = 384-d fused representation
        self.classifier = _make_head(fusion_dim * 3, n_classes, dropout)

    def _encode_images(self, images, image_mask):
        B, N, C, H, W = images.shape
        with torch.no_grad():
            feat = self.vision_features(images.reshape(B * N, C, H, W)).reshape(B, N, -1)
        return self.vis_pool(feat, image_mask)

    def forward(self, bert_ids, bert_mask, gpt_ids, gpt_mask, images, image_mask, **_):
        bert_hs = self.bert(input_ids=bert_ids, attention_mask=bert_mask).last_hidden_state[:, 0, :]
        gpt_hs  = self.gpt2(input_ids=gpt_ids,  attention_mask=gpt_mask).last_hidden_state[:, -1, :]

        h_bert = self.bert_proj(bert_hs)
        h_gpt  = self.gpt_proj(gpt_hs)
        
        vis_hs = self._encode_images(images, image_mask)
        h_vis  = self.vis_proj(vis_hs)

        mm_g   = torch.sigmoid(self.mm_gate)
        h_vis_gated = mm_g * h_vis

        fused = torch.cat([h_bert, h_gpt, h_vis_gated], dim=1)

        return self.classifier(fused)

# ─── Ablation variants ────────────────────────────────────────────────────────
class AblationBertOnly(nn.Module):
    def __init__(self, n_classes, fusion_dim=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        self.bert  = AutoModel.from_pretrained("bert-base-uncased")
        h = self.bert.config.hidden_size
        self.proj  = nn.Sequential(nn.Linear(h, fusion_dim),
                                   nn.LayerNorm(fusion_dim), nn.GELU())
        self.classifier = _make_head(fusion_dim, n_classes, dropout)

    def forward(self, bert_ids, bert_mask, **_):
        hs = self.bert(input_ids=bert_ids, attention_mask=bert_mask).last_hidden_state[:, 0, :]
        return self.classifier(self.proj(hs))

class AblationGpt2Only(nn.Module):
    def __init__(self, n_classes, fusion_dim=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        self.gpt2     = GPT2Model.from_pretrained("gpt2")
        h = self.gpt2.config.hidden_size
        self.proj     = nn.Sequential(nn.Linear(h, fusion_dim),
                                      nn.LayerNorm(fusion_dim), nn.GELU())
        self.classifier = _make_head(fusion_dim, n_classes, dropout)

    def forward(self, gpt_ids, gpt_mask, **_):
        hs = self.gpt2(input_ids=gpt_ids, attention_mask=gpt_mask).last_hidden_state[:, -1, :]
        return self.classifier(self.proj(hs))

class AblationResNetOnly(nn.Module):
    def __init__(self, n_classes, fusion_dim=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.vision_features = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.vision_features.parameters():
            p.requires_grad = False
        self.vis_pool   = VisualAttentionPool(512)
        self.proj       = nn.Sequential(nn.Linear(512, fusion_dim),
                                        nn.LayerNorm(fusion_dim), nn.GELU())
        self.classifier = _make_head(fusion_dim, n_classes, dropout)

    def forward(self, images, image_mask, **_):
        B, N, C, H, W = images.shape
        with torch.no_grad():
            feat = self.vision_features(images.reshape(B * N, C, H, W)).reshape(B, N, -1)
        h = self.proj(self.vis_pool(feat, image_mask))
        return self.classifier(h)

class AblationBertGpt2Only(nn.Module):
    def __init__(self, n_classes, fusion_dim=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        self.bert   = AutoModel.from_pretrained("bert-base-uncased")
        self.gpt2   = GPT2Model.from_pretrained("gpt2")
        h_bert = self.bert.config.hidden_size
        h_gpt = self.gpt2.config.hidden_size
        self.bert_proj  = nn.Sequential(nn.Linear(h_bert, fusion_dim),
                                       nn.LayerNorm(fusion_dim), nn.GELU())
        self.gpt_proj  = nn.Sequential(nn.Linear(h_gpt, fusion_dim),
                                       nn.LayerNorm(fusion_dim), nn.GELU())
        self.classifier = _make_head(fusion_dim * 2, n_classes, dropout)

    def forward(self, bert_ids, bert_mask, gpt_ids, gpt_mask, **_):
        bert_hs = self.bert(input_ids=bert_ids, attention_mask=bert_mask).last_hidden_state[:, 0, :]
        gpt_hs  = self.gpt2(input_ids=gpt_ids,  attention_mask=gpt_mask).last_hidden_state[:, -1, :]
        h_bert  = self.bert_proj(bert_hs)
        h_gpt   = self.gpt_proj(gpt_hs)
        return self.classifier(torch.cat([h_bert, h_gpt], dim=1))

# ── LLRD Optimiser ────────────────────────────────────────────────────────
def llrd_optimizer(model, base_lr=5e-5, decay=0.9, wd=0.01):
    groups = []
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith(("bert.", "gpt2.")) and p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": base_lr, "weight_decay": wd})
    for attr, lr_scale in [("bert", 0.5), ("gpt2", 0.25)]:
        if not hasattr(model, attr): continue
        encoder = getattr(model, attr)
        layers  = list(getattr(encoder, "encoder", encoder).layer
                       if hasattr(getattr(encoder, "encoder", None), "layer")
                       else getattr(encoder, "h", []))
        for i, layer in enumerate(layers):
            groups.append({"params": list(layer.parameters()),
                           "lr": base_lr * lr_scale * (decay ** (len(layers) - 1 - i)),
                           "weight_decay": wd})
    return torch.optim.AdamW(groups) if groups else torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, weight_decay=wd)

# ── Load Data ─────────────────────────────────────────────────────────────
texts, paths = load_corpus(pdf_dir)
labels, _    = labels_sbert_kmeans(texts, n_clusters=N_CLUSTERS, seed=42)
n_classes    = int(np.max(labels)) + 1

bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt_tok = AutoTokenizer.from_pretrained("gpt2")
gpt_tok.pad_token = gpt_tok.eos_token
gpt_tok.padding_side = "left"
image_index = extract_images_from_pdfs(paths, cache_dir=IMG_CACHE,
                                       max_images_per_doc=N_IMAGES)

# ── Dataset Builders ───────────────────────────────────────────────────────
texts_arr  = np.asarray(texts, dtype=object)
labels_arr = np.asarray(labels)

def build_mm_loader(idx, shuffle):
    ds = MultimodalDataset(
        texts_arr[idx].tolist(), labels_arr[idx].tolist(),
        [paths[i] for i in idx], image_index,
        bert_tok, gpt_tok, max_len=MAX_LEN, n_images=N_IMAGES)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      pin_memory=True)

# ── Generic Evaluation Loops ───────────────────────────────────────────────
def eval_multimodal_model(model, loader):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch["bert_ids"], batch["bert_mask"],
                           batch["gpt_ids"],  batch["gpt_mask"],
                           batch["images"],   batch["image_mask"])
            probs  = torch.softmax(logits, 1).cpu().numpy()
            y_probs.append(probs)
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(batch["label"].cpu().numpy().tolist())
    return (np.array(y_true), np.array(y_pred), np.vstack(y_probs))

def eval_ablation_model(model, loader, model_type):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            if model_type == "bert":
                logits = model(bert_ids=batch["bert_ids"],
                               bert_mask=batch["bert_mask"])
            elif model_type == "gpt2":
                logits = model(gpt_ids=batch["gpt_ids"],
                               gpt_mask=batch["gpt_mask"])
            elif model_type == "resnet":
                logits = model(images=batch["images"],
                               image_mask=batch["image_mask"])
            elif model_type == "bert_gpt2":
                logits = model(bert_ids=batch["bert_ids"],
                               bert_mask=batch["bert_mask"],
                               gpt_ids=batch["gpt_ids"],
                               gpt_mask=batch["gpt_mask"])
            else:
                logits = model(batch["bert_ids"], batch["bert_mask"],
                               batch["gpt_ids"],  batch["gpt_mask"],
                               batch["images"],   batch["image_mask"])
            probs = torch.softmax(logits, 1).cpu().numpy()
            y_probs.append(probs)
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(batch["label"].cpu().numpy().tolist())
    return (np.array(y_true), np.array(y_pred), np.vstack(y_probs))

def train_nn_model(model, loader, optimizer, scheduler, loss_fn, epochs,
                   val_loader=None, track_loss=False):
    train_losses, val_losses = [], []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for batch in loader:
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["bert_ids"], batch["bert_mask"],
                           batch["gpt_ids"],  batch["gpt_mask"],
                           batch["images"],   batch["image_mask"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            ep_loss += loss.item()
        train_losses.append(ep_loss / len(loader))
        if val_loader is not None:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch  = {k: v.to(DEVICE) for k, v in batch.items()}
                    logits = model(batch["bert_ids"], batch["bert_mask"],
                                   batch["gpt_ids"],  batch["gpt_mask"],
                                   batch["images"],   batch["image_mask"])
                    vl += loss_fn(logits, batch["label"]).item()
            val_losses.append(vl / len(val_loader))

def train_ablation_nn(model, loader, model_type, optimizer, scheduler,
                      loss_fn, epochs):
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            if model_type == "bert":
                logits = model(bert_ids=batch["bert_ids"],
                               bert_mask=batch["bert_mask"])
            elif model_type == "gpt2":
                logits = model(gpt_ids=batch["gpt_ids"],
                               gpt_mask=batch["gpt_mask"])
            elif model_type == "resnet":
                logits = model(images=batch["images"],
                               image_mask=batch["image_mask"])
            elif model_type == "bert_gpt2":
                logits = model(bert_ids=batch["bert_ids"],
                               bert_mask=batch["bert_mask"],
                               gpt_ids=batch["gpt_ids"],
                               gpt_mask=batch["gpt_mask"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            ep_loss += loss.item()

# ── Proposed Multimodal (7 seeds) ────────────────────────────────────────
mm_name    = "Multimodal (BERT+GPT-2+ResNet-18)"
mm_pairs   = []
mm_per_seed = {}
loss_fn_mm = nn.CrossEntropyLoss(label_smoothing=0.1)

for seed in SEEDS:
    set_seed(seed)
    idx = np.arange(len(texts))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                      stratify=labels_arr, random_state=seed)
    dl_tr = build_mm_loader(idx_tr, shuffle=True)
    dl_te = build_mm_loader(idx_te, shuffle=False)

    model = MultimodalBertGptResNet(n_classes=n_classes).to(DEVICE)
    opt   = llrd_optimizer(model, base_lr=5e-5)
    total = len(dl_tr) * MM_EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    train_nn_model(model, dl_tr, opt, sched, loss_fn_mm, MM_EPOCHS)

    y_t, y_p, y_s = eval_multimodal_model(model, dl_te)
    acc  = accuracy_score(y_t, y_p)
    _, _, f1, _ = precision_recall_fscore_support(y_t, y_p, average="weighted",
                                                   zero_division=0)
    mm_pairs.append((y_t, y_p, y_s))
    mm_per_seed[seed] = {
        "acc":   acc, "f1": f1,
        "auc":   compute_auc(y_t, y_s, n_classes),
        "brier": compute_brier(y_t, y_s, n_classes),
        "ece":   compute_ece(y_t, y_s),
    }
    del model; torch.cuda.empty_cache()

mm_summary, _ = metrics_from_pairs(mm_pairs, n_classes)

# ── Ablation Study (7 seeds each) ────────────────────────────────────────
ablation_configs = [
    ("BERT only",          AblationBertOnly,    "bert"),
    ("GPT-2 only",         AblationGpt2Only,    "gpt2"),
    ("ResNet-18 only",     AblationResNetOnly,  "resnet"),
    ("BERT + GPT-2",       AblationBertGpt2Only,"bert_gpt2"),
]
ablation_results  = {}

for abl_name, ModelClass, mtype in ablation_configs:
    pairs = []
    for seed in SEEDS:
        set_seed(seed)
        idx = np.arange(len(texts))
        idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                          stratify=labels_arr, random_state=seed)
        dl_tr = build_mm_loader(idx_tr, shuffle=True)
        dl_te = build_mm_loader(idx_te, shuffle=False)
        model = ModelClass(n_classes=n_classes).to(DEVICE)
        opt   = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-5, weight_decay=0.01)
        total = len(dl_tr) * MM_EPOCHS
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        train_ablation_nn(model, dl_tr, mtype, opt, sched, loss_fn_mm, MM_EPOCHS)
        
        y_t, y_p, y_s = eval_ablation_model(model, dl_te, mtype)
        pairs.append((y_t, y_p, y_s))
        del model; torch.cuda.empty_cache()
        
    summary, _ = metrics_from_pairs(pairs, n_classes)
    ablation_results[abl_name]  = summary

ablation_results[mm_name] = mm_summary

# ── Five-Fold Cross-Validation (proposed model) ──────────────────────────
skf           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results  = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(texts_arr, labels_arr)):
    set_seed(42 + fold)
    dl_tr = build_mm_loader(tr_idx, shuffle=True)
    dl_va = build_mm_loader(va_idx, shuffle=False)
    
    model  = MultimodalBertGptResNet(n_classes=n_classes).to(DEVICE)
    opt    = llrd_optimizer(model, base_lr=5e-5)
    total  = len(dl_tr) * MM_EPOCHS
    sched  = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    
    train_nn_model(model, dl_tr, opt, sched, loss_fn_mm,
                             MM_EPOCHS, val_loader=dl_va, track_loss=True)
    y_t, y_p, y_s = eval_multimodal_model(model, dl_va)
    
    acc  = accuracy_score(y_t, y_p)
    _, _, f1, _ = precision_recall_fscore_support(y_t, y_p, average="weighted",
                                                   zero_division=0)
    auc_v   = compute_auc(y_t, y_s, n_classes)
    brier_v = compute_brier(y_t, y_s, n_classes)
    ece_v   = compute_ece(y_t, y_s)
    fold_results.append({"fold": fold+1, "acc": acc, "f1": f1,
                          "auc": auc_v, "brier": brier_v, "ece": ece_v})
    del model; torch.cuda.empty_cache()
