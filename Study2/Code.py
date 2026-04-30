

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import (
    BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer,
)


# ===========================================================================
# Configuration
# ===========================================================================

# Same seven seeds as Study 3, for methodological consistency
SEEDS = (42, 7, 123, 999, 2023, 8888, 7777)

# Adjust these paths for your Kaggle notebook
EXCEL_PATH = "/kaggle/input/breast-lesions-usg/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
IMG_DIR    = "/kaggle/input/breast-lesions-usg/images/BrEaST-Lesions_USG-images_and_masks"
SHEET_NAME = "BrEaST-Lesions-USG clinical dat"

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PER_SEED_CSV = f"{OUTPUT_DIR}/study2_per_seed_results.csv"

EPOCHS = 5
BATCH_SIZE = 16
MAX_LENGTH = 128
LR = 2e-5

MODELS_TO_RUN = ("BERT-only", "GPT2-only", "BERT+GPT2", "Multimodal")
PROPOSED = "Multimodal"


# ===========================================================================
# Calibration metric: Expected Calibration Error
# ===========================================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Standard binned ECE for binary classification.

    Lower is better; 0 means perfectly calibrated.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1
                                   else y_prob < hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = y_prob[in_bin].mean()
        ece += (in_bin.sum() / len(y_prob)) * abs(bin_acc - bin_conf)
    return ece


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
# Image transform (matches the original Study 2 code)
# ===========================================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ===========================================================================
# Datasets — text-only and text+image
# ===========================================================================

class TextOnlyDataset(Dataset):
    """For BERT-only, GPT2-only, BERT+GPT2 (no image)."""

    def __init__(self, df, bert_tok, gpt_tok, max_length=128):
        self.df = df.reset_index(drop=True)
        self.bert_tok = bert_tok
        self.gpt_tok = gpt_tok
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"{row['Symptoms']} {row['Diagnosis']}"

        b = self.bert_tok.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding="max_length",
            return_attention_mask=True, return_tensors="pt", truncation=True,
        )
        g = self.gpt_tok.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            padding="max_length", return_attention_mask=True,
            return_tensors="pt", truncation=True,
        )
        label = 1 if row["Classification"] == "malignant" else 0
        return {
            "bert_input_ids":      b["input_ids"].squeeze(0),
            "bert_attention_mask": b["attention_mask"].squeeze(0),
            "gpt_input_ids":       g["input_ids"].squeeze(0),
            "gpt_attention_mask":  g["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class TextImageDataset(Dataset):
    """For the multimodal fusion. Same structure as the original
    CustomTextImageDataset in your Study 2 code."""

    def __init__(self, df, img_dir, bert_tok, gpt_tok, transform,
                 max_length=128):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.bert_tok = bert_tok
        self.gpt_tok = gpt_tok
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"{row['Symptoms']} {row['Diagnosis']}"

        b = self.bert_tok.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding="max_length",
            return_attention_mask=True, return_tensors="pt", truncation=True,
        )
        g = self.gpt_tok.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            padding="max_length", return_attention_mask=True,
            return_tensors="pt", truncation=True,
        )

        img_path = os.path.join(self.img_dir, row["Image_filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = 1 if row["Classification"] == "malignant" else 0
        return {
            "bert_input_ids":      b["input_ids"].squeeze(0),
            "bert_attention_mask": b["attention_mask"].squeeze(0),
            "gpt_input_ids":       g["input_ids"].squeeze(0),
            "gpt_attention_mask":  g["attention_mask"].squeeze(0),
            "image":               image,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ===========================================================================
# Models
# ===========================================================================

class BertOnlyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.bert.config.hidden_size, num_classes),
        )

    def forward(self, bert_input_ids, bert_attention_mask, **_):
        out = self.bert(input_ids=bert_input_ids,
                        attention_mask=bert_attention_mask)
        return self.classifier(out.pooler_output)


class GPT2OnlyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained("gpt2")
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.gpt.config.n_embd, num_classes),
        )

    def forward(self, gpt_input_ids, gpt_attention_mask, **_):
        out = self.gpt(input_ids=gpt_input_ids,
                       attention_mask=gpt_attention_mask)
        # Last-token pooling, matching your original Study 2 design
        return self.classifier(out.last_hidden_state[:, -1, :])


class BertGPT2Fusion(nn.Module):
    """Text-only fusion: BERT [CLS] + GPT-2 last token, projected to 128 each
    and concatenated to a 256-dim vector before classification.

    Same architecture as the multimodal model but with the ResNet stream
    removed, so the comparison isolates the contribution of the visual
    stream specifically."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.gpt = GPT2Model.from_pretrained("gpt2")
        self.bert_proj = nn.Linear(self.bert.config.hidden_size, 128)
        self.gpt_proj  = nn.Linear(self.gpt.config.n_embd, 128)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 2, num_classes),
        )

    def forward(self, bert_input_ids, bert_attention_mask,
                gpt_input_ids, gpt_attention_mask, **_):
        b = self.bert(input_ids=bert_input_ids,
                      attention_mask=bert_attention_mask)
        bf = self.bert_proj(b.pooler_output)
        g = self.gpt(input_ids=gpt_input_ids,
                     attention_mask=gpt_attention_mask)
        gf = self.gpt_proj(g.last_hidden_state[:, -1, :])
        combined = torch.cat((bf, gf), dim=1)
        return self.classifier(combined)


class MultimodalClassifier(nn.Module):
    """The proposed model: BERT + GPT-2 + ResNet-18, mid-level concatenation
    fusion to a 384-dim vector before classification.

    This is identical to the BERTGPT2ImageClassifier in your original code.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.gpt = GPT2Model.from_pretrained("gpt2")
        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)
        self.bert_proj = nn.Linear(self.bert.config.hidden_size, 128)
        self.gpt_proj  = nn.Linear(self.gpt.config.n_embd, 128)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3, num_classes),
        )

    def forward(self, bert_input_ids, bert_attention_mask,
                gpt_input_ids, gpt_attention_mask, image, **_):
        b = self.bert(input_ids=bert_input_ids,
                      attention_mask=bert_attention_mask)
        bf = self.bert_proj(b.pooler_output)
        g = self.gpt(input_ids=gpt_input_ids,
                     attention_mask=gpt_attention_mask)
        gf = self.gpt_proj(g.last_hidden_state[:, -1, :])
        imf = self.resnet(image)
        combined = torch.cat((bf, gf, imf), dim=1)
        return self.classifier(combined)


# ===========================================================================
# Train and evaluate one model on one seed
# ===========================================================================

def train_and_evaluate(model_name, train_loader, val_loader, device):
    """Build the named model, train for EPOCHS, then evaluate on val_loader.
    Returns a metrics dict with accuracy, roc_auc, brier_score, ece.
    """
    if model_name == "BERT-only":
        model = BertOnlyClassifier(num_classes=2)
        needs_image = False
    elif model_name == "GPT2-only":
        model = GPT2OnlyClassifier(num_classes=2)
        needs_image = False
    elif model_name == "BERT+GPT2":
        model = BertGPT2Fusion(num_classes=2)
        needs_image = False
    elif model_name == "Multimodal":
        model = MultimodalClassifier(num_classes=2)
        needs_image = True
    else:
        raise ValueError(model_name)

    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    for ep in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimiser.zero_grad()
            kwargs = {
                "bert_input_ids":      batch["bert_input_ids"].to(device),
                "bert_attention_mask": batch["bert_attention_mask"].to(device),
                "gpt_input_ids":       batch["gpt_input_ids"].to(device),
                "gpt_attention_mask":  batch["gpt_attention_mask"].to(device),
            }
            if needs_image:
                kwargs["image"] = batch["image"].to(device)
            logits = model(**kwargs)
            loss = loss_fn(logits, batch["label"].to(device))
            loss.backward()
            optimiser.step()

    # Evaluate
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            kwargs = {
                "bert_input_ids":      batch["bert_input_ids"].to(device),
                "bert_attention_mask": batch["bert_attention_mask"].to(device),
                "gpt_input_ids":       batch["gpt_input_ids"].to(device),
                "gpt_attention_mask":  batch["gpt_attention_mask"].to(device),
            }
            if needs_image:
                kwargs["image"] = batch["image"].to(device)
            logits = model(**kwargs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            y_prob.extend(probs.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "roc_auc":     roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece":         expected_calibration_error(y_true, y_prob),
    }

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    return metrics


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Study 2 (USG) multi-seed evaluation with calibration + Wilcoxon")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Models: {MODELS_TO_RUN}")
    print(f"Epochs: {EPOCHS}, batch size: {BATCH_SIZE}, lr: {LR}")
    print()

    # Load data once
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    df = df.dropna(subset=["Image_filename", "Classification"])
    df = df[df["Classification"].isin(["benign", "malignant"])]
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} cases ({(df['Classification'] == 'malignant').sum()} "
          f"malignant, {(df['Classification'] == 'benign').sum()} benign)\n")

    # Tokenisers (load once, reuse across seeds)
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    gpt_tok  = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tok.pad_token = gpt_tok.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Resume from existing CSV if present
    if os.path.exists(PER_SEED_CSV):
        existing = pd.read_csv(PER_SEED_CSV)
        completed = set(zip(existing["seed"], existing["model"]))
        print(f"Resuming: {len(completed)} (seed, model) combos already done.")
    else:
        existing = None
        completed = set()

    rows_so_far = ([] if existing is None
                   else existing.to_dict(orient="records"))

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)

        # ONE stratified split shared by all four models, so paired
        # comparison via Wilcoxon is valid.
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=seed,
            stratify=df["Classification"],
        )

        # Two dataset variants per seed (text-only and text+image)
        train_text_ds = TextOnlyDataset(train_df, bert_tok, gpt_tok, MAX_LENGTH)
        val_text_ds   = TextOnlyDataset(val_df,   bert_tok, gpt_tok, MAX_LENGTH)
        train_text_loader = DataLoader(train_text_ds, batch_size=BATCH_SIZE,
                                       shuffle=True)
        val_text_loader   = DataLoader(val_text_ds,   batch_size=BATCH_SIZE,
                                       shuffle=False)

        train_mm_ds = TextImageDataset(train_df, IMG_DIR, bert_tok, gpt_tok,
                                       image_transform, MAX_LENGTH)
        val_mm_ds   = TextImageDataset(val_df,   IMG_DIR, bert_tok, gpt_tok,
                                       image_transform, MAX_LENGTH)
        train_mm_loader = DataLoader(train_mm_ds, batch_size=BATCH_SIZE,
                                     shuffle=True)
        val_mm_loader   = DataLoader(val_mm_ds,   batch_size=BATCH_SIZE,
                                     shuffle=False)

        for model_name in MODELS_TO_RUN:
            if (seed, model_name) in completed:
                print(f"  [skip] {model_name} (already done)")
                continue

            train_loader = (train_mm_loader if model_name == "Multimodal"
                            else train_text_loader)
            val_loader   = (val_mm_loader   if model_name == "Multimodal"
                            else val_text_loader)

            print(f"  Training {model_name}...", end=" ", flush=True)
            try:
                metrics = train_and_evaluate(
                    model_name, train_loader, val_loader, device,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            print(f"acc={metrics['accuracy']:.4f}  "
                  f"roc={metrics['roc_auc']:.4f}  "
                  f"brier={metrics['brier_score']:.4f}  "
                  f"ece={metrics['ece']:.4f}")

            row = {"seed": seed, "model": model_name, **metrics}
            rows_so_far.append(row)

            # Checkpoint to CSV after every (seed, model)
            pd.DataFrame(rows_so_far).to_csv(PER_SEED_CSV, index=False)

    # ======================================================================
    # Summary
    # ======================================================================
    df_per_seed = pd.DataFrame(rows_so_far)
    summary_rows = []
    for name in df_per_seed["model"].unique():
        sub = df_per_seed[df_per_seed["model"] == name]
        summary_rows.append({
            "model":          name,
            "accuracy_mean":  sub["accuracy"].mean(),
            "accuracy_std":   sub["accuracy"].std(),
            "roc_auc_mean":   sub["roc_auc"].mean(),
            "roc_auc_std":    sub["roc_auc"].std(),
            "brier_mean":     sub["brier_score"].mean(),
            "brier_std":      sub["brier_score"].std(),
            "ece_mean":       sub["ece"].mean(),
            "ece_std":        sub["ece"].std(),
        })
    df_summary = (pd.DataFrame(summary_rows)
                  .sort_values("accuracy_mean", ascending=False)
                  .reset_index(drop=True))

    print("\n" + "=" * 70)
    print(f"Summary across {len(SEEDS)} seeds")
    print("=" * 70)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200):
        print(df_summary.to_string(index=False))
    df_summary.to_csv(f"{OUTPUT_DIR}/study2_summary.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR}/study2_summary.csv")

    # ======================================================================
    # Wilcoxon: Multimodal vs each baseline (paired by seed)
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"Wilcoxon signed-rank tests ({PROPOSED} vs each baseline)")
    print("alternative='greater' tests whether Multimodal > baseline")
    print("=" * 70)

    proposed_accs = (df_per_seed[df_per_seed["model"] == PROPOSED]
                     .sort_values("seed")["accuracy"].values)

    wilcoxon_rows = []
    for name in MODELS_TO_RUN:
        if name == PROPOSED:
            continue
        baseline_accs = (df_per_seed[df_per_seed["model"] == name]
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
                                   alternative="greater",
                                   zero_method="wilcox")
                note = ""
            except ValueError as e:
                stat, p, note = np.nan, np.nan, f"wilcoxon error: {e}"

        wilcoxon_rows.append({
            "baseline":             name,
            "multimodal_mean_acc":  proposed_accs.mean(),
            "baseline_mean_acc":    baseline_accs.mean(),
            "diff":                 proposed_accs.mean() - baseline_accs.mean(),
            "wilcoxon_statistic":   stat,
            "p_value":              p,
            "significant_at_005":   (not np.isnan(p)) and (p < 0.05),
            "note":                 note,
        })

    df_wilcoxon = pd.DataFrame(wilcoxon_rows)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200):
        print(df_wilcoxon.to_string(index=False))
    df_wilcoxon.to_csv(f"{OUTPUT_DIR}/study2_wilcoxon.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR}/study2_wilcoxon.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
