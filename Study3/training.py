
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)


# ----------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimiser, loss_fn, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimiser.zero_grad()
        logits = model(**{k: v for k, v in batch.items() if k != "label"})
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        optimiser.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k: v for k, v in batch.items() if k != "label"})
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(batch["label"].cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "y_true": np.array(y_true), "y_pred": np.array(y_pred),
    }


def fit_transformer(model, train_loader, test_loader, device,
                    epochs: int = 3, lr: float = 5e-5,
                    weight_decay: float = 0.01, verbose: bool = True) -> Dict:
    """Fit on train_loader, evaluate on test_loader, return metrics."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        tl = train_one_epoch(model, train_loader, opt, loss_fn, device)
        if verbose:
            print(f"    epoch {ep + 1}: loss = {tl:.4f}")
    return evaluate(model, test_loader, device)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def mcnemar_test(y_true: Sequence[int],
                 pred_a: Sequence[int],
                 pred_b: Sequence[int],
                 exact: bool = True) -> Dict:
    """McNemar's test for paired classifier predictions on the same test set.

    Counts discordant pairs:
      b = model A correct, model B wrong
      c = model A wrong, model B correct

    Exact (binomial) test recommended when b + c < 25, chi-square otherwise.
    Returns {'b': int, 'c': int, 'statistic': float, 'p_value': float}.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)

    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c

    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "note": "no discordant pairs"}

    if exact and n < 25:
        # Exact two-sided binomial test (p=0.5)
        from math import comb
        k = min(b, c)
        p = 0.0
        for i in range(k + 1):
            p += comb(n, i) * (0.5 ** n)
        p_value = min(1.0, 2.0 * p)
        return {"b": b, "c": c, "statistic": float(k),
                "p_value": float(p_value), "note": "exact binomial"}

    # Chi-square with continuity correction
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    from math import erf, sqrt
    # Chi-square df=1 survival: P(X > stat) = erfc(sqrt(stat)/sqrt(2))
    # Implemented via erf to avoid importing scipy:
    p_value = 1 - erf(np.sqrt(stat / 2.0))
    return {"b": b, "c": c, "statistic": float(stat),
            "p_value": float(p_value), "note": "chi-square w/ continuity"}


def paired_bootstrap_ci(y_true: Sequence[int],
                        pred_a: Sequence[int],
                        pred_b: Sequence[int],
                        n_boot: int = 10000,
                        alpha: float = 0.05,
                        seed: int = 42) -> Dict:
    """Paired bootstrap CI for (accuracy_A - accuracy_B).

    Resamples test indices with replacement n_boot times, computes the
    accuracy difference on each resample, reports the (alpha/2, 1-alpha/2)
    percentile interval.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (pred_a[idx] == y_true[idx]).mean() - (pred_b[idx] == y_true[idx]).mean()

    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    point = float((pred_a == y_true).mean() - (pred_b == y_true).mean())
    return {
        "point_diff": point,
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
        "n_boot": n_boot,
        "significant_at_alpha": (lo > 0) or (hi < 0),
    }


def summarise_multiseed(metrics: List[Dict]) -> Dict:
    """Given a list of per-seed metric dicts, return mean/std for each scalar key."""
    keys = ["accuracy", "precision", "recall", "f1"]
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics if k in m]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
    return out


# ----------------------------------------------------------------------------
# Seeding
# ----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
