
from __future__ import annotations

import copy
import warnings
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader

from .data import (
    DualTokenDataset, SingleTokenDataset, ChunkedDualTokenDataset,
)
from .models import (
    BertClassifier, GPT2Classifier,
    BertGPT2Fusion, BertGPT2CNNFusion, HierarchicalBertGPT2Fusion,
    FusionOp,
)
from .training import (
    fit_transformer, evaluate, set_seed,
    mcnemar_test, paired_bootstrap_ci, summarise_multiseed,
)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Classical / ensemble baselines
# ----------------------------------------------------------------------------

def _make_classical_models(seed: int = 42):
    """All 9 sklearn estimators referenced in the chapter.

    xgboost/lightgbm/catboost are imported lazily so the script works
    even if they're not installed (they'll just be skipped).
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "SVM":                SVC(kernel="linear", random_state=seed),
        "DecisionTree":       DecisionTreeClassifier(random_state=seed),
        "KNN":                KNeighborsClassifier(n_neighbors=5),
        "GradientBoosting":   GradientBoostingClassifier(random_state=seed),
        "AdaBoost":           AdaBoostClassifier(random_state=seed, algorithm="SAMME"),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(random_state=seed,
                                          use_label_encoder=False,
                                          eval_metric="mlogloss",
                                          verbosity=0)
    except ImportError:
        print("[note] xgboost not installed, skipping")
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(random_state=seed, verbose=-1)
    except ImportError:
        print("[note] lightgbm not installed, skipping")
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(random_state=seed, verbose=0)
    except ImportError:
        print("[note] catboost not installed, skipping")
    return models


def run_classical_baselines(X: np.ndarray, y: np.ndarray,
                            seeds: Sequence[int] = (42, 7, 123, 2024, 1),
                            test_size: float = 0.2) -> Dict:
    """Run every classical/ensemble model across multiple seeds.

    Returns {model_name: {accuracy_mean, accuracy_std, predictions_per_seed, ...}}.
    """
    y = np.asarray(y)
    per_model = {name: [] for name in _make_classical_models(seeds[0]).keys()}
    preds_per_model = {name: {} for name in per_model}
    y_test_per_seed = {}

    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed,
        )
        y_test_per_seed[seed] = y_te
        for name, clf in _make_classical_models(seed).items():
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            acc = accuracy_score(y_te, pred)
            p, r, f1, _ = precision_recall_fscore_support(
                y_te, pred, average="weighted", zero_division=0
            )
            per_model[name].append({
                "accuracy": acc, "precision": p, "recall": r, "f1": f1, "seed": seed,
            })
            preds_per_model[name][seed] = pred

    summary = {}
    for name, runs in per_model.items():
        if not runs:
            continue
        summary[name] = {
            **summarise_multiseed(runs),
            "per_seed_accuracy": [r["accuracy"] for r in runs],
            "predictions_per_seed": preds_per_model[name],
            "y_test_per_seed": y_test_per_seed,
        }
    return summary


# ----------------------------------------------------------------------------
# Transformer multi-seed runs  —  fix for issue #3
# ----------------------------------------------------------------------------

def _make_dataloaders(texts, labels, bert_tok, gpt_tok, idx_train, idx_test,
                      dataset_cls, batch_size=16, **ds_kwargs):
    arr = np.asarray(texts, dtype=object)
    lab = np.asarray(labels)

    # Build a single-tokenizer dataset or a dual/chunked one depending on class
    def build(idx):
        kw = dict(ds_kwargs)
        if dataset_cls is SingleTokenDataset:
            # Which tokenizer? Caller passes it in ds_kwargs["tokenizer"]
            return dataset_cls(arr[idx].tolist(), lab[idx], **kw)
        return dataset_cls(arr[idx].tolist(), lab[idx],
                           bert_tok=bert_tok, gpt_tok=gpt_tok, **kw)

    ds_tr, ds_te = build(idx_train), build(idx_test)
    return (DataLoader(ds_tr, batch_size=batch_size, shuffle=True),
            DataLoader(ds_te, batch_size=batch_size, shuffle=False))


def run_transformer_multiseed(
    name: str,
    model_factory: Callable,
    texts: List[str], labels: np.ndarray,
    bert_tok, gpt_tok,
    dataset_cls=DualTokenDataset,
    dataset_kwargs: Optional[Dict] = None,
    seeds: Sequence[int] = (42, 7, 123, 2024, 1),
    test_size: float = 0.2,
    epochs: int = 3, batch_size: int = 16,
    lr: float = 5e-5, weight_decay: float = 0.01,
    device: Optional[torch.device] = None,
) -> Dict:
    """Train+evaluate one model family across `seeds`. Returns aggregated metrics."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_kwargs = dataset_kwargs or {}
    labels = np.asarray(labels)

    per_seed = []
    preds_per_seed = {}
    y_test_per_seed = {}

    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")

    for seed in seeds:
        set_seed(seed)
        idx = np.arange(len(texts))
        idx_tr, idx_te = train_test_split(
            idx, test_size=test_size, stratify=labels, random_state=seed,
        )
        y_test_per_seed[seed] = labels[idx_te]

        dl_tr, dl_te = _make_dataloaders(
            texts, labels, bert_tok, gpt_tok, idx_tr, idx_te,
            dataset_cls=dataset_cls, batch_size=batch_size, **dataset_kwargs,
        )

        model = model_factory()
        m = fit_transformer(model, dl_tr, dl_te, device,
                            epochs=epochs, lr=lr, weight_decay=weight_decay,
                            verbose=False)
        per_seed.append({k: m[k] for k in ("accuracy", "precision", "recall", "f1")})
        preds_per_seed[seed] = m["y_pred"]
        print(f"  seed {seed:>4}: acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

        del model
        torch.cuda.empty_cache()

    summary = summarise_multiseed(per_seed)
    print(f"  mean acc = {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    return {
        **summary,
        "per_seed": per_seed,
        "predictions_per_seed": preds_per_seed,
        "y_test_per_seed": y_test_per_seed,
    }


# ----------------------------------------------------------------------------
# Pairwise statistical comparison
# ----------------------------------------------------------------------------

def compare_models(name_a: str, results_a: Dict,
                   name_b: str, results_b: Dict,
                   seed_for_test: int = 42) -> Dict:
    """Run McNemar + paired bootstrap CI for a specific seed's test set.

    Only works when both models were evaluated on the *same* train/test split.
    Since run_transformer_multiseed and run_classical_baselines both use
    train_test_split with the seed as random_state, results from the same
    seed are directly comparable.
    """
    y_true = results_a["y_test_per_seed"][seed_for_test]
    pred_a = results_a["predictions_per_seed"][seed_for_test]
    pred_b = results_b["predictions_per_seed"][seed_for_test]

    mc  = mcnemar_test(y_true, pred_a, pred_b)
    boot = paired_bootstrap_ci(y_true, pred_a, pred_b)
    return {
        "pair": f"{name_a} vs {name_b}",
        "seed": seed_for_test,
        "mcnemar": mc,
        "bootstrap": boot,
    }
