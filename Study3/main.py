
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .data import (
    load_corpus, get_tfidf_features,
    labels_tfidf_kmeans, labels_sbert_kmeans, labels_from_nus_categories,
    DualTokenDataset, SingleTokenDataset, ChunkedDualTokenDataset,
)
from .models import (
    BertClassifier, GPT2Classifier,
    BertGPT2Fusion, BertGPT2CNNFusion,
    FusionOp,
)
from .experiments import (
    run_classical_baselines, run_transformer_multiseed,
    compare_models, build_results_table,
)


def _get_labels(texts, paths, strategy: str, n_clusters: int, seed: int):
    if strategy == "tfidf_kmeans":
        print("[labels] TF-IDF + KMeans (original; has circularity concern)")
        labels, _ = labels_tfidf_kmeans(texts, n_clusters=n_clusters, seed=seed)
    elif strategy == "sbert_kmeans":
        print("[labels] Sentence-BERT + KMeans (breaks TF-IDF circularity)")
        labels, _ = labels_sbert_kmeans(texts, n_clusters=n_clusters, seed=seed)
    elif strategy == "nus_categories":
        print("[labels] NUS author-assigned categories (true external labels)")
        labels, names = labels_from_nus_categories(paths)
        print(f"  found {len(set(labels))} distinct categories: {names}")
    else:
        raise ValueError(f"unknown label strategy: {strategy}")
    return labels


def run_everything(
    pdf_directory: str,
    label_strategy: str = "tfidf_kmeans",    
    n_clusters: int = 5,
    seeds=(42, 7, 123, 2024, 1),             
    epochs: int = 3,
    batch_size: int = 16,
    max_len: int = 128,                     
    run_all_fusion_ops: bool = True,         
    mode: str = "all",
) -> Dict:
    """One-call orchestrator. Runs everything consistent with the fixes."""
    from transformers import BertTokenizer, GPT2Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Load corpus ---
    texts, paths = load_corpus(pdf_directory)

    labels = _get_labels(texts, paths, label_strategy, n_clusters, seeds[0])
    n_classes = int(np.max(labels)) + 1
    print(f"n_classes = {n_classes}, n_docs = {len(texts)}")

    # --- 3. Tokenisers (GPT-2 needs EOS as pad, left-padding) ---
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    gpt_tok  = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tok.pad_token = gpt_tok.eos_token
    gpt_tok.padding_side = "left"

    results = {"classical": {}, "transformer": {}, "stats": []: None}

    # ---------------------------------------------------------------------
    # Classical baselines
    # ---------------------------------------------------------------------
    if mode in ("classical", "all"):
        print("\n" + "#" * 70)
        print("# Classical / ensemble baselines (multi-seed)")
        print("#" * 70)
        X = get_tfidf_features(texts)
        results["classical"] = run_classical_baselines(X, labels, seeds=seeds)

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    if mode in ("transformers", "all"):
        print("\n" + "#" * 70)
        print("# Transformer and fusion models (multi-seed)")
        print("#" * 70)

        # 1. BERT only (missing baseline)
        results["transformer"]["BERT-only"] = run_transformer_multiseed(
            "BERT-only",
            lambda: BertClassifier(n_classes),
            texts, labels, bert_tok, gpt_tok,
            dataset_cls=SingleTokenDataset,
            dataset_kwargs={"tokenizer": bert_tok, "max_len": max_len},
            seeds=seeds, epochs=epochs, batch_size=batch_size, device=device,
        )

        # 2. GPT-2 only (missing baseline)
        results["transformer"]["GPT2-only"] = run_transformer_multiseed(
            "GPT2-only",
            lambda: GPT2Classifier(n_classes),
            texts, labels, bert_tok, gpt_tok,
            dataset_cls=SingleTokenDataset,
            dataset_kwargs={"tokenizer": gpt_tok, "max_len": max_len},
            seeds=seeds, epochs=epochs, batch_size=batch_size, device=device,
        )

        if run_all_fusion_ops:
            for op, tag in [
                (FusionOp.CONCAT,          "Fusion (Concat)"),
                (FusionOp.CROSS_ATTENTION, "Fusion (CrossAttn)"),
                (FusionOp.GATED,           "Fusion (Gated)"),
            ]:
                results["transformer"][tag] = run_transformer_multiseed(
                    tag,
                    lambda op=op: BertGPT2Fusion(n_classes, fusion_op=op,
                                                 use_sequence_cross_attn=(op == FusionOp.CROSS_ATTENTION)),
                    texts, labels, bert_tok, gpt_tok,
                    dataset_kwargs={"max_len": max_len},
                    seeds=seeds, epochs=epochs, batch_size=batch_size, device=device,
                )

        # 7. Triple hybrid with CNN
        results["transformer"]["Fusion+CNN"] = run_transformer_multiseed(
            "Fusion+CNN",
            lambda: BertGPT2CNNFusion(n_classes),
            texts, labels, bert_tok, gpt_tok,
            dataset_kwargs={"max_len": max_len},
            seeds=seeds, epochs=epochs, batch_size=batch_size, device=device,
        )

    
    # ---------------------------------------------------------------------
    # Statistical tests 
    # ---------------------------------------------------------------------
    if mode in ("stats", "all") and results["classical"] and results["transformer"]:
        print("\n" + "#" * 70)
        print("# Pairwise statistical comparisons")
        print("#" * 70)

        # Best classical vs best transformer on seed=seeds[0]
        best_cls = max(results["classical"].items(),
                       key=lambda kv: kv[1]["accuracy_mean"])
        best_tr  = max(results["transformer"].items(),
                       key=lambda kv: kv[1]["accuracy_mean"])

        pair = compare_models(best_cls[0], best_cls[1],
                              best_tr[0],  best_tr[1],
                              seed_for_test=seeds[0])
        results["stats"].append(pair)
        print(f"\n{pair['pair']}  (seed {pair['seed']})")
        print(f"  McNemar: b={pair['mcnemar']['b']} c={pair['mcnemar']['c']} "
              f"p={pair['mcnemar']['p_value']:.4f} ({pair['mcnemar']['note']})")
        b = pair["bootstrap"]
        print(f"  Paired bootstrap: point={b['point_diff']:+.4f} "
              f"CI=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  "
              f"{'(significant)' if b['significant_at_alpha'] else '(not significant)'}")

        # Fusion (WeightedSum) vs BERT-only — does the fusion actually help?
        if "Fusion (WeightedSum)" in results["transformer"] and \
           "BERT-only" in results["transformer"]:
            pair = compare_models(
                "BERT-only", results["transformer"]["BERT-only"],
                "Fusion (WeightedSum)", results["transformer"]["Fusion (WeightedSum)"],
                seed_for_test=seeds[0],
            )
            results["stats"].append(pair)
            print(f"\n{pair['pair']}  (seed {pair['seed']})")
            print(f"  McNemar: b={pair['mcnemar']['b']} c={pair['mcnemar']['c']} "
                  f"p={pair['mcnemar']['p_value']:.4f}")
            b = pair["bootstrap"]
            print(f"  Paired bootstrap: point={b['point_diff']:+.4f} "
                  f"CI=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  "
                  f"{'(significant)' if b['significant_at_alpha'] else '(not significant)'}")

    