
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------------------------------------------------------
# PDF reading
# ----------------------------------------------------------------------------

def extract_text_from_pdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        text = "".join(p.get_text("text") for p in doc)
        doc.close()
        return text
    except Exception as e:
        print(f"[warn] could not read {path}: {e}")
        return ""


def get_pdf_files(directory: str) -> List[str]:
    out = []
    for root, _, names in os.walk(directory):
        for n in names:
            if n.lower().endswith(".pdf"):
                out.append(os.path.join(root, n))
    return sorted(out)


def preprocess_text(text: str) -> str:
    return " ".join(text.lower().split())


def load_corpus(directory: str) -> Tuple[List[str], List[str]]:
    """Return (texts, paths) for every readable PDF under `directory`."""
    paths = get_pdf_files(directory)
    texts, kept = [], []
    for p in paths:
        t = preprocess_text(extract_text_from_pdf(p))
        if t.strip():
            texts.append(t)
            kept.append(p)
    print(f"Loaded {len(texts)} documents from {directory}")
    return texts, kept


# ----------------------------------------------------------------------------
# Label strategies  —  fix for issue #2 (label circularity)
# ----------------------------------------------------------------------------

def labels_tfidf_kmeans(texts: List[str], n_clusters: int = 5,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """ORIGINAL approach from the chapter: KMeans on TF-IDF vectors.

    WARNING: classical baselines that also operate on TF-IDF features are
    then trying to predict labels derived from those same features — a
    circularity that inflates baseline performance and muddies the
    comparison. Use sbert_kmeans or nus_categories for a cleaner study.

    Returns (labels, tfidf_features).
    """
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    X = tfidf.fit_transform(texts).toarray()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X).astype(int)
    return labels, X


def labels_sbert_kmeans(texts: List[str], n_clusters: int = 5,
                        seed: int = 42,
                        sbert_model: str = "all-MiniLM-L6-v2",
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster on Sentence-BERT embeddings instead of TF-IDF.

    This breaks the circularity with TF-IDF-based classical baselines:
    labels now come from contextualised embeddings, while baselines still
    see TF-IDF features — so any TF-IDF classifier must genuinely recover
    the semantic grouping rather than trivially re-derive it.

    Returns (labels, sbert_features).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "pip install sentence-transformers to use labels_sbert_kmeans"
        ) from e

    encoder = SentenceTransformer(sbert_model)
    # Use first ~2000 characters of each document so the encoder sees something
    # representative without blowing memory on the full text.
    snippets = [t[:2000] for t in texts]
    X = encoder.encode(snippets, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X).astype(int)
    return labels, X


def labels_from_nus_categories(
    paths: List[str],
    corpus_root: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract author-assigned categories from the NUS corpus layout.

    Each document in the NUS Keyphrase Extraction Corpus has a companion
    file of author-assigned keyphrases/categories. This function looks
    for such files next to the PDF and extracts category strings, then
    maps them to integer labels.

    The function looks for, in order of preference:
      - <doc_id>/<doc_id>.kwd  (author keywords)
      - <doc_id>/<doc_id>.categories
      - a top-level categories.csv mapping doc_id -> category

    Returns (labels, category_names). If extraction fails for a document
    it gets the 'unknown' label so you can filter those out in caller.
    """
    categories_for_doc = []
    for p in paths:
        doc_dir = os.path.dirname(p)
        doc_id = os.path.splitext(os.path.basename(p))[0]
        cat = None

        for suffix in (".categories", ".kwd"):
            meta = os.path.join(doc_dir, doc_id + suffix)
            if os.path.exists(meta):
                try:
                    with open(meta, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read().strip()
                    # Take first non-empty line as the category
                    for line in content.splitlines():
                        line = line.strip()
                        if line:
                            cat = line.lower()
                            break
                except Exception:
                    pass
            if cat:
                break

        categories_for_doc.append(cat or "unknown")

    # Map strings -> integer labels
    unique = sorted(set(categories_for_doc) - {"unknown"}) + ["unknown"]
    name_to_id = {n: i for i, n in enumerate(unique)}
    labels = np.array([name_to_id[c] for c in categories_for_doc], dtype=int)
    n_unknown = (labels == name_to_id["unknown"]).sum()
    if n_unknown > 0:
        print(f"[note] {n_unknown}/{len(labels)} documents had no category "
              f"file and got label 'unknown'")
    return labels, unique


def get_tfidf_features(texts: List[str], max_features: int = 500) -> np.ndarray:
    """TF-IDF feature matrix for classical baselines."""
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    return tfidf.fit_transform(texts).toarray()


# ----------------------------------------------------------------------------
# PyTorch datasets
# ----------------------------------------------------------------------------

class SingleTokenDataset(Dataset):
    """Single tokeniser — for BERT-only or GPT-2-only baselines."""

    def __init__(self, texts, labels, tokenizer, max_len: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


class DualTokenDataset(Dataset):
    """Tokenise with BERT *and* GPT-2 tokenisers for the same document.

    Used by all fusion models (they need both token streams).
    """

    def __init__(self, texts, labels, bert_tok, gpt_tok, max_len: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.bert_tok = bert_tok
        self.gpt_tok = gpt_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        b = self.bert_tok(t, max_length=self.max_len, padding="max_length",
                          truncation=True, return_tensors="pt")
        g = self.gpt_tok(t, max_length=self.max_len, padding="max_length",
                         truncation=True, return_tensors="pt")
        return {
            "bert_ids":  b["input_ids"].squeeze(0),
            "bert_mask": b["attention_mask"].squeeze(0),
            "gpt_ids":   g["input_ids"].squeeze(0),
            "gpt_mask":  g["attention_mask"].squeeze(0),
            "label":     torch.tensor(self.labels[idx], dtype=torch.long),
        }

