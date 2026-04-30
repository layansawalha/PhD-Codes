# Study 3: Multimodal Scientific PDF Classification

This directory contains the codebase for evaluating the multimodal fusion of BERT, GPT-2, and a ResNet-18 vision encoder on full-text scientific PDFs[cite: 13, 20]. 

## Overview
This study tests whether a true multimodal architecture can outperform single-modality and classical baselines on complex scientific documents. It extracts both the text and embedded images (figures, tables) from PDFs to classify them into thematic clusters. To maintain a clean and focused evaluation, this pipeline specifically tests the primary multimodal hypothesis and omits exploratory ablation sweeps.

## Key Components

1.  **Data Extraction & Processing:** 
    Uses PyMuPDF to extract text and up to 4 embedded images per document[cite: 20]. Text is tokenized for BERT and GPT-2, while images are normalized and resized for ResNet-18[cite: 20].
2.  **Baseline Models:** 
    *   *Classical/Ensemble:* TF-IDF features trained on Logistic Regression, SVM, Random Forest, XGBoost, etc[cite: 15].
    *   *Single-Transformer:* BERT-only and GPT-2-only baselines to measure the exact value added by the fusion mechanism[cite: 15].
3.  **Proposed Multimodal Fusion:** 
    A unified `MultimodalFusion` network that projects BERT (contextual text), GPT-2 (sequential text), and ResNet-18 (visual features) outputs into a shared latent space for joint classification[cite: 20].

## Validation Strategy
The study uses a multi-seed protocol (seeds 42, 7, and 123) with strict train/test splits[cite: 20]. Pre-trained backbones are fine-tuned with a cautious learning rate to prevent catastrophic forgetting, and metrics (Accuracy, Precision, Recall, F1-score) are reported as a mean across all seeds to ensure robust, reproducible results[cite: 20, 22].

## Usage (Colab / Jupyter)

To run the streamlined multimodal experiment without the overhead of ablation studies, upload this directory to your environment and run:
```python
# 1. Install requirements
!pip install pymupdf transformers scikit-learn torchvision --quiet

# 2. Run the core multimodal experiment
from multimodal import run_multimodal_experiment

results = run_multimodal_experiment(
    pdf_directory="/content/data",
    image_cache_dir="/content/pdf_images",
    label_strategy="sbert_kmeans", 
    seeds=(42, 7, 123),
    epochs=5,
    batch_size=8
)