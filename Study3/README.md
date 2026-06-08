# Study 3: Multimodal Scientific PDF Classification

This directory contains the code to test whether a mid-level fusion of BERT, GPT-2 and ResNet-18 representations outperforms single-modality baselines on scientific PDF classification using extracted text and images.

## Overview

The `Code.py` script evaluates the proposed multimodal architecture across multiple random seeds to validate performance on the NUS-WIDE scientific PDF dataset.

## Key Features

- **Data Handling**: Extracts text and embedded images from scientific PDFs (with fallback to page rendering).
- **Proposed Method Only**: Implements BERT + GPT-2 + ResNet-18 multimodal fusion with weighted sum aggregation.
- **Metrics**: Records accuracy, precision, recall, and F1-score for comprehensive classification evaluation.
- **Checkpointing**: Automatically saves results to a CSV after every seed so interrupted Kaggle sessions can resume seamlessly.
- **Statistical Testing**: Runs Wilcoxon signed-rank tests across 7 seeds to measure statistical significance of performance.
- **GPU Support**: Automatic detection and acceleration on T4 GPU.

## Usage

This script is configured for a Kaggle notebook environment with a T4 GPU. Update the `PDF_DIR` and `IMAGE_CACHE_DIR` variables in the configuration section to point to the correct Kaggle input directories.

```python
# Run directly in Kaggle notebook
exec(open("Code.py").read())
```

Or run locally:

```bash
python Code.py
```

## Outputs

The script generates three CSV files in the `/kaggle/working/` directory:

- **`study3_per_seed_results.csv`**: Raw results for each seed with accuracy, precision, recall, and F1-score.
- **`study3_summary.csv`**: Mean and standard deviation of metrics across all 7 seeds.
- **`study3_wilcoxon.csv`**: Wilcoxon signed-rank test results comparing the proposed multimodal model against a random (50%) baseline.

## Architecture

### Proposed Multimodal Fusion

```
Scientific PDF
    ├── Text Extraction (BERT tokenization)
    │   └── BERT [CLS] token → 768 dims → Project to fusion_dim
    │
    ├── Text Extraction (GPT-2 tokenization)
    │   └── GPT-2 last token → 768 dims → Project to fusion_dim
    │
    └── Image Extraction & Processing
        ├── Extract up to 4 embedded images (figures, tables)
        ├── Fallback: render first pages if insufficient embedded images
        └── ResNet-18 (frozen) → 512 dims → Project to fusion_dim
        
                    ↓
            Weighted Sum Fusion: fusion_dim
                    ↓
    Classification Head (dropout + linear) → n_classes
```

## Configuration

Key hyperparameters in `Code.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEEDS` | (42, 7, 123, 999, 2023, 8888, 7777) | 7 random seeds for reproducibility |
| `EPOCHS` | 5 | Training epochs per seed |
| `BATCH_SIZE` | 8 | Batch size (smaller due to image memory) |
| `MAX_LENGTH` | 128 | Max token length for text |
| `LEARNING_RATE` | 5e-5 | AdamW learning rate |
| `FUSION_DIM` | 256 | Shared fusion dimension |
| `N_IMAGES_PER_PDF` | 4 | Max images to extract per PDF |

## Requirements

See `../Requirements.txt` for full dependency list. Key packages:
- `torch`, `torchvision`
- `transformers` (BERT, GPT-2)
- `pymupdf` (PDF image extraction)
- `scikit-learn`
- `pandas`, `numpy`, `Pillow`

## Citation

Part of PhD research on hybrid multimodal and quantum machine learning architectures.
