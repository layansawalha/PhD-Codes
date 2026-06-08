Study 2: Multimodal Breast Lesion Classification Using Clinical Text and Ultrasound Images

This directory contains the code to evaluate whether a mid-level fusion of BERT, GPT-2 and ResNet-18 representations outperforms text-only baselines for breast lesion classification using clinical text descriptions and diagnostic ultrasound images.

Overview

The study2_full_analysis.py script evaluates four different architectures across multiple random seeds to validate the proposed multimodal fusion framework on the BrEaST-Lesions-USG dataset.

Key Features

Data Handling: Processes both clinical text descriptions (symptoms and diagnosis information) and ultrasound images from the BrEaST-Lesions-USG dataset.

Model Comparison: Evaluates four architectures:

* BERT-only (clinical text)
* GPT-2-only (clinical text)
* Resnet-only (images)
* Proposed Multimodal (BERT+GPT2+ResNet-18)

Mid-Level Fusion: Combines projected feature representations from BERT, GPT-2, and ResNet-18 through feature concatenation before classification.

Metrics: Records accuracy, ROC-AUC, Brier Score, and Expected Calibration Error (ECE) to evaluate both classification performance and probability calibration.

Checkpointing: Automatically saves results after every model-seed combination, enabling interrupted Kaggle sessions to resume without loss of progress.

Statistical Testing: Performs Wilcoxon signed-rank tests across 7 random seeds to assess whether the proposed multimodal model significantly outperforms the text-only baselines.

GPU Support: Automatic CUDA detection and acceleration for Kaggle T4 GPU environments.

Usage

This script is configured for a Kaggle notebook environment with a T4 GPU. Update the EXCEL_PATH and IMG_DIR variables in the configuration section to point to the correct dataset locations before execution.

Run directly in a Kaggle notebook:

exec(open("study2_full_analysis.py").read())

Or run locally:

python study2_full_analysis.py

Outputs

The script generates three CSV files in the /kaggle/working/ directory:

study2_per_seed_results.csv

* Raw results for each model and random seed.
* Includes accuracy, ROC-AUC, Brier Score, and Expected Calibration Error.

study2_summary.csv

* Mean and standard deviation of all metrics across the 7 random seeds.
* Provides overall model comparison.

study2_wilcoxon.csv

* Wilcoxon signed-rank test results comparing the proposed multimodal model against each baseline model.
* Reports statistical significance and effect direction.

Architecture

BERT-only Baseline

Clinical Text
↓
BERT Encoder
↓
[CLS] Representation (768 dims)
↓
Dropout + Linear Layer
↓
Classification

GPT-2-only Baseline

Clinical Text
↓
GPT-2 Encoder
↓
Last Token Representation (768 dims)
↓
Dropout + Linear Layer
↓
Classification

BERT+GPT2 Text Fusion

Clinical Text
├── BERT Encoder
│   └── [CLS] Representation → Project to 128 dims
│
└── GPT-2 Encoder
└── Last Token Representation → Project to 128 dims

```
            ↓
    Concatenation (256 dims)
            ↓
  Dropout + Linear Layer
            ↓
      Classification
```

Proposed Multimodal Fusion

Clinical Text + Ultrasound Image
├── BERT Encoder
│   └── [CLS] Representation → Project to 128 dims
│
├── GPT-2 Encoder
│   └── Last Token Representation → Project to 128 dims
│
└── Ultrasound Image
└── ResNet-18 → 128 dims

```
            ↓
 Mid-Level Feature Fusion
 Concatenation (384 dims)
            ↓
  Dropout + Linear Layer
            ↓
      Classification
```

Configuration

Key hyperparameters in study2_full_analysis.py:

Parameter                Value
SEEDS                    (42, 7, 123, 999, 2023, 8888, 7777)
EPOCHS                   5
BATCH_SIZE               16
MAX_LENGTH               128
LEARNING_RATE            2e-5
IMAGE_SIZE               224 × 224
BERT_PROJECTION          128
GPT2_PROJECTION          128
RESNET_OUTPUT            128

Dataset

BrEaST-Lesions-USG Dataset

Input Modalities:

* Clinical text descriptions (Symptoms and Diagnosis fields)
* Diagnostic ultrasound images

Classification Task:

* Benign vs Malignant breast lesion classification

Image Processing:

* Resize to 224 × 224 pixels
* ImageNet normalization
* RGB conversion

Calibration Metrics

In addition to conventional classification metrics, this study evaluates prediction calibration using:

* Brier Score (lower is better)
* Expected Calibration Error (ECE) (lower is better)

These metrics assess the reliability of model confidence estimates, which is particularly important in medical decision-support applications.

Requirements

See ../Requirements.txt for full dependency information.

Key packages:

* torch, torchvision
* transformers (BERT, GPT-2)
* scikit-learn
* scipy
* pandas
* numpy
* Pillow

Citation

Part of PhD research investigating multimodal deep learning architectures for medical data classification through the integration of transformer-based language models and convolutional neural networks.
