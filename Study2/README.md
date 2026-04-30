Study 2: Mid-Level Multimodal Fusion (USG & Clinical Text)

This directory contains the code to test whether a mid-level fusion of BERT, GPT-2 and ResNet-18 representations outperforms single-modality baselines using combined clinical text and ultrasound images.

Overview

The study2_full_analysis.py script evaluates four different models across multiple random seeds to validate the proposed multimodal architecture using the Breast Lesions Ultrasound (USG) Collection. 

Key Features

Data Handling: Processes both clinical text descriptions and diagnostic ultrasound images from the BrEaST-Lesions-USG dataset.
Model Comparison: Evaluates BERT-only (text), GPT2-only (text), a BERT+GPT2 text fusion and the proposed Multimodal (BERT+GPT2+ResNet) architecture.
Metrics: Records accuracy, ROC-AUC, Brier score and Expected Calibration Error (ECE) to assess both discrimination and calibration.
Checkpointing: Automatically saves results to a CSV after every seed and model combination so interrupted Kaggle sessions can resume seamlessly.
Statistical Testing: Runs Wilcoxon signed-rank tests across 7 seeds to measure the significance of the multimodal model's performance against the text-only baselines.

Usage

This script is configured for a Kaggle notebook environment with a T4 GPU. Update the EXCEL_PATH and IMG_DIR variables in the configuration section to point to the correct Kaggle input directories before running.

Outputs

The script generates three CSV files in the /kaggle/working/ directory:
study2_per_seed_results.csv: The raw results for each model-seed combination.
study2_summary.csv: The mean and standard deviation of metrics across all seeds.
study2_wilcoxon.csv: The statistical significance results comparing the proposed Multimodal model to baselines.
