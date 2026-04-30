{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Study 2: Mid-Level Multimodal Fusion (USG & Clinical Text)\
\
This directory contains the code to test whether a mid-level fusion of BERT, GPT-2 and ResNet-18 representations outperforms single-modality baselines using combined clinical text and ultrasound images[cite: 1].\
\
## Overview\
The `study2_full_analysis.py` script evaluates four different models across multiple random seeds to validate the proposed multimodal architecture using the Breast Lesions Ultrasound (USG) Collection[cite: 1]. \
\
## Key Features\
*   **Data Handling:** Processes both clinical text descriptions and diagnostic ultrasound images from the BrEaST-Lesions-USG dataset[cite: 1].\
*   **Model Comparison:** Evaluates BERT-only (text), GPT2-only (text), a BERT+GPT2 text fusion and the proposed Multimodal (BERT+GPT2+ResNet) architecture[cite: 1].\
*   **Metrics:** Records accuracy, ROC-AUC, Brier score and Expected Calibration Error (ECE) to assess both discrimination and calibration[cite: 1].\
*   **Checkpointing:** Automatically saves results to a CSV after every seed and model combination so interrupted Kaggle sessions can resume seamlessly[cite: 1].\
*   **Statistical Testing:** Runs Wilcoxon signed-rank tests across 7 seeds to measure the significance of the multimodal model's performance against the text-only baselines[cite: 1].\
\
## Usage\
This script is configured for a Kaggle notebook environment with a T4 GPU[cite: 1]. Update the `EXCEL_PATH` and `IMG_DIR` variables in the configuration section to point to the correct Kaggle input directories before running[cite: 1].\
\
## Outputs\
The script generates three CSV files in the `/kaggle/working/` directory[cite: 1]:\
*   `study2_per_seed_results.csv`: The raw results for each model-seed combination[cite: 1].\
*   `study2_summary.csv`: The mean and standard deviation of metrics across all seeds[cite: 1].\
*   `study2_wilcoxon.csv`: The statistical significance results comparing the proposed Multimodal model to baselines[cite: 1].}