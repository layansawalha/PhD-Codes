{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww30040\viewh16460\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Study 1: Quantum and Classical SVM Fusion for Breast Cancer Classification\
\
This directory contains the experimental code to evaluate whether fusing a Quantum Support Vector Machine (QSVM) with a classical SVM through ensemble methods improves breast cancer classification accuracy[cite: 12].\
\
## Overview\
The `study1_full_analysis.py` script runs a complete multi-seed evaluation pipeline on the Wisconsin Breast Cancer Dataset (WBCD)[cite: 12]. It is designed to run efficiently in a single Kaggle notebook cell[cite: 12].\
\
## Key Features\
*   **Data Processing:** Loads the WBCD and scales the data using `MinMaxScaler`[cite: 12].\
*   **Dimensionality Reduction:** Applies a Restricted Boltzmann Machine (RBM) to reduce the feature space to 4 dimensions to match the quantum circuit's qubit constraints[cite: 12].\
*   **Model Training:** Builds and evaluates six distinct models per seed: classical SVM, QSVM, Bagging, Stacking, Voting and a master Hybrid ensemble[cite: 12].\
*   **Metrics:** Computes accuracy, ROC-AUC, Brier score and Expected Calibration Error (ECE) for every model[cite: 12].\
*   **Statistical Testing:** Performs Wilcoxon signed-rank tests across 7 random seeds (42, 7, 123, 999, 2023, 8888 and 7777) to compare the Hybrid model against all baselines[cite: 12].\
\
## Usage\
Upload the `study1_full_analysis.py` script to a Kaggle environment and execute it[cite: 12]. Ensure the `qiskit` and `qiskit-machine-learning` libraries are installed[cite: 12]. \
\
## Outputs\
The script generates three output files in the `/kaggle/working` directory[cite: 12]:\
*   `study1_per_seed_results.csv`: Raw metric numbers for every model across each seed[cite: 12].\
*   `study1_summary.csv`: Aggregated mean and standard deviation for all metrics[cite: 12].\
*   `study1_wilcoxon.csv`: Paired statistical test results comparing the Hybrid model against baselines[cite: 12].}