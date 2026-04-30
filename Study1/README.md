Study 1: Quantum and Classical SVM Fusion for Breast Cancer Classification

This directory contains the experimental code to evaluate whether fusing a Quantum Support Vector Machine (QSVM) with a classical SVM through ensemble methods improves breast cancer classification accuracy.

Overview

The study1_full_analysis.py script runs a complete multi-seed evaluation pipeline on the Wisconsin Breast Cancer Dataset (WBCD). It is designed to run efficiently in a single Kaggle notebook cell.

Key Features

Data Processing: Loads the WBCD and scales the data using MinMaxScaler.
Dimensionality Reduction: Applies a Restricted Boltzmann Machine (RBM) to reduce the feature space to 4 dimensions to match the quantum circuit's qubit constraints.
Model Training: Builds and evaluates six distinct models per seed: classical SVM, QSVM, Bagging, Stacking, Voting and a master Hybrid ensemble.
Metrics: Computes accuracy, ROC-AUC, Brier score and Expected Calibration Error (ECE) for every model.
Statistical Testing: Performs Wilcoxon signed-rank tests across 7 random seeds (42, 7, 123, 999, 2023, 8888 and 7777) to compare the Hybrid model against all baselines.

Usage

Upload the study1_full_analysis.py script to a Kaggle environment and execute it. Ensure the qiskit and qiskit-machine-learning libraries are installed. 

Outputs

The script generates three output files in the /kaggle/working directory:
study1_per_seed_results.csv: Raw metric numbers for every model across each seed.
study1_summary.csv: Aggregated mean and standard deviation for all metrics.
study1_wilcoxon.csv: Paired statistical test results comparing the Hybrid model against baselines.
