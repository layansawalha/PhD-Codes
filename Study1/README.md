# Study 1: Quantum and Classical SVM Fusion for Breast Cancer Classification

This directory contains code to evaluate whether fusing a Quantum Support Vector Machine (QSVM) with a classical SVM through ensemble methods improves breast cancer classification on the Wisconsin Breast Cancer Dataset (WBCD).

## Overview

The `Code.py` script runs a complete multi-seed evaluation pipeline that constructs and compares six distinct models using quantum and classical machine learning fusion strategies.

## Key Features

- **Data Processing**: Loads the Wisconsin Breast Cancer Dataset (WBCD, 569 samples) and scales features using MinMaxScaler.
- **Dimensionality Reduction**: Applies Restricted Boltzmann Machine (RBM) to reduce 30 features to 4 dimensions (matching quantum circuit qubit count).
- **Six Model Architectures**:
  1. **Classical SVM**: Radial basis function (RBF) kernel SVM
  2. **QSVM**: Quantum Support Vector Machine with ZZ feature map
  3. **Bagging Ensemble**: 5 QSVM estimators with bagging
  4. **Stacking Ensemble**: QSVM + SVM with Random Forest (50 trees, max_depth=3) meta-learner
  5. **Voting Ensemble**: Soft-voting combination of QSVM and SVM
  6. **Hybrid Ensemble**: Master soft-voting combination of Bagging, Stacking, and Voting ensembles
- **Metrics**: Records accuracy, ROC-AUC, Brier score, and Expected Calibration Error (ECE) for discrimination and calibration assessment.
- **Statistical Testing**: Performs Wilcoxon signed-rank tests across 7 seeds to measure statistical significance of Hybrid model performance against all baselines.
- **Multi-Seed Reproducibility**: Uses 7 random seeds (42, 7, 123, 999, 2023, 8888, 7777) for rigorous validation.

## Usage

Run in a Kaggle notebook (recommended):

```python
# Ensure qiskit packages are installed
!pip install qiskit qiskit-machine-learning qiskit-algorithms

# Run the script
exec(open("Code.py").read())
```

Or run locally:

```bash
pip install qiskit qiskit-machine-learning qiskit-algorithms scikit-learn numpy pandas scipy
python Code.py
```

## Outputs

The script generates three CSV files in the `/kaggle/working/` directory:

- **`study1_per_seed_results.csv`**: Raw results for each model on each seed (42 rows = 6 models × 7 seeds)
  - Columns: seed, model, accuracy, roc_auc, brier_score, ece, fit_time_s
  
- **`study1_summary.csv`**: Aggregated statistics across all seeds for each model
  - Columns: model, accuracy_mean, accuracy_std, roc_auc_mean, roc_auc_std, brier_mean, brier_std, ece_mean, ece_std
  
- **`study1_wilcoxon.csv`**: Pairwise statistical comparisons of Hybrid vs baseline models
  - Columns: baseline, hybrid_mean_acc, baseline_mean_acc, diff, wilcoxon_statistic, p_value, significant_at_005, note

## Architecture

### Quantum-Classical Fusion Pipeline

```
Wisconsin Breast Cancer Dataset (569 samples, 30 features)
    ├── MinMax Scaling [0, 1]
    │   ↓
    └── RBM Dimensionality Reduction (30 → 4 dims)
            ↓
    Train-Test Split (80-20, stratified)
            ↓
    ┌───────────────────────────────────────────────────┐
    │ Six Model Architectures:                          │
    │                                                   │
    ├─ Classical SVM                                    │
    │   └─ RBF kernel, probability=True                │
    │                                                   │
    ├─ Quantum SVM (QSVM)                              │
    │   ├─ ZZ Feature Map (reps=2, linear entangle)   │
    │   ├─ Fidelity Quantum Kernel                     │
    │   └─ Statevector Sampler backend                │
    │                                                   │
    ├─ Bagging Ensemble                                │
    │   └─ 5 × QSVM estimators                         │
    │                                                   │
    ├─ Stacking Ensemble                               │
    │   ├─ Base learners: QSVM + SVM                   │
    │   └─ Meta-learner: Random Forest (50 trees)      │
    │                                                   │
    ├─ Voting Ensemble                                 │
    │   ├─ QSVM + SVM                                  │
    │   └─ Soft voting aggregation                     │
    │                                                   │
    └─ Hybrid Ensemble                                 │
        ├─ Bagging Ensemble                            │
        ├─ Stacking Ensemble                           │
        ├─ Voting Ensemble                             │
        └─ Soft voting aggregation (master ensemble)   │
            ↓
    Evaluation on 20% Test Split
            ↓
    Metrics: Accuracy, ROC-AUC, Brier, ECE
```

### Probability Extraction Strategy

For models without native `predict_proba()`:
- QSVC uses min-max rescaled decision function as probability proxy
- Provides calibration metrics for all models on equal footing

## Configuration

Key parameters in `Code.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEEDS` | (42, 7, 123, 999, 2023, 8888, 7777) | 7 random seeds for reproducibility |
| `N_FEATURES` | 4 | RBM output and quantum qubit count |
| **RBM** | | Dimensionality reduction |
| `n_components` | 4 | Output dimensions |
| `n_iter` | 20 | Training iterations |
| `learning_rate` | 0.01 | RBM learning rate |
| **SVM** | | Classical SVM |
| `kernel` | rbf | Radial basis function |
| `C` | 1.0 | Regularization parameter |
| `gamma` | scale | Kernel coefficient |
| **Bagging** | | Ensemble method |
| `n_estimators` | 5 | Number of base estimators |
| **Stacking** | | Ensemble method |
| `cv` | 5 | Cross-validation folds |
| Meta-learner | Random Forest (50 trees) | Final combiner |
| **Quantum** | | |
| Feature map | ZZ Feature Map | Quantum encoding |
| `reps` | 2 | Repetitions |
| `entanglement` | linear | Qubit connectivity |
| Backend | StatevectorSampler | Quantum simulator |

## Dataset

**Wisconsin Breast Cancer Dataset (WBCD)**
- **Samples**: 569 instances (357 benign, 212 malignant)
- **Features**: 30 input features → reduced to 4 via RBM
- **Task**: Binary classification (malignant vs benign)
- **Train-Test Split**: 80-20 stratified split per seed

## Metrics Explained

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | [0, 1] | Proportion of correct predictions (higher is better) |
| **ROC-AUC** | [0, 1] | Area under the Receiver Operating Characteristic curve (higher is better) |
| **Brier Score** | [0, 1] | Mean squared error of probabilities (lower is better) |
| **ECE** | [0, 1] | Expected Calibration Error (lower is better; 0 = perfect calibration) |

## Requirements

See `../Requirements.txt` for full dependency list. Key packages:
- `qiskit` (quantum circuits and algorithms)
- `qiskit-machine-learning` (quantum machine learning algorithms)
- `qiskit-algorithms` (state fidelity estimation)
- `scikit-learn` (classical ML, SVM, ensemble methods, RBM)
- `pandas`, `numpy` (data handling and numerical computation)
- `scipy` (Wilcoxon statistical testing)

## Key Hypotheses

The study investigates:
1. **Does quantum-classical fusion outperform purely classical approaches?**
   - Compares QSVM and Hybrid ensemble vs classical SVM baselines
2. **Do ensemble methods improve hybrid model performance?**
   - Compares Bagging, Stacking, and Voting vs individual models
3. **Can a master Hybrid ensemble (Bagging + Stacking + Voting) provide the best robustness?**
   - Tests Hybrid ensemble against all individual and intermediate ensemble approaches

## Statistical Testing

**Wilcoxon Signed-Rank Test**:
- **Paired comparison**: Same 7 train-test splits used for all models
- **Alternative hypothesis**: "greater" tests if Hybrid accuracy > baseline accuracy
- **Significance level**: α = 0.05
- **Null hypothesis**: No difference in median accuracy between Hybrid and baseline

## Citation

Part of PhD research on hybrid multimodal and quantum machine learning architectures.
