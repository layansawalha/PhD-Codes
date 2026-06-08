# Study 4: Advanced Regression Fusion Strategies (Construction Costs)

This directory contains code to evaluate multiple machine learning and ensemble regression approaches for predicting construction costs using both original and augmented datasets.

## Overview

The experimental pipeline compares classical regression models, tree-based ensembles, hypertuned models, and advanced neural-ensemble fusion strategies on construction cost prediction tasks.

## Key Features

- **Data Handling**: Processes construction costs dataset with categorical and numerical features (floor_area, storeys, building_function_code, type_of_work, main_construction).
- **Data Augmentation**: Applies Gaussian noise augmentation to key features to generate synthetic training data (2x dataset expansion).
- **Multiple Regression Approaches**:
  - Classical ML models (Linear Regression, SVR, KNN, MLP)
  - Tree-based models (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost)
  - Ensemble Stacking with meta-learners (XGBoost, Ridge)
  - Weighted averaging ensembles
  - Hybrid deep learning fusion (MLP + XGBoost + CatBoost)
- **Metrics**: Records R² score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) for comprehensive regression evaluation.
- **Hyperparameter Optimization**: RandomizedSearchCV for tuning base models before ensemble construction.
- **Feature Engineering**: Derives new features (floor_area_per_storey, floor_area_x_storeys, floor_area_squared) without data leakage.
- **Comprehensive Evaluation**: Compares simple ensembles, stacking ensembles, and neural network fusion architectures.

## Usage

Run the augmented dataset experiments (recommended):

```python
# Run in Kaggle notebook
exec(open("augmented_dataset.py").read())
```

Or run the original dataset experiments:

```python
exec(open("original_dataset.py").read())
```

Run locally:

```bash
python augmented_dataset.py
python original_dataset.py
```

## Outputs

Each script reports results for multiple models and ensemble strategies:

**From `augmented_dataset.py`:**
1. **Basic ML Models**: Linear Regression, SVR, Random Forest, Gradient Boosting, XGBoost, KNN, MLP
2. **Tree-Based Ensembles**: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
3. **Stacking Ensemble**: Base learners + Linear Regression meta-learner
4. **Hypertuned Models**: RandomizedSearchCV optimized Random Forest and XGBoost
5. **Advanced Stacking**: Tuned base models + XGBoost meta-learner
6. **Weighted Averaging**: Simple average of tuned model predictions
7. **Hybrid Stacking**: CatBoost + XGBoost + MLP + Ridge meta-learner
8. **Deep Learning Fusion**: MLP + XGBoost + CatBoost predictions fused via deep neural network

**From `original_dataset.py`:**
- Same model evaluations as augmented dataset (for comparison)

Each script outputs:
- **R² Score**: Coefficient of determination (higher is better)
- **RMSE**: Root Mean Squared Error in cost units
- **MAE**: Mean Absolute Error in cost units

## Architecture

### Fusion Strategies

```
Construction Cost Features
├── Raw Features
│   ├── floor_area
│   ├── storeys
│   ├── building_function_code
│   ├── type_of_work
│   └── main_construction
│
├── Feature Engineering (no leakage)
│   ├── floor_area_per_storey
│   ├── floor_area_x_storeys
│   └── floor_area_squared
│
└── Ensemble Fusion Strategies
    ├── Simple Ensemble
    │   └── Average predictions from multiple models
    │
    ├── Stacking Ensemble
    │   ├── Base learners (RF, GB, XGBoost, LightGBM, CatBoost)
    │   └── Meta-learner (Linear Regression / XGBoost)
    │
    ├── Hypertuned + Stacking
    │   ├── Tuned RF, GB, XGBoost with hyperparameter search
    │   └── XGBoost meta-learner
    │
    └── Deep Learning Fusion
        ├── MLP stream (scaled features)
        ├── XGBoost stream (base predictions)
        ├── CatBoost stream (base predictions)
        └── Merged with Dense layers → final regression
                    ↓
        Regression Output (Construction Cost Prediction)
        
        Metrics: R², RMSE, MAE
```

## Configuration

Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TEST_SIZE` | 0.2 | Train-test split ratio |
| `RANDOM_STATE` | 42 | Reproducibility seed |
| **Data Augmentation** | | |
| `AUGMENTATION_FACTOR` | 2 | Double original dataset size |
| `NOISE_SCALE` | 1% of feature std | Gaussian noise level |
| **Models** | | |
| `XGBoost` n_estimators | 100-500 | Number of boosting rounds |
| `Random Forest` n_estimators | 100-500 | Number of trees |
| `MLP hidden layers` | (128, 64, 32) | Neural network architecture |
| `Meta-learner` | Ridge / XGBoost | Ensemble combiner |

## Datasets

- **`Finaldataset9.csv`**: Original construction costs dataset (2,835 samples)
- **`augmented_dataset_final.csv`**: Augmented dataset with 2x samples via Gaussian noise

### Features

**Input Features (5 columns):**
- `type_of_work` (categorical)
- `building_function_code` (categorical)
- `floor_area` (numerical)
- `main_construction` (categorical)
- `storeys` (numerical)

**Target:**
- `cost_rebased` (numerical, in currency units)

## Requirements

See `../Requirements.txt` for full dependency list. Key packages:
- `pandas`, `numpy`
- `scikit-learn` (ML models, preprocessing, ensembles)
- `xgboost`, `lightgbm`, `catboost` (gradient boosting)
- `tensorflow`, `keras` (deep learning)
- `scipy` (statistical distributions)
- `matplotlib` (visualization)

## Files

- **`augmented_dataset.py`**: Experiments on augmented construction costs dataset (Gaussian noise augmentation)
- **`original_dataset.py`**: Experiments on original construction costs dataset
- **`Code`**: Placeholder file
- **`Dataset/`**: Directory for storing dataset files

## Key Findings

The scripts evaluate whether:
1. **Data augmentation** improves model robustness
2. **Ensemble methods** outperform individual models
3. **Hyperparameter tuning** enhances regression performance
4. **Neural network fusion** can combine complementary model predictions effectively

## Citation

Part of PhD research on hybrid multimodal and quantum machine learning architectures.
