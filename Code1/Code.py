
import warnings
warnings.filterwarnings("ignore")

import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    BaggingClassifier, StackingClassifier, VotingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# ===========================================================================
# Configuration
# ===========================================================================

# Same seven seeds as Study 3, so the experimental population is consistent
# across studies and the Wilcoxon test has identical statistical power.
SEEDS = (42, 7, 123, 999, 2023, 8888, 7777)

N_FEATURES = 4   # RBM output dimension; matches qubit count

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# Calibration metric: Expected Calibration Error
# ===========================================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Standard binned ECE for binary classification.

    Splits predicted probabilities into n_bins equal-width buckets,
    computes |mean_accuracy - mean_confidence| in each bucket, weights
    by bucket size, and sums. Lower is better; 0 means perfect
    calibration.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Last bin is closed on the right to capture probabilities of exactly 1.0
        if i == n_bins - 1:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob < hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = y_prob[in_bin].mean()
        ece += (in_bin.sum() / len(y_prob)) * abs(bin_acc - bin_conf)
    return ece


# ===========================================================================
# Build all six models for one seed
# ===========================================================================

def build_models(seed):
    """Construct all six models with the given random seed where applicable.

    The quantum kernel is rebuilt every call. Note that QSVC itself does
    not accept a random_state argument; the underlying scikit-learn SVC
    inside it uses its own internal seed.
    """
    feature_map = ZZFeatureMap(feature_dimension=N_FEATURES, reps=2,
                               entanglement="linear")
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    qsvc = QSVC(quantum_kernel=qkernel)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
              random_state=seed)

    bagging = BaggingClassifier(
        estimator=qsvc, n_estimators=5, random_state=seed,
    )
    stacking = StackingClassifier(
        estimators=[("qsvm", qsvc), ("svm", svm)],
        final_estimator=RandomForestClassifier(
            n_estimators=50, max_depth=3, random_state=seed,
        ),
        cv=5,
    )
    voting = VotingClassifier(
        estimators=[("qsvm", qsvc), ("svm", svm)], voting="soft",
    )
    hybrid = VotingClassifier(
        estimators=[
            ("bagging_model", bagging),
            ("stacking_model", stacking),
            ("voting_model", voting),
        ],
        voting="soft",
    )

    return {
        "SVM":      svm,
        "QSVM":     qsvc,
        "Bagging":  bagging,
        "Stacking": stacking,
        "Voting":   voting,
        "Hybrid":   hybrid,
    }


# ===========================================================================
# Probability extraction (handles QSVC without predict_proba)
# ===========================================================================

def get_proba(model, X):
    """Return P(class=1) for each row of X.

    QSVC does not implement predict_proba in a useful way out of the box.
    For QSVC we fall back to a min-max-rescaled decision function as a
    rough probability proxy so that ROC-AUC and calibration metrics can
    still be computed. This proxy is acceptable because calibration is
    being reported as an exploratory metric, not a deployment claim.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).ravel()
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-12:
            return np.full_like(scores, 0.5)
        return (scores - s_min) / (s_max - s_min)
    # Last resort: hard prediction as 0/1
    return model.predict(X).astype(float)


# ===========================================================================
# Run one seed end-to-end
# ===========================================================================

def run_one_seed(seed):
    """Train and evaluate all six models on a single seed.

    Returns a dict mapping model name to a metrics dict.
    """
    # 1. Load and scale data
    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. RBM dimensionality reduction (deterministic given seed)
    rbm = BernoulliRBM(
        n_components=N_FEATURES, learning_rate=0.01, n_iter=20,
        random_state=seed,
    )
    X_reduced = rbm.fit_transform(X_scaled)

    # 3. Stratified train-test split (the same split is used for every model)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_reduced, y, test_size=0.2, random_state=seed, stratify=y,
    )

    # 4. Build and train every model on this same split
    models = build_models(seed)
    metrics = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = get_proba(model, X_te)

        try:
            roc = roc_auc_score(y_te, y_prob)
        except Exception:
            roc = np.nan
        try:
            brier = brier_score_loss(y_te, y_prob)
        except Exception:
            brier = np.nan
        try:
            ece = expected_calibration_error(y_te, y_prob)
        except Exception:
            ece = np.nan

        metrics[name] = {
            "accuracy":    accuracy_score(y_te, y_pred),
            "roc_auc":     roc,
            "brier_score": brier,
            "ece":         ece,
            "fit_time_s":  time.time() - t0,
        }
        print(f"  {name:<10}  acc={metrics[name]['accuracy']:.4f}  "
              f"roc={metrics[name]['roc_auc']:.4f}  "
              f"brier={metrics[name]['brier_score']:.4f}  "
              f"ece={metrics[name]['ece']:.4f}  "
              f"({metrics[name]['fit_time_s']:.1f}s)")
    return metrics


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Study 1 (WBCD) multi-seed evaluation with calibration metrics")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"RBM components / qubits: {N_FEATURES}")
    print()

    # Run every seed
    per_seed_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        metrics = run_one_seed(seed)
        for name, m in metrics.items():
            per_seed_results.append({"seed": seed, "model": name, **m})

    # Save per-seed results
    df_per_seed = pd.DataFrame(per_seed_results)
    df_per_seed.to_csv(f"{OUTPUT_DIR}/study1_per_seed_results.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR}/study1_per_seed_results.csv")

    # Summary across seeds
    summary_rows = []
    for name in df_per_seed["model"].unique():
        sub = df_per_seed[df_per_seed["model"] == name]
        summary_rows.append({
            "model":          name,
            "accuracy_mean":  sub["accuracy"].mean(),
            "accuracy_std":   sub["accuracy"].std(),
            "roc_auc_mean":   sub["roc_auc"].mean(),
            "roc_auc_std":    sub["roc_auc"].std(),
            "brier_mean":     sub["brier_score"].mean(),
            "brier_std":      sub["brier_score"].std(),
            "ece_mean":       sub["ece"].mean(),
            "ece_std":        sub["ece"].std(),
        })
    df_summary = (pd.DataFrame(summary_rows)
                  .sort_values("accuracy_mean", ascending=False)
                  .reset_index(drop=True))

    print("\n" + "=" * 70)
    print(f"Summary across {len(SEEDS)} seeds")
    print("=" * 70)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200):
        print(df_summary.to_string(index=False))
    df_summary.to_csv(f"{OUTPUT_DIR}/study1_summary.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR}/study1_summary.csv")

    # Wilcoxon: Hybrid vs every baseline, paired by seed
    print("\n" + "=" * 70)
    print("Wilcoxon signed-rank tests (Hybrid vs each baseline)")
    print("alternative='greater' tests whether Hybrid > baseline")
    print("=" * 70)

    hybrid_accs = (df_per_seed[df_per_seed["model"] == "Hybrid"]
                   .sort_values("seed")["accuracy"].values)

    wilcoxon_rows = []
    for name in ("SVM", "QSVM", "Bagging", "Stacking", "Voting"):
        baseline_accs = (df_per_seed[df_per_seed["model"] == name]
                         .sort_values("seed")["accuracy"].values)

        diffs = hybrid_accs - baseline_accs
        if np.allclose(diffs, 0):
            stat, p = np.nan, 1.0
            note = "all paired differences are zero"
        else:
            try:
                stat, p = wilcoxon(hybrid_accs, baseline_accs,
                                   alternative="greater",
                                   zero_method="wilcox")
                note = ""
            except ValueError as e:
                stat, p = np.nan, np.nan
                note = f"wilcoxon error: {e}"

        wilcoxon_rows.append({
            "baseline":           name,
            "hybrid_mean_acc":    hybrid_accs.mean(),
            "baseline_mean_acc":  baseline_accs.mean(),
            "diff":               hybrid_accs.mean() - baseline_accs.mean(),
            "wilcoxon_statistic": stat,
            "p_value":            p,
            "significant_at_005": (not np.isnan(p)) and (p < 0.05),
            "note":               note,
        })

    df_wilcoxon = pd.DataFrame(wilcoxon_rows)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200):
        print(df_wilcoxon.to_string(index=False))
    df_wilcoxon.to_csv(f"{OUTPUT_DIR}/study1_wilcoxon.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR}/study1_wilcoxon.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
