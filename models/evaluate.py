"""
Contract Clause Classification — Comprehensive Evaluation
==========================================================
Loads saved model and test data, computes per-class and aggregate metrics,
generates confusion matrix, calibration analysis, and saves results.

Usage:
    python -m models.evaluate
    python -m models.evaluate --config configs/baseline.yaml
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"
RESULTS_PATH = PROJECT_ROOT / "models" / "evaluation_results.json"


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "baseline.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Per-class Metrics
# ──────────────────────────────────────────────

def compute_per_class_metrics(y_true, y_pred, label_names):
    """Compute precision, recall, F1 for all 100 classes."""
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = []
    for i, name in enumerate(label_names):
        if name in report:
            per_class.append({
                "class_id": i,
                "class_name": name,
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1_score": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            })

    per_class.sort(key=lambda x: x["f1_score"])
    return per_class, report


# ──────────────────────────────────────────────
# Confusion Matrix Analysis
# ──────────────────────────────────────────────

def analyze_confusion_matrix(y_true, y_pred, label_names):
    """Find top 20 most confused class pairs."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    # Zero out diagonal (correct predictions)
    np.fill_diagonal(cm, 0)

    # Find top confused pairs
    confused_pairs = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": label_names[i],
                    "predicted_class": label_names[j],
                    "true_class_id": i,
                    "predicted_class_id": j,
                    "count": int(cm[i, j]),
                })

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    return confused_pairs[:20], cm


def plot_confusion_matrix_top20(cm, label_names, confused_pairs, save_path):
    """Plot confusion matrix heatmap for the top-20 most involved classes."""
    # Collect unique class IDs from top confused pairs
    involved_ids = set()
    for pair in confused_pairs:
        involved_ids.add(pair["true_class_id"])
        involved_ids.add(pair["predicted_class_id"])
        if len(involved_ids) >= 20:
            break

    involved_ids = sorted(list(involved_ids))[:20]
    involved_names = [label_names[i] for i in involved_ids]

    # Extract sub-matrix
    sub_cm = cm[np.ix_(involved_ids, involved_ids)]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        sub_cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=involved_names,
        yticklabels=involved_names,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title("Confusion Matrix -- Top 20 Most Confused Classes", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix plot to {save_path}")


# ──────────────────────────────────────────────
# Calibration Analysis
# ──────────────────────────────────────────────

def calibration_analysis(y_true, y_proba, n_bins=10):
    """
    Bin predictions by confidence (max predicted probability), then check
    actual accuracy within each bin.
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration_data = []

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        mask = (confidences >= low) & (confidences < high)
        if i == n_bins - 1:  # include right edge for last bin
            mask = (confidences >= low) & (confidences <= high)

        n_in_bin = mask.sum()
        if n_in_bin > 0:
            avg_confidence = float(confidences[mask].mean())
            avg_accuracy = float(correct[mask].mean())
        else:
            avg_confidence = (low + high) / 2
            avg_accuracy = 0.0

        calibration_data.append({
            "bin_low": float(low),
            "bin_high": float(high),
            "avg_confidence": avg_confidence,
            "avg_accuracy": avg_accuracy,
            "n_samples": int(n_in_bin),
        })

    return calibration_data


def plot_calibration_curve(calibration_data, save_path):
    """Plot reliability diagram (calibration curve)."""
    confs = [d["avg_confidence"] for d in calibration_data if d["n_samples"] > 0]
    accs = [d["avg_accuracy"] for d in calibration_data if d["n_samples"] > 0]
    counts = [d["n_samples"] for d in calibration_data if d["n_samples"] > 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.plot(confs, accs, "o-", color="steelblue", linewidth=2, markersize=8, label="Model")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax1.set_ylabel("Actual Accuracy", fontsize=12)
    ax1.set_title("Calibration Curve (Reliability Diagram)", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions per bin
    bin_centers = confs
    ax2.bar(bin_centers, counts, width=0.08, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Prediction Distribution by Confidence Bin", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibration plot to {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(config_path: str = None):
    config = load_config(config_path)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and label names
    model_path = PROJECT_ROOT / config["output"]["model_path"]
    label_path = PROJECT_ROOT / config["output"]["label_names_path"]

    print("Loading model and label names...")
    model = joblib.load(model_path)
    label_names = joblib.load(label_path)
    print(f"  Model loaded from {model_path}")
    print(f"  {len(label_names)} class labels loaded")

    # Load test predictions (saved during training)
    preds_path = PROJECT_ROOT / "models" / "saved" / "test_predictions.npz"
    if preds_path.exists():
        print(f"  Loading cached test predictions from {preds_path}")
        preds = np.load(preds_path)
        y_true = preds["y_true"]
        y_pred = preds["y_pred"]
        y_proba = preds["y_proba"]
    else:
        # Re-generate predictions from test data
        print("  Regenerating test predictions...")
        test_path = PROJECT_ROOT / config["data"]["test_path"]
        test_df = pd.read_parquet(test_path)
        y_true = test_df["label"].values

        vec_path = PROJECT_ROOT / config["output"]["vectorizer_path"]
        vectorizer = joblib.load(vec_path)
        X_test = vectorizer.transform(test_df["text"].values)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    num_classes = len(label_names)

    # ── Per-class metrics ──
    print(f"\n{'='*60}")
    print("  Per-Class Metrics (all {0} classes)".format(num_classes))
    print(f"{'='*60}")

    per_class, report = compute_per_class_metrics(y_true, y_pred, label_names)

    print(f"\n  {'Class':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>6}")
    print(f"  {'-'*70}")
    for cls in per_class:
        print(f"  {cls['class_name']:<45} {cls['precision']:>6.3f} {cls['recall']:>6.3f} "
              f"{cls['f1_score']:>6.3f} {cls['support']:>6}")

    # ── Aggregate metrics ──
    print(f"\n{'='*60}")
    print("  Aggregate Metrics")
    print(f"{'='*60}")

    test_acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_rec = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    test_logloss = log_loss(y_true, y_proba, labels=list(range(num_classes)))

    aggregate = {
        "accuracy": float(test_acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "micro_f1": float(micro_f1),
        "log_loss": float(test_logloss),
    }

    for k, v in aggregate.items():
        print(f"  {k:<25} {v:.4f}")

    # ── Confusion matrix analysis ──
    print(f"\n{'='*60}")
    print("  Confusion Matrix -- Top 20 Most Confused Pairs")
    print(f"{'='*60}")

    confused_pairs, cm = analyze_confusion_matrix(y_true, y_pred, label_names)

    print(f"\n  {'True Class':<35} {'Predicted':<35} {'Count':>6}")
    print(f"  {'-'*78}")
    for pair in confused_pairs:
        print(f"  {pair['true_class']:<35} {pair['predicted_class']:<35} {pair['count']:>6}")

    # Plot confusion matrix
    cm_plot_path = PLOTS_DIR / "confusion_matrix_top20.png"
    plot_confusion_matrix_top20(cm, label_names, confused_pairs, cm_plot_path)

    # ── Calibration analysis ──
    print(f"\n{'='*60}")
    print("  Calibration Analysis")
    print(f"{'='*60}")

    cal_data = calibration_analysis(y_true, y_proba)

    print(f"\n  {'Bin':<15} {'Avg Conf':>10} {'Avg Acc':>10} {'N':>8}")
    print(f"  {'-'*45}")
    for d in cal_data:
        bin_label = f"[{d['bin_low']:.1f}, {d['bin_high']:.1f})"
        print(f"  {bin_label:<15} {d['avg_confidence']:>10.3f} {d['avg_accuracy']:>10.3f} {d['n_samples']:>8}")

    # Compute Expected Calibration Error (ECE)
    total_samples = sum(d["n_samples"] for d in cal_data)
    ece = sum(
        d["n_samples"] * abs(d["avg_accuracy"] - d["avg_confidence"])
        for d in cal_data if d["n_samples"] > 0
    ) / total_samples if total_samples > 0 else 0.0
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")

    # Plot calibration curve
    cal_plot_path = PLOTS_DIR / "calibration_curve.png"
    plot_calibration_curve(cal_data, cal_plot_path)

    # ── Save all results ──
    results = {
        "aggregate_metrics": aggregate,
        "expected_calibration_error": float(ece),
        "per_class_metrics": per_class,
        "top_confused_pairs": confused_pairs,
        "calibration_bins": cal_data,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved evaluation results to {RESULTS_PATH}")

    print(f"\n{'='*60}")
    print("  Evaluation Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    config_path = None
    if len(sys.argv) > 1 and sys.argv[1] == "--config" and len(sys.argv) > 2:
        config_path = sys.argv[2]
    main(config_path)
