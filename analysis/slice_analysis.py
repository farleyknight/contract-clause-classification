"""
Contract Clause Classification — Slice Analysis
=================================================
Analyzes model performance across different data slices:
- Clause length (short / medium / long)
- Class rarity (frequent / moderate / rare)
- Vocabulary overlap between classes

Saves slice performance plots to analysis/plots/.

Usage:
    python -m analysis.slice_analysis
    python -m analysis.slice_analysis --config configs/baseline.yaml
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score


# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "baseline.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Slice Definitions
# ──────────────────────────────────────────────

def assign_length_slice(texts):
    """Assign each text to a length bucket: short / medium / long."""
    word_counts = np.array([len(t.split()) for t in texts])
    p33 = np.percentile(word_counts, 33)
    p66 = np.percentile(word_counts, 66)

    slices = []
    for wc in word_counts:
        if wc <= p33:
            slices.append("short")
        elif wc <= p66:
            slices.append("medium")
        else:
            slices.append("long")

    return np.array(slices), word_counts, p33, p66


def assign_rarity_slice(labels, train_label_counts, label_names):
    """
    Assign each sample's class to a rarity bucket based on training frequency.

    Buckets:
    - frequent: top third of classes by count
    - moderate: middle third
    - rare: bottom third
    """
    # Sort classes by training count
    sorted_classes = sorted(train_label_counts.items(), key=lambda x: x[1], reverse=True)
    n_classes = len(sorted_classes)
    third = n_classes // 3

    frequent_classes = set(c for c, _ in sorted_classes[:third])
    rare_classes = set(c for c, _ in sorted_classes[2 * third:])
    moderate_classes = set(c for c, _ in sorted_classes[third:2 * third])

    slices = []
    for label in labels:
        if label in frequent_classes:
            slices.append("frequent")
        elif label in rare_classes:
            slices.append("rare")
        else:
            slices.append("moderate")

    return np.array(slices)


def compute_class_vocab_overlaps(train_texts, train_labels, label_names):
    """
    Compute pairwise Jaccard similarity of class vocabularies.
    Returns a matrix of shape (n_classes, n_classes).
    """
    n_classes = len(label_names)

    # Build vocabulary per class
    class_vocabs = {}
    for text, label in zip(train_texts, train_labels):
        if label not in class_vocabs:
            class_vocabs[label] = set()
        class_vocabs[label].update(text.lower().split())

    # Compute pairwise Jaccard
    overlap_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(i, n_classes):
            vocab_i = class_vocabs.get(i, set())
            vocab_j = class_vocabs.get(j, set())
            if vocab_i or vocab_j:
                intersection = len(vocab_i & vocab_j)
                union = len(vocab_i | vocab_j)
                jaccard = intersection / union if union > 0 else 0.0
            else:
                jaccard = 0.0
            overlap_matrix[i, j] = jaccard
            overlap_matrix[j, i] = jaccard

    return overlap_matrix


def assign_overlap_slice(labels, overlap_matrix):
    """
    Assign each sample to a vocab-overlap bucket based on its class's
    average vocabulary overlap with all other classes.

    Buckets: low_overlap / medium_overlap / high_overlap
    """
    # Average off-diagonal overlap for each class
    n_classes = overlap_matrix.shape[0]
    avg_overlaps = np.zeros(n_classes)
    for i in range(n_classes):
        others = [overlap_matrix[i, j] for j in range(n_classes) if j != i]
        avg_overlaps[i] = np.mean(others) if others else 0.0

    p33 = np.percentile(avg_overlaps, 33)
    p66 = np.percentile(avg_overlaps, 66)

    slices = []
    for label in labels:
        ov = avg_overlaps[label]
        if ov <= p33:
            slices.append("low_overlap")
        elif ov <= p66:
            slices.append("medium_overlap")
        else:
            slices.append("high_overlap")

    return np.array(slices), avg_overlaps


# ──────────────────────────────────────────────
# Metrics per Slice
# ──────────────────────────────────────────────

def compute_slice_metrics(y_true, y_pred, slice_labels, slice_name):
    """Compute accuracy and macro-F1 for each value in the slice."""
    unique_slices = sorted(set(slice_labels))
    metrics = []

    for s in unique_slices:
        mask = slice_labels == s
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        macro_f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        micro_f1 = f1_score(y_true[mask], y_pred[mask], average="micro", zero_division=0)
        metrics.append({
            "slice_name": slice_name,
            "slice_value": s,
            "n_samples": int(mask.sum()),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
        })

    return metrics


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_slice_comparison(all_metrics, save_path):
    """Plot grouped bar charts of accuracy and macro-F1 across slices."""
    df = pd.DataFrame(all_metrics)

    slice_groups = df["slice_name"].unique()
    n_groups = len(slice_groups)

    fig, axes = plt.subplots(n_groups, 2, figsize=(14, 5 * n_groups))
    if n_groups == 1:
        axes = axes.reshape(1, -1)

    for idx, slice_name in enumerate(slice_groups):
        group = df[df["slice_name"] == slice_name].copy()

        # Order slices meaningfully
        if slice_name == "clause_length":
            order = ["short", "medium", "long"]
        elif slice_name == "class_rarity":
            order = ["frequent", "moderate", "rare"]
        elif slice_name == "vocab_overlap":
            order = ["low_overlap", "medium_overlap", "high_overlap"]
        else:
            order = group["slice_value"].tolist()

        group["slice_value"] = pd.Categorical(group["slice_value"], categories=order, ordered=True)
        group = group.sort_values("slice_value")

        # Accuracy plot
        ax = axes[idx, 0]
        bars = ax.bar(
            group["slice_value"].astype(str),
            group["accuracy"],
            color="steelblue",
            edgecolor="black",
            alpha=0.8,
        )
        for bar, n in zip(bars, group["n_samples"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={n:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(f"{slice_name} -- Accuracy", fontsize=12)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        # Macro F1 plot
        ax = axes[idx, 1]
        bars = ax.bar(
            group["slice_value"].astype(str),
            group["macro_f1"],
            color="coral",
            edgecolor="black",
            alpha=0.8,
        )
        for bar, n in zip(bars, group["n_samples"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={n:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(f"{slice_name} -- Macro F1", fontsize=12)
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved slice performance plot to {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(config_path: str = None):
    config = load_config(config_path)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load label names
    label_path = PROJECT_ROOT / config["output"]["label_names_path"]
    label_names = joblib.load(label_path)

    # Load test predictions
    preds_path = PROJECT_ROOT / "models" / "saved" / "test_predictions.npz"
    preds = np.load(preds_path)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]

    # Load test text data
    test_path = PROJECT_ROOT / config["data"]["test_path"]
    test_df = pd.read_parquet(test_path)
    texts = test_df["text"].values

    # Load training data for rarity + vocab stats
    train_path = PROJECT_ROOT / config["data"]["train_path"]
    train_df = pd.read_parquet(train_path)
    train_label_counts = Counter(train_df["label"].values)

    all_metrics = []

    # ── 1. Performance by clause length ──
    print(f"\n{'='*60}")
    print("  Slice Analysis: Clause Length")
    print(f"{'='*60}")

    length_slices, word_counts, p33, p66 = assign_length_slice(texts)
    print(f"  Thresholds: short <= {p33:.0f} words, medium <= {p66:.0f} words, long > {p66:.0f} words")

    length_metrics = compute_slice_metrics(y_true, y_pred, length_slices, "clause_length")
    all_metrics.extend(length_metrics)

    for m in length_metrics:
        print(f"    {m['slice_value']:<15} n={m['n_samples']:>6,}  "
              f"acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    # ── 2. Performance by class rarity ──
    print(f"\n{'='*60}")
    print("  Slice Analysis: Class Rarity")
    print(f"{'='*60}")

    rarity_slices = assign_rarity_slice(y_true, train_label_counts, label_names)
    rarity_metrics = compute_slice_metrics(y_true, y_pred, rarity_slices, "class_rarity")
    all_metrics.extend(rarity_metrics)

    for m in rarity_metrics:
        print(f"    {m['slice_value']:<15} n={m['n_samples']:>6,}  "
              f"acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    # ── 3. Performance by vocabulary overlap ──
    print(f"\n{'='*60}")
    print("  Slice Analysis: Vocabulary Overlap Between Classes")
    print(f"{'='*60}")

    print("  Computing pairwise class vocabulary Jaccard similarities...")
    overlap_matrix = compute_class_vocab_overlaps(
        train_df["text"].values, train_df["label"].values, label_names
    )

    overlap_slices, avg_overlaps = assign_overlap_slice(y_true, overlap_matrix)
    overlap_metrics = compute_slice_metrics(y_true, y_pred, overlap_slices, "vocab_overlap")
    all_metrics.extend(overlap_metrics)

    for m in overlap_metrics:
        print(f"    {m['slice_value']:<20} n={m['n_samples']:>6,}  "
              f"acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    # Show top-10 highest average overlap classes
    print(f"\n  Top 10 classes by average vocabulary overlap:")
    top_overlap_classes = np.argsort(avg_overlaps)[::-1][:10]
    for cid in top_overlap_classes:
        print(f"    {label_names[cid]:<45} avg_overlap={avg_overlaps[cid]:.3f}")

    # ── Save plots ──
    plot_slice_comparison(all_metrics, PLOTS_DIR / "slice_performance.png")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  Slice Analysis Summary")
    print(f"{'='*60}")

    summary_df = pd.DataFrame(all_metrics)
    for slice_name in summary_df["slice_name"].unique():
        group = summary_df[summary_df["slice_name"] == slice_name]
        best = group.loc[group["accuracy"].idxmax()]
        worst = group.loc[group["accuracy"].idxmin()]
        gap = best["accuracy"] - worst["accuracy"]
        print(f"\n  {slice_name}:")
        print(f"    Best:  {best['slice_value']} (acc={best['accuracy']:.4f})")
        print(f"    Worst: {worst['slice_value']} (acc={worst['accuracy']:.4f})")
        print(f"    Gap:   {gap:.4f}")

    print(f"\n{'='*60}")
    print("  Slice Analysis Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    config_path = None
    if len(sys.argv) > 1 and sys.argv[1] == "--config" and len(sys.argv) > 2:
        config_path = sys.argv[2]
    main(config_path)
