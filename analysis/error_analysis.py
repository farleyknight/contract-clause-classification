"""
Contract Clause Classification — Error Analysis
=================================================
Identifies the worst predictions (high-confidence misclassifications),
categorizes failure modes, and prints a detailed error report.

Usage:
    python -m analysis.error_analysis
    python -m analysis.error_analysis --config configs/baseline.yaml
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import yaml
import joblib


# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "baseline.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Failure Mode Categorization
# ──────────────────────────────────────────────

def compute_class_vocabularies(texts, labels, label_names):
    """Build a vocabulary set for each class from the training data."""
    class_vocabs = {}
    for text, label in zip(texts, labels):
        name = label_names[label]
        if name not in class_vocabs:
            class_vocabs[name] = set()
        words = set(text.lower().split())
        class_vocabs[name].update(words)
    return class_vocabs


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def categorize_failure(
    text, true_label, pred_label, confidence, label_names,
    class_vocabs, class_counts, median_length
):
    """
    Categorize a misclassification into failure mode(s).

    Failure modes:
    - vocabulary_overlap: high Jaccard similarity between true and predicted class vocabularies
    - class_ambiguity: predicted class is semantically close (high vocab overlap)
    - short_text: text length is below median
    - rare_class: true class has low support in training data
    """
    modes = []

    true_name = label_names[true_label]
    pred_name = label_names[pred_label]

    # Vocabulary overlap
    if true_name in class_vocabs and pred_name in class_vocabs:
        jaccard = jaccard_similarity(class_vocabs[true_name], class_vocabs[pred_name])
        if jaccard > 0.3:
            modes.append("vocabulary_overlap")
        if jaccard > 0.2:
            modes.append("class_ambiguity")

    # Short text
    if len(text.split()) < median_length:
        modes.append("short_text")

    # Rare class
    true_count = class_counts.get(true_label, 0)
    total = sum(class_counts.values())
    if true_count < total / (len(label_names) * 2):
        modes.append("rare_class")

    if not modes:
        modes.append("other")

    return modes


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(config_path: str = None):
    config = load_config(config_path)

    # Load label names
    label_path = PROJECT_ROOT / config["output"]["label_names_path"]
    label_names = joblib.load(label_path)

    # Load test predictions
    preds_path = PROJECT_ROOT / "models" / "saved" / "test_predictions.npz"
    preds = np.load(preds_path)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    y_proba = preds["y_proba"]

    # Load test text data
    test_path = PROJECT_ROOT / config["data"]["test_path"]
    test_df = pd.read_parquet(test_path)
    texts = test_df["text"].values

    # Load train data for vocabulary analysis
    train_path = PROJECT_ROOT / config["data"]["train_path"]
    train_df = pd.read_parquet(train_path)
    train_texts = train_df["text"].values
    train_labels = train_df["label"].values

    # Build class vocabularies and statistics
    print("Building class vocabularies from training data...")
    class_vocabs = compute_class_vocabularies(train_texts, train_labels, label_names)
    class_counts = Counter(train_labels)

    # Compute median text length (in words)
    text_lengths = [len(t.split()) for t in texts]
    median_length = float(np.median(text_lengths))

    # Find misclassifications
    confidences = np.max(y_proba, axis=1)
    wrong_mask = y_true != y_pred
    wrong_indices = np.where(wrong_mask)[0]

    if len(wrong_indices) == 0:
        print("No misclassifications found. Model is perfect on test set.")
        return

    # Sort by confidence (descending) to find worst predictions
    wrong_confidences = confidences[wrong_indices]
    sorted_order = np.argsort(wrong_confidences)[::-1]
    worst_indices = wrong_indices[sorted_order][:30]

    print(f"\n{'='*80}")
    print(f"  Error Analysis: Top 30 Highest-Confidence Misclassifications")
    print(f"{'='*80}")
    print(f"\n  Total test samples:          {len(y_true):,}")
    print(f"  Total misclassifications:    {len(wrong_indices):,}")
    print(f"  Error rate:                  {len(wrong_indices)/len(y_true)*100:.1f}%")
    print(f"  Median text length (words):  {median_length:.0f}")

    # Categorize all errors
    all_failure_modes = Counter()
    for idx in wrong_indices:
        modes = categorize_failure(
            texts[idx], y_true[idx], y_pred[idx], confidences[idx],
            label_names, class_vocabs, class_counts, median_length
        )
        for mode in modes:
            all_failure_modes[mode] += 1

    print(f"\n  Failure Mode Distribution (across all {len(wrong_indices)} errors):")
    print(f"  {'-'*50}")
    for mode, count in all_failure_modes.most_common():
        pct = 100.0 * count / len(wrong_indices)
        print(f"    {mode:<25} {count:>6} ({pct:>5.1f}%)")

    # Detailed report for top 30
    print(f"\n{'='*80}")
    print(f"  Detailed Error Report (Top 30)")
    print(f"{'='*80}")

    for rank, idx in enumerate(worst_indices, 1):
        true_name = label_names[y_true[idx]]
        pred_name = label_names[y_pred[idx]]
        conf = confidences[idx]
        text = texts[idx]

        modes = categorize_failure(
            text, y_true[idx], y_pred[idx], conf,
            label_names, class_vocabs, class_counts, median_length
        )

        # Compute vocab overlap for context
        vocab_overlap = 0.0
        if true_name in class_vocabs and pred_name in class_vocabs:
            vocab_overlap = jaccard_similarity(class_vocabs[true_name], class_vocabs[pred_name])

        print(f"\n  [{rank}] Confidence: {conf:.4f}")
        print(f"      True class:      {true_name}")
        print(f"      Predicted class:  {pred_name}")
        print(f"      Failure modes:    {', '.join(modes)}")
        print(f"      Vocab overlap:    {vocab_overlap:.3f}")
        print(f"      Text length:      {len(text.split())} words")
        print(f"      Text (first 300): {text[:300]}{'...' if len(text) > 300 else ''}")

    # Summary by failure mode with examples
    print(f"\n{'='*80}")
    print(f"  Failure Mode Examples")
    print(f"{'='*80}")

    mode_examples = {
        "vocabulary_overlap": [],
        "class_ambiguity": [],
        "short_text": [],
        "rare_class": [],
        "other": [],
    }

    for idx in worst_indices:
        modes = categorize_failure(
            texts[idx], y_true[idx], y_pred[idx], confidences[idx],
            label_names, class_vocabs, class_counts, median_length
        )
        for mode in modes:
            if mode in mode_examples and len(mode_examples[mode]) < 3:
                mode_examples[mode].append({
                    "text": texts[idx][:200],
                    "true": label_names[y_true[idx]],
                    "pred": label_names[y_pred[idx]],
                    "confidence": float(confidences[idx]),
                })

    for mode, examples in mode_examples.items():
        if examples:
            print(f"\n  -- {mode.upper()} --")
            for i, ex in enumerate(examples, 1):
                print(f"    Example {i}:")
                print(f"      True: {ex['true']} | Pred: {ex['pred']} (conf: {ex['confidence']:.4f})")
                print(f"      Text: {ex['text']}...")

    print(f"\n{'='*80}")
    print("  Error Analysis Complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    config_path = None
    if len(sys.argv) > 1 and sys.argv[1] == "--config" and len(sys.argv) > 2:
        config_path = sys.argv[2]
    main(config_path)
