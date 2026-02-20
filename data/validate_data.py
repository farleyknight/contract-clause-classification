"""
Validate the downloaded LEDGAR dataset.

Checks:
- Schema: verify text and label columns exist with correct types
- Class distribution: plot and print, flag severe imbalances
- Duplicate detection: within and across splits
- Text length distribution: plot histogram
- Save plots to data/plots/
"""

import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PLOTS_DIR = PROJECT_ROOT / "data" / "plots"

SPLITS = ["train", "validation", "test"]


def load_splits():
    """Load all raw parquet splits into a dict of DataFrames."""
    dfs = {}
    for split in SPLITS:
        path = RAW_DIR / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run data/processed/download.py first."
            )
        dfs[split] = pd.read_parquet(path)
    return dfs


def validate_schema(dfs):
    """Verify text and label columns exist with correct types."""
    print("=" * 60)
    print("SCHEMA VALIDATION")
    print("=" * 60)
    all_ok = True
    for split, df in dfs.items():
        issues = []
        if "text" not in df.columns:
            issues.append("missing 'text' column")
        elif df["text"].dtype != object:
            issues.append(f"'text' column has dtype {df['text'].dtype}, expected object/string")

        if "label" not in df.columns:
            issues.append("missing 'label' column")
        elif not np.issubdtype(df["label"].dtype, np.integer):
            issues.append(f"'label' column has dtype {df['label'].dtype}, expected int")

        # Check for nulls
        if "text" in df.columns and df["text"].isna().any():
            n_null = df["text"].isna().sum()
            issues.append(f"{n_null} null values in 'text'")
        if "label" in df.columns and df["label"].isna().any():
            n_null = df["label"].isna().sum()
            issues.append(f"{n_null} null values in 'label'")

        if issues:
            all_ok = False
            print(f"  [{split}] ISSUES: {'; '.join(issues)}")
        else:
            print(f"  [{split}] OK -- {len(df):,} rows, columns: {list(df.columns)}")

    if all_ok:
        print("\n  All schema checks passed.")
    else:
        print("\n  WARNING: Some schema checks failed!")
    return all_ok


def analyze_class_distribution(dfs):
    """Analyze and plot class distribution, flag severe imbalances."""
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for split, df in dfs.items():
        label_counts = Counter(df["label"])
        sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        counts = [c for _, c in sorted_counts]

        total = len(df)
        top_count = counts[0]
        bot_count = counts[-1]
        imbalance_ratio = top_count / bot_count if bot_count > 0 else float("inf")

        print(f"\n  [{split}]")
        print(f"    Unique classes: {len(label_counts)}")
        print(f"    Imbalance ratio: {imbalance_ratio:.1f}x")
        print(f"    Largest class:  label={sorted_counts[0][0]} ({top_count:,} samples, {100*top_count/total:.1f}%)")
        print(f"    Smallest class: label={sorted_counts[-1][0]} ({bot_count:,} samples, {100*bot_count/total:.1f}%)")

        # Flag severe imbalances (>100x)
        if imbalance_ratio > 100:
            print(f"    *** SEVERE IMBALANCE DETECTED ({imbalance_ratio:.0f}x) ***")
        elif imbalance_ratio > 50:
            print(f"    ** Moderate imbalance ({imbalance_ratio:.0f}x) **")

        # Classes with fewer than 10 examples
        rare_classes = [(lid, c) for lid, c in sorted_counts if c < 10]
        if rare_classes:
            print(f"    Classes with <10 examples: {len(rare_classes)}")

        # Plot class distribution
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(counts)), counts, color="steelblue", alpha=0.8)
        ax.set_xlabel("Class (sorted by frequency)")
        ax.set_ylabel("Count")
        ax.set_title(f"Class Distribution -- {split} ({len(label_counts)} classes)")
        ax.set_yscale("log")
        plt.tight_layout()
        plot_path = PLOTS_DIR / f"class_distribution_{split}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"    Plot saved: {plot_path}")


def detect_duplicates(dfs):
    """Check for duplicate texts within and across splits."""
    print("\n" + "=" * 60)
    print("DUPLICATE DETECTION")
    print("=" * 60)

    # Within-split duplicates
    for split, df in dfs.items():
        n_total = len(df)
        n_unique = df["text"].nunique()
        n_dupes = n_total - n_unique
        pct = 100 * n_dupes / n_total if n_total > 0 else 0
        if n_dupes > 0:
            print(f"  [{split}] {n_dupes:,} duplicate texts ({pct:.1f}%) out of {n_total:,}")
        else:
            print(f"  [{split}] No duplicates found ({n_total:,} rows)")

    # Cross-split duplicates
    print("\n  Cross-split overlap:")
    split_names = list(dfs.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            texts_1 = set(dfs[s1]["text"])
            texts_2 = set(dfs[s2]["text"])
            overlap = texts_1 & texts_2
            if overlap:
                print(f"    {s1} <-> {s2}: {len(overlap):,} shared texts")
            else:
                print(f"    {s1} <-> {s2}: No overlap")


def analyze_text_lengths(dfs):
    """Plot text length distributions."""
    print("\n" + "=" * 60)
    print("TEXT LENGTH DISTRIBUTION")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(dfs), figsize=(5 * len(dfs), 4), squeeze=False)
    axes = axes[0]

    for idx, (split, df) in enumerate(dfs.items()):
        lengths = df["text"].str.len()
        print(f"\n  [{split}]")
        print(f"    Min:    {lengths.min():,} chars")
        print(f"    Max:    {lengths.max():,} chars")
        print(f"    Mean:   {lengths.mean():,.0f} chars")
        print(f"    Median: {lengths.median():,.0f} chars")
        print(f"    Std:    {lengths.std():,.0f} chars")

        # Plot
        ax = axes[idx]
        ax.hist(lengths, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Text Length (chars)")
        ax.set_ylabel("Count")
        ax.set_title(f"Text Length -- {split}")

    plt.tight_layout()
    plot_path = PLOTS_DIR / "text_length_distribution.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")


def run_all_validations():
    """Run all validation checks."""
    print("Loading raw data...\n")
    dfs = load_splits()

    validate_schema(dfs)
    analyze_class_distribution(dfs)
    detect_duplicates(dfs)
    analyze_text_lengths(dfs)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_validations()
