"""
Download LEDGAR dataset from Hugging Face and save splits as parquet files.

Dataset: coastalcph/lex_glue, config "ledgar"
100-class single-label classification of SEC contract provisions.
"""

import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# Resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_and_save():
    """Download LEDGAR dataset and save each split to data/raw/ as parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading LEDGAR dataset (coastalcph/lex_glue, config='ledgar')...")
    ds = load_dataset("coastalcph/lex_glue", "ledgar")

    label_names = ds["train"].features["label"].names

    # Save each split as parquet
    for split_name in ds:
        out_path = RAW_DIR / f"{split_name}.parquet"
        ds[split_name].to_parquet(str(out_path))
        print(f"  Saved {split_name} -> {out_path} ({len(ds[split_name]):,} rows)")

    # Print dataset statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nSplits: {list(ds.keys())}")
    for split_name in ds:
        print(f"  {split_name}: {len(ds[split_name]):,} examples")

    print(f"\nNumber of classes: {len(label_names)}")

    # Label distribution for each split
    for split_name in ds:
        labels = ds[split_name]["label"]
        label_counts = Counter(labels)
        sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\n-- Label distribution ({split_name}) --")
        print(f"  {'Rank':<5} {'Label':<45} {'Count':>6} {'%':>6}")
        print(f"  {'-' * 65}")
        for rank, (label_id, count) in enumerate(sorted_counts[:15], 1):
            pct = 100 * count / len(labels)
            print(f"  {rank:<5} {label_names[label_id]:<45} {count:>6} {pct:>5.1f}%")
        if len(sorted_counts) > 15:
            print(f"  ... ({len(sorted_counts) - 15} more classes)")

        top_count = sorted_counts[0][1]
        bot_count = sorted_counts[-1][1]
        print(f"\n  Imbalance ratio (largest/smallest): {top_count / bot_count:.1f}x")
        print(f"  Largest class:  {label_names[sorted_counts[0][0]]} ({top_count})")
        print(f"  Smallest class: {label_names[sorted_counts[-1][0]]} ({bot_count})")

    # Text length stats
    print(f"\n-- Text length stats (train) --")
    text_lengths = [len(t) for t in ds["train"]["text"]]
    print(f"  Min:    {min(text_lengths):,} chars")
    print(f"  Max:    {max(text_lengths):,} chars")
    print(f"  Mean:   {sum(text_lengths) / len(text_lengths):,.0f} chars")
    sorted_lengths = sorted(text_lengths)
    median_idx = len(sorted_lengths) // 2
    print(f"  Median: {sorted_lengths[median_idx]:,} chars")

    print("\nDone.")


if __name__ == "__main__":
    download_and_save()
