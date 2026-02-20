"""
Process raw LEDGAR data: clean text, add metadata features, save to parquet.

Input:  data/raw/{split}.parquet
Output: data/processed/{split}.parquet
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["train", "validation", "test"]


def clean_text(text: str) -> str:
    """Normalize whitespace, strip leading/trailing space, collapse internal whitespace."""
    if not isinstance(text, str):
        return ""
    # Replace various whitespace chars (tabs, newlines, etc.) with single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add metadata columns: text_length, word_count, avg_word_length."""
    df = df.copy()
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    # avg_word_length: total chars in words / number of words
    df["avg_word_length"] = df["text"].apply(
        lambda t: (
            sum(len(w) for w in t.split()) / len(t.split())
            if t and len(t.split()) > 0
            else 0.0
        )
    )
    return df


def process_split(split_name: str) -> pd.DataFrame:
    """Load, clean, and enrich a single split."""
    raw_path = RAW_DIR / f"{split_name}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Run data/processed/download.py first."
        )

    df = pd.read_parquet(raw_path)
    n_before = len(df)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Remove empty texts
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    n_after = len(df)
    n_removed = n_before - n_after

    # Add metadata
    df = add_metadata(df)

    return df, n_removed


def process_all():
    """Process all splits and save to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Processing raw LEDGAR data...\n")
    for split in SPLITS:
        df, n_removed = process_split(split)
        out_path = PROCESSED_DIR / f"{split}.parquet"
        df.to_parquet(out_path, index=False)

        print(f"  [{split}]")
        print(f"    Rows: {len(df):,} (removed {n_removed} empty)")
        print(f"    Columns: {list(df.columns)}")
        print(f"    Text length -- mean: {df['text_length'].mean():.0f}, "
              f"median: {df['text_length'].median():.0f}")
        print(f"    Word count  -- mean: {df['word_count'].mean():.0f}, "
              f"median: {df['word_count'].median():.0f}")
        print(f"    Saved: {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    process_all()
