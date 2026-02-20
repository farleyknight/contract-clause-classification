"""
Feature build orchestrator for contract clause classification.

Loads processed data, calls all feature builders, concatenates into a
feature matrix, and saves the result.

Usage:
    python -m features.build_features
    python -m features.build_features --config configs/baseline.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack, issparse, csr_matrix

from features.text_features import build_all_text_features
from features.structural_features import build_all_structural_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"


def load_config(config_path: str | Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_processed_data(split: str = "train") -> pd.DataFrame:
    """Load processed parquet data for a split."""
    path = PROJECT_ROOT / "data" / "processed" / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run data/processed/process_data.py first."
        )
    return pd.read_parquet(path)


def build_features(config_path: str | Path = None):
    """Main feature building pipeline.

    1. Load processed data
    2. Build text features (TF-IDF, stats, keywords)
    3. Build structural features (if enabled)
    4. Concatenate into single feature matrix
    5. Save to features/feature_matrix.parquet
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG

    config = load_config(config_path)
    features_config = config.get("features", {})

    print("=" * 60)
    print(f"BUILDING FEATURES ({config.get('experiment_name', 'default')})")
    print("=" * 60)

    # Load processed data
    print("\nLoading processed data...")
    df_train = load_processed_data("train")
    print(f"  Train: {df_train.shape}")

    # Build text features
    print("\nBuilding text features...")
    text_features, vectorizer, dense_col_names = build_all_text_features(
        df_train, config, fit=True
    )
    print(f"  Text features shape: {text_features.shape}")

    # Build structural features if enabled
    structural_df = None
    structural_col_names = []
    if features_config.get("use_structural", False):
        print("\nBuilding structural features...")
        structural_df = build_all_structural_features(df_train)
        structural_col_names = structural_df.columns.tolist()
        print(f"  Structural features shape: {structural_df.shape}")

    # Combine all features
    print("\nCombining features...")
    if structural_df is not None:
        struct_sparse = csr_matrix(structural_df.values)
        if issparse(text_features):
            combined = hstack([text_features, struct_sparse])
        else:
            combined = np.hstack([text_features, structural_df.values])
    else:
        combined = text_features

    print(f"  Combined feature matrix shape: {combined.shape}")

    # Save feature matrix as parquet (convert sparse to dense for storage)
    output_dir = PROJECT_ROOT / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_matrix.parquet"

    if issparse(combined):
        dense_matrix = combined.toarray()
    else:
        dense_matrix = np.array(combined)

    # Generate column names
    n_tfidf_cols = (
        text_features.shape[1]
        - len(dense_col_names)
        if not features_config.get("use_tfidf", True)
        else (
            text_features.shape[1]
            - len(dense_col_names)
        )
    )
    # Actually, text_features already includes TF-IDF + dense cols
    # We need to reconstruct the column names
    tfidf_col_count = combined.shape[1] - len(dense_col_names) - len(structural_col_names)
    col_names = (
        [f"tfidf_{i}" for i in range(tfidf_col_count)]
        + dense_col_names
        + structural_col_names
    )

    feature_df = pd.DataFrame(dense_matrix, columns=col_names)

    # Add label column for convenience
    feature_df["label"] = df_train["label"].values

    feature_df.to_parquet(output_path, index=False)
    print(f"\n  Saved feature matrix: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    print(f"  Total samples: {feature_df.shape[0]:,}")
    print(f"  Total features: {combined.shape[1]:,}")
    if tfidf_col_count > 0:
        print(f"  TF-IDF features: {tfidf_col_count:,}")
    if dense_col_names:
        print(f"  Dense text features: {len(dense_col_names)}")
        print(f"    Columns: {dense_col_names}")
    if structural_col_names:
        print(f"  Structural features: {len(structural_col_names)}")
        print(f"    Columns: {structural_col_names}")

    # Basic statistics of dense features
    if dense_col_names:
        print(f"\n  Dense feature statistics (train):")
        for col in dense_col_names:
            vals = feature_df[col]
            print(f"    {col}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
                  f"min={vals.min():.3f}, max={vals.max():.3f}")

    print("\nDone.")
    return feature_df


def main():
    parser = argparse.ArgumentParser(description="Build feature matrix")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    build_features(args.config)


if __name__ == "__main__":
    main()
