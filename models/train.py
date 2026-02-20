"""
Contract Clause Classification — Full Training Pipeline
========================================================
Loads TF-IDF features, trains a LightGBM multiclass classifier with
5-fold stratified cross-validation, logs experiments to MLflow, and
saves the trained model.

Usage:
    python -m models.train
    python -m models.train --config configs/baseline.yaml
"""

import sys
import time
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import yaml
import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from scipy import sparse


# ──────────────────────────────────────────────
# Helpers
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


def ensure_dirs():
    """Create output directories if they don't exist."""
    (PROJECT_ROOT / "models" / "saved").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "analysis" / "plots").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "features").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)


def prepare_data(config: dict):
    """
    Load data from parquet if available, otherwise fetch from HuggingFace
    and build the feature matrix + processed data from scratch.

    Returns:
        X_train, X_val, X_test: sparse TF-IDF feature matrices
        y_train, y_val, y_test: label arrays
        label_names: list of class names
        vectorizer: fitted TfidfVectorizer
    """
    feature_path = PROJECT_ROOT / config["output"]["feature_matrix_path"]
    train_path = PROJECT_ROOT / config["data"]["train_path"]
    val_path = PROJECT_ROOT / config["data"]["validation_path"]
    test_path = PROJECT_ROOT / config["data"]["test_path"]

    # Check if processed data exists
    if feature_path.exists() and train_path.exists():
        print("Loading pre-built feature matrix and labels...")
        feature_df = pd.read_parquet(feature_path)
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

        y_train = train_df["label"].values
        y_val = val_df["label"].values
        y_test = test_df["label"].values
        label_names = (
            sorted(train_df["label_name"].unique().tolist())
            if "label_name" in train_df.columns
            else None
        )

        # Reconstruct sparse matrices from feature_df
        if "split" in feature_df.columns:
            train_mask = feature_df["split"] == "train"
            val_mask = feature_df["split"] == "validation"
            test_mask = feature_df["split"] == "test"

            feat_cols = [c for c in feature_df.columns if c not in ("split", "label")]
            X_train = sparse.csr_matrix(feature_df.loc[train_mask, feat_cols].values)
            X_val = sparse.csr_matrix(feature_df.loc[val_mask, feat_cols].values)
            X_test = sparse.csr_matrix(feature_df.loc[test_mask, feat_cols].values)

            vectorizer = None
            return X_train, X_val, X_test, y_train, y_val, y_test, label_names, vectorizer

    # Fallback: load from HuggingFace and build everything
    print("Feature matrix not found. Building from HuggingFace dataset...")
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    label_names = ds["train"].features["label"].names

    # Save processed label data
    for split_name in ["train", "validation", "test"]:
        split_df = pd.DataFrame({
            "text": ds[split_name]["text"],
            "label": ds[split_name]["label"],
            "label_name": [label_names[l] for l in ds[split_name]["label"]],
        })
        split_path = PROJECT_ROOT / config["data"][f"{split_name}_path"]
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_parquet(split_path, index=False)
        print(f"  Saved {split_name} data to {split_path} ({len(split_df):,} rows)")

    # Build TF-IDF features
    tfidf_cfg = config.get("tfidf", {})
    ngram_range = tuple(tfidf_cfg.get("ngram_range", [1, 2]))

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 10000),
        ngram_range=ngram_range,
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        min_df=tfidf_cfg.get("min_df", 3),
        max_df=tfidf_cfg.get("max_df", 0.95),
    )

    X_train = vectorizer.fit_transform(ds["train"]["text"])
    X_val = vectorizer.transform(ds["validation"]["text"])
    X_test = vectorizer.transform(ds["test"]["text"])

    y_train = np.array(ds["train"]["label"])
    y_val = np.array(ds["validation"]["label"])
    y_test = np.array(ds["test"]["label"])

    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"  X_train shape: {X_train.shape}")

    # Save vectorizer
    vec_path = PROJECT_ROOT / config["output"]["vectorizer_path"]
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vec_path)
    print(f"  Saved vectorizer to {vec_path}")

    # Save feature matrix metadata
    feature_meta = {
        "n_features": X_train.shape[1],
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        "n_test": X_test.shape[0],
        "tfidf_config": tfidf_cfg,
    }
    feature_meta_path = feature_path.with_suffix(".json")
    with open(feature_meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)
    print(f"  Saved feature metadata to {feature_meta_path}")

    return X_train, X_val, X_test, y_train, y_val, y_test, label_names, vectorizer


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute balanced class weights inversely proportional to class frequency."""
    counts = Counter(y)
    n_samples = len(y)
    weights = np.ones(num_classes)
    for cls, count in counts.items():
        weights[cls] = n_samples / (num_classes * count)
    return weights


# ──────────────────────────────────────────────
# Cross-Validation
# ──────────────────────────────────────────────

def build_lgbm_params(config: dict) -> dict:
    """Extract LightGBM parameters from config, handling the nested structure."""
    model_params = config["model"]["params"].copy()
    # Remove class_weight since we handle it via sample_weight
    model_params.pop("class_weight", None)
    return model_params


def run_cross_validation(X_train, y_train, config: dict, class_weights: np.ndarray):
    """Run stratified k-fold cross-validation and return fold metrics."""
    train_cfg = config["training"]
    n_folds = train_cfg.get("cv_folds", 5)
    num_classes = 100

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    lgbm_params = build_lgbm_params(config)

    fold_metrics = []

    print(f"\n{'='*60}")
    print(f"  {n_folds}-Fold Stratified Cross-Validation")
    print(f"{'='*60}")

    sample_weights = np.array([class_weights[label] for label in y_train])

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        t0 = time.time()

        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        w_fold_train = sample_weights[train_idx]

        fold_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            **lgbm_params,
        )

        fold_model.fit(
            X_fold_train, y_fold_train,
            sample_weight=w_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            eval_metric=train_cfg.get("eval_metric", "multi_logloss"),
            callbacks=[
                lgb.early_stopping(train_cfg.get("early_stopping_rounds", 50), verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_fold_pred = fold_model.predict(X_fold_val)
        y_fold_proba = fold_model.predict_proba(X_fold_val)

        fold_acc = accuracy_score(y_fold_val, y_fold_pred)
        fold_macro_f1 = f1_score(y_fold_val, y_fold_pred, average="macro", zero_division=0)
        fold_micro_f1 = f1_score(y_fold_val, y_fold_pred, average="micro", zero_division=0)
        fold_logloss = log_loss(y_fold_val, y_fold_proba, labels=list(range(num_classes)))

        elapsed = time.time() - t0
        best_iter = getattr(fold_model, "best_iteration_", lgbm_params.get("n_estimators", 500))

        fold_metrics.append({
            "fold": fold_idx + 1,
            "accuracy": fold_acc,
            "macro_f1": fold_macro_f1,
            "micro_f1": fold_micro_f1,
            "log_loss": fold_logloss,
            "best_iteration": best_iter,
            "time_s": elapsed,
        })

        print(f"  Accuracy:    {fold_acc:.4f}")
        print(f"  Macro F1:    {fold_macro_f1:.4f}")
        print(f"  Micro F1:    {fold_micro_f1:.4f}")
        print(f"  Log Loss:    {fold_logloss:.4f}")
        print(f"  Best iter:   {best_iter}")
        print(f"  Time:        {elapsed:.1f}s")

    return fold_metrics


# ──────────────────────────────────────────────
# Final Training
# ──────────────────────────────────────────────

def train_final_model(X_train, y_train, X_val, y_val, config: dict, class_weights: np.ndarray):
    """Train the final model on the full training set with validation-based early stopping."""
    train_cfg = config["training"]
    lgbm_params = build_lgbm_params(config)
    num_classes = 100

    print(f"\n{'='*60}")
    print("  Training Final Model")
    print(f"{'='*60}")

    sample_weights = np.array([class_weights[label] for label in y_train])

    t0 = time.time()

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        **lgbm_params,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        eval_metric=train_cfg.get("eval_metric", "multi_logloss"),
        callbacks=[
            lgb.early_stopping(train_cfg.get("early_stopping_rounds", 50), verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    elapsed = time.time() - t0
    best_iter = getattr(model, "best_iteration_", lgbm_params.get("n_estimators", 500))

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Best iteration: {best_iter}")

    return model, elapsed, best_iter


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(config_path: str = None):
    config = load_config(config_path)
    ensure_dirs()

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, label_names, vectorizer = prepare_data(config)

    # If label_names came back None (loaded from parquet without label_name column),
    # try loading from the dataset directly
    if label_names is None:
        ds = load_dataset("coastalcph/lex_glue", "ledgar")
        label_names = ds["train"].features["label"].names

    num_classes = 100

    # Compute class weights (balanced)
    class_weights = compute_class_weights(y_train, num_classes)

    print(f"\nDataset summary:")
    print(f"  Train:      {X_train.shape[0]:,} samples, {X_train.shape[1]:,} features")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Test:       {X_test.shape[0]:,} samples")
    print(f"  Classes:    {num_classes}")

    # MLflow setup
    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(str(PROJECT_ROOT / mlflow_cfg["tracking_uri"]))
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=config.get("experiment_name", "baseline")):
        # Log config parameters
        model_params = config["model"]["params"]
        mlflow.log_params({
            "experiment_name": config.get("experiment_name", "baseline"),
            "model_type": config["model"]["type"],
            "n_estimators": model_params.get("n_estimators"),
            "learning_rate": model_params.get("learning_rate"),
            "num_leaves": model_params.get("num_leaves"),
            "max_depth": model_params.get("max_depth"),
            "cv_folds": config["training"].get("cv_folds", 5),
            "n_train_samples": X_train.shape[0],
            "n_features": X_train.shape[1],
            "num_classes": num_classes,
            "tfidf_max_features": config.get("tfidf", {}).get("max_features"),
        })

        # Cross-validation
        fold_metrics = run_cross_validation(X_train, y_train, config, class_weights)

        # Aggregate CV metrics
        cv_accs = [m["accuracy"] for m in fold_metrics]
        cv_f1s = [m["macro_f1"] for m in fold_metrics]
        cv_logloss = [m["log_loss"] for m in fold_metrics]

        cv_summary = {
            "cv_accuracy_mean": float(np.mean(cv_accs)),
            "cv_accuracy_std": float(np.std(cv_accs)),
            "cv_macro_f1_mean": float(np.mean(cv_f1s)),
            "cv_macro_f1_std": float(np.std(cv_f1s)),
            "cv_log_loss_mean": float(np.mean(cv_logloss)),
            "cv_log_loss_std": float(np.std(cv_logloss)),
        }

        mlflow.log_metrics(cv_summary)

        print(f"\n{'='*60}")
        print("  Cross-Validation Summary")
        print(f"{'='*60}")
        print(f"  Accuracy:  {cv_summary['cv_accuracy_mean']:.4f} +/- {cv_summary['cv_accuracy_std']:.4f}")
        print(f"  Macro F1:  {cv_summary['cv_macro_f1_mean']:.4f} +/- {cv_summary['cv_macro_f1_std']:.4f}")
        print(f"  Log Loss:  {cv_summary['cv_log_loss_mean']:.4f} +/- {cv_summary['cv_log_loss_std']:.4f}")

        # Train final model
        model, train_time, best_iter = train_final_model(
            X_train, y_train, X_val, y_val, config, class_weights
        )

        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        test_acc = accuracy_score(y_test, y_test_pred)
        test_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
        test_micro_f1 = f1_score(y_test, y_test_pred, average="micro", zero_division=0)
        test_logloss = log_loss(y_test, y_test_proba, labels=list(range(num_classes)))

        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "test_micro_f1": test_micro_f1,
            "test_log_loss": test_logloss,
            "train_time_s": train_time,
            "best_iteration": best_iter,
        })

        print(f"\n{'='*60}")
        print("  Test Set Results")
        print(f"{'='*60}")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Macro F1:  {test_macro_f1:.4f}")
        print(f"  Micro F1:  {test_micro_f1:.4f}")
        print(f"  Log Loss:  {test_logloss:.4f}")

        # Save model and label names
        model_path = PROJECT_ROOT / config["output"]["model_path"]
        label_path = PROJECT_ROOT / config["output"]["label_names_path"]
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)
        joblib.dump(label_names, label_path)

        print(f"\n  Saved model to {model_path}")
        print(f"  Saved label names to {label_path}")

        # Save test predictions for downstream analysis
        preds_path = PROJECT_ROOT / "models" / "saved" / "test_predictions.npz"
        np.savez(
            preds_path,
            y_true=y_test,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
        )
        print(f"  Saved test predictions to {preds_path}")

        # Log artifacts
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(label_path))

        # Final summary
        print(f"\n{'='*60}")
        print("  Training Summary")
        print(f"{'='*60}")
        print(f"  Model:           {config['model']['type']}")
        print(f"  CV Accuracy:     {cv_summary['cv_accuracy_mean']:.4f} +/- {cv_summary['cv_accuracy_std']:.4f}")
        print(f"  CV Macro F1:     {cv_summary['cv_macro_f1_mean']:.4f} +/- {cv_summary['cv_macro_f1_std']:.4f}")
        print(f"  Test Accuracy:   {test_acc:.4f}")
        print(f"  Test Macro F1:   {test_macro_f1:.4f}")
        print(f"  Best Iteration:  {best_iter}")
        print(f"  Training Time:   {train_time:.1f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    config_path = None
    if len(sys.argv) > 1 and sys.argv[1] == "--config" and len(sys.argv) > 2:
        config_path = sys.argv[2]
    main(config_path)
