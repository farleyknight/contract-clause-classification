"""
Contract Clause Classification — Single-Clause Inference
=========================================================
Load a trained model and vectorizer to predict clause type for a
given text input.

Usage:
    python -m models.predict "This agreement shall be governed by the laws of..."
    python -m models.predict --file clause.txt
"""

import sys
import json
from pathlib import Path

import numpy as np
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


class ClausePredictor:
    """Wrapper for clause classification inference."""

    def __init__(self, config_path: str = None):
        config = load_config(config_path)

        model_path = PROJECT_ROOT / config["output"]["model_path"]
        label_path = PROJECT_ROOT / config["output"]["label_names_path"]
        vec_path = PROJECT_ROOT / config["output"]["vectorizer_path"]

        self.model = joblib.load(model_path)
        self.label_names = joblib.load(label_path)
        self.vectorizer = joblib.load(vec_path)

    def predict_clause(self, text: str) -> dict:
        """
        Predict the clause type for a given text.

        Args:
            text: The contract clause text to classify.

        Returns:
            dict with keys:
                - predicted_class: str, the predicted class name
                - predicted_class_id: int, the predicted class index
                - confidence: float, probability of the top prediction
                - top5: list of dicts, each with 'class_name', 'class_id', 'probability'
        """
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]

        # Top-5 predictions sorted by probability
        top5_indices = np.argsort(proba)[::-1][:5]

        top5 = []
        for idx in top5_indices:
            top5.append({
                "class_name": self.label_names[idx],
                "class_id": int(idx),
                "probability": float(proba[idx]),
            })

        predicted_idx = top5_indices[0]

        return {
            "predicted_class": self.label_names[predicted_idx],
            "predicted_class_id": int(predicted_idx),
            "confidence": float(proba[predicted_idx]),
            "top5": top5,
        }


# ──────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────

_predictor = None


def predict_clause(text: str, config_path: str = None) -> dict:
    """
    Predict clause type for a single text string.

    Caches the predictor for repeated calls.
    """
    global _predictor
    if _predictor is None:
        _predictor = ClausePredictor(config_path)
    return _predictor.predict_clause(text)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict contract clause type from text."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="The clause text to classify (quoted string).",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a text file containing the clause.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    args = parser.parse_args()

    # Get input text
    if args.file:
        with open(args.file, "r") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        print("Error: Provide clause text as argument or via --file.")
        print("Usage: python -m models.predict \"clause text here\"")
        print("       python -m models.predict --file clause.txt")
        sys.exit(1)

    # Predict
    predictor = ClausePredictor(args.config)
    result = predictor.predict_clause(text)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nInput text: {text[:150]}{'...' if len(text) > 150 else ''}")
        print(f"\nPredicted class: {result['predicted_class']}")
        print(f"Confidence:      {result['confidence']:.4f}")
        print(f"\nTop-5 predictions:")
        print(f"  {'Rank':<5} {'Class':<45} {'Probability':>12}")
        print(f"  {'-'*64}")
        for rank, pred in enumerate(result["top5"], 1):
            print(f"  {rank:<5} {pred['class_name']:<45} {pred['probability']:>12.4f}")


if __name__ == "__main__":
    main()
