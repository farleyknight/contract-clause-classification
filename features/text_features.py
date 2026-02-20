"""
Text feature extraction for contract clause classification.

Functions:
- build_tfidf_features: TF-IDF vectorization with configurable params from YAML
- extract_text_stats: word count, char count, avg word length, sentence count, vocab richness
- extract_legal_keyword_counts: count occurrences of common legal terms
- build_all_text_features: combine all text features into a single DataFrame/matrix
"""

import re
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Legal keywords to count
LEGAL_KEYWORDS = [
    "shall",
    "hereinafter",
    "notwithstanding",
    "whereas",
    "hereby",
    "pursuant",
    "indemnify",
    "terminate",
    "governing law",
    "confidential",
    "waiver",
    "amendment",
    "liability",
    "warranty",
    "arbitration",
]


def build_tfidf_features(
    texts: list[str],
    config: dict[str, Any],
    fit: bool = True,
    vectorizer: TfidfVectorizer | None = None,
) -> tuple:
    """Build TF-IDF feature matrix from texts using config params.

    Args:
        texts: List of text strings to vectorize.
        config: Dict with TF-IDF parameters (from YAML config under 'tfidf' key).
        fit: If True, fit a new vectorizer. If False, use the provided vectorizer.
        vectorizer: Pre-fitted vectorizer to use when fit=False.

    Returns:
        Tuple of (sparse feature matrix, fitted TfidfVectorizer).
    """
    tfidf_config = config.get("tfidf", {})

    if fit:
        ngram_range = tuple(tfidf_config.get("ngram_range", [1, 2]))
        vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get("max_features", 10000),
            ngram_range=ngram_range,
            min_df=tfidf_config.get("min_df", 3),
            max_df=tfidf_config.get("max_df", 0.95),
            sublinear_tf=tfidf_config.get("sublinear_tf", True),
        )
        X = vectorizer.fit_transform(texts)

        # Save fitted vectorizer
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        print(f"  Saved TF-IDF vectorizer to {vectorizer_path}")
    else:
        if vectorizer is None:
            raise ValueError("Must provide a fitted vectorizer when fit=False")
        X = vectorizer.transform(texts)

    return X, vectorizer


def extract_text_stats(texts: list[str]) -> pd.DataFrame:
    """Extract text statistics from a list of texts.

    Features:
    - word_count: number of whitespace-delimited tokens
    - char_count: total character count
    - avg_word_length: mean length of words
    - sentence_count: approximate sentence count (split on '.', '!', '?')
    - vocab_richness: unique words / total words (type-token ratio)

    Args:
        texts: List of text strings.

    Returns:
        DataFrame with one row per text and columns for each statistic.
    """
    stats = []
    for text in texts:
        if not text or not isinstance(text, str):
            stats.append({
                "word_count": 0,
                "char_count": 0,
                "avg_word_length": 0.0,
                "sentence_count": 0,
                "vocab_richness": 0.0,
            })
            continue

        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_length = (
            sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
        )

        # Sentence count: split on sentence-ending punctuation
        sentences = re.split(r"[.!?]+", text)
        sentence_count = len([s for s in sentences if s.strip()])

        # Vocab richness: type-token ratio
        unique_words = set(w.lower() for w in words)
        vocab_richness = len(unique_words) / word_count if word_count > 0 else 0.0

        stats.append({
            "word_count": word_count,
            "char_count": char_count,
            "avg_word_length": avg_word_length,
            "sentence_count": sentence_count,
            "vocab_richness": vocab_richness,
        })

    return pd.DataFrame(stats)


def extract_legal_keyword_counts(texts: list[str]) -> pd.DataFrame:
    """Count occurrences of common legal keywords in each text.

    Keywords counted: shall, hereinafter, notwithstanding, whereas, hereby,
    pursuant, indemnify, terminate, governing law, confidential, waiver,
    amendment, liability, warranty, arbitration.

    Args:
        texts: List of text strings.

    Returns:
        DataFrame with one row per text and one column per keyword
        (prefixed with 'kw_').
    """
    counts = []
    for text in texts:
        if not text or not isinstance(text, str):
            row = {f"kw_{kw.replace(' ', '_')}": 0 for kw in LEGAL_KEYWORDS}
            counts.append(row)
            continue

        text_lower = text.lower()
        row = {}
        for kw in LEGAL_KEYWORDS:
            # Use word boundary matching for single-word keywords,
            # simple count for multi-word phrases
            if " " in kw:
                row[f"kw_{kw.replace(' ', '_')}"] = text_lower.count(kw)
            else:
                # Count whole-word matches
                pattern = r"\b" + re.escape(kw) + r"\b"
                row[f"kw_{kw.replace(' ', '_')}"] = len(re.findall(pattern, text_lower))
        counts.append(row)

    return pd.DataFrame(counts)


def build_all_text_features(
    df: pd.DataFrame,
    config: dict[str, Any],
    fit: bool = True,
    vectorizer: TfidfVectorizer | None = None,
) -> tuple:
    """Build all text features: TF-IDF + text stats + legal keyword counts.

    Args:
        df: DataFrame with a 'text' column.
        config: Full config dict (with 'tfidf' and 'features' keys).
        fit: Whether to fit a new TF-IDF vectorizer.
        vectorizer: Pre-fitted vectorizer (used when fit=False).

    Returns:
        Tuple of (combined feature matrix as sparse/array, fitted vectorizer,
                  list of dense feature column names).
    """
    texts = df["text"].tolist()
    features_config = config.get("features", {})
    dense_parts = []
    dense_col_names = []

    # TF-IDF features
    tfidf_matrix = None
    if features_config.get("use_tfidf", True):
        print("  Building TF-IDF features...")
        tfidf_matrix, vectorizer = build_tfidf_features(
            texts, config, fit=fit, vectorizer=vectorizer
        )
        print(f"    TF-IDF shape: {tfidf_matrix.shape}")

    # Text stats
    if features_config.get("use_text_stats", True):
        print("  Extracting text statistics...")
        text_stats_df = extract_text_stats(texts)
        dense_parts.append(text_stats_df.values)
        dense_col_names.extend(text_stats_df.columns.tolist())
        print(f"    Text stats shape: {text_stats_df.shape}")

    # Legal keyword counts
    if features_config.get("use_legal_keywords", True):
        print("  Extracting legal keyword counts...")
        keyword_df = extract_legal_keyword_counts(texts)
        dense_parts.append(keyword_df.values)
        dense_col_names.extend(keyword_df.columns.tolist())
        print(f"    Legal keywords shape: {keyword_df.shape}")

    # Combine all features
    if tfidf_matrix is not None and dense_parts:
        from scipy.sparse import csr_matrix
        dense_combined = np.hstack(dense_parts)
        dense_sparse = csr_matrix(dense_combined)
        combined = hstack([tfidf_matrix, dense_sparse])
    elif tfidf_matrix is not None:
        combined = tfidf_matrix
    elif dense_parts:
        combined = np.hstack(dense_parts)
    else:
        raise ValueError("No features enabled in config!")

    return combined, vectorizer, dense_col_names
