"""
Pytest tests for feature extraction modules.

Tests:
- TF-IDF output shape and no NaN
- Legal keyword counts are non-negative
- Text stats produce expected columns
- Structural features handle edge cases
- Uses small synthetic test data
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from features.text_features import (
    build_tfidf_features,
    extract_text_stats,
    extract_legal_keyword_counts,
    build_all_text_features,
)
from features.structural_features import (
    extract_clause_position,
    extract_section_header_presence,
    extract_formatting_features,
    build_all_structural_features,
)


# ──────────────────────────────────────────────
# Fixtures / sample data
# ──────────────────────────────────────────────

SAMPLE_TEXTS = [
    (
        "This Agreement shall be governed by and construed in accordance "
        "with the laws of the State of Delaware, notwithstanding any "
        "conflict of laws provisions."
    ),
    (
        'The "Company" hereby agrees to indemnify and hold harmless the '
        "Investor against any and all losses pursuant to Section 5.2 of "
        "this Agreement."
    ),
    (
        "ARTICLE IV. CONFIDENTIALITY. (a) Each party shall maintain the "
        "confidentiality of all proprietary information; (b) disclosure "
        "shall require prior written consent; (c) obligations hereinafter "
        "shall survive termination."
    ),
    (
        "IN WITNESS WHEREOF, the parties have executed this Agreement as "
        "of the date first written above by their duly authorized "
        "representatives."
    ),
    "Short clause.",
]

SAMPLE_CONFIG = {
    "tfidf": {
        "max_features": 100,
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_df": 1.0,
        "sublinear_tf": True,
    },
    "features": {
        "use_tfidf": True,
        "use_text_stats": True,
        "use_legal_keywords": True,
        "use_structural": True,
    },
}


@pytest.fixture
def sample_df():
    return pd.DataFrame({"text": SAMPLE_TEXTS})


# ──────────────────────────────────────────────
# TF-IDF tests
# ──────────────────────────────────────────────


class TestTfidfFeatures:
    def test_tfidf_output_shape(self):
        """TF-IDF matrix should have n_samples rows and <= max_features cols."""
        X, vectorizer = build_tfidf_features(SAMPLE_TEXTS, SAMPLE_CONFIG, fit=True)
        assert X.shape[0] == len(SAMPLE_TEXTS)
        assert X.shape[1] <= SAMPLE_CONFIG["tfidf"]["max_features"]

    def test_tfidf_no_nan(self):
        """TF-IDF matrix should contain no NaN values."""
        X, vectorizer = build_tfidf_features(SAMPLE_TEXTS, SAMPLE_CONFIG, fit=True)
        if issparse(X):
            assert not np.isnan(X.data).any()
        else:
            assert not np.isnan(X).any()

    def test_tfidf_sparse(self):
        """TF-IDF output should be a sparse matrix."""
        X, vectorizer = build_tfidf_features(SAMPLE_TEXTS, SAMPLE_CONFIG, fit=True)
        assert issparse(X)

    def test_tfidf_transform_mode(self):
        """Transform mode should produce same number of columns as fit."""
        X_fit, vectorizer = build_tfidf_features(SAMPLE_TEXTS, SAMPLE_CONFIG, fit=True)
        X_transform, _ = build_tfidf_features(
            SAMPLE_TEXTS[:3], SAMPLE_CONFIG, fit=False, vectorizer=vectorizer
        )
        assert X_transform.shape[1] == X_fit.shape[1]
        assert X_transform.shape[0] == 3

    def test_tfidf_requires_vectorizer_for_transform(self):
        """Transform mode without vectorizer should raise ValueError."""
        with pytest.raises(ValueError, match="Must provide a fitted vectorizer"):
            build_tfidf_features(SAMPLE_TEXTS, SAMPLE_CONFIG, fit=False, vectorizer=None)


# ──────────────────────────────────────────────
# Legal keyword tests
# ──────────────────────────────────────────────


class TestLegalKeywords:
    def test_keyword_counts_non_negative(self):
        """All keyword counts should be >= 0."""
        df = extract_legal_keyword_counts(SAMPLE_TEXTS)
        assert (df >= 0).all().all()

    def test_keyword_counts_shape(self):
        """Output should have n_samples rows and one column per keyword."""
        df = extract_legal_keyword_counts(SAMPLE_TEXTS)
        assert df.shape[0] == len(SAMPLE_TEXTS)
        assert df.shape[1] == 15  # 15 legal keywords

    def test_keyword_counts_known_values(self):
        """Check that known keywords are detected."""
        df = extract_legal_keyword_counts(SAMPLE_TEXTS)
        # First text contains "shall" and "notwithstanding"
        assert df.iloc[0]["kw_shall"] >= 1
        assert df.iloc[0]["kw_notwithstanding"] >= 1
        # Second text contains "indemnify" and "pursuant"
        assert df.iloc[1]["kw_indemnify"] >= 1
        assert df.iloc[1]["kw_pursuant"] >= 1

    def test_keyword_counts_empty_text(self):
        """Empty text should produce all zeros."""
        df = extract_legal_keyword_counts(["", None, "   "])
        assert df.iloc[0].sum() == 0
        assert df.iloc[1].sum() == 0

    def test_keyword_column_names(self):
        """Column names should be prefixed with 'kw_'."""
        df = extract_legal_keyword_counts(SAMPLE_TEXTS[:1])
        assert all(col.startswith("kw_") for col in df.columns)


# ──────────────────────────────────────────────
# Text stats tests
# ──────────────────────────────────────────────


class TestTextStats:
    def test_text_stats_columns(self):
        """Should produce expected columns."""
        df = extract_text_stats(SAMPLE_TEXTS)
        expected_cols = {
            "word_count",
            "char_count",
            "avg_word_length",
            "sentence_count",
            "vocab_richness",
        }
        assert set(df.columns) == expected_cols

    def test_text_stats_shape(self):
        """Output should have n_samples rows."""
        df = extract_text_stats(SAMPLE_TEXTS)
        assert df.shape[0] == len(SAMPLE_TEXTS)

    def test_text_stats_non_negative(self):
        """All stats should be >= 0."""
        df = extract_text_stats(SAMPLE_TEXTS)
        assert (df >= 0).all().all()

    def test_text_stats_word_count(self):
        """Word count should be positive for non-empty texts."""
        df = extract_text_stats(SAMPLE_TEXTS)
        for i, text in enumerate(SAMPLE_TEXTS):
            if text and text.strip():
                assert df.iloc[i]["word_count"] > 0

    def test_text_stats_empty_input(self):
        """Empty input should produce zeros."""
        df = extract_text_stats(["", None])
        assert df.iloc[0]["word_count"] == 0
        assert df.iloc[1]["word_count"] == 0

    def test_vocab_richness_range(self):
        """Vocab richness should be between 0 and 1."""
        df = extract_text_stats(SAMPLE_TEXTS)
        for i, text in enumerate(SAMPLE_TEXTS):
            if text and text.strip():
                assert 0 <= df.iloc[i]["vocab_richness"] <= 1.0


# ──────────────────────────────────────────────
# Structural features tests
# ──────────────────────────────────────────────


class TestStructuralFeatures:
    def test_clause_position_shape(self):
        """Clause position should have correct shape."""
        df = extract_clause_position(SAMPLE_TEXTS)
        assert df.shape[0] == len(SAMPLE_TEXTS)
        assert "has_preamble_language" in df.columns
        assert "has_closing_language" in df.columns
        assert "has_definition_language" in df.columns

    def test_clause_position_detects_closing(self):
        """Should detect closing language in the witness clause."""
        df = extract_clause_position(SAMPLE_TEXTS)
        # Fourth text has "IN WITNESS WHEREOF" and "duly authorized"
        assert df.iloc[3]["has_closing_language"] == 1

    def test_section_header_detects_article(self):
        """Should detect ARTICLE IV in the third sample."""
        df = extract_section_header_presence(SAMPLE_TEXTS)
        assert df.iloc[2]["has_article_header"] == 1

    def test_section_header_detects_enumeration(self):
        """Should detect letter enumeration (a), (b), (c) in third sample."""
        df = extract_section_header_presence(SAMPLE_TEXTS)
        assert df.iloc[2]["has_letter_enum"] == 1

    def test_section_header_detects_section_ref(self):
        """Should detect Section 5.2 reference in second sample."""
        df = extract_section_header_presence(SAMPLE_TEXTS)
        assert df.iloc[1]["has_numeric_section"] == 1
        assert df.iloc[1]["section_ref_count"] >= 1

    def test_formatting_caps_ratio(self):
        """Caps ratio should be between 0 and 1."""
        df = extract_formatting_features(SAMPLE_TEXTS)
        assert (df["caps_ratio"] >= 0).all()
        assert (df["caps_ratio"] <= 1).all()

    def test_formatting_defined_terms(self):
        """Should detect defined terms in quotes."""
        df = extract_formatting_features(SAMPLE_TEXTS)
        # Second text has '"Company"'
        assert df.iloc[1]["has_defined_terms"] == 1
        assert df.iloc[1]["defined_term_count"] >= 1

    def test_formatting_semicolons(self):
        """Should count semicolons in the third sample."""
        df = extract_formatting_features(SAMPLE_TEXTS)
        # Third text has semicolons separating list items
        assert df.iloc[2]["semicolon_count"] >= 2

    def test_all_caps_detection(self):
        """Should detect ALL CAPS words."""
        df = extract_formatting_features(SAMPLE_TEXTS)
        # Third text has "ARTICLE" and "CONFIDENTIALITY"
        assert df.iloc[2]["has_all_caps_words"] == 1
        assert df.iloc[2]["all_caps_word_count"] >= 1

    def test_build_all_structural(self, sample_df):
        """Combined structural features should have correct row count."""
        result = build_all_structural_features(sample_df)
        assert result.shape[0] == len(SAMPLE_TEXTS)
        # Should have columns from all three extractors
        assert result.shape[1] >= 10

    def test_edge_case_empty_text(self):
        """All structural extractors should handle empty text."""
        empty_texts = ["", None, "   "]
        pos = extract_clause_position(empty_texts)
        hdr = extract_section_header_presence(empty_texts)
        fmt = extract_formatting_features(empty_texts)

        assert pos.shape[0] == 3
        assert hdr.shape[0] == 3
        assert fmt.shape[0] == 3

        # All values should be zero for empty texts
        assert pos.iloc[0].sum() == 0
        assert hdr.iloc[0].sum() == 0
        assert fmt.iloc[0].sum() == 0

    def test_edge_case_single_word(self):
        """Structural features should handle single-word input."""
        pos = extract_clause_position(["hello"])
        hdr = extract_section_header_presence(["hello"])
        fmt = extract_formatting_features(["hello"])

        assert pos.shape[0] == 1
        assert hdr.shape[0] == 1
        assert fmt.shape[0] == 1


# ──────────────────────────────────────────────
# Combined feature tests
# ──────────────────────────────────────────────


class TestCombinedFeatures:
    def test_build_all_text_features(self, sample_df):
        """Combined text features should produce correct shape."""
        combined, vectorizer, dense_cols = build_all_text_features(
            sample_df, SAMPLE_CONFIG, fit=True
        )
        assert combined.shape[0] == len(SAMPLE_TEXTS)
        # Should have TF-IDF cols + text stats (5) + keyword counts (15)
        assert combined.shape[1] > 20

    def test_build_all_text_features_no_nan(self, sample_df):
        """Combined features should contain no NaN."""
        combined, vectorizer, dense_cols = build_all_text_features(
            sample_df, SAMPLE_CONFIG, fit=True
        )
        if issparse(combined):
            dense = combined.toarray()
        else:
            dense = combined
        assert not np.isnan(dense).any()
