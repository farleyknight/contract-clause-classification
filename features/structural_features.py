"""
Structural feature extraction for contract clause classification.

Extracts features based on the structure and formatting of clause text,
rather than its semantic content.

Functions:
- extract_clause_position: detect if clause is early/middle/late
- extract_section_header_presence: detect section headers / numbering patterns
- extract_formatting_features: ALL CAPS ratio, defined terms, enumeration patterns
- build_all_structural_features: combine all structural features
"""

import re

import numpy as np
import pandas as pd


def extract_clause_position(texts: list[str]) -> pd.DataFrame:
    """Detect structural position indicators in clause text.

    Since individual clauses do not carry explicit document position,
    we use heuristic proxies:
    - has_preamble_language: presence of preamble/recital indicators
      ("whereas", "witnesseth", "recitals", "NOW THEREFORE")
    - has_closing_language: presence of closing indicators
      ("IN WITNESS WHEREOF", "executed as of", "duly authorized")
    - has_definition_language: presence of definition section markers
      ("as defined", "means", "shall mean", "as used herein")

    Args:
        texts: List of text strings.

    Returns:
        DataFrame with boolean indicator columns.
    """
    preamble_patterns = [
        r"\bwhereas\b",
        r"\bwitnesseth\b",
        r"\brecitals?\b",
        r"\bnow\s*,?\s*therefore\b",
    ]
    closing_patterns = [
        r"\bin\s+witness\s+whereof\b",
        r"\bexecuted\s+as\s+of\b",
        r"\bduly\s+authorized\b",
        r"\bsignature\s+page\b",
    ]
    definition_patterns = [
        r"\bas\s+defined\b",
        r"\bshall\s+mean\b",
        r"\bas\s+used\s+herein\b",
        r'"[A-Z][^"]+"\s+means\b',
    ]

    rows = []
    for text in texts:
        if not text or not isinstance(text, str):
            rows.append({
                "has_preamble_language": 0,
                "has_closing_language": 0,
                "has_definition_language": 0,
            })
            continue

        text_lower = text.lower()
        rows.append({
            "has_preamble_language": int(
                any(re.search(p, text_lower) for p in preamble_patterns)
            ),
            "has_closing_language": int(
                any(re.search(p, text_lower) for p in closing_patterns)
            ),
            "has_definition_language": int(
                any(re.search(p, text, re.IGNORECASE) for p in definition_patterns)
            ),
        })

    return pd.DataFrame(rows)


def extract_section_header_presence(texts: list[str]) -> pd.DataFrame:
    """Detect presence of section headers and numbering patterns.

    Patterns detected:
    - has_letter_enum: "(a)", "(b)", "(i)", "(ii)", etc.
    - has_numeric_section: "Section 1", "Section 3.2", etc.
    - has_article_header: "ARTICLE I", "ARTICLE IV", "Article 5", etc.
    - has_numbered_list: "1.", "2.", "3." at start of line/text
    - section_ref_count: total count of section/article references

    Args:
        texts: List of text strings.

    Returns:
        DataFrame with indicator and count columns.
    """
    rows = []
    for text in texts:
        if not text or not isinstance(text, str):
            rows.append({
                "has_letter_enum": 0,
                "has_numeric_section": 0,
                "has_article_header": 0,
                "has_numbered_list": 0,
                "section_ref_count": 0,
            })
            continue

        # (a), (b), (c), (i), (ii), (iii), (iv), (v)
        has_letter_enum = int(
            bool(re.search(r"\([a-z]\)|\([ivxlcdm]+\)", text, re.IGNORECASE))
        )

        # Section 1, Section 3.2, Section 12(a)
        has_numeric_section = int(
            bool(re.search(r"\bSection\s+\d+", text, re.IGNORECASE))
        )

        # ARTICLE I, ARTICLE IV, Article 5
        has_article_header = int(
            bool(re.search(r"\bARTICLE\s+[IVXLCDM\d]+\b|\bArticle\s+\d+", text))
        )

        # Numbered lists: "1.", "2.", "3." etc. (at word boundary)
        has_numbered_list = int(
            bool(re.search(r"(?:^|\n)\s*\d+\.", text))
        )

        # Count all section/article references
        section_refs = re.findall(
            r"\b(?:Section|Article|Clause|Paragraph)\s+[\d.]+",
            text,
            re.IGNORECASE,
        )
        section_ref_count = len(section_refs)

        rows.append({
            "has_letter_enum": has_letter_enum,
            "has_numeric_section": has_numeric_section,
            "has_article_header": has_article_header,
            "has_numbered_list": has_numbered_list,
            "section_ref_count": section_ref_count,
        })

    return pd.DataFrame(rows)


def extract_formatting_features(texts: list[str]) -> pd.DataFrame:
    """Extract formatting-related features from texts.

    Features:
    - caps_ratio: fraction of alphabetic characters that are uppercase
    - has_defined_terms: presence of terms in quotes (e.g., "Agreement", "Party")
    - defined_term_count: count of quoted capitalized terms
    - has_enumeration: presence of enumeration patterns (e.g., "(i)", "1)", "a.")
    - parenthetical_count: number of parenthetical expressions
    - semicolon_count: number of semicolons (common in legal lists)
    - has_all_caps_words: presence of ALL CAPS words (3+ letters)
    - all_caps_word_count: count of ALL CAPS words

    Args:
        texts: List of text strings.

    Returns:
        DataFrame with formatting feature columns.
    """
    rows = []
    for text in texts:
        if not text or not isinstance(text, str):
            rows.append({
                "caps_ratio": 0.0,
                "has_defined_terms": 0,
                "defined_term_count": 0,
                "has_enumeration": 0,
                "parenthetical_count": 0,
                "semicolon_count": 0,
                "has_all_caps_words": 0,
                "all_caps_word_count": 0,
            })
            continue

        # Caps ratio
        alpha_chars = [c for c in text if c.isalpha()]
        upper_chars = [c for c in alpha_chars if c.isupper()]
        caps_ratio = len(upper_chars) / len(alpha_chars) if alpha_chars else 0.0

        # Defined terms in quotes: "SomeCapitalizedTerm"
        defined_terms = re.findall(r'"([A-Z][^"]*)"', text)
        defined_term_count = len(defined_terms)

        # Enumeration patterns: (i), (a), 1), a., (1)
        has_enumeration = int(
            bool(re.search(r"\([a-z]\)|\(\d+\)|\d+\)|\b[a-z]\.", text))
        )

        # Parenthetical expressions
        parenthetical_count = len(re.findall(r"\([^)]+\)", text))

        # Semicolons
        semicolon_count = text.count(";")

        # ALL CAPS words (3+ letters, excluding common abbreviations)
        all_caps_words = re.findall(r"\b[A-Z]{3,}\b", text)
        all_caps_word_count = len(all_caps_words)

        rows.append({
            "caps_ratio": caps_ratio,
            "has_defined_terms": int(defined_term_count > 0),
            "defined_term_count": defined_term_count,
            "has_enumeration": has_enumeration,
            "parenthetical_count": parenthetical_count,
            "semicolon_count": semicolon_count,
            "has_all_caps_words": int(all_caps_word_count > 0),
            "all_caps_word_count": all_caps_word_count,
        })

    return pd.DataFrame(rows)


def build_all_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all structural features and combine into a single DataFrame.

    Args:
        df: DataFrame with a 'text' column.

    Returns:
        DataFrame with all structural feature columns.
    """
    texts = df["text"].tolist()

    print("  Extracting clause position features...")
    position_df = extract_clause_position(texts)
    print(f"    Position features shape: {position_df.shape}")

    print("  Extracting section header features...")
    header_df = extract_section_header_presence(texts)
    print(f"    Section header features shape: {header_df.shape}")

    print("  Extracting formatting features...")
    formatting_df = extract_formatting_features(texts)
    print(f"    Formatting features shape: {formatting_df.shape}")

    # Combine all structural features
    combined = pd.concat([position_df, header_df, formatting_df], axis=1)
    print(f"    Combined structural features shape: {combined.shape}")

    return combined
