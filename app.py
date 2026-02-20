"""
Contract Clause Classification — Streamlit App
Interactive classification of SEC contract provisions using TF-IDF + LightGBM.
Dataset: LEDGAR (100-class single-label classification).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent

EXAMPLE_CLAUSES = {
    "Termination": (
        "Either party may terminate this Agreement upon thirty (30) days' "
        "prior written notice to the other party. In the event of a material "
        "breach by either party, the non-breaching party may terminate this "
        "Agreement immediately upon written notice, provided that the breaching "
        "party has failed to cure such breach within fifteen (15) days after "
        "receipt of written notice thereof."
    ),
    "Indemnification": (
        "The Indemnifying Party shall indemnify, defend, and hold harmless the "
        "Indemnified Party and its officers, directors, employees, agents, "
        "successors, and assigns from and against any and all losses, damages, "
        "liabilities, deficiencies, claims, actions, judgments, settlements, "
        "interest, awards, penalties, fines, costs, or expenses of whatever kind, "
        "including reasonable attorneys' fees, arising out of or resulting from "
        "any breach of any representation, warranty, or obligation under this "
        "Agreement."
    ),
    "Governing Law": (
        "This Agreement and all related documents including all exhibits attached "
        "hereto and all matters arising out of or relating to this Agreement, "
        "whether sounding in contract, tort, or statute, shall be governed by "
        "and construed in accordance with the internal laws of the State of "
        "Delaware, without giving effect to the conflict of laws provisions thereof "
        "to the extent such principles or rules would require or permit the "
        "application of the laws of any jurisdiction other than those of the "
        "State of Delaware."
    ),
    "Confidentiality": (
        "Each party agrees that all information, whether written, oral, or "
        "otherwise, provided by the Disclosing Party to the Receiving Party "
        "in connection with this Agreement that is designated as confidential "
        "or that reasonably should be understood to be confidential given the "
        "nature of the information and the circumstances of disclosure shall be "
        "treated as confidential and shall not be disclosed to any third party "
        "without the prior written consent of the Disclosing Party for a period "
        "of five (5) years following disclosure."
    ),
}

CONFIDENCE_THRESHOLDS = {"high": 0.8, "medium": 0.5}


# ──────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────


def _find_file(*candidates: str) -> Path | None:
    """Return the first path that exists among *candidates* (relative to PROJECT_DIR)."""
    for candidate in candidates:
        p = PROJECT_DIR / candidate
        if p.exists():
            return p
    return None


@st.cache_resource
def load_model():
    """Load the LightGBM model, TF-IDF vectorizer, and label names.

    Returns (model, vectorizer, label_names) or (None, None, None) if not found.
    """
    model_path = _find_file("models/saved/lgbm_model.joblib", "lgbm_model.joblib")
    vec_path = _find_file("models/tfidf_vectorizer.pkl", "tfidf_vectorizer.joblib")
    label_path = _find_file("models/saved/label_names.joblib", "label_names.joblib")

    if model_path is None or vec_path is None or label_path is None:
        return None, None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    label_names = joblib.load(label_path)
    return model, vectorizer, label_names


def load_evaluation_results() -> dict | None:
    """Load evaluation results JSON if it exists."""
    path = _find_file("models/evaluation_results.json")
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Classification helpers
# ──────────────────────────────────────────────


def classify_text(text: str, model, vectorizer, label_names):
    """Classify a single text and return (predicted_label, probabilities)."""
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    return label_names[pred_idx], probs


def confidence_badge(score: float) -> str:
    """Return a colored markdown badge for the confidence level."""
    if score >= CONFIDENCE_THRESHOLDS["high"]:
        return f":green[High ({score:.1%})]"
    elif score >= CONFIDENCE_THRESHOLDS["medium"]:
        return f":orange[Medium ({score:.1%})]"
    else:
        return f":red[Low ({score:.1%})]"


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────


def render_sidebar(label_names):
    """Render the sidebar with model info and evaluation results."""
    with st.sidebar:
        st.header("Model Information")
        st.markdown("**Algorithm:** TF-IDF + LightGBM")
        st.markdown("**Dataset:** LEDGAR (SEC contract provisions)")
        st.markdown(f"**Number of clause types:** {len(label_names) if label_names else 100}")

        # Evaluation results
        eval_results = load_evaluation_results()
        if eval_results is not None:
            st.divider()
            st.subheader("Model Performance")

            if "accuracy" in eval_results:
                st.metric("Accuracy", f"{eval_results['accuracy']:.4f}")
            if "macro_f1" in eval_results:
                st.metric("Macro F1", f"{eval_results['macro_f1']:.4f}")

            # Per-class F1 scores
            per_class = eval_results.get("per_class", {})
            if per_class:
                class_f1s = [
                    (name, info["f1-score"])
                    for name, info in per_class.items()
                    if isinstance(info, dict) and "f1-score" in info
                ]
                class_f1s.sort(key=lambda x: x[1], reverse=True)

                if class_f1s:
                    st.divider()
                    st.subheader("Top 10 Classes by F1")
                    top_df = pd.DataFrame(class_f1s[:10], columns=["Class", "F1"])
                    top_df["F1"] = top_df["F1"].map(lambda x: f"{x:.3f}")
                    st.dataframe(top_df, hide_index=True, use_container_width=True)

                    st.subheader("Bottom 10 Classes by F1")
                    bottom_df = pd.DataFrame(class_f1s[-10:], columns=["Class", "F1"])
                    bottom_df["F1"] = bottom_df["F1"].map(lambda x: f"{x:.3f}")
                    st.dataframe(bottom_df, hide_index=True, use_container_width=True)

        st.divider()
        st.caption(
            "**Disclaimer:** This is an automated classification tool intended "
            "for informational purposes only. Predictions should not be relied "
            "upon as legal advice. Always consult a qualified attorney for "
            "contract review and interpretation. Model accuracy varies by "
            "clause type and may produce incorrect classifications."
        )


# ──────────────────────────────────────────────
# UI: Single clause tab
# ──────────────────────────────────────────────


def render_single_tab(model, vectorizer, label_names):
    """Render the single-clause classification tab."""
    st.subheader("Classify a Contract Clause")

    # Example clauses
    st.markdown("**Try an example clause:**")
    cols = st.columns(len(EXAMPLE_CLAUSES))
    for col, (clause_type, clause_text) in zip(cols, EXAMPLE_CLAUSES.items()):
        with col:
            if st.button(clause_type, key=f"example_{clause_type}", use_container_width=True):
                st.session_state["clause_input"] = clause_text

    # Text input
    clause_text = st.text_area(
        "Paste a contract clause below:",
        value=st.session_state.get("clause_input", ""),
        height=200,
        key="clause_text_area",
        placeholder="Enter or paste a contract clause here...",
    )

    if st.button("Classify", type="primary", use_container_width=True):
        if not clause_text or not clause_text.strip():
            st.warning("Please enter a clause to classify.")
            return

        if len(clause_text.strip()) < 10:
            st.warning(
                "The input text is very short (fewer than 10 characters). "
                "Please provide a more complete clause for accurate classification."
            )
            return

        with st.spinner("Classifying..."):
            predicted_label, probs = classify_text(clause_text, model, vectorizer, label_names)
            confidence = float(np.max(probs))

        # Primary prediction
        st.divider()
        st.markdown("### Prediction")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"## {predicted_label}")
        with col2:
            st.markdown(f"**Confidence:** {confidence_badge(confidence)}")

        # Top 5 predictions
        st.divider()
        st.markdown("### Top 5 Predictions")
        top_indices = np.argsort(probs)[::-1][:5]
        top_data = []
        for idx in top_indices:
            top_data.append(
                {
                    "Clause Type": label_names[idx],
                    "Probability": float(probs[idx]),
                }
            )

        top_df = pd.DataFrame(top_data)

        for _, row in top_df.iterrows():
            prob = row["Probability"]
            label = row["Clause Type"]
            col_label, col_bar = st.columns([1, 2])
            with col_label:
                st.markdown(f"**{label}**")
            with col_bar:
                st.progress(prob, text=f"{prob:.1%}")


# ──────────────────────────────────────────────
# UI: Batch mode tab
# ──────────────────────────────────────────────


def render_batch_tab(model, vectorizer, label_names):
    """Render the batch classification tab."""
    st.subheader("Batch Classification")
    st.markdown("Paste multiple clauses, **one per line**, to classify them all at once.")

    batch_text = st.text_area(
        "Clauses (one per line):",
        height=250,
        key="batch_text_area",
        placeholder=(
            "Paste each clause on its own line.\n"
            "Example:\n"
            "This Agreement shall be governed by the laws of Delaware.\n"
            "Either party may terminate this Agreement upon 30 days notice."
        ),
    )

    if st.button("Classify All", type="primary", key="batch_classify", use_container_width=True):
        if not batch_text or not batch_text.strip():
            st.warning("Please enter at least one clause.")
            return

        lines = [line.strip() for line in batch_text.strip().split("\n") if line.strip()]
        if not lines:
            st.warning("No valid clauses found. Please enter one clause per line.")
            return

        results = []
        progress_bar = st.progress(0, text="Classifying clauses...")

        for i, line in enumerate(lines):
            if len(line) < 10:
                results.append(
                    {
                        "Clause Text": line,
                        "Predicted Type": "(too short)",
                        "Confidence": 0.0,
                    }
                )
            else:
                predicted_label, probs = classify_text(line, model, vectorizer, label_names)
                confidence = float(np.max(probs))
                truncated = line[:100] + ("..." if len(line) > 100 else "")
                results.append(
                    {
                        "Clause Text": truncated,
                        "Predicted Type": predicted_label,
                        "Confidence": confidence,
                    }
                )
            progress_bar.progress((i + 1) / len(lines), text=f"Classified {i + 1}/{len(lines)}")

        progress_bar.empty()

        results_df = pd.DataFrame(results)

        st.markdown(f"### Results ({len(results_df)} clauses)")
        st.dataframe(
            results_df.style.format({"Confidence": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )

        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="clause_classification_results.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="Contract Clause Classifier",
        page_icon="",
        layout="wide",
    )

    st.title("Contract Clause Classification")
    st.markdown(
        "Classify SEC contract provisions into **100 clause types** using the "
        "[LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) dataset "
        "and a TF-IDF + LightGBM model."
    )

    # Load model
    model, vectorizer, label_names = load_model()

    # Sidebar (always rendered)
    render_sidebar(label_names)

    # Gate: if models are not available, show instructions and stop
    if model is None or vectorizer is None or label_names is None:
        st.error("Trained model not found.")
        st.markdown(
            """
### How to train the model

Run the training script to generate the required model files:

```bash
cd contract-clause-classification
python train.py
```

This will create the following files:
- `lgbm_model.joblib` — trained LightGBM classifier
- `tfidf_vectorizer.joblib` — fitted TF-IDF vectorizer
- `label_names.joblib` — list of 100 clause type labels

Alternatively, place the files in `models/saved/` (for the model and label names)
and `models/` (for the vectorizer).

Once the files are available, refresh this page.
"""
        )
        return

    # Tabs
    tab_single, tab_batch = st.tabs(["Single Clause", "Batch Mode"])

    with tab_single:
        render_single_tab(model, vectorizer, label_names)

    with tab_batch:
        render_batch_tab(model, vectorizer, label_names)


if __name__ == "__main__":
    main()
