# Contract Clause Classification

Multiclass classification of SEC contract provisions using the [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) dataset (100 classes, ~80k examples).

## Dataset

- **Source:** `coastalcph/lex_glue`, config `ledgar`
- **Task:** Single-label classification of contract clause type
- **Splits:** 60k train / 10k validation / 10k test
- **Classes:** 100 (e.g., Governing Laws, Notices, Severability, Amendments)
- **Imbalance:** 137x ratio between largest and smallest class

## Current Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| TF-IDF + LightGBM (default params) | 0.086 | 0.043 |

The baseline model is severely undertrained — LightGBM defaults are a poor fit for sparse, high-dimensional TF-IDF features with 100 classes.

## Usage

```bash
pip install datasets lightgbm scikit-learn joblib
python train.py
```

Saves `tfidf_vectorizer.joblib`, `lgbm_model.joblib`, and `label_names.joblib` to disk.

## Next Steps

TODO
