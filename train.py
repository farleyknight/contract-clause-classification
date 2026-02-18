"""
Contract Clause Classification — TF-IDF + LightGBM Baseline
Dataset: LEDGAR (coastalcph/lex_glue, config "ledgar")
100-class single-label classification of SEC contract provisions.
"""

import time
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
import lightgbm as lgb
import joblib


# ──────────────────────────────────────────────
# 1. Load data & sanity-check
# ──────────────────────────────────────────────

print("Loading LEDGAR dataset...")
ds = load_dataset("coastalcph/lex_glue", "ledgar")
label_names = ds["train"].features["label"].names

print(f"\nSplits: {list(ds.keys())}")
for split in ds:
    print(f"  {split}: {len(ds[split]):,} examples")

print(f"\nNumber of classes: {len(label_names)}")

# Label distribution (train)
from collections import Counter

train_labels = ds["train"]["label"]
label_counts = Counter(train_labels)
sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

print("\n── Label distribution (train) ──")
print(f"{'Rank':<5} {'Label':<45} {'Count':>6} {'%':>6}")
print("-" * 65)
for rank, (label_id, count) in enumerate(sorted_counts, 1):
    pct = 100 * count / len(train_labels)
    print(f"{rank:<5} {label_names[label_id]:<45} {count:>6} {pct:>5.1f}%")

# Flag imbalance
top_count = sorted_counts[0][1]
bot_count = sorted_counts[-1][1]
print(f"\nImbalance ratio (largest / smallest class): {top_count / bot_count:.1f}x")
print(f"Largest class:  {label_names[sorted_counts[0][0]]} ({top_count})")
print(f"Smallest class: {label_names[sorted_counts[-1][0]]} ({bot_count})")

# Show a few examples
print("\n── Sample examples ──")
for i in range(3):
    ex = ds["train"][i]
    print(f"\n[{i}] Label: {label_names[ex['label']]}")
    print(f"    Text:  {ex['text'][:200]}...")


# ──────────────────────────────────────────────
# 2. TF-IDF + LightGBM baseline
# ──────────────────────────────────────────────

print("\n\n══════════════════════════════════════════════")
print("Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer()

X_train = tfidf.fit_transform(ds["train"]["text"])
X_val = tfidf.transform(ds["validation"]["text"])
X_test = tfidf.transform(ds["test"]["text"])

y_train = np.array(ds["train"]["label"])
y_val = np.array(ds["validation"]["label"])
y_test = np.array(ds["test"]["label"])

print(f"Vocabulary size: {len(tfidf.vocabulary_):,}")
print(f"X_train shape: {X_train.shape}")

print("\nTraining LightGBM multiclass classifier...")
t0 = time.time()

model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(label_names),
    n_estimators=200,
    verbosity=-1,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="multi_logloss",
)

elapsed = time.time() - t0
print(f"Training time: {elapsed:.1f}s")


# ──────────────────────────────────────────────
# 3. Evaluation
# ──────────────────────────────────────────────

print("\n\n══════════════════════════════════════════════")
print("Evaluating on test set...")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print(f"\nOverall accuracy: {acc:.4f}")
print(f"Macro F1:         {macro_f1:.4f}")

# Full classification report
report = classification_report(
    y_test, y_pred,
    target_names=label_names,
    output_dict=True,
    zero_division=0,
)
print("\n── Full classification report ──")
print(classification_report(
    y_test, y_pred,
    target_names=label_names,
    zero_division=0,
))

# 10 worst-performing classes by F1
print("\n── 10 worst classes by F1 ──")
class_f1s = []
for name in label_names:
    if name in report:
        class_f1s.append((name, report[name]["f1-score"], int(report[name]["support"])))

class_f1s.sort(key=lambda x: x[1])
print(f"{'Class':<45} {'F1':>6} {'Support':>8}")
print("-" * 62)
for name, f1, support in class_f1s[:10]:
    print(f"{name:<45} {f1:>6.3f} {support:>8}")


# ──────────────────────────────────────────────
# 4. Save model & vectorizer
# ──────────────────────────────────────────────

print("\n\n══════════════════════════════════════════════")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
joblib.dump(model, "lgbm_model.joblib")
# Save label names for convenience
joblib.dump(label_names, "label_names.joblib")

print("Saved:")
print("  tfidf_vectorizer.joblib")
print("  lgbm_model.joblib")
print("  label_names.joblib")
print("\nDone.")
