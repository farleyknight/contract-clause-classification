"""
Contract Clause Classification — Class Confusion Analysis
===========================================================
Analyzes which clause types are most confused with each other:
- Builds a confusion graph (nodes=classes, edges=confusion frequency)
- Identifies confusion clusters (groups of classes that confuse each other)
- For top 10 confused pairs, shows example misclassified texts with explanations
- Saves confusion analysis plot

Usage:
    python -m analysis.class_confusion
    python -m analysis.class_confusion --config configs/baseline.yaml
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "baseline.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Confusion Graph
# ──────────────────────────────────────────────

def build_confusion_graph(y_true, y_pred, label_names):
    """
    Build a confusion graph where nodes are classes and edges are
    bidirectional confusion counts (i confused as j + j confused as i).
    """
    n_classes = len(label_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Zero out diagonal
    np.fill_diagonal(cm, 0)

    # Build symmetric confusion (bidirectional)
    edges = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            bidirectional = cm[i, j] + cm[j, i]
            if bidirectional > 0:
                edges.append({
                    "class_a": label_names[i],
                    "class_b": label_names[j],
                    "class_a_id": i,
                    "class_b_id": j,
                    "a_as_b": int(cm[i, j]),
                    "b_as_a": int(cm[j, i]),
                    "total_confusion": int(bidirectional),
                })

    edges.sort(key=lambda x: x["total_confusion"], reverse=True)
    return edges, cm


def find_confusion_clusters(edges, min_confusion=2, max_clusters=10):
    """
    Identify confusion clusters using a simple connected-component approach
    on edges with confusion >= min_confusion.

    Uses union-find for connected components.
    """
    # Filter edges by min confusion
    strong_edges = [e for e in edges if e["total_confusion"] >= min_confusion]

    if not strong_edges:
        return []

    # Union-Find
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for edge in strong_edges:
        union(edge["class_a"], edge["class_b"])

    # Group by root
    clusters = defaultdict(set)
    for edge in strong_edges:
        root = find(edge["class_a"])
        clusters[root].add(edge["class_a"])
        clusters[root].add(edge["class_b"])

    # Sort clusters by size (descending)
    cluster_list = sorted(clusters.values(), key=lambda x: len(x), reverse=True)

    # Annotate with internal confusion count
    result = []
    for cluster in cluster_list[:max_clusters]:
        cluster_names = cluster
        # Sum confusion within cluster
        internal_confusion = sum(
            e["total_confusion"]
            for e in strong_edges
            if e["class_a"] in cluster_names and e["class_b"] in cluster_names
        )
        result.append({
            "classes": sorted(cluster_names),
            "size": len(cluster_names),
            "internal_confusion": internal_confusion,
        })

    return result


# ──────────────────────────────────────────────
# Example Retrieval
# ──────────────────────────────────────────────

def get_confused_examples(y_true, y_pred, texts, label_names, class_a_id, class_b_id, n=3):
    """Get example texts that were confused between class_a and class_b."""
    examples = []

    # a misclassified as b
    mask_ab = (y_true == class_a_id) & (y_pred == class_b_id)
    ab_indices = np.where(mask_ab)[0][:n]
    for idx in ab_indices:
        examples.append({
            "direction": f"{label_names[class_a_id]} -> {label_names[class_b_id]}",
            "true_class": label_names[class_a_id],
            "pred_class": label_names[class_b_id],
            "text": texts[idx][:300],
        })

    # b misclassified as a
    mask_ba = (y_true == class_b_id) & (y_pred == class_a_id)
    ba_indices = np.where(mask_ba)[0][:n]
    for idx in ba_indices:
        examples.append({
            "direction": f"{label_names[class_b_id]} -> {label_names[class_a_id]}",
            "true_class": label_names[class_b_id],
            "pred_class": label_names[class_a_id],
            "text": texts[idx][:300],
        })

    return examples


def explain_confusion(class_a, class_b, class_vocabs):
    """Generate an explanation for why two classes might be confused."""
    vocab_a = class_vocabs.get(class_a, set())
    vocab_b = class_vocabs.get(class_b, set())

    if not vocab_a or not vocab_b:
        return "Insufficient vocabulary data for analysis."

    intersection = vocab_a & vocab_b
    union = vocab_a | vocab_b
    jaccard = len(intersection) / len(union) if union else 0.0

    # Find shared distinctive words (exclude very common ones)
    shared_words = sorted(intersection, key=lambda w: len(w), reverse=True)[:20]

    # Unique to each
    unique_a = vocab_a - vocab_b
    unique_b = vocab_b - vocab_a

    explanation_parts = []
    explanation_parts.append(f"Vocabulary overlap (Jaccard): {jaccard:.3f}")
    explanation_parts.append(f"Shared vocabulary size: {len(intersection):,} words")
    explanation_parts.append(f"Unique to '{class_a}': {len(unique_a):,} words")
    explanation_parts.append(f"Unique to '{class_b}': {len(unique_b):,} words")

    if jaccard > 0.4:
        explanation_parts.append("HIGH OVERLAP: These classes share very similar legal language.")
    elif jaccard > 0.2:
        explanation_parts.append("MODERATE OVERLAP: Significant shared vocabulary suggests semantic similarity.")
    else:
        explanation_parts.append("LOW OVERLAP: Confusion may stem from structural rather than lexical similarity.")

    if shared_words:
        explanation_parts.append(f"Key shared terms: {', '.join(shared_words[:10])}")

    return "\n      ".join(explanation_parts)


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_confusion_analysis(top_edges, clusters, save_path):
    """
    Plot:
    - Top-20 confused pairs as horizontal bar chart
    - Confusion clusters as a heatmap of cluster sizes
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [2, 1]})

    # ── Left: Top confused pairs ──
    ax = axes[0]
    top20 = top_edges[:20]
    pair_labels = [f"{e['class_a'][:25]} <> {e['class_b'][:25]}" for e in top20]
    pair_labels.reverse()
    counts = [e["total_confusion"] for e in top20]
    counts.reverse()

    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(counts)))
    ax.barh(pair_labels, counts, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Bidirectional Confusion Count", fontsize=12)
    ax.set_title("Top 20 Most Confused Class Pairs", fontsize=14)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)

    # ── Right: Confusion clusters ──
    ax = axes[1]
    if clusters:
        cluster_data = []
        for i, cluster in enumerate(clusters[:10]):
            cluster_data.append({
                "Cluster": f"C{i+1}",
                "Size": cluster["size"],
                "Confusion": cluster["internal_confusion"],
            })
        cdf = pd.DataFrame(cluster_data)

        ax.barh(cdf["Cluster"], cdf["Size"], color="steelblue", edgecolor="black", alpha=0.8, label="Classes")
        ax2 = ax.twiny()
        ax2.barh(
            [i - 0.2 for i in range(len(cdf))],
            cdf["Confusion"],
            height=0.4,
            color="coral",
            edgecolor="black",
            alpha=0.7,
            label="Confusion",
        )
        ax.set_xlabel("Number of Classes", fontsize=11)
        ax2.set_xlabel("Internal Confusion Count", fontsize=11, color="coral")
        ax.set_title("Confusion Clusters", fontsize=14)
        ax.legend(loc="lower right")
        ax2.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No confusion clusters found", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Confusion Clusters", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion analysis plot to {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(config_path: str = None):
    config = load_config(config_path)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load label names
    label_path = PROJECT_ROOT / config["output"]["label_names_path"]
    label_names = joblib.load(label_path)

    # Load test predictions
    preds_path = PROJECT_ROOT / "models" / "saved" / "test_predictions.npz"
    preds = np.load(preds_path)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]

    # Load test text data
    test_path = PROJECT_ROOT / config["data"]["test_path"]
    test_df = pd.read_parquet(test_path)
    texts = test_df["text"].values

    # Build class vocabularies from training data for explanations
    train_path = PROJECT_ROOT / config["data"]["train_path"]
    train_df = pd.read_parquet(train_path)
    class_vocabs = {}
    for text, label in zip(train_df["text"].values, train_df["label"].values):
        name = label_names[label]
        if name not in class_vocabs:
            class_vocabs[name] = set()
        class_vocabs[name].update(text.lower().split())

    # ── Build confusion graph ──
    print(f"\n{'='*60}")
    print("  Class Confusion Analysis")
    print(f"{'='*60}")

    edges, cm = build_confusion_graph(y_true, y_pred, label_names)

    print(f"\n  Total confusion edges (bidirectional): {len(edges)}")
    total_confusion = sum(e["total_confusion"] for e in edges)
    print(f"  Total confusion count: {total_confusion:,}")

    # ── Top confused pairs ──
    print(f"\n{'='*60}")
    print("  Top 20 Most Confused Class Pairs")
    print(f"{'='*60}")

    print(f"\n  {'Class A':<30} {'Class B':<30} {'A->B':>5} {'B->A':>5} {'Total':>6}")
    print(f"  {'-'*80}")
    for edge in edges[:20]:
        print(f"  {edge['class_a']:<30} {edge['class_b']:<30} "
              f"{edge['a_as_b']:>5} {edge['b_as_a']:>5} {edge['total_confusion']:>6}")

    # ── Confusion clusters ──
    print(f"\n{'='*60}")
    print("  Confusion Clusters")
    print(f"{'='*60}")

    clusters = find_confusion_clusters(edges, min_confusion=2)

    if clusters:
        for i, cluster in enumerate(clusters):
            print(f"\n  Cluster {i+1} ({cluster['size']} classes, "
                  f"internal confusion: {cluster['internal_confusion']}):")
            for name in cluster["classes"]:
                print(f"    - {name}")
    else:
        print("  No confusion clusters found with min_confusion >= 2.")

    # ── Top 10 confused pairs: examples + explanations ──
    print(f"\n{'='*60}")
    print("  Top 10 Confused Pairs: Examples and Explanations")
    print(f"{'='*60}")

    for rank, edge in enumerate(edges[:10], 1):
        class_a = edge["class_a"]
        class_b = edge["class_b"]
        class_a_id = edge["class_a_id"]
        class_b_id = edge["class_b_id"]

        print(f"\n  [{rank}] {class_a} <-> {class_b}")
        print(f"      Total confusion: {edge['total_confusion']} "
              f"({edge['a_as_b']} + {edge['b_as_a']})")

        # Explanation
        explanation = explain_confusion(class_a, class_b, class_vocabs)
        print(f"      {explanation}")

        # Example misclassified texts
        examples = get_confused_examples(
            y_true, y_pred, texts, label_names,
            class_a_id, class_b_id, n=2
        )

        if examples:
            print(f"      Example misclassifications:")
            for ex in examples:
                print(f"        Direction: {ex['direction']}")
                print(f"        Text: {ex['text']}{'...' if len(ex['text']) >= 300 else ''}")
                print()
        else:
            print("      No examples found in test set.")

    # ── Which classes are most often confused (as source) ──
    print(f"\n{'='*60}")
    print("  Classes Most Often Misclassified (as source)")
    print(f"{'='*60}")

    # Sum confusion where each class appears as true class
    class_confusion_total = defaultdict(int)
    for edge in edges:
        class_confusion_total[edge["class_a"]] += edge["a_as_b"]
        class_confusion_total[edge["class_b"]] += edge["b_as_a"]

    sorted_confused = sorted(class_confusion_total.items(), key=lambda x: x[1], reverse=True)[:15]

    print(f"\n  {'Class':<45} {'Times Misclassified':>20}")
    print(f"  {'-'*67}")
    for name, count in sorted_confused:
        print(f"  {name:<45} {count:>20}")

    # ── Save plot ──
    plot_confusion_analysis(edges, clusters, PLOTS_DIR / "class_confusion_analysis.png")

    print(f"\n{'='*60}")
    print("  Class Confusion Analysis Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    config_path = None
    if len(sys.argv) > 1 and sys.argv[1] == "--config" and len(sys.argv) > 2:
        config_path = sys.argv[2]
    main(config_path)
