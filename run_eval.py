# run_eval.py
# Simple evaluation runner for Vibe Matcher (offline/local SBERT embeddings)
# Outputs: outputs/eval_results.csv and outputs/latency_plot.png

import os
import timeit
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Local imports from your project
from src.recommender import load_products, build_embeddings_for_products
from src.embedding_utils import local_sbert_embed

# --- Config ---
QUERIES = [
    "energetic urban chic",
    "relaxed cozy loungewear",
    "boho festival earthy tones"
]
TOP_K = 3
GOOD_THRESHOLD = 0.7
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helpers ---
def top_k_matches(query_vec, product_vectors, df, k=3):
    sims = cosine_similarity(query_vec.reshape(1, -1), product_vectors)[0]
    idx = np.argsort(-sims)[:k]
    results = []
    for i in idx:
        results.append({
            "id": int(df.iloc[i]["id"]),
            "name": df.iloc[i]["name"],
            "score": float(sims[i])
        })
    return results, sims

# --- Load data & build product embeddings (local SBERT) ---
print("Loading products...")
df = load_products()                     # uses data/products.json in your repo
print(f"Loaded {len(df)} products")

print("Building product embeddings (SBERT local)...")
product_vectors = build_embeddings_for_products(df, use_openai=False)  # numpy array
print("Product embeddings shape:", product_vectors.shape)

# --- Run queries, measure time, collect metrics ---
rows = []
latency_samples = {}

for q in QUERIES:
    print("\n--- Query:", q)
    # 1) Embed query with same local model
    q_vec = local_sbert_embed([q])[0]   # returns list of vectors
    q_vec = np.array(q_vec)

    # 2) Single-run similarity & top-k (for printing & metrics)
    results, sims = top_k_matches(q_vec, product_vectors, df, k=TOP_K)
    top_score = results[0]["score"]
    is_good = top_score > GOOD_THRESHOLD

    print(f"Top-{TOP_K} results:")
    for r in results:
        print(f" - {r['name']} (score={r['score']:.3f})")
    print("Top score:", round(top_score, 3), "| Good match? ", is_good)

    # 3) Measure latency using timeit (3 repeats)
    def run_once():
        _ = top_k_matches(q_vec, product_vectors, df, k=TOP_K)
    times = timeit.repeat(lambda: run_once(), repeat=5, number=1)
    latency_samples[q] = times
    avg_latency = sum(times)/len(times)
    print(f"Latency samples (s): {['{:.4f}'.format(t) for t in times]}. Avg={avg_latency:.4f}s")

    # 4) Save row-level metrics
    rows.append({
        "query": q,
        "top_score": top_score,
        "good_match": is_good,
        "avg_latency_s": avg_latency
    })

    # Also store detailed top-k
    for i, r in enumerate(results):
        rows.append({
            "query": q,
            "rank": i+1,
            "product_id": r["id"],
            "product_name": r["name"],
            "score": r["score"]
        })

# --- Save metrics CSV ---
metrics_df = pd.DataFrame([r for r in rows if "top_score" in r or "score" in r])
metrics_path = os.path.join(OUTPUT_DIR, "eval_results.csv")
metrics_df.to_csv(metrics_path, index=False)
print("\nSaved evaluation CSV to:", metrics_path)

# --- Plot latency distributions ---
plt.figure(figsize=(8,5))
for q in QUERIES:
    s = sorted(latency_samples[q])
    plt.plot(s, label=q[:30])
plt.xlabel("Sorted run index")
plt.ylabel("Latency (s)")
plt.title("Latency samples per query (local evaluation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "latency_plot.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print("Saved latency plot to:", plot_path)

# --- Summary printed at end ---
summary_df = pd.DataFrame(rows)
print("\nSummary (showing top metrics rows):")
print(metrics_df.head(10))
print("\nDone.")
