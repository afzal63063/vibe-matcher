# test_run.py
# -----------------------------------------------
# This script runs your full Vibe Matcher pipeline
# -----------------------------------------------

from src.recommender import (
    load_products,
    build_embeddings_for_products,
    upsert_to_pinecone,
    search,
)

# 1️⃣ Load sample product data
df = load_products()
print(f"Loaded {len(df)} products.")

# 2️⃣ Build local embeddings (no API key needed)
embs = build_embeddings_for_products(df, use_openai=False)
print("✅ Embeddings generated:", embs.shape)

# 3️⃣ Push to Pinecone (if API key and .env set)
try:
    idx = upsert_to_pinecone(df, embs)
except Exception as e:
    print("⚠️ Skipping Pinecone step:", e)
    idx = None

# 4️⃣ Run a test vibe query
search("energetic urban chic", index=idx)
