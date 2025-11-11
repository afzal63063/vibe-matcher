# src/recommender.py

import os
import json
import numpy as np
import pandas as pd

from src.embedding_utils import openai_embed, local_sbert_embed
from src.pinecone_utils import init_pinecone, query as pinecone_query, upsert_items
from src.faiss_utils import build_faiss_index, search as faiss_search


def load_products(path="data/products.json"):
    """Load product dataset"""
    return pd.read_json(path)


def build_embeddings_for_products(df, use_openai=True):
    """Generate embeddings for each product description"""
    texts = df["desc"].tolist()
    if use_openai:
        embs = openai_embed(texts)
    else:
        embs = local_sbert_embed(texts)
    return np.array(embs)


def upsert_to_pinecone(df, vectors, index=None):
    """Push product embeddings into Pinecone"""
    idx = index or init_pinecone()
    items = []
    for i, v in enumerate(vectors):
        metadata = {
            "name": df.iloc[i]["name"],
            "tags": df.iloc[i]["tags"],
            "desc": df.iloc[i]["desc"],
        }
        items.append((str(df.iloc[i]["id"]), v.tolist(), metadata))
    upsert_items(idx, items)
    print(f"✅ Upserted {len(items)} items to Pinecone index.")
    return idx


def search(query_text, index=None, use_openai_query=True, top_k=3):
    """Search similar products by vibe query"""
    if use_openai_query:
        qv = openai_embed([query_text])[0]
        if index:
            res = pinecone_query(index, qv, top_k=top_k)
            print("🔍 Top matches:")
            for match in res.matches:
                print(f"- {match['metadata']['name']} (score={match['score']:.3f})")
            return res
    # Otherwise fallback to FAISS/local search (if implemented)
    print("⚠️ Local search fallback not implemented yet.")
