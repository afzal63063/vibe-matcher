# src/embedding_utils.py
import os
from typing import List
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY

def openai_embed(texts: List[str], model: str="text-embedding-ada-002"):
    """Return list of embeddings from OpenAI"""
    # Batched requests recommended for many inputs
    resp = openai.Embedding.create(model=model, input=texts)
    return [item['embedding'] for item in resp['data']]

# Local fallback using sentence-transformers (dense vectors)
def local_sbert_embed(texts: List[str], model_name: str="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    return m.encode(texts, show_progress_bar=False).tolist()
