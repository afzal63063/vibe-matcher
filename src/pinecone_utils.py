# src/pinecone_utils.py
import os
import pinecone
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vibe-matcher")

def init_pinecone():
    pinecone.init(api_key=API_KEY, environment=ENV)
    if INDEX_NAME not in pinecone.list_indexes():
        # create index; choose dimension after you create embeddings (e.g. 1536 for OpenAI)
        pinecone.create_index(index_name=INDEX_NAME, dimension=1536, metric='cosine')
    return pinecone.Index(INDEX_NAME)

def upsert_items(index, items):
    # items: list of (id, vector, metadata)
    index.upsert(vectors=items)

def query(index, vector, top_k=3):
    res = index.query(vector=vector, top_k=top_k, include_metadata=True, include_values=False)
    return res
