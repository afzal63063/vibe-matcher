# src/faiss_utils.py
import faiss
import numpy as np
import pickle

def build_faiss_index(vectors: np.ndarray, nlist=50):
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    return index

def save_index(index, path):
    faiss.write_index(index, path)

def load_index(path):
    return faiss.read_index(path)

def search(index, query_vec, top_k=3):
    D, I = index.search(query_vec, top_k)
    return D, I
