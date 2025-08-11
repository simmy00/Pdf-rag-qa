# rag/embeddings_index.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Dict, Tuple
import os
import pickle

class FAISSIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model = SentenceTransformer(model_name)
        self.dim = dim
        self.index = None
        self.metadatas = []  # parallel list to vectors
        self.ids = []

    def _normalize(self, vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms==0] = 1e-10
        return vectors / norms

    def build(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        vectors = self._normalize(vectors).astype('float32')
        self.index = faiss.IndexFlatIP(self.dim)  # cosine via normalized vectors dot-product
        self.index.add(vectors)
        self.metadatas = chunks
        self.ids = list(range(len(chunks)))

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        q_vec = self.model.encode([query_text], convert_to_numpy=True)
        q_vec = self._normalize(q_vec).astype('float32')
        D, I = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((float(score), self.metadatas[idx]))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadatas.pkl"), "wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadatas.pkl"), "rb") as f:
            self.metadatas = pickle.load(f)
