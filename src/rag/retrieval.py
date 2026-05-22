import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

INDEX_DIR = "src/rag"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        
        if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise RuntimeError(f"Index or metadata not found in {INDEX_DIR}. Please run index.py first.")
            
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r") as f:
            # json dict keys are strings, convert to int to match faiss output
            metadata_str_keys = json.load(f)
            self.metadata = {int(k): v for k, v in metadata_str_keys.items()}
            
    def retrieve(self, query: str, top_k: int = 3, filter_product: str = None) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Retrieves top_k chunks. Returns chunks and a boolean indicating if retrieval is weak.
        """
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # We fetch more if we need to filter
        search_k = top_k * 5 if filter_product else top_k
        distances, indices = self.index.search(query_embedding, search_k)
        
        results = []
        is_weak = True
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
                
            meta = self.metadata.get(idx)
            if not meta:
                continue
                
            if filter_product and meta['product'] != filter_product:
                continue
                
            results.append({
                "score": float(dist),
                "metadata": meta
            })
            
            if len(results) == top_k:
                break
                
        # A simple heuristic for L2 distance with all-MiniLM-L6-v2
        # Lower distance is better. Typically < 1.0 means good semantic overlap.
        # If the best result is > 1.2, we consider it weak.
        if results and results[0]["score"] < 1.2:
            is_weak = False
            
        return results, is_weak

# Global retriever instance to avoid reloading
_retriever = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
