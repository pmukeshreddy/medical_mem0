"""
Hybrid BM25 + Dense Retrieval

Combines sparse (BM25) and dense (vector) retrieval with score fusion.
"""

import os
import time
import math
from typing import List, Dict, Tuple
from collections import Counter

from mem0 import Memory
from .base import RetrievalStrategy


class HybridBM25(RetrievalStrategy):
    """Hybrid BM25 + dense vector search."""
    
    name = "hybrid_bm25"
    description = "BM25 sparse + dense vector fusion"
    
    def __init__(self, config: Dict = None, bm25_weight: float = 0.3):
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.bm25_weight = bm25_weight
    
    def _default_config(self) -> Dict:
        return {
            "vector_store": {
                "provider": "pinecone",
                "config": {
                    "api_key": os.getenv("PINECONE_API_KEY"),
                    "collection_name": os.getenv("PINECONE_INDEX_NAME", "medmem0"),
                    "embedding_model_dims": 1536,
                    "serverless_config": {"cloud": "aws", "region": "us-east-1"}
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small", "api_key": os.getenv("OPENAI_API_KEY")}
            },
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
            }
        }
    
    def _compute_bm25_score(self, query: str, content: str, k1: float = 1.5, b: float = 0.75) -> float:
        """Compute BM25-like score."""
        query_terms = query.lower().split()
        doc_terms = content.lower().split()
        term_freq = Counter(doc_terms)
        doc_len = len(doc_terms)
        avg_doc_len = 100  # Approximate
        
        score = 0.0
        for term in query_terms:
            tf = term_freq.get(term, 0)
            if tf > 0:
                # Simplified BM25
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                score += math.log(1 + tf) * (numerator / denominator)
        
        return score
    
    def search(self, query: str, patient_id: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Hybrid search with BM25 + dense fusion."""
        start = time.perf_counter()
        
        # Get more candidates for reranking
        results = self.memory.search(query=query, user_id=patient_id, limit=k * 3)
        
        # Handle Mem0 response format {'results': [...]}
        if isinstance(results, dict):
            results = results.get('results', [])
        
        if not results:
            return [], (time.perf_counter() - start) * 1000
        
        # Score with BM25 + dense fusion
        scored = []
        max_bm25 = 0.001  # Avoid division by zero
        
        for i, doc in enumerate(results):
            content = doc.get("memory", doc.get("content", ""))
            bm25 = self._compute_bm25_score(query, content)
            max_bm25 = max(max_bm25, bm25)
            scored.append({"doc": doc, "bm25": bm25, "rank": i})
        
        # Normalize and combine scores
        for item in scored:
            dense_score = 1.0 - (item["rank"] / len(results))
            bm25_norm = item["bm25"] / max_bm25
            item["final"] = (1 - self.bm25_weight) * dense_score + self.bm25_weight * bm25_norm
        
        # Sort by final score
        scored.sort(key=lambda x: x["final"], reverse=True)
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        memories = []
        for item in scored[:k]:
            doc = item["doc"]
            memories.append({
                "id": doc.get("id", ""),
                "content": doc.get("memory", doc.get("content", "")),
                "metadata": doc.get("metadata", {}),
                "score": item["final"]
            })
        
        return memories, latency_ms
