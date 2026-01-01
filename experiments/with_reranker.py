"""
With Reranker - Cohere Rerank v3

Two-stage retrieval: dense retrieval + neural reranking.
"""

import os
import time
from typing import List, Dict, Tuple

from mem0 import Memory
from .base import RetrievalStrategy


class WithReranker(RetrievalStrategy):
    """Dense retrieval + Cohere reranking."""
    
    name = "with_reranker"
    description = "Dense retrieval + Cohere Rerank v3"
    
    def __init__(self, config: Dict = None, cohere_api_key: str = None):
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self.cohere_client = None
        
        if self.cohere_api_key:
            try:
                import cohere
                self.cohere_client = cohere.Client(self.cohere_api_key)
            except ImportError:
                print("WARNING: cohere package not installed. pip install cohere")
    
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
    
    def _rerank_with_cohere(self, query: str, documents: List[str], top_n: int) -> List[int]:
        """Rerank documents using Cohere."""
        if not self.cohere_client:
            return list(range(min(top_n, len(documents))))
        
        response = self.cohere_client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=top_n
        )
        
        return [r.index for r in response.results]
    
    def _fallback_rerank(self, query: str, documents: List[Dict], top_n: int) -> List[int]:
        """Fallback reranking using keyword overlap."""
        query_terms = set(query.lower().split())
        
        scores = []
        for i, doc in enumerate(documents):
            content = doc.get("memory", doc.get("content", "")).lower()
            doc_terms = set(content.split())
            overlap = len(query_terms & doc_terms)
            scores.append((overlap, i))
        
        scores.sort(reverse=True)
        return [idx for _, idx in scores[:top_n]]
    
    def search(self, query: str, patient_id: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Two-stage retrieval with reranking."""
        start = time.perf_counter()
        
        # Stage 1: Dense retrieval (get more candidates)
        results = self.memory.search(query=query, user_id=patient_id, limit=k * 4)
        
        if not results:
            return [], (time.perf_counter() - start) * 1000
        
        # Stage 2: Rerank
        if self.cohere_client:
            documents = [r.get("memory", r.get("content", "")) for r in results]
            reranked_indices = self._rerank_with_cohere(query, documents, k)
        else:
            reranked_indices = self._fallback_rerank(query, results, k)
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        memories = []
        for idx in reranked_indices:
            if idx < len(results):
                doc = results[idx]
                memories.append({
                    "id": doc.get("id", ""),
                    "content": doc.get("memory", doc.get("content", "")),
                    "metadata": doc.get("metadata", {}),
                    "score": 1.0 - (len(memories) / k)  # Rank-based score
                })
        
        return memories, latency_ms
