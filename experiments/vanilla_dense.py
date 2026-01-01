"""
Vanilla Dense Retrieval - Baseline Strategy

Simple vector similarity search using Mem0's default behavior.
This is the baseline all other strategies are compared against.
"""

import os
import time
from typing import List, Dict, Tuple

from mem0 import Memory
from .base import RetrievalStrategy


class VanillaDense(RetrievalStrategy):
    """Baseline dense vector search."""
    
    name = "vanilla_dense"
    description = "Baseline dense vector similarity search"
    
    def __init__(self, config: Dict = None):
        """Initialize with Mem0 config."""
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
    
    def _default_config(self) -> Dict:
        return {
            "vector_store": {
                "provider": "pinecone",
                "config": {
                    "api_key": os.getenv("PINECONE_API_KEY"),
                    "collection_name": os.getenv("PINECONE_INDEX_NAME", "medmem0"),
                    "embedding_model_dims": 1536,
                    "serverless_config": {
                        "cloud": "aws",
                        "region": "us-east-1",
                    }
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            }
        }
    
    def search(
        self, 
        query: str, 
        patient_id: str, 
        k: int = 5
    ) -> Tuple[List[Dict], float]:
        """Simple dense vector search."""
        start = time.perf_counter()
        
        results = self.memory.search(
            query=query,
            user_id=patient_id,
            limit=k
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Handle Mem0 response format {'results': [...]}
        if isinstance(results, dict):
            results = results.get('results', [])
        
        # Normalize format
        memories = []
        for r in (results or []):
            if isinstance(r, dict):
                memories.append({
                    "id": r.get("id", ""),
                    "content": r.get("memory", r.get("content", "")),
                    "metadata": r.get("metadata", {}),
                    "score": r.get("score")
                })
        
        return memories, latency_ms
