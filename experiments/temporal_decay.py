"""
Temporal Decay Retrieval

Boosts recent memories, decays older ones.
"""

import os
import time
import math
from typing import List, Dict, Tuple
from datetime import datetime

from mem0 import Memory
from .base import RetrievalStrategy


class TemporalDecay(RetrievalStrategy):
    """Retrieval with temporal decay weighting."""
    
    name = "temporal_decay"
    description = "Dense retrieval with exponential time decay"
    
    def __init__(self, config: Dict = None, decay_rate: float = 0.1):
        """
        Args:
            decay_rate: Higher = faster decay. 0.1 means half-life ~7 years
        """
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.decay_rate = decay_rate
    
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
    
    def _compute_temporal_weight(self, date_str: str, reference: datetime) -> float:
        """Compute exponential decay weight based on age."""
        if not date_str:
            return 0.5  # Default for undated
        
        try:
            if isinstance(date_str, str):
                date_str = date_str.replace("Z", "+00:00")
                if "T" in date_str:
                    doc_date = datetime.fromisoformat(date_str)
                else:
                    doc_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                doc_date = doc_date.replace(tzinfo=None)
            else:
                return 0.5
            
            days_old = (reference - doc_date).days
            if days_old < 0:
                days_old = 0
            
            # Exponential decay: 2^(-decay_rate * days/365)
            weight = math.pow(2, -self.decay_rate * days_old / 365)
            return max(0.1, weight)  # Floor at 0.1
            
        except:
            return 0.5
    
    def search(self, query: str, patient_id: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Search with temporal decay reranking."""
        start = time.perf_counter()
        now = datetime.now()
        
        # Get more candidates
        results = self.memory.search(query=query, user_id=patient_id, limit=k * 2)
        
        # Handle Mem0 response format {'results': [...]}
        if isinstance(results, dict):
            results = results.get('results', [])
        
        if not results:
            return [], (time.perf_counter() - start) * 1000
        
        # Apply temporal weighting
        scored = []
        for i, doc in enumerate(results):
            base_score = 1.0 - (i / len(results))
            
            metadata = doc.get("metadata", {})
            date_str = metadata.get("date")
            temporal_weight = self._compute_temporal_weight(date_str, now)
            
            final_score = base_score * (1 + temporal_weight)
            scored.append({"doc": doc, "score": final_score, "temporal": temporal_weight})
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        memories = []
        for item in scored[:k]:
            doc = item["doc"]
            memories.append({
                "id": doc.get("id", ""),
                "content": doc.get("memory", doc.get("content", "")),
                "metadata": doc.get("metadata", {}),
                "score": item["score"],
                "temporal_weight": item["temporal"]
            })
        
        return memories, latency_ms
