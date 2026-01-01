"""Memory service wrapping Mem0."""

import time
from typing import List, Dict, Any, Optional
from mem0 import Memory
from config import get_settings


class MemoryService:
    """Service for memory operations using Mem0."""
    
    def __init__(self):
        settings = get_settings()
        
        self.memory = Memory.from_config({
            "vector_store": {
                "provider": "pinecone",
                "config": {
                    "api_key": settings.pinecone_api_key,
                    "collection_name": settings.pinecone_index_name,
                    "embedding_model_dims": 1536,
                    "serverless_config": {
                        "cloud": settings.pinecone_cloud,
                        "region": settings.pinecone_region,
                    }
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": settings.embedding_model,
                    "api_key": settings.openai_api_key,
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": settings.llm_model,
                    "api_key": settings.openai_api_key,
                }
            }
        })
    
    def add(self, patient_id: str, content: str, metadata: Dict[str, Any] = None) -> Dict:
        """Add memory for a patient."""
        result = self.memory.add(
            messages=[{"role": "assistant", "content": content}],
            user_id=patient_id,
            metadata=metadata or {}
        )
        return result
    
    def search(
        self, 
        patient_id: str, 
        query: str, 
        limit: int = 5,
        strategy: str = "vanilla"
    ) -> tuple[List[Dict], float]:
        """
        Search memories for a patient.
        
        Returns:
            (memories, latency_ms)
        """
        start = time.perf_counter()
        
        # Base search
        results = self.memory.search(
            query=query,
            user_id=patient_id,
            limit=limit * 2 if strategy != "vanilla" else limit
        )
        
        # Apply strategy
        if strategy == "hybrid":
            results = self._apply_hybrid_bm25(query, results, limit)
        elif strategy == "temporal":
            results = self._apply_temporal_decay(results, limit)
        elif strategy == "entity":
            results = self._apply_entity_filter(query, results, limit)
        else:
            results = results[:limit] if results else []
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Normalize results
        memories = []
        for r in results:
            memories.append({
                "id": r.get("id", ""),
                "content": r.get("memory", r.get("content", "")),
                "metadata": r.get("metadata", {}),
                "score": r.get("score")
            })
        
        return memories, latency_ms
    
    def get_all(self, patient_id: str) -> List[Dict]:
        """Get all memories for a patient."""
        results = self.memory.get_all(user_id=patient_id)
        return results if results else []
    
    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        try:
            self.memory.delete(memory_id)
            return True
        except:
            return False
    
    # --- Retrieval Strategies ---
    
    def _apply_hybrid_bm25(
        self, 
        query: str, 
        results: List[Dict], 
        limit: int,
        bm25_weight: float = 0.3
    ) -> List[Dict]:
        """Apply BM25 + dense fusion."""
        from collections import Counter
        import math
        
        if not results:
            return []
        
        query_terms = query.lower().split()
        
        scored = []
        for i, doc in enumerate(results):
            content = doc.get("memory", doc.get("content", "")).lower()
            doc_terms = content.split()
            term_freq = Counter(doc_terms)
            
            # BM25-like score
            bm25_score = 0
            for term in query_terms:
                tf = term_freq.get(term, 0)
                if tf > 0:
                    bm25_score += math.log(1 + tf)
            
            # Dense score (position-based proxy)
            dense_score = 1.0 - (i / len(results))
            
            # Normalize BM25
            max_bm25 = max(1, bm25_score)
            norm_bm25 = bm25_score / max_bm25
            
            # Combine
            final_score = (1 - bm25_weight) * dense_score + bm25_weight * norm_bm25
            scored.append((final_score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:limit]]
    
    def _apply_temporal_decay(
        self, 
        results: List[Dict], 
        limit: int,
        decay_factor: float = 0.1
    ) -> List[Dict]:
        """Boost recent memories with temporal decay."""
        from datetime import datetime
        
        if not results:
            return []
        
        now = datetime.now()
        scored = []
        
        for i, doc in enumerate(results):
            base_score = 1.0 - (i / len(results))
            
            # Get date from metadata
            metadata = doc.get("metadata", {})
            date_str = metadata.get("date")
            
            temporal_boost = 1.0
            if date_str:
                try:
                    doc_date = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
                    days_old = (now - doc_date.replace(tzinfo=None)).days
                    temporal_boost = 2.0 ** (-decay_factor * days_old / 365)
                except:
                    pass
            
            final_score = base_score * (1 + temporal_boost)
            scored.append((final_score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:limit]]
    
    def _apply_entity_filter(
        self, 
        query: str, 
        results: List[Dict], 
        limit: int
    ) -> List[Dict]:
        """Pre-filter by medical entities."""
        
        MEDICAL_TERMS = {
            "diabetes", "hypertension", "copd", "asthma", "cardiac", "heart",
            "kidney", "renal", "liver", "cancer", "depression", "anxiety",
            "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin"
        }
        
        query_lower = query.lower()
        query_entities = [t for t in MEDICAL_TERMS if t in query_lower]
        
        if not query_entities:
            return results[:limit]
        
        # Boost results containing query entities
        scored = []
        for i, doc in enumerate(results):
            content = doc.get("memory", doc.get("content", "")).lower()
            base_score = 1.0 - (i / len(results))
            
            entity_matches = sum(1 for e in query_entities if e in content)
            entity_boost = 1 + (entity_matches * 0.2)
            
            final_score = base_score * entity_boost
            scored.append((final_score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:limit]]


# Singleton instance
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get or create memory service singleton."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
