"""Memory service wrapping Mem0."""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from mem0 import Memory
from openai import OpenAI
from config import get_settings


class LLMMedicalExpander:
    """LLM-based medical term expansion."""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.cache = {}
    
    def expand(self, query: str) -> List[str]:
        if query in self.cache:
            return self.cache[query]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """Extract ALL medical concepts from the query and provide clinical synonyms/abbreviations.
Return ONLY a comma-separated list of terms that would appear in clinical notes.
Focus on: vital signs, lab values, measurements, conditions, medications, abbreviations.
Keep it to 10-12 most relevant terms. No explanations."""
            }, {
                "role": "user",
                "content": query
            }],
            temperature=0.3,
            max_tokens=100
        )
        
        result = response.choices[0].message.content.strip()
        terms = [t.strip() for t in result.split(",") if t.strip()][:10]
        self.cache[query] = terms
        return terms


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
        
        # Initialize expander for expanded_only strategy
        self.expander = LLMMedicalExpander(OpenAI(api_key=settings.openai_api_key))
    
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
    ) -> Tuple[List[Dict], float]:
        """
        Search memories for a patient.
        
        Strategies:
        - vanilla: Basic dense vector search
        - enhanced: LLM medical term expansion
        
        Returns:
            (memories, latency_ms)
        """
        start = time.perf_counter()
        
        search_query = query
        
        # Apply enhanced strategy
        if strategy == "enhanced":
            expansions = self.expander.expand(query)
            search_query = f"{query} {' '.join(expansions)}"
        
        # Execute search
        results = self.memory.search(
            query=search_query,
            user_id=patient_id,
            limit=limit
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Normalize results - handle different mem0 response formats
        memories = []
        if results:
            # Handle if results is a dict with 'results' key
            if isinstance(results, dict) and "results" in results:
                results = results["results"]
            
            for r in results:
                # Handle string results
                if isinstance(r, str):
                    memories.append({
                        "id": "",
                        "content": r,
                        "metadata": {},
                        "score": None
                    })
                # Handle dict results
                elif isinstance(r, dict):
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


# Singleton instance
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get or create memory service singleton."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service