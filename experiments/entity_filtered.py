"""
Entity Filtered Retrieval

Pre-filters by medical entities, boosts matching content.
"""

import os
import time
from typing import List, Dict, Tuple, Set

from mem0 import Memory
from .base import RetrievalStrategy


# Medical entity vocabulary
MEDICAL_ENTITIES = {
    "conditions": {
        "diabetes", "hypertension", "copd", "asthma", "cardiac", "heart failure",
        "kidney disease", "renal", "liver", "hepatic", "cancer", "depression",
        "anxiety", "obesity", "stroke", "pneumonia", "arthritis", "anemia"
    },
    "medications": {
        "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin",
        "omeprazole", "levothyroxine", "albuterol", "prednisone", "aspirin",
        "ibuprofen", "gabapentin", "sertraline", "metoprolol", "furosemide"
    },
    "vitals": {
        "blood pressure", "heart rate", "temperature", "weight", "bmi",
        "pulse", "respiratory rate", "oxygen saturation"
    },
    "labs": {
        "glucose", "a1c", "cholesterol", "ldl", "hdl", "creatinine",
        "hemoglobin", "platelet", "wbc", "bun", "potassium", "sodium"
    }
}


class EntityFiltered(RetrievalStrategy):
    """Retrieval with medical entity filtering and boosting."""
    
    name = "entity_filtered"
    description = "Entity extraction + filtered retrieval"
    
    def __init__(self, config: Dict = None, entity_boost: float = 0.3):
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.entity_boost = entity_boost
        
        # Flatten entity vocabulary
        self.all_entities: Set[str] = set()
        for category in MEDICAL_ENTITIES.values():
            self.all_entities.update(category)
    
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
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract medical entities from text."""
        text_lower = text.lower()
        found = set()
        
        for entity in self.all_entities:
            if entity in text_lower:
                found.add(entity)
        
        return found
    
    def _compute_entity_score(self, query_entities: Set[str], doc_content: str) -> float:
        """Compute entity overlap score."""
        if not query_entities:
            return 0.0
        
        doc_entities = self._extract_entities(doc_content)
        overlap = len(query_entities & doc_entities)
        
        return overlap / len(query_entities)
    
    def search(self, query: str, patient_id: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Search with entity-based filtering and boosting."""
        start = time.perf_counter()
        
        # Extract entities from query
        query_entities = self._extract_entities(query)
        
        # Enhance query with entities
        if query_entities:
            enhanced_query = f"{query} {' '.join(query_entities)}"
        else:
            enhanced_query = query
        
        # Get candidates
        results = self.memory.search(query=enhanced_query, user_id=patient_id, limit=k * 2)
        
        if not results:
            return [], (time.perf_counter() - start) * 1000
        
        # Score with entity boosting
        scored = []
        for i, doc in enumerate(results):
            content = doc.get("memory", doc.get("content", ""))
            
            base_score = 1.0 - (i / len(results))
            entity_score = self._compute_entity_score(query_entities, content)
            
            final_score = base_score + (self.entity_boost * entity_score)
            scored.append({
                "doc": doc, 
                "score": final_score, 
                "entity_score": entity_score,
                "matched_entities": list(self._extract_entities(content) & query_entities)
            })
        
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
                "matched_entities": item["matched_entities"]
            })
        
        return memories, latency_ms
