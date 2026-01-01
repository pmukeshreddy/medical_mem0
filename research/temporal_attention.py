"""
Temporal Attention: Learned Decay Weights for Memory Retrieval

Key Idea:
- Not all memories decay equally over time
- Chronic conditions (diabetes) → slow decay, always relevant
- Acute episodes (flu) → fast decay, less relevant after resolution
- Learn optimal decay rates per category

Instead of:
    score = base_score * time_decay(days_old)

We use:
    score = base_score * learned_decay(category, days_old)
"""

import os
import time
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from mem0 import Memory


@dataclass
class DecayConfig:
    """Decay configuration for a category."""
    category: str
    half_life_days: float  # Days until relevance halves
    min_relevance: float = 0.1  # Floor (never goes below this)


# Learned decay rates (in production, these would be trained)
DEFAULT_DECAY_CONFIGS = {
    # Chronic conditions - very slow decay
    "chronic": DecayConfig("chronic", half_life_days=365 * 5, min_relevance=0.5),
    
    # Medications - medium decay (current meds most relevant)
    "medication": DecayConfig("medication", half_life_days=180, min_relevance=0.2),
    
    # Vitals - fast decay (recent vitals most relevant)
    "vitals": DecayConfig("vitals", half_life_days=30, min_relevance=0.1),
    
    # Labs - medium decay
    "labs": DecayConfig("labs", half_life_days=90, min_relevance=0.15),
    
    # Acute episodes - very fast decay
    "acute": DecayConfig("acute", half_life_days=14, min_relevance=0.05),
    
    # Visits - medium decay
    "visit": DecayConfig("visit", half_life_days=60, min_relevance=0.1),
    
    # Default
    "default": DecayConfig("default", half_life_days=90, min_relevance=0.1),
}

# Keywords to determine category
CATEGORY_KEYWORDS = {
    "chronic": [
        "diabetes", "hypertension", "copd", "asthma", "heart failure",
        "chronic kidney", "arthritis", "cancer", "hiv", "epilepsy"
    ],
    "acute": [
        "flu", "cold", "infection", "acute", "injury", "fracture",
        "sprain", "stitches", "fever"
    ],
    "vitals": [
        "blood pressure", "heart rate", "temperature", "weight", 
        "height", "bmi", "pulse", "respiratory"
    ],
    "labs": [
        "a1c", "glucose", "cholesterol", "creatinine", "hemoglobin",
        "platelet", "wbc", "bun", "liver function", "thyroid"
    ],
    "medication": [
        "prescribed", "medication", "drug", "dose", "mg", "tablet",
        "capsule", "injection", "insulin"
    ],
}


class TemporalAttentionRetriever:
    """
    Memory retrieval with learned temporal decay.
    
    Different categories decay at different rates:
    - Chronic conditions: Very slow decay (always relevant)
    - Acute episodes: Fast decay (less relevant after resolved)
    - Vitals: Fast decay (recent most important)
    """
    
    def __init__(self, config: Dict = None, decay_configs: Dict[str, DecayConfig] = None):
        """Initialize with Mem0 config and optional custom decay rates."""
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.decay_configs = decay_configs or DEFAULT_DECAY_CONFIGS
    
    def _default_config(self) -> Dict:
        """Default Mem0 config."""
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
    
    def _categorize(self, content: str) -> str:
        """Determine category from content."""
        content_lower = content.lower()
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                return category
        
        return "default"
    
    def _compute_temporal_weight(
        self, 
        content: str, 
        date_str: Optional[str],
        reference_date: datetime = None
    ) -> float:
        """
        Compute temporal weight using learned decay rates.
        
        Formula: weight = max(min_relevance, 0.5 ^ (days / half_life))
        """
        reference_date = reference_date or datetime.now()
        
        # Get category and decay config
        category = self._categorize(content)
        decay_config = self.decay_configs.get(category, self.decay_configs["default"])
        
        # If no date, use moderate weight
        if not date_str:
            return 0.5
        
        # Parse date
        try:
            if isinstance(date_str, str):
                # Handle various date formats
                date_str = date_str.replace("Z", "+00:00")
                if "T" in date_str:
                    memory_date = datetime.fromisoformat(date_str)
                else:
                    memory_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                memory_date = memory_date.replace(tzinfo=None)
            else:
                return 0.5
        except:
            return 0.5
        
        # Compute days old
        days_old = (reference_date - memory_date).days
        if days_old < 0:
            days_old = 0
        
        # Exponential decay: 0.5 ^ (days / half_life)
        decay = math.pow(0.5, days_old / decay_config.half_life_days)
        
        # Apply floor
        weight = max(decay_config.min_relevance, decay)
        
        return weight
    
    def search(
        self, 
        query: str, 
        patient_id: str, 
        k: int = 5,
        reference_date: datetime = None
    ) -> Tuple[List[Dict], float]:
        """
        Search with temporal attention.
        
        Returns:
            (memories, latency_ms)
        """
        start = time.perf_counter()
        reference_date = reference_date or datetime.now()
        
        # Get more candidates for reranking
        results = self.memory.search(
            query=query,
            user_id=patient_id,
            limit=k * 3
        )
        
        # Handle Mem0 response format {'results': [...]}
        if isinstance(results, dict):
            results = results.get('results', [])
        
        if not results:
            return [], (time.perf_counter() - start) * 1000
        
        # Apply temporal weights
        scored = []
        for i, mem in enumerate(results):
            content = mem.get("memory", mem.get("content", ""))
            metadata = mem.get("metadata", {})
            date_str = metadata.get("date")
            
            # Base score from vector similarity (position-based proxy)
            base_score = 1.0 - (i / len(results))
            
            # Temporal weight
            temporal_weight = self._compute_temporal_weight(
                content, date_str, reference_date
            )
            
            # Combined score
            final_score = base_score * temporal_weight
            
            scored.append({
                "memory": mem,
                "base_score": base_score,
                "temporal_weight": temporal_weight,
                "final_score": final_score,
                "category": self._categorize(content)
            })
        
        # Sort by final score
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Return top k
        memories = []
        for item in scored[:k]:
            mem = item["memory"]
            memories.append({
                "id": mem.get("id", ""),
                "content": mem.get("memory", mem.get("content", "")),
                "memory": mem.get("memory", mem.get("content", "")),
                "metadata": mem.get("metadata", {}),
                "score": item["final_score"],
                "temporal_weight": item["temporal_weight"],
                "category": item["category"]
            })
        
        latency_ms = (time.perf_counter() - start) * 1000
        return memories, latency_ms
    
    def explain_decay(self, content: str, date_str: str) -> Dict:
        """Explain why a memory has a certain temporal weight."""
        category = self._categorize(content)
        decay_config = self.decay_configs.get(category, self.decay_configs["default"])
        weight = self._compute_temporal_weight(content, date_str)
        
        # Parse date for explanation
        try:
            if "T" in date_str:
                memory_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                memory_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
            days_old = (datetime.now() - memory_date.replace(tzinfo=None)).days
        except:
            days_old = "unknown"
        
        return {
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "detected_category": category,
            "half_life_days": decay_config.half_life_days,
            "min_relevance": decay_config.min_relevance,
            "days_old": days_old,
            "temporal_weight": weight,
            "explanation": f"Category '{category}' has half-life of {decay_config.half_life_days} days. "
                          f"Memory is {days_old} days old, resulting in weight {weight:.3f}"
        }
    
    def get_decay_configs(self) -> Dict[str, Dict]:
        """Return current decay configurations."""
        return {
            name: {
                "half_life_days": config.half_life_days,
                "min_relevance": config.min_relevance
            }
            for name, config in self.decay_configs.items()
        }
    
    def update_decay_config(self, category: str, half_life_days: float, min_relevance: float = 0.1):
        """Update decay config for a category (for tuning)."""
        self.decay_configs[category] = DecayConfig(
            category=category,
            half_life_days=half_life_days,
            min_relevance=min_relevance
        )


# Convenience function
def create_temporal_retriever(config: Dict = None) -> TemporalAttentionRetriever:
    """Create a TemporalAttentionRetriever instance."""
    return TemporalAttentionRetriever(config)
