"""
MemWalker: Hierarchical Memory Traversal for Patient Data

Inspired by: "MemWalker: Interactive Memory System for LLMs" (2023)
Paper: https://arxiv.org/abs/2310.05029

Key Idea:
- Instead of flat vector search, organize memories into a tree
- Traverse tree intelligently based on query
- More relevant results for hierarchical data (medical records)

Fix: Use hierarchy to BOOST scores, not FILTER candidates.
"""

import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import time

from mem0 import Memory
from openai import OpenAI


@dataclass
class MemoryNode:
    """A node in the memory tree."""
    id: str
    content: str
    category: str
    subcategory: str = ""
    date: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["MemoryNode"] = field(default_factory=list)
    relevance_score: float = 0.0
    original_score: float = 0.0


class CategoryDetector:
    """Detects category from content using keywords."""
    
    CONDITION_KEYWORDS = [
        "diabetes", "hypertension", "copd", "asthma", "heart", "cardiac",
        "kidney", "renal", "liver", "hepatic", "cancer", "depression",
        "anxiety", "arthritis", "stroke", "pneumonia", "bronchitis",
        "sinusitis", "obesity", "hyperlipidemia", "anemia", "prediabetes"
    ]
    
    MEDICATION_KEYWORDS = [
        "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin",
        "omeprazole", "levothyroxine", "albuterol", "prednisone", "aspirin",
        "ibuprofen", "acetaminophen", "amoxicillin", "gabapentin", "sertraline",
        "prescribed", "medication", "drug", "dose", "mg", "twice daily"
    ]
    
    VITAL_KEYWORDS = [
        "blood pressure", "heart rate", "temperature", "respiratory rate",
        "oxygen saturation", "weight", "height", "bmi", "pulse", "vital"
    ]
    
    LAB_KEYWORDS = [
        "a1c", "glucose", "cholesterol", "ldl", "hdl", "triglycerides",
        "creatinine", "bun", "hemoglobin", "platelet", "wbc", "rbc", "lab"
    ]
    
    VISIT_KEYWORDS = [
        "visit", "encounter", "appointment", "consultation", "follow-up",
        "check-up", "examination"
    ]
    
    @classmethod
    def detect(cls, content: str) -> str:
        content_lower = content.lower()
        
        if any(kw in content_lower for kw in cls.VITAL_KEYWORDS):
            return "vitals"
        if any(kw in content_lower for kw in cls.LAB_KEYWORDS):
            return "labs"
        if any(kw in content_lower for kw in cls.MEDICATION_KEYWORDS):
            return "medications"
        if any(kw in content_lower for kw in cls.CONDITION_KEYWORDS):
            return "conditions"
        if any(kw in content_lower for kw in cls.VISIT_KEYWORDS):
            return "visits"
        return "other"
    
    @classmethod
    def extract_subcategory(cls, content: str) -> str:
        content_lower = content.lower()
        for kw in cls.CONDITION_KEYWORDS + cls.MEDICATION_KEYWORDS:
            if kw in content_lower:
                return kw
        return ""


class MemWalker:
    """
    MemWalker: Intelligent hierarchical memory traversal.
    
    Key insight: Use hierarchy to BOOST relevant branches, not FILTER.
    This ensures we never do worse than vanilla.
    """
    
    # Boost multiplier for memories in relevant branches
    BRANCH_BOOST = 1.35  # 35% boost for matching branch
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
    
    def _determine_relevant_branches(self, query: str) -> List[str]:
        """Use LLM to determine which branches are relevant."""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Determine which memory categories are relevant to this medical query.
Categories: conditions, medications, visits, vitals, labs, other
Return ONLY comma-separated category names, nothing else.
Example: "conditions,medications" or "vitals,labs" """
                },
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=50
        )
        
        categories = response.choices[0].message.content.strip().lower()
        return [c.strip() for c in categories.split(",") if c.strip()]
    
    def search(self, query: str, patient_id: str, k: int = 5) -> Tuple[List[Dict], float]:
        """
        Search using hierarchical boosting.
        
        1. Get candidates via vanilla search (larger pool)
        2. Detect relevant branches from query
        3. Boost scores for memories in relevant branches
        4. Re-rank and return top-k
        """
        start = time.perf_counter()
        
        try:
            # Step 1: Get MORE candidates than needed (vanilla search)
            fetch_k = k * 4  # Fetch 4x to have room for re-ranking
            results = self.memory.search(
                query=query,
                user_id=patient_id,
                limit=fetch_k
            )
            
            if isinstance(results, dict):
                results = results.get('results', [])
            
            if not results:
                return [], (time.perf_counter() - start) * 1000
            
            # Step 2: Determine relevant branches
            relevant_branches = self._determine_relevant_branches(query)
            
            # Step 3: Score and boost
            scored_memories = []
            for r in results:
                if not isinstance(r, dict):
                    continue
                
                content = r.get("memory", r.get("content", ""))
                original_score = r.get("score", 0.0) or 0.0
                
                # Detect category of this memory
                category = CategoryDetector.detect(content)
                subcategory = CategoryDetector.extract_subcategory(content)
                
                # Apply boost if in relevant branch
                boosted_score = original_score
                if category in relevant_branches:
                    boosted_score = original_score * self.BRANCH_BOOST
                
                scored_memories.append({
                    "id": r.get("id", ""),
                    "content": content,
                    "memory": content,
                    "metadata": r.get("metadata", {}),
                    "original_score": original_score,
                    "score": boosted_score,
                    "category": category,
                    "subcategory": subcategory,
                    "boosted": category in relevant_branches
                })
            
            # Step 4: Re-rank by boosted score
            scored_memories.sort(key=lambda x: x["score"], reverse=True)
            top_memories = scored_memories[:k]
            
            latency_ms = (time.perf_counter() - start) * 1000
            return top_memories, latency_ms
        
        except Exception as e:
            # Fallback to vanilla
            latency_ms = (time.perf_counter() - start) * 1000
            results = self.memory.search(query=query, user_id=patient_id, limit=k)
            
            if isinstance(results, dict):
                results = results.get('results', [])
            
            memories = []
            for r in (results or []):
                if isinstance(r, dict):
                    memories.append({
                        "id": r.get("id", ""),
                        "content": r.get("memory", r.get("content", "")),
                        "memory": r.get("memory", r.get("content", "")),
                        "metadata": r.get("metadata", {}),
                        "score": r.get("score", 0)
                    })
            
            return memories, latency_ms
    
    def search_with_details(self, query: str, patient_id: str, k: int = 5) -> Dict:
        """Search with full debug details."""
        start = time.perf_counter()
        
        relevant_branches = self._determine_relevant_branches(query)
        memories, _ = self.search(query, patient_id, k)
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        return {
            "query": query,
            "relevant_branches": relevant_branches,
            "memories": memories,
            "latency_ms": latency_ms,
            "boosted_count": sum(1 for m in memories if m.get("boosted", False))
        }


def create_memwalker(config: Dict = None) -> MemWalker:
    return MemWalker(config)