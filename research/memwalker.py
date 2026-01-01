"""
MemWalker: Hierarchical Memory Traversal for Patient Data

Inspired by: "MemWalker: Interactive Memory System for LLMs" (2023)
Paper: https://arxiv.org/abs/2310.05029

Key Idea:
- Instead of flat vector search, organize memories into a tree
- Traverse tree intelligently based on query
- More relevant results for hierarchical data (medical records)
"""

import os
from typing import List, Dict, Any
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


@dataclass
class MemoryTree:
    """Hierarchical memory tree for a patient."""
    patient_id: str
    root: Dict[str, List[MemoryNode]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.root = {
            "conditions": [],
            "medications": [],
            "visits": [],
            "vitals": [],
            "labs": [],
            "other": []
        }


class HierarchicalPatientMemory:
    """Builds hierarchical tree from flat Mem0 memories."""
    
    CONDITION_KEYWORDS = [
        "diabetes", "hypertension", "copd", "asthma", "heart", "cardiac",
        "kidney", "renal", "liver", "hepatic", "cancer", "depression",
        "anxiety", "arthritis", "stroke", "pneumonia", "bronchitis",
        "sinusitis", "obesity", "hyperlipidemia", "anemia", "prediabetes"
    ]
    
    MEDICATION_KEYWORDS = [
        "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin",
        "omeprazole", "levothyroxine", "albuterol", "prednisone", "aspirin",
        "ibuprofen", "acetaminophen", "amoxicillin", "gabapentin", "sertraline"
    ]
    
    VITAL_KEYWORDS = [
        "blood pressure", "heart rate", "temperature", "respiratory rate",
        "oxygen saturation", "weight", "height", "bmi", "pulse"
    ]
    
    LAB_KEYWORDS = [
        "a1c", "glucose", "cholesterol", "ldl", "hdl", "triglycerides",
        "creatinine", "bun", "hemoglobin", "platelet", "wbc", "rbc"
    ]
    
    def __init__(self, memories: List[Dict]) -> None:
        self.memories = memories
        self.tree = None
    
    def build_tree(self, patient_id: str) -> MemoryTree:
        tree = MemoryTree(patient_id=patient_id)
        
        for mem in self.memories:
            try:
                node = self._create_node(mem)
                if node and node.content:
                    category = self._categorize(node.content)
                    tree.root[category].append(node)
            except:
                continue
        
        for category in tree.root:
            tree.root[category].sort(key=lambda x: x.date or "", reverse=True)
        
        self.tree = tree
        return tree
    
    def _create_node(self, memory) -> MemoryNode:
        if isinstance(memory, str):
            return MemoryNode(id="", content=memory, category="other")
        
        content = memory.get("memory", memory.get("content", ""))
        metadata = memory.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        
        return MemoryNode(
            id=memory.get("id", ""),
            content=content,
            category=metadata.get("type", "other"),
            subcategory=self._extract_subcategory(content),
            date=metadata.get("date", ""),
            metadata=metadata
        )
    
    def _categorize(self, content: str) -> str:
        content_lower = content.lower()
        
        if any(kw in content_lower for kw in self.VITAL_KEYWORDS):
            return "vitals"
        if any(kw in content_lower for kw in self.LAB_KEYWORDS):
            return "labs"
        if any(kw in content_lower for kw in self.MEDICATION_KEYWORDS):
            return "medications"
        if any(kw in content_lower for kw in self.CONDITION_KEYWORDS):
            return "conditions"
        if "visit" in content_lower or "encounter" in content_lower:
            return "visits"
        return "other"
    
    def _extract_subcategory(self, content: str) -> str:
        content_lower = content.lower()
        for kw in self.CONDITION_KEYWORDS + self.MEDICATION_KEYWORDS:
            if kw in content_lower:
                return kw
        return ""


class MemWalker:
    """MemWalker: Intelligent hierarchical memory traversal."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tree_cache: Dict[str, MemoryTree] = {}
    
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
    
    def _get_tree(self, patient_id: str) -> MemoryTree:
        """Get or build memory tree using search (not get_all which is broken)."""
        if patient_id not in self.tree_cache:
            all_memories = []
            seen_ids = set()
            
            # Search with broad terms to get diverse memories
            search_terms = ["patient", "diagnosed", "visit", "medication", "vital", "lab", "condition"]
            
            for term in search_terms:
                try:
                    results = self.memory.search(query=term, user_id=patient_id, limit=30)
                    
                    if isinstance(results, dict):
                        results = results.get('results', [])
                    
                    for r in (results or []):
                        if isinstance(r, dict):
                            mem_id = r.get('id', '')
                            if mem_id and mem_id not in seen_ids:
                                seen_ids.add(mem_id)
                                all_memories.append(r)
                except:
                    continue
            
            builder = HierarchicalPatientMemory(all_memories)
            self.tree_cache[patient_id] = builder.build_tree(patient_id)
        
        return self.tree_cache[patient_id]
    
    def _determine_relevant_branches(self, query: str) -> List[str]:
        """Use LLM to determine which branches to traverse."""
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
    
    def _score_nodes(self, nodes: List[MemoryNode], query: str) -> List[MemoryNode]:
        """Score nodes by relevance using embeddings."""
        if not nodes:
            return []
        
        query_resp = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = query_resp.data[0].embedding
        
        contents = [n.content for n in nodes if n.content]
        if not contents:
            return []
        
        content_resp = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=contents
        )
        
        import math
        for i, node in enumerate(nodes):
            if i < len(content_resp.data):
                node_embedding = content_resp.data[i].embedding
                dot = sum(x * y for x, y in zip(query_embedding, node_embedding))
                norm_a = math.sqrt(sum(x * x for x in query_embedding))
                norm_b = math.sqrt(sum(x * x for x in node_embedding))
                node.relevance_score = dot / (norm_a * norm_b) if norm_a and norm_b else 0
        
        return sorted(nodes, key=lambda x: x.relevance_score, reverse=True)
    
    def search(self, query: str, patient_id: str, k: int = 5) -> tuple[List[Dict], float]:
        """Search using hierarchical traversal."""
        start = time.perf_counter()
        
        try:
            tree = self._get_tree(patient_id)
            relevant_branches = self._determine_relevant_branches(query)
            
            candidate_nodes = []
            for branch in relevant_branches:
                if branch in tree.root:
                    candidate_nodes.extend(tree.root[branch])
            
            if not candidate_nodes:
                for branch_nodes in tree.root.values():
                    candidate_nodes.extend(branch_nodes)
            
            if candidate_nodes:
                scored_nodes = self._score_nodes(candidate_nodes, query)
                top_nodes = scored_nodes[:k]
            else:
                top_nodes = []
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            memories = []
            for node in top_nodes:
                memories.append({
                    "id": node.id,
                    "content": node.content,
                    "memory": node.content,
                    "metadata": node.metadata,
                    "score": node.relevance_score,
                    "category": node.category,
                    "subcategory": node.subcategory
                })
            
            return memories, latency_ms
        
        except Exception as e:
            # Fallback to regular search
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
    
    def invalidate_cache(self, patient_id: str = None):
        if patient_id:
            self.tree_cache.pop(patient_id, None)
        else:
            self.tree_cache.clear()
    
    def get_tree_stats(self, patient_id: str) -> Dict:
        tree = self._get_tree(patient_id)
        stats = {"patient_id": patient_id, "categories": {}}
        for category, nodes in tree.root.items():
            stats["categories"][category] = {
                "count": len(nodes),
                "subcategories": list(set(n.subcategory for n in nodes if n.subcategory))
            }
        stats["total_memories"] = sum(len(nodes) for nodes in tree.root.values())
        return stats


def create_memwalker(config: Dict = None) -> MemWalker:
    return MemWalker(config)