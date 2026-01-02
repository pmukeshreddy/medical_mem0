"""
RAG-Fusion: Multi-Query Retrieval with Reciprocal Rank Fusion

Paper: "RAG-Fusion: a new take on Retrieval-Augmented Generation"

Key Idea:
- Single query might miss relevant docs due to vocabulary mismatch
- Generate multiple query perspectives
- Search all, merge with Reciprocal Rank Fusion (RRF)

Flow:
    Query → LLM (expand to 3-4 variants) → Search each → RRF merge → Top-K

No hardcoding - LLM generates variants, RRF is pure math.
"""

import os
import time
from typing import List, Dict, Tuple
from mem0 import Memory
from openai import OpenAI


class RAGFusion:
    """
    RAG-Fusion: Multi-query retrieval with intelligent merging.
    
    1. LLM expands query into multiple variants
    2. Search each variant
    3. Reciprocal Rank Fusion to merge results
    """
    
    name = "rag_fusion"
    description = "Multi-query expansion with Reciprocal Rank Fusion"
    
    # RRF constant (standard value from literature)
    RRF_K = 60
    
    def __init__(self, config: Dict = None, num_variants: int = 3):
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.num_variants = num_variants
    
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
    
    def _expand_query(self, query: str) -> List[str]:
        """Use LLM to generate query variants."""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""Generate {self.num_variants} different ways to ask the same question.
Each variant should:
- Capture the same intent
- Use different words/synonyms
- Be a complete question

Return ONLY the questions, one per line, no numbering."""
                },
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        variants = response.choices[0].message.content.strip().split("\n")
        # Clean and filter
        variants = [v.strip() for v in variants if v.strip()]
        return variants[:self.num_variants]
    
    def _reciprocal_rank_fusion(self, results_by_query: Dict[str, List[Dict]], k: int) -> List[Dict]:
        """
        Merge results using Reciprocal Rank Fusion.
        
        RRF_score(doc) = Σ 1 / (RRF_K + rank_i)
        
        Docs appearing in multiple queries get higher scores.
        """
        # Collect all unique docs with their ranks per query
        doc_scores = {}  # doc_id -> {"score": float, "doc": dict}
        
        for query, results in results_by_query.items():
            for rank, doc in enumerate(results):
                doc_id = doc.get("id") or doc.get("memory", "")[:50]  # Fallback to content
                
                rrf_score = 1.0 / (self.RRF_K + rank + 1)  # +1 because rank is 0-indexed
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "score": 0.0,
                        "doc": doc,
                        "appeared_in": 0
                    }
                
                doc_scores[doc_id]["score"] += rrf_score
                doc_scores[doc_id]["appeared_in"] += 1
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # Return top-k with enriched info
        results = []
        for item in sorted_docs[:k]:
            doc = item["doc"].copy()
            doc["rrf_score"] = item["score"]
            doc["appeared_in_queries"] = item["appeared_in"]
            results.append(doc)
        
        return results
    
    def search(self, query: str, patient_id: str, k: int = 5) -> Tuple[List[Dict], float]:
        """
        Search using RAG-Fusion.
        
        1. Expand query into variants
        2. Search each variant
        3. Merge with RRF
        """
        start = time.perf_counter()
        
        try:
            # Step 1: Generate query variants
            variants = self._expand_query(query)
            all_queries = [query] + variants  # Include original
            
            # Step 2: Search each query
            results_by_query = {}
            fetch_per_query = k * 2  # Fetch more for better fusion
            
            for q in all_queries:
                results = self.memory.search(
                    query=q,
                    user_id=patient_id,
                    limit=fetch_per_query
                )
                
                if isinstance(results, dict):
                    results = results.get('results', [])
                
                # Normalize format
                normalized = []
                for r in (results or []):
                    if isinstance(r, dict):
                        normalized.append({
                            "id": r.get("id", ""),
                            "content": r.get("memory", r.get("content", "")),
                            "memory": r.get("memory", r.get("content", "")),
                            "metadata": r.get("metadata", {}),
                            "score": r.get("score", 0)
                        })
                
                results_by_query[q] = normalized
            
            # Step 3: RRF merge
            fused_results = self._reciprocal_rank_fusion(results_by_query, k)
            
            latency_ms = (time.perf_counter() - start) * 1000
            return fused_results, latency_ms
        
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
        """Search with debug details."""
        start = time.perf_counter()
        
        variants = self._expand_query(query)
        results, _ = self.search(query, patient_id, k)
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        return {
            "original_query": query,
            "variants": variants,
            "results": results,
            "latency_ms": latency_ms
        }


def create_rag_fusion(config: Dict = None, num_variants: int = 3) -> RAGFusion:
    """Create a RAG-Fusion instance."""
    return RAGFusion(config, num_variants)
