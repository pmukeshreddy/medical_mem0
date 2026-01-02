"""
Advanced Medical Retrieval - Research Implementations

Based on recent papers:
- Query2Doc (Microsoft, 2023): Generate pseudo-document for better matching
- Step-Back Prompting (Google, 2023): Abstract queries for broader recall
- ColBERT Late Interaction (Stanford): Token-level similarity
- Self-RAG (2023): Self-reflective retrieval verification
- Medical Entity Linking: Domain-aware query expansion
- Adaptive Retrieval: Iterative refinement

Target: Improve Recall@5 from ~10% to 40%+ while keeping latency <3s
"""

import os
import time
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from mem0 import Memory
from openai import OpenAI


class LLMMedicalExpander:
    """
    LLM-based medical term expansion.
    Dynamically generates synonyms, abbreviations, and related clinical terms.
    """
    
    def __init__(self, openai_client: OpenAI, cache_enabled: bool = True):
        self.client = openai_client
        self.cache = {} if cache_enabled else None
    
    def expand(self, query: str) -> List[str]:
        """Extract and expand medical terms from query using LLM."""
        # Check cache
        if self.cache is not None and query in self.cache:
            return self.cache[query]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """Extract medical concepts from the query and provide clinical synonyms/abbreviations.

Return ONLY a comma-separated list of terms that would appear in clinical notes.
Focus on: abbreviations, measurement units, clinical terminology, ICD/SNOMED-style terms.

Examples:
- "blood pressure" → systolic, diastolic, BP, mmHg, hypertension, HTN
- "kidney function" → renal, creatinine, eGFR, GFR, CKD, BUN, nephro
- "heart problems" → cardiac, cardiovascular, CHF, arrhythmia, EKG, ECG

Keep it to 8-10 most relevant terms. No explanations."""
            }, {
                "role": "user",
                "content": query
            }],
            temperature=0.3,
            max_tokens=100
        )
        
        result = response.choices[0].message.content.strip()
        terms = [t.strip() for t in result.split(",") if t.strip()][:10]
        
        # Cache result
        if self.cache is not None:
            self.cache[query] = terms
        
        return terms


class Query2Doc:
    """
    Query2Doc: Query Expansion with LLM-Generated Pseudo-Documents
    Paper: https://arxiv.org/abs/2303.07678
    
    Instead of expanding queries, generate a hypothetical document
    that would answer the query, then use it for retrieval.
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def generate(self, query: str) -> str:
        """Generate pseudo-document that answers the query."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """You are a medical records system. Generate a realistic clinical note 
that would answer this query. Include specific values, units, dates, and medical terminology.
Keep it to 3-4 sentences. Write as actual clinical documentation, not as an explanation."""
            }, {
                "role": "user", 
                "content": query
            }],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()


class StepBackRetrieval:
    """
    Step-Back Prompting for better recall
    Paper: https://arxiv.org/abs/2310.06117
    
    Generate more abstract/general queries to capture broader context,
    then combine with original query results.
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def get_stepback_query(self, query: str) -> str:
        """Generate a more general/abstract version of the query."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """Given a specific medical query, generate a more general/abstract version 
that would help find relevant background information.
Return ONLY the abstracted query, nothing else.

Examples:
- "What is the patient's current blood pressure?" → "patient vital signs measurements"
- "Has the patient been diagnosed with diabetes?" → "patient metabolic conditions and lab results"
- "What medications is the patient taking for pain?" → "patient medications and treatments" """
            }, {
                "role": "user",
                "content": query
            }],
            temperature=0.2,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()


class MedicalEntityExpander:
    """
    LLM-based medical entity expansion.
    Wrapper around LLMMedicalExpander for backward compatibility.
    """
    
    def __init__(self, openai_client: OpenAI = None):
        client = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._expander = LLMMedicalExpander(client)
    
    def expand(self, query: str) -> List[str]:
        return self._expander.expand(query)


class AdaptiveRetriever:
    """
    Adaptive/Iterative Retrieval
    
    1. Initial retrieval with original query
    2. Analyze results to identify gaps
    3. Generate targeted follow-up queries
    4. Merge results with deduplication
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def analyze_and_expand(self, query: str, initial_results: List[str]) -> Optional[str]:
        """Analyze initial results and generate targeted follow-up if needed."""
        if not initial_results:
            return None
        
        results_text = "\n".join(initial_results[:3])
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """Analyze if the retrieved medical records answer the query.
If important information is missing, generate ONE specific follow-up search query.
If the results are sufficient, respond with "SUFFICIENT".
Return ONLY the follow-up query or "SUFFICIENT"."""
            }, {
                "role": "user",
                "content": f"Query: {query}\n\nRetrieved:\n{results_text}"
            }],
            temperature=0.2,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        return None if "SUFFICIENT" in result.upper() else result


class SelfRAGVerifier:
    """
    Self-RAG: Learning to Retrieve, Generate, and Critique
    Paper: https://arxiv.org/abs/2310.11511
    
    Verify retrieved documents are actually relevant before returning.
    Filter out false positives.
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def verify_relevance(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Score and filter documents by relevance."""
        if not documents:
            return []
        
        # Batch verify for efficiency
        doc_texts = [f"[{i}] {d.get('content', '')[:200]}" for i, d in enumerate(documents[:10])]
        docs_str = "\n".join(doc_texts)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """Rate each document's relevance to the query on a scale of 0-2:
0 = Not relevant
1 = Partially relevant
2 = Highly relevant

Return comma-separated scores in order. Example: 2,1,0,2,1"""
            }, {
                "role": "user",
                "content": f"Query: {query}\n\nDocuments:\n{docs_str}"
            }],
            temperature=0,
            max_tokens=50
        )
        
        try:
            scores = [int(s.strip()) for s in response.choices[0].message.content.split(",")]
            # Filter and sort by relevance
            scored_docs = list(zip(documents[:len(scores)], scores))
            relevant = [(d, s) for d, s in scored_docs if s > 0]
            relevant.sort(key=lambda x: x[1], reverse=True)
            return [d for d, _ in relevant]
        except:
            return documents


class LateInteractionScorer:
    """
    ColBERT-inspired late interaction scoring.
    
    Instead of single vector comparison, compare token-level embeddings
    for finer-grained matching. Approximated here without full ColBERT.
    """
    
    def __init__(self):
        # Token importance weights (medical domain)
        self.important_tokens = {
            "blood", "pressure", "glucose", "heart", "rate", "pain",
            "medication", "mg", "ml", "tablet", "injection", "diagnosis",
            "systolic", "diastolic", "hemoglobin", "cholesterol", "bmi",
            "weight", "height", "respiratory", "temperature", "chronic",
            "acute", "disorder", "finding", "observation"
        }
    
    def score(self, query: str, document: str) -> float:
        """Compute late-interaction style score."""
        query_tokens = set(query.lower().split())
        doc_tokens = set(document.lower().split())
        
        # Exact matches
        exact_matches = query_tokens & doc_tokens
        
        # Important token matches (higher weight)
        important_matches = exact_matches & self.important_tokens
        
        # Compute score
        if not query_tokens:
            return 0.0
        
        base_score = len(exact_matches) / len(query_tokens)
        importance_bonus = len(important_matches) * 0.2
        
        return min(1.0, base_score + importance_bonus)


class AdvancedMedicalRetriever:
    """
    Combined advanced retrieval pipeline.
    
    Pipeline:
    1. Medical entity expansion
    2. Query2Doc pseudo-document generation
    3. Step-back abstract query
    4. Multi-query retrieval with RRF
    5. Late interaction reranking
    6. Self-RAG verification
    7. Optional adaptive follow-up
    """
    
    name = "advanced_medical"
    description = "Research-backed medical retrieval (Query2Doc + StepBack + SelfRAG)"
    
    RRF_K = 60
    
    def __init__(
        self,
        config: Dict = None,
        use_query2doc: bool = True,
        use_stepback: bool = True,
        use_verification: bool = True,
        use_adaptive: bool = False,  # Adds latency
        fast_mode: bool = False  # Skip expensive operations
    ):
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.use_query2doc = use_query2doc
        self.use_stepback = use_stepback
        self.use_verification = use_verification
        self.use_adaptive = use_adaptive
        self.fast_mode = fast_mode
        
        # Initialize components
        self.entity_expander = MedicalEntityExpander(self.openai)
        self.query2doc = Query2Doc(self.openai) if use_query2doc else None
        self.stepback = StepBackRetrieval(self.openai) if use_stepback else None
        self.verifier = SelfRAGVerifier(self.openai) if use_verification else None
        self.adaptive = AdaptiveRetriever(self.openai) if use_adaptive else None
        self.late_scorer = LateInteractionScorer()
    
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
    
    def _normalize_results(self, results) -> List[Dict]:
        """Normalize Mem0 results."""
        if isinstance(results, dict):
            results = results.get('results', [])
        
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
        return normalized
    
    def _rrf_merge(self, results_list: List[List[Dict]]) -> List[Dict]:
        """Reciprocal Rank Fusion merge."""
        doc_scores = {}
        
        for results in results_list:
            for rank, doc in enumerate(results):
                doc_id = doc.get("id") or hash(doc.get("content", "")[:100])
                rrf_score = 1.0 / (self.RRF_K + rank + 1)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"score": 0.0, "doc": doc, "count": 0}
                
                doc_scores[doc_id]["score"] += rrf_score
                doc_scores[doc_id]["count"] += 1
        
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        return [
            {**item["doc"], "rrf_score": item["score"], "query_count": item["count"]}
            for item in sorted_docs
        ]
    
    def _build_queries(self, query: str) -> Dict[str, str]:
        """Build all query variants."""
        queries = {"original": query}
        
        # Entity expansion - build enhanced query
        expansions = self.entity_expander.expand(query)
        if expansions:
            queries["expanded"] = f"{query} {' '.join(expansions[:5])}"
        
        # Query2Doc - parallel if not fast mode
        if self.query2doc and not self.fast_mode:
            queries["query2doc"] = self.query2doc.generate(query)
        
        # Step-back - parallel if not fast mode
        if self.stepback and not self.fast_mode:
            queries["stepback"] = self.stepback.get_stepback_query(query)
        
        return queries
    
    def search(self, query: str, patient_id: str, k: int = 5) -> Tuple[List[Dict], float]:
        """
        Advanced multi-stage retrieval.
        """
        start = time.perf_counter()
        
        try:
            # Stage 1: Build query variants
            queries = self._build_queries(query)
            
            # Stage 2: Retrieve from all queries
            all_results = []
            fetch_per_query = k * 2
            
            for q_name, q_text in queries.items():
                results = self.memory.search(
                    query=q_text,
                    user_id=patient_id,
                    limit=fetch_per_query
                )
                normalized = self._normalize_results(results)
                all_results.append(normalized)
            
            # Stage 3: RRF merge
            merged = self._rrf_merge(all_results)
            
            # Stage 4: Late interaction reranking
            for doc in merged:
                doc["late_score"] = self.late_scorer.score(query, doc.get("content", ""))
            
            # Combined scoring
            for doc in merged:
                doc["final_score"] = doc.get("rrf_score", 0) * 0.7 + doc.get("late_score", 0) * 0.3
            
            merged.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            
            # Stage 5: Self-RAG verification (optional)
            if self.verifier and not self.fast_mode:
                merged = self.verifier.verify_relevance(query, merged[:k*2])
            
            # Stage 6: Adaptive follow-up (optional)
            if self.adaptive and len(merged) < k:
                contents = [d.get("content", "") for d in merged]
                followup = self.adaptive.analyze_and_expand(query, contents)
                if followup:
                    extra_results = self.memory.search(
                        query=followup, user_id=patient_id, limit=k
                    )
                    extra_normalized = self._normalize_results(extra_results)
                    # Add new results
                    existing_ids = {d.get("id") for d in merged}
                    for doc in extra_normalized:
                        if doc.get("id") not in existing_ids:
                            merged.append(doc)
            
            latency_ms = (time.perf_counter() - start) * 1000
            return merged[:k], latency_ms
            
        except Exception as e:
            print(f"Error in AdvancedRetriever: {e}")
            latency_ms = (time.perf_counter() - start) * 1000
            # Fallback to vanilla
            results = self.memory.search(query=query, user_id=patient_id, limit=k)
            return self._normalize_results(results), latency_ms
    
    def search_with_details(self, query: str, patient_id: str, k: int = 5) -> Dict:
        """Search with full debug information."""
        start = time.perf_counter()
        
        queries = self._build_queries(query)
        results, search_latency = self.search(query, patient_id, k)
        
        return {
            "original_query": query,
            "query_variants": queries,
            "results": results,
            "num_results": len(results),
            "latency_ms": (time.perf_counter() - start) * 1000
        }


class FastMedicalRetriever(AdvancedMedicalRetriever):
    """
    Optimized version for lower latency.
    Uses only entity expansion and late interaction (no LLM calls in retrieval).
    """
    
    name = "fast_medical"
    description = "Fast medical retrieval (entity expansion + late interaction only)"
    
    def __init__(self, config: Dict = None):
        super().__init__(
            config=config,
            use_query2doc=False,
            use_stepback=False,
            use_verification=False,
            use_adaptive=False,
            fast_mode=True
        )


class BalancedMedicalRetriever(AdvancedMedicalRetriever):
    """
    Balanced version - Query2Doc + StepBack without verification.
    Good tradeoff between recall and latency.
    """
    
    name = "balanced_medical"
    description = "Balanced medical retrieval (Query2Doc + StepBack)"
    
    def __init__(self, config: Dict = None):
        super().__init__(
            config=config,
            use_query2doc=True,
            use_stepback=True,
            use_verification=False,
            use_adaptive=False,
            fast_mode=False
        )


# Export for strategy registry
__all__ = [
    "AdvancedMedicalRetriever",
    "FastMedicalRetriever", 
    "BalancedMedicalRetriever",
    "LLMMedicalExpander",
    "Query2Doc",
    "StepBackRetrieval",
    "MedicalEntityExpander",
    "SelfRAGVerifier",
    "LateInteractionScorer"
]
