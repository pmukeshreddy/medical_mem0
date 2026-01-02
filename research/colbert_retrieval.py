"""
ColBERT-Style Late Interaction Retrieval

Approximates ColBERT's MaxSim operation without requiring the full model.
Uses token-level matching with medical term weighting.

Paper: https://arxiv.org/abs/2004.12832
"""

import os
import time
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from mem0 import Memory


# Medical term categories with importance weights
TERM_WEIGHTS = {
    # Critical clinical values (weight 3.0)
    "critical": {
        "systolic", "diastolic", "glucose", "hemoglobin", "creatinine",
        "potassium", "sodium", "hba1c", "a1c", "egfr", "gfr", "inr",
        "troponin", "bnp", "lactate", "ph"
    },
    # Important measurements (weight 2.0)
    "measurements": {
        "blood", "pressure", "heart", "rate", "pulse", "respiratory",
        "temperature", "weight", "height", "bmi", "oxygen", "saturation",
        "mmhg", "mg/dl", "mg", "ml", "units", "bpm"
    },
    # Conditions/Diagnoses (weight 2.5)
    "conditions": {
        "diabetes", "hypertension", "ckd", "chronic", "acute", "disorder",
        "disease", "syndrome", "failure", "insufficiency", "infection",
        "cancer", "malignant", "benign", "stenosis", "anemia"
    },
    # Medications (weight 2.0)
    "medications": {
        "medication", "drug", "tablet", "capsule", "injection", "dose",
        "prescription", "metformin", "lisinopril", "aspirin", "insulin",
        "statin", "antibiotic", "analgesic"
    },
    # Body parts (weight 1.5)
    "anatomy": {
        "kidney", "renal", "heart", "cardiac", "lung", "pulmonary",
        "liver", "hepatic", "brain", "neural", "oral", "dental"
    }
}

# Flatten for quick lookup
WEIGHTED_TERMS = {}
for category, terms in TERM_WEIGHTS.items():
    weight = {"critical": 3.0, "measurements": 2.0, "conditions": 2.5, 
              "medications": 2.0, "anatomy": 1.5}[category]
    for term in terms:
        WEIGHTED_TERMS[term] = weight


class ColBERTApproximator:
    """
    Approximates ColBERT late interaction without the model.
    
    ColBERT computes: sum over query tokens of max(similarity to each doc token)
    We approximate with: weighted term matching + fuzzy matching
    """
    
    def __init__(self):
        self.term_weights = WEIGHTED_TERMS
        self.stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                          "have", "has", "had", "do", "does", "did", "will", "would",
                          "could", "should", "may", "might", "must", "shall", "can",
                          "to", "of", "in", "for", "on", "with", "at", "by", "from",
                          "as", "into", "through", "during", "before", "after",
                          "above", "below", "between", "under", "again", "further",
                          "then", "once", "here", "there", "when", "where", "why",
                          "how", "all", "each", "few", "more", "most", "other", "some",
                          "such", "no", "nor", "not", "only", "own", "same", "so",
                          "than", "too", "very", "just", "and", "but", "or", "if",
                          "because", "until", "while", "what", "which", "who", "whom",
                          "this", "that", "these", "those", "am", "any", "both"}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text."""
        # Lowercase and split
        text = text.lower()
        # Keep alphanumeric and some medical chars
        tokens = re.findall(r'[a-z0-9]+(?:\.[0-9]+)?', text)
        # Filter stopwords
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]
    
    def compute_maxsim(self, query_tokens: List[str], doc_tokens: Set[str]) -> float:
        """
        Compute MaxSim-style score.
        For each query token, find max similarity to any doc token.
        """
        if not query_tokens or not doc_tokens:
            return 0.0
        
        total_score = 0.0
        
        for q_token in query_tokens:
            # Get weight for this query token
            q_weight = self.term_weights.get(q_token, 1.0)
            
            # Find best match in document
            best_match = 0.0
            
            # Exact match
            if q_token in doc_tokens:
                best_match = 1.0
            else:
                # Fuzzy matching for medical terms
                for d_token in doc_tokens:
                    # Prefix match (e.g., "glucose" matches "glucos")
                    if d_token.startswith(q_token[:4]) or q_token.startswith(d_token[:4]):
                        best_match = max(best_match, 0.7)
                    # Substring match
                    elif q_token in d_token or d_token in q_token:
                        best_match = max(best_match, 0.5)
            
            total_score += best_match * q_weight
        
        # Normalize by query length
        return total_score / len(query_tokens)
    
    def score(self, query: str, document: str) -> float:
        """Score query-document pair."""
        query_tokens = self.tokenize(query)
        doc_tokens = set(self.tokenize(document))
        return self.compute_maxsim(query_tokens, doc_tokens)


class ColBERTRetriever:
    """
    Two-stage retrieval with ColBERT-style reranking.
    
    Stage 1: Dense retrieval (Mem0)
    Stage 2: Late interaction reranking
    """
    
    name = "colbert_approx"
    description = "ColBERT-style late interaction retrieval"
    
    def __init__(self, config: Dict = None, rerank_depth: int = 20):
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.scorer = ColBERTApproximator()
        self.rerank_depth = rerank_depth
    
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
                    "dense_score": r.get("score", 0)
                })
        return normalized
    
    def search(self, query: str, patient_id: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Two-stage retrieval with ColBERT reranking."""
        start = time.perf_counter()
        
        # Stage 1: Dense retrieval
        results = self.memory.search(
            query=query,
            user_id=patient_id,
            limit=self.rerank_depth
        )
        candidates = self._normalize_results(results)
        
        if not candidates:
            return [], (time.perf_counter() - start) * 1000
        
        # Stage 2: ColBERT-style reranking
        for doc in candidates:
            doc["colbert_score"] = self.scorer.score(query, doc.get("content", ""))
        
        # Combined score (weighted)
        max_dense = max((d.get("dense_score", 0) or 0.001) for d in candidates)
        for doc in candidates:
            dense_norm = (doc.get("dense_score", 0) or 0) / max_dense
            colbert = doc.get("colbert_score", 0)
            # ColBERT gets more weight for recall improvement
            doc["final_score"] = dense_norm * 0.4 + colbert * 0.6
        
        # Sort by final score
        candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        latency_ms = (time.perf_counter() - start) * 1000
        return candidates[:k], latency_ms


class HybridColBERTBM25:
    """
    Hybrid retrieval combining:
    - Dense vectors (semantic)
    - ColBERT late interaction (token-level)
    - BM25 (lexical)
    
    Uses Reciprocal Rank Fusion for combination.
    """
    
    name = "hybrid_colbert_bm25"
    description = "Hybrid dense + ColBERT + BM25"
    
    RRF_K = 60
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.colbert = ColBERTApproximator()
    
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
    
    def _bm25_score(self, query: str, doc: str) -> float:
        """Simple BM25-like scoring."""
        query_terms = set(query.lower().split())
        doc_terms = doc.lower().split()
        doc_term_set = set(doc_terms)
        
        if not query_terms:
            return 0.0
        
        matches = query_terms & doc_term_set
        
        # TF component
        tf_sum = sum(doc_terms.count(t) for t in matches)
        
        # IDF approximation (rare terms score higher)
        idf_sum = sum(1.0 / (1 + doc_terms.count(t)) for t in matches)
        
        return (tf_sum * 0.5 + idf_sum * 0.5) / len(query_terms)
    
    def search(self, query: str, patient_id: str, k: int = 5) -> Tuple[List[Dict], float]:
        """Hybrid search with three-way fusion."""
        start = time.perf_counter()
        
        # Get candidates
        results = self.memory.search(query=query, user_id=patient_id, limit=k * 4)
        candidates = self._normalize_results(results)
        
        if not candidates:
            return [], (time.perf_counter() - start) * 1000
        
        # Score each candidate with all methods
        for doc in candidates:
            content = doc.get("content", "")
            doc["colbert_score"] = self.colbert.score(query, content)
            doc["bm25_score"] = self._bm25_score(query, content)
        
        # Create rankings for each method
        dense_ranked = sorted(candidates, key=lambda x: x.get("score", 0) or 0, reverse=True)
        colbert_ranked = sorted(candidates, key=lambda x: x.get("colbert_score", 0), reverse=True)
        bm25_ranked = sorted(candidates, key=lambda x: x.get("bm25_score", 0), reverse=True)
        
        # RRF fusion
        doc_scores = defaultdict(lambda: {"score": 0.0, "doc": None})
        
        for rankings in [dense_ranked, colbert_ranked, bm25_ranked]:
            for rank, doc in enumerate(rankings):
                doc_id = doc.get("id") or id(doc)
                doc_scores[doc_id]["score"] += 1.0 / (self.RRF_K + rank + 1)
                doc_scores[doc_id]["doc"] = doc
        
        # Sort by fused score
        fused = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        final_results = [item["doc"] for item in fused[:k]]
        
        for i, doc in enumerate(final_results):
            doc["rrf_score"] = fused[i]["score"]
        
        latency_ms = (time.perf_counter() - start) * 1000
        return final_results, latency_ms


__all__ = ["ColBERTRetriever", "ColBERTApproximator", "HybridColBERTBM25"]
