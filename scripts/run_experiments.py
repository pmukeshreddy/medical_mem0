"""
Run retrieval experiments comparing different strategies.

Compares:
1. Vanilla Dense (baseline)
2. Hybrid BM25 + Dense
3. With Cohere Reranker
4. Temporal Decay (boost recent)
5. Entity Filtered
6. Graph Augmented

Outputs metrics: Recall@K, MRR, Latency, Token usage

Usage:
    python run_experiments.py                     # Run all strategies
    python run_experiments.py --strategy vanilla  # Single strategy
    python run_experiments.py --output results/   # Custom output dir
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import statistics
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class EvalCase:
    """A single evaluation case with ground truth."""
    case_id: str
    patient_id: str
    query: str
    expected_memory_ids: List[str]  # Ground truth memory IDs
    expected_keywords: List[str]     # Keywords that should appear
    category: str                    # diabetes, cardiac, etc.


@dataclass 
class RetrievalResult:
    """Result from a single retrieval."""
    query: str
    retrieved_ids: List[str]
    retrieved_contents: List[str]
    latency_ms: float
    strategy: str


@dataclass
class StrategyMetrics:
    """Aggregated metrics for a strategy."""
    strategy: str
    recall_at_3: float
    recall_at_5: float
    mrr: float  # Mean Reciprocal Rank
    avg_latency_ms: float
    p95_latency_ms: float
    total_cases: int
    

# ============================================================
# RETRIEVAL STRATEGIES
# ============================================================

class RetrievalStrategy(ABC):
    """Base class for retrieval strategies."""
    
    name: str = "base"
    
    @abstractmethod
    def search(self, query: str, patient_id: str, k: int = 5) -> List[Dict]:
        """Search for relevant memories."""
        pass


class VanillaDense(RetrievalStrategy):
    """Baseline dense vector search."""
    
    name = "vanilla_dense"
    
    def __init__(self, memory):
        self.memory = memory
    
    def search(self, query: str, patient_id: str, k: int = 5) -> List[Dict]:
        results = self.memory.search(
            query=query,
            user_id=patient_id,
            limit=k
        )
        return results


class HybridBM25(RetrievalStrategy):
    """Hybrid BM25 + Dense fusion."""
    
    name = "hybrid_bm25"
    
    def __init__(self, memory, bm25_weight: float = 0.3):
        self.memory = memory
        self.bm25_weight = bm25_weight
        self.bm25_index = {}  # patient_id -> BM25 index
    
    def _get_bm25_scores(self, query: str, patient_id: str, docs: List[Dict]) -> Dict[str, float]:
        """Simple BM25 scoring (simplified implementation)."""
        from collections import Counter
        import math
        
        query_terms = query.lower().split()
        scores = {}
        
        for doc in docs:
            doc_id = doc.get("id", "")
            content = doc.get("memory", doc.get("content", "")).lower()
            doc_terms = content.split()
            term_freq = Counter(doc_terms)
            
            score = 0
            for term in query_terms:
                tf = term_freq.get(term, 0)
                if tf > 0:
                    # Simplified BM25
                    score += math.log(1 + tf)
            
            scores[doc_id] = score
        
        return scores
    
    def search(self, query: str, patient_id: str, k: int = 5) -> List[Dict]:
        # Get dense results (fetch more for reranking)
        dense_results = self.memory.search(
            query=query,
            user_id=patient_id,
            limit=k * 2
        )
        
        if not dense_results:
            return []
        
        # Get BM25 scores
        bm25_scores = self._get_bm25_scores(query, patient_id, dense_results)
        
        # Normalize and combine scores
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        
        combined = []
        for i, doc in enumerate(dense_results):
            doc_id = doc.get("id", "")
            dense_score = 1.0 - (i / len(dense_results))  # Position-based
            bm25_score = bm25_scores.get(doc_id, 0) / max_bm25 if max_bm25 > 0 else 0
            
            final_score = (1 - self.bm25_weight) * dense_score + self.bm25_weight * bm25_score
            combined.append((final_score, doc))
        
        # Sort by combined score
        combined.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in combined[:k]]


class WithReranker(RetrievalStrategy):
    """Dense + Cohere Reranker."""
    
    name = "with_reranker"
    
    def __init__(self, memory):
        self.memory = memory
        self.cohere_key = os.getenv("COHERE_API_KEY")
        self.cohere_client = None
        
        if self.cohere_key:
            try:
                import cohere
                self.cohere_client = cohere.Client(self.cohere_key)
            except ImportError:
                print("WARNING: cohere not installed. Run: pip install cohere")
    
    def search(self, query: str, patient_id: str, k: int = 5) -> List[Dict]:
        # Get more candidates for reranking
        candidates = self.memory.search(
            query=query,
            user_id=patient_id,
            limit=k * 3
        )
        
        if not candidates or not self.cohere_client:
            return candidates[:k] if candidates else []
        
        # Rerank with Cohere
        try:
            docs = [c.get("memory", c.get("content", "")) for c in candidates]
            rerank_response = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs,
                top_n=k
            )
            
            reranked = []
            for r in rerank_response.results:
                reranked.append(candidates[r.index])
            
            return reranked
            
        except Exception as e:
            print(f"Rerank error: {e}")
            return candidates[:k]


class TemporalDecay(RetrievalStrategy):
    """Boost recent visits with temporal decay."""
    
    name = "temporal_decay"
    
    def __init__(self, memory, decay_factor: float = 0.1):
        self.memory = memory
        self.decay_factor = decay_factor
    
    def search(self, query: str, patient_id: str, k: int = 5) -> List[Dict]:
        results = self.memory.search(
            query=query,
            user_id=patient_id,
            limit=k * 2
        )
        
        if not results:
            return []
        
        # Apply temporal decay
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
                    doc_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    days_old = (now - doc_date.replace(tzinfo=None)).days
                    # Exponential decay
                    temporal_boost = 2.0 ** (-self.decay_factor * days_old / 365)
                except:
                    pass
            
            final_score = base_score * (1 + temporal_boost)
            scored.append((final_score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]


class EntityFiltered(RetrievalStrategy):
    """Pre-filter by medical entities before search."""
    
    name = "entity_filtered"
    
    # Common medical entities for filtering
    CONDITION_KEYWORDS = {
        "diabetes", "hypertension", "copd", "asthma", "cardiac", "heart",
        "kidney", "renal", "liver", "hepatic", "cancer", "tumor",
        "depression", "anxiety", "arthritis", "stroke", "pneumonia"
    }
    
    MEDICATION_KEYWORDS = {
        "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin",
        "omeprazole", "levothyroxine", "albuterol", "prednisone", "aspirin"
    }
    
    def __init__(self, memory):
        self.memory = memory
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract medical entities from query."""
        query_lower = query.lower()
        entities = []
        
        for kw in self.CONDITION_KEYWORDS | self.MEDICATION_KEYWORDS:
            if kw in query_lower:
                entities.append(kw)
        
        return entities
    
    def search(self, query: str, patient_id: str, k: int = 5) -> List[Dict]:
        entities = self._extract_entities(query)
        
        # If entities found, enhance query
        if entities:
            enhanced_query = f"{query} {' '.join(entities)}"
        else:
            enhanced_query = query
        
        results = self.memory.search(
            query=enhanced_query,
            user_id=patient_id,
            limit=k
        )
        
        return results


# ============================================================
# EVALUATION HARNESS
# ============================================================

class EvalHarness:
    """Run experiments and compute metrics."""
    
    def __init__(self, strategies: List[RetrievalStrategy], eval_cases: List[EvalCase]):
        self.strategies = {s.name: s for s in strategies}
        self.eval_cases = eval_cases
    
    def _compute_recall_at_k(self, retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
        """Compute Recall@K."""
        if not expected_ids:
            return 1.0
        
        retrieved_set = set(retrieved_ids[:k])
        expected_set = set(expected_ids)
        
        hits = len(retrieved_set & expected_set)
        return hits / len(expected_set)
    
    def _compute_mrr(self, retrieved_ids: List[str], expected_ids: List[str]) -> float:
        """Compute Mean Reciprocal Rank."""
        if not expected_ids:
            return 1.0
        
        for i, rid in enumerate(retrieved_ids):
            if rid in expected_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _keyword_recall(self, retrieved_contents: List[str], expected_keywords: List[str]) -> float:
        """Compute keyword-based recall (fallback when IDs not available)."""
        if not expected_keywords:
            return 1.0
        
        combined_content = " ".join(retrieved_contents).lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in combined_content)
        
        return hits / len(expected_keywords)
    
    def run_strategy(self, strategy_name: str) -> StrategyMetrics:
        """Run evaluation for a single strategy."""
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        recall_3_scores = []
        recall_5_scores = []
        mrr_scores = []
        latencies = []
        
        print(f"\n  Running {strategy_name}...")
        
        for case in self.eval_cases:
            # Time the search
            start = time.perf_counter()
            results = strategy.search(case.query, case.patient_id, k=5)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            
            # Extract IDs and contents
            retrieved_ids = [r.get("id", "") for r in results]
            retrieved_contents = [r.get("memory", r.get("content", "")) for r in results]
            
            # Compute metrics
            if case.expected_memory_ids:
                recall_3 = self._compute_recall_at_k(retrieved_ids, case.expected_memory_ids, 3)
                recall_5 = self._compute_recall_at_k(retrieved_ids, case.expected_memory_ids, 5)
                mrr = self._compute_mrr(retrieved_ids, case.expected_memory_ids)
            else:
                # Fallback to keyword matching
                recall_3 = self._keyword_recall(retrieved_contents[:3], case.expected_keywords)
                recall_5 = self._keyword_recall(retrieved_contents[:5], case.expected_keywords)
                mrr = recall_3  # Approximate
            
            recall_3_scores.append(recall_3)
            recall_5_scores.append(recall_5)
            mrr_scores.append(mrr)
        
        # Aggregate
        return StrategyMetrics(
            strategy=strategy_name,
            recall_at_3=statistics.mean(recall_3_scores) if recall_3_scores else 0,
            recall_at_5=statistics.mean(recall_5_scores) if recall_5_scores else 0,
            mrr=statistics.mean(mrr_scores) if mrr_scores else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            total_cases=len(self.eval_cases)
        )
    
    def run_all(self) -> List[StrategyMetrics]:
        """Run all strategies and return metrics."""
        results = []
        for name in self.strategies:
            metrics = self.run_strategy(name)
            results.append(metrics)
        return results


# ============================================================
# SAMPLE EVAL CASES GENERATOR
# ============================================================

def generate_sample_eval_cases(patients_file: Path, n_cases: int = 50) -> List[EvalCase]:
    """Generate sample eval cases from patient data."""
    
    if not patients_file.exists():
        print(f"WARNING: {patients_file} not found, using synthetic cases")
        return generate_synthetic_eval_cases(n_cases)
    
    cases = []
    patient_ids = []
    
    # Load patient IDs
    with open(patients_file) as f:
        for line in f:
            patient = json.loads(line)
            patient_ids.append(patient["id"])
            if len(patient_ids) >= n_cases:
                break
    
    # Generate queries for each patient
    query_templates = [
        ("What medications is the patient currently taking?", ["medication", "taking", "prescribed"]),
        ("Has the patient had any cardiac issues?", ["cardiac", "heart", "cardiovascular"]),
        ("What are the patient's active conditions?", ["condition", "diagnosis", "disorder"]),
        ("When was the patient's last visit?", ["visit", "encounter", "appointment"]),
        ("Does the patient have diabetes?", ["diabetes", "diabetic", "glucose", "A1c"]),
    ]
    
    for i, pid in enumerate(patient_ids):
        query, keywords = query_templates[i % len(query_templates)]
        cases.append(EvalCase(
            case_id=f"case_{i}",
            patient_id=pid,
            query=query,
            expected_memory_ids=[],  # Would need manual labeling
            expected_keywords=keywords,
            category="general"
        ))
    
    return cases


def generate_synthetic_eval_cases(n_cases: int = 50) -> List[EvalCase]:
    """Generate synthetic eval cases for testing."""
    import uuid
    
    cases = []
    queries = [
        ("diabetes management history", ["diabetes", "A1c", "glucose", "metformin"]),
        ("cardiac events and treatments", ["cardiac", "heart", "ecg", "cardiology"]),
        ("current medication list", ["medication", "prescription", "drug"]),
        ("recent lab results", ["lab", "blood", "test", "results"]),
        ("chronic conditions", ["chronic", "condition", "diagnosis"]),
    ]
    
    for i in range(n_cases):
        query, keywords = queries[i % len(queries)]
        cases.append(EvalCase(
            case_id=f"synth_{i}",
            patient_id=str(uuid.uuid4()),
            query=query,
            expected_memory_ids=[],
            expected_keywords=keywords,
            category=["diabetes", "cardiac", "medication", "labs", "chronic"][i % 5]
        ))
    
    return cases


# ============================================================
# OUTPUT
# ============================================================

def print_results(results: List[StrategyMetrics]):
    """Print results as table."""
    print("\n" + "=" * 80)
    print("RETRIEVAL EXPERIMENT RESULTS")
    print("=" * 80)
    
    # Header
    print(f"\n{'Strategy':<20} {'Recall@3':<10} {'Recall@5':<10} {'MRR':<10} {'Latency(ms)':<12} {'P95(ms)':<10}")
    print("-" * 80)
    
    # Sort by Recall@5
    results_sorted = sorted(results, key=lambda x: x.recall_at_5, reverse=True)
    
    for r in results_sorted:
        print(f"{r.strategy:<20} {r.recall_at_3:<10.3f} {r.recall_at_5:<10.3f} {r.mrr:<10.3f} {r.avg_latency_ms:<12.1f} {r.p95_latency_ms:<10.1f}")
    
    print("-" * 80)
    print(f"Total eval cases: {results[0].total_cases if results else 0}")


def save_results(results: List[StrategyMetrics], output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results]
    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run retrieval experiments")
    parser.add_argument("--strategy", type=str, default=None, help="Run single strategy")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--n-cases", type=int, default=50, help="Number of eval cases")
    parser.add_argument("--dry-run", action="store_true", help="Test without Mem0")
    args = parser.parse_args()
    
    print("=== MedMem0 Retrieval Experiments ===\n")
    
    DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
    OUTPUT_DIR = Path(__file__).parent.parent / args.output
    
    # Generate eval cases
    print("Loading eval cases...")
    eval_cases = generate_sample_eval_cases(DATA_DIR / "patients.jsonl", args.n_cases)
    print(f"  Generated {len(eval_cases)} eval cases")
    
    if args.dry_run:
        print("\n[DRY RUN] Would run experiments with these cases:")
        for case in eval_cases[:3]:
            print(f"  - {case.query}")
        return
    
    # Initialize Mem0 with Pinecone
    print("\nInitializing Mem0 with Pinecone...")
    try:
        from mem0 import Memory
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not pinecone_api_key or not openai_api_key:
            print("ERROR: Set PINECONE_API_KEY and OPENAI_API_KEY in .env")
            return
        
        memory = Memory.from_config({
            "vector_store": {
                "provider": "pinecone",
                "config": {
                    "api_key": pinecone_api_key,
                    "index_name": os.getenv("PINECONE_INDEX_NAME", "medmem0"),
                    "embedding_model_dims": 1536,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": openai_api_key,
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": openai_api_key,
                }
            }
        })
    except Exception as e:
        print(f"ERROR initializing Mem0: {e}")
        print("Run seed_patients.py first or check .env")
        return
    
    # Initialize strategies
    print("Initializing strategies...")
    strategies = [
        VanillaDense(memory),
        HybridBM25(memory),
        TemporalDecay(memory),
        EntityFiltered(memory),
    ]
    
    # Add reranker if available
    if os.getenv("COHERE_API_KEY"):
        strategies.append(WithReranker(memory))
        print("  + WithReranker (Cohere)")
    else:
        print("  - WithReranker skipped (no COHERE_API_KEY)")
    
    # Filter to single strategy if specified
    if args.strategy:
        strategies = [s for s in strategies if s.name == args.strategy]
        if not strategies:
            print(f"ERROR: Unknown strategy '{args.strategy}'")
            return
    
    print(f"  Running {len(strategies)} strategies")
    
    # Run experiments
    harness = EvalHarness(strategies, eval_cases)
    
    if args.strategy:
        results = [harness.run_strategy(args.strategy)]
    else:
        results = harness.run_all()
    
    # Output
    print_results(results)
    save_results(results, OUTPUT_DIR)
    
    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
