"""
Evaluation Harness for MedMem0 Retrieval Benchmarks.

Runs gold dataset against all retrieval strategies and produces metrics.

Usage:
    python -m eval.harness                    # Run all
    python -m eval.harness --strategy vanilla # Single strategy
    python -m eval.harness --export results/  # Export results
"""

import json
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import asdict

from dotenv import load_dotenv
load_dotenv()

from .metrics import (
    evaluate_single, 
    aggregate_metrics, 
    format_metrics_table,
    format_detailed_report,
    AggregatedMetrics
)


# Path to gold dataset
GOLD_DATASET_PATH = Path(__file__).parent / "gold_dataset" / "cases.json"


def load_gold_dataset() -> List[Dict]:
    """Load gold evaluation cases."""
    with open(GOLD_DATASET_PATH) as f:
        data = json.load(f)
    return data["cases"]


def get_patient_ids(limit: int = None) -> List[str]:
    """Get patient IDs from processed data."""
    patients_file = Path(__file__).parent.parent / "data" / "processed" / "patients.jsonl"
    
    if not patients_file.exists():
        print(f"WARNING: {patients_file} not found")
        return []
    
    patient_ids = []
    with open(patients_file) as f:
        for line in f:
            patient = json.loads(line)
            patient_ids.append(patient["id"])
            if limit and len(patient_ids) >= limit:
                break
    
    return patient_ids


def init_strategies() -> Dict:
    """Initialize all retrieval strategies."""
    from mem0 import Memory
    
    # Mem0 config
    config = {
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
    
    # Import strategies
    from backend.core.memory_service import MemoryService
    
    # Import research strategies
    try:
        from research.memwalker import MemWalker
        from research.temporal_attention import TemporalAttentionRetriever
        from research.medical_entity_graph import MedicalEntityGraph
        has_research = True
    except ImportError:
        has_research = False
        print("WARNING: Research strategies not found")
    
    strategies = {
        "vanilla": ("Vanilla Dense", lambda q, p, k: run_vanilla(q, p, k, config)),
        "hybrid": ("Hybrid BM25", lambda q, p, k: run_hybrid(q, p, k, config)),
        "temporal": ("Temporal Decay", lambda q, p, k: run_temporal(q, p, k, config)),
        "entity": ("Entity Filter", lambda q, p, k: run_entity(q, p, k, config)),
    }
    
    if has_research:
        strategies["memwalker"] = ("MemWalker", lambda q, p, k: run_memwalker(q, p, k, config))
        strategies["temporal_attention"] = ("Temporal Attention", lambda q, p, k: run_temporal_attention(q, p, k, config))
        strategies["entity_graph"] = ("Entity Graph", lambda q, p, k: run_entity_graph(q, p, k, config))
    
    return strategies


# Strategy runners
_memory_cache = {}

def _get_memory(config):
    """Get cached memory instance."""
    from mem0 import Memory
    key = str(config)
    if key not in _memory_cache:
        _memory_cache[key] = Memory.from_config(config)
    return _memory_cache[key]


def run_vanilla(query: str, patient_id: str, k: int, config: Dict):
    """Run vanilla dense search."""
    memory = _get_memory(config)
    start = time.perf_counter()
    
    results = memory.search(query=query, user_id=patient_id, limit=k)
    
    latency = (time.perf_counter() - start) * 1000
    contents = [r.get("memory", r.get("content", "")) for r in (results or [])]
    
    return contents, latency


def run_hybrid(query: str, patient_id: str, k: int, config: Dict):
    """Run hybrid BM25 + dense."""
    from backend.core.memory_service import MemoryService
    
    # Use memory service which has hybrid built-in
    memory = _get_memory(config)
    start = time.perf_counter()
    
    results = memory.search(query=query, user_id=patient_id, limit=k * 2)
    
    if results:
        # Apply BM25 reranking
        from collections import Counter
        import math
        
        query_terms = query.lower().split()
        scored = []
        
        for i, doc in enumerate(results):
            content = doc.get("memory", doc.get("content", "")).lower()
            term_freq = Counter(content.split())
            
            bm25_score = sum(math.log(1 + term_freq.get(t, 0)) for t in query_terms)
            dense_score = 1.0 - (i / len(results))
            
            final = 0.7 * dense_score + 0.3 * (bm25_score / max(1, bm25_score))
            scored.append((final, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [d for _, d in scored[:k]]
    
    latency = (time.perf_counter() - start) * 1000
    contents = [r.get("memory", r.get("content", "")) for r in (results or [])]
    
    return contents, latency


def run_temporal(query: str, patient_id: str, k: int, config: Dict):
    """Run with temporal decay."""
    from datetime import datetime
    import math
    
    memory = _get_memory(config)
    start = time.perf_counter()
    
    results = memory.search(query=query, user_id=patient_id, limit=k * 2)
    
    if results:
        now = datetime.now()
        scored = []
        
        for i, doc in enumerate(results):
            base_score = 1.0 - (i / len(results))
            
            metadata = doc.get("metadata", {})
            date_str = metadata.get("date")
            
            temporal_boost = 1.0
            if date_str:
                try:
                    doc_date = datetime.fromisoformat(str(date_str).replace("Z", ""))
                    days_old = (now - doc_date).days
                    temporal_boost = math.pow(0.5, days_old / 365)
                except:
                    pass
            
            final = base_score * (1 + temporal_boost)
            scored.append((final, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [d for _, d in scored[:k]]
    
    latency = (time.perf_counter() - start) * 1000
    contents = [r.get("memory", r.get("content", "")) for r in (results or [])]
    
    return contents, latency


def run_entity(query: str, patient_id: str, k: int, config: Dict):
    """Run with entity filtering."""
    MEDICAL_TERMS = {
        "diabetes", "hypertension", "copd", "cardiac", "kidney",
        "metformin", "insulin", "lisinopril", "a1c", "glucose"
    }
    
    memory = _get_memory(config)
    start = time.perf_counter()
    
    # Extract entities from query
    query_lower = query.lower()
    entities = [t for t in MEDICAL_TERMS if t in query_lower]
    
    # Enhance query
    if entities:
        enhanced = f"{query} {' '.join(entities)}"
    else:
        enhanced = query
    
    results = memory.search(query=enhanced, user_id=patient_id, limit=k)
    
    latency = (time.perf_counter() - start) * 1000
    contents = [r.get("memory", r.get("content", "")) for r in (results or [])]
    
    return contents, latency


def run_memwalker(query: str, patient_id: str, k: int, config: Dict):
    """Run MemWalker hierarchical search."""
    from research.memwalker import MemWalker
    
    walker = MemWalker(config)
    results, latency = walker.search(query, patient_id, k)
    
    contents = [r.get("content", r.get("memory", "")) for r in results]
    return contents, latency


def run_temporal_attention(query: str, patient_id: str, k: int, config: Dict):
    """Run temporal attention search."""
    from research.temporal_attention import TemporalAttentionRetriever
    
    retriever = TemporalAttentionRetriever(config)
    results, latency = retriever.search(query, patient_id, k)
    
    contents = [r.get("content", r.get("memory", "")) for r in results]
    return contents, latency


def run_entity_graph(query: str, patient_id: str, k: int, config: Dict):
    """Run entity graph search."""
    from research.medical_entity_graph import MedicalEntityGraph
    
    graph = MedicalEntityGraph(config)
    results, latency, _ = graph.search(query, patient_id, k)
    
    contents = [r.get("content", r.get("memory", "")) for r in results]
    return contents, latency


class EvalHarness:
    """Main evaluation harness."""
    
    def __init__(self, strategies: Dict = None):
        """Initialize with strategies."""
        self.strategies = strategies or init_strategies()
        self.gold_cases = load_gold_dataset()
        self.patient_ids = get_patient_ids(limit=50)
        
        if not self.patient_ids:
            raise ValueError("No patient IDs found. Run seed_patients.py first.")
    
    def run_strategy(
        self, 
        strategy_key: str, 
        k: int = 5,
        max_cases: int = None
    ) -> AggregatedMetrics:
        """Run evaluation for a single strategy."""
        
        if strategy_key not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_key}")
        
        name, runner = self.strategies[strategy_key]
        print(f"\n  Running {name}...")
        
        results = []
        cases = self.gold_cases[:max_cases] if max_cases else self.gold_cases
        
        for i, case in enumerate(cases):
            # Pick a patient (round-robin)
            patient_id = self.patient_ids[i % len(self.patient_ids)]
            
            # Run retrieval
            try:
                contents, latency = runner(case["query"], patient_id, k)
            except Exception as e:
                print(f"    Error on case {case['id']}: {e}")
                continue
            
            # Evaluate
            metrics = evaluate_single(
                contents,
                case["expected_keywords"],
                latency
            )
            
            results.append({
                "case": case,
                "patient_id": patient_id,
                "metrics": metrics
            })
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i + 1}/{len(cases)}")
        
        # Aggregate
        return aggregate_metrics(results, name)
    
    def run_all(self, k: int = 5, max_cases: int = None) -> List[AggregatedMetrics]:
        """Run all strategies."""
        print("=== Running Evaluation Harness ===")
        print(f"  Cases: {len(self.gold_cases)}")
        print(f"  Patients: {len(self.patient_ids)}")
        print(f"  Strategies: {len(self.strategies)}")
        
        results = []
        for key in self.strategies:
            try:
                metrics = self.run_strategy(key, k, max_cases)
                if metrics:
                    results.append(metrics)
            except Exception as e:
                print(f"  ERROR on {key}: {e}")
        
        return results
    
    def export_results(
        self, 
        results: List[AggregatedMetrics], 
        output_dir: Path
    ):
        """Export results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON export
        json_file = output_dir / f"eval_results_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "results": [asdict(r) for r in results]
            }, f, indent=2)
        
        # Text report
        report_file = output_dir / f"eval_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write("=" * 72 + "\n")
            f.write("MEDMEM0 RETRIEVAL EVALUATION REPORT\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 72 + "\n\n")
            
            f.write("SUMMARY TABLE\n")
            f.write(format_metrics_table(results))
            f.write("\n\n")
            
            f.write("DETAILED RESULTS\n")
            f.write("-" * 72 + "\n")
            for r in results:
                f.write(format_detailed_report(r))
                f.write("\n\n")
        
        print(f"\nResults exported:")
        print(f"  {json_file}")
        print(f"  {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Run MedMem0 evaluation")
    parser.add_argument("--strategy", type=str, default=None, help="Single strategy to run")
    parser.add_argument("--k", type=int, default=5, help="Top-k results")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases")
    parser.add_argument("--export", type=str, default="results", help="Export directory")
    args = parser.parse_args()
    
    # Initialize
    harness = EvalHarness()
    
    # Run
    if args.strategy:
        results = [harness.run_strategy(args.strategy, args.k, args.max_cases)]
    else:
        results = harness.run_all(args.k, args.max_cases)
    
    # Print summary
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(format_metrics_table(results))
    
    # Export
    harness.export_results(results, Path(args.export))


if __name__ == "__main__":
    main()
