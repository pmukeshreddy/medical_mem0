"""
Run benchmarks across experiments AND research strategies.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from experiments import get_strategy, STRATEGIES as EXP_STRATEGIES


def get_all_strategies():
    strategies = {}
    for name in EXP_STRATEGIES:
        strategies[name] = lambda n=name: get_strategy(n)
    
    # Research strategies
    try:
        from research.rag_fusion import RAGFusion
        strategies['rag_fusion'] = lambda: RAGFusion()
    except ImportError as e:
        print(f"Warning: Could not load RAGFusion: {e}")
    
    try:
        from research.advanced_retrieval import (
            AdvancedMedicalRetriever,
            FastMedicalRetriever,
            BalancedMedicalRetriever
        )
        strategies['advanced'] = lambda: AdvancedMedicalRetriever()
        strategies['fast_medical'] = lambda: FastMedicalRetriever()
        strategies['balanced'] = lambda: BalancedMedicalRetriever()
    except ImportError as e:
        print(f"Warning: Could not load advanced retrieval: {e}")
    
    try:
        from research.colbert_retrieval import ColBERTRetriever, HybridColBERTBM25
        strategies['colbert'] = lambda: ColBERTRetriever()
        strategies['hybrid_colbert'] = lambda: HybridColBERTBM25()
    except ImportError as e:
        print(f"Warning: Could not load ColBERT: {e}")
    
    return strategies


def load_gold_dataset():
    gold_path = Path(__file__).parent / "eval" / "gold_dataset" / "cases.json"
    if not gold_path.exists():
        print(f"ERROR: Gold dataset not found at {gold_path}")
        sys.exit(1)
    with open(gold_path) as f:
        return json.load(f)["cases"]


def compute_recall(retrieved_contents, expected_keywords, k=None):
    if not expected_keywords:
        return 1.0
    if k:
        retrieved_contents = retrieved_contents[:k]
    combined = " ".join(retrieved_contents).lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return found / len(expected_keywords)


def run_strategy(strategy_name, strategy_factory, cases, limit=None):
    print(f"\n  Running {strategy_name}...")
    try:
        strategy = strategy_factory()
    except Exception as e:
        print(f"    ERROR initializing {strategy_name}: {e}")
        return None
    
    results = []
    cases_to_run = cases[:limit] if limit else cases
    
    for i, case in enumerate(cases_to_run):
        try:
            memories, latency = strategy.search(
                query=case["query"],
                patient_id=case["patient_id"],
                k=5
            )
            if isinstance(memories, dict):
                memories = memories.get('results', [])
            
            contents = []
            for m in memories:
                if isinstance(m, dict):
                    contents.append(m.get("content", m.get("memory", "")))
                elif isinstance(m, str):
                    contents.append(m)
            
            recall_3 = compute_recall(contents, case["expected_keywords"], k=3)
            recall_5 = compute_recall(contents, case["expected_keywords"], k=5)
            
            results.append({
                "case_id": case["id"],
                "recall_3": recall_3,
                "recall_5": recall_5,
                "latency_ms": latency,
                "num_results": len(memories)
            })
        except Exception as e:
            print(f"    Error on {case['id']}: {e}")
            results.append({"case_id": case["id"], "error": str(e)})
        
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{len(cases_to_run)}")
    
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None
    
    return {
        "strategy": strategy_name,
        "num_cases": len(valid),
        "recall_3": sum(r["recall_3"] for r in valid) / len(valid),
        "recall_5": sum(r["recall_5"] for r in valid) / len(valid),
        "avg_latency_ms": sum(r["latency_ms"] for r in valid) / len(valid),
        "p95_latency_ms": sorted([r["latency_ms"] for r in valid])[int(len(valid) * 0.95)] if len(valid) > 1 else valid[0]["latency_ms"],
    }


def print_results_table(results):
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Strategy':<15} {'Recall@3':<10} {'Recall@5':<10} {'Avg Latency':<12} {'P95 Latency':<12}")
    print("-" * 80)
    
    sorted_results = sorted(results, key=lambda x: x["recall_5"], reverse=True)
    for r in sorted_results:
        print(f"{r['strategy']:<15} {r['recall_3']:<10.3f} {r['recall_5']:<10.3f} {r['avg_latency_ms']:<12.1f} {r['p95_latency_ms']:<12.1f}")
    print("-" * 80)
    
    best = sorted_results[0]
    baseline = next((r for r in sorted_results if r["strategy"] == "vanilla"), sorted_results[-1])
    improvement = ((best["recall_5"] - baseline["recall_5"]) / baseline["recall_5"]) * 100 if baseline["recall_5"] > 0 else 0
    print(f"\nBest: {best['strategy']} (Recall@5: {best['recall_5']:.3f})")
    if best['strategy'] != 'vanilla':
        print(f"Improvement over vanilla: +{improvement:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    
    all_strategies = get_all_strategies()
    
    if args.list:
        print("Experiments:", list(EXP_STRATEGIES.keys()))
        print("Research:", [k for k in all_strategies.keys() if k not in EXP_STRATEGIES])
        return
    
    print("=" * 80)
    print("MEDMEM0 BENCHMARKS")
    print("=" * 80)
    
    cases = load_gold_dataset()
    print(f"\nLoaded {len(cases)} cases")
    
    strategy_names = args.strategies or list(all_strategies.keys())
    print(f"Strategies: {strategy_names}")
    
    results = []
    for name in strategy_names:
        if name not in all_strategies:
            print(f"  WARNING: Unknown '{name}'")
            continue
        try:
            result = run_strategy(name, all_strategies[name], cases, args.limit)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {name} failed - {e}")
    
    if results:
        print_results_table(results)


if __name__ == "__main__":
    main()