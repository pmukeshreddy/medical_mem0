"""
Run benchmarks across experiments AND research strategies.

Usage:
    python run_benchmarks.py
    python run_benchmarks.py --strategies vanilla hybrid hyde
    python run_benchmarks.py --limit 10
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experiments import get_strategy, STRATEGIES as EXP_STRATEGIES


def get_all_strategies():
    """Get both experiment and research strategies."""
    strategies = {}
    
    # Experiment strategies
    for name in EXP_STRATEGIES:
        strategies[name] = lambda n=name: get_strategy(n)
    
    # Research strategies
    try:
        from research.hyde_retrieval import HyDERetrieval
        strategies['hyde'] = lambda: HyDERetrieval()
    except ImportError as e:
        print(f"Warning: Could not load HyDE: {e}")
    
    try:
        from research.memwalker import MemWalker
        strategies['memwalker'] = lambda: MemWalker()
    except ImportError as e:
        print(f"Warning: Could not load MemWalker: {e}")
    
    try:
        from research.temporal_attention import TemporalAttentionRetriever
        strategies['temporal_attn'] = lambda: TemporalAttentionRetriever()
    except ImportError as e:
        print(f"Warning: Could not load TemporalAttention: {e}")
    
    return strategies


def load_gold_dataset():
    """Load gold evaluation cases."""
    gold_path = Path(__file__).parent / "eval" / "gold_dataset" / "cases.json"
    
    if not gold_path.exists():
        print(f"ERROR: Gold dataset not found at {gold_path}")
        print("Run: python eval/generate_gold_dataset.py --n 50")
        sys.exit(1)
    
    with open(gold_path) as f:
        data = json.load(f)
    
    return data["cases"]


def compute_recall(retrieved_contents, expected_keywords, k=None):
    """Compute keyword recall."""
    if not expected_keywords:
        return 1.0
    
    if k:
        retrieved_contents = retrieved_contents[:k]
    
    combined = " ".join(retrieved_contents).lower()
    
    found = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return found / len(expected_keywords)


def run_strategy(strategy_name, strategy_factory, cases, limit=None):
    """Run a single strategy across all cases."""
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
            
            # Handle Mem0 response format {'results': [...]}
            if isinstance(memories, dict):
                memories = memories.get('results', [])
            
            # Extract content
            contents = []
            for m in memories:
                if isinstance(m, dict):
                    contents.append(m.get("content", m.get("memory", "")))
                elif isinstance(m, str):
                    contents.append(m)
            
            # Compute recall
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
            results.append({
                "case_id": case["id"],
                "recall_3": 0,
                "recall_5": 0,
                "latency_ms": 0,
                "num_results": 0,
                "error": str(e)
            })
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{len(cases_to_run)}")
    
    # Aggregate
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
    """Print results as ASCII table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    header = f"{'Strategy':<15} {'Recall@3':<10} {'Recall@5':<10} {'Avg Latency':<12} {'P95 Latency':<12}"
    print(header)
    print("-" * 80)
    
    # Sort by recall@5
    sorted_results = sorted(results, key=lambda x: x["recall_5"], reverse=True)
    
    for r in sorted_results:
        row = f"{r['strategy']:<15} {r['recall_3']:<10.3f} {r['recall_5']:<10.3f} {r['avg_latency_ms']:<12.1f} {r['p95_latency_ms']:<12.1f}"
        print(row)
    
    print("-" * 80)
    
    # Winner
    best = sorted_results[0]
    baseline = next((r for r in sorted_results if r["strategy"] == "vanilla"), sorted_results[-1])
    
    improvement = ((best["recall_5"] - baseline["recall_5"]) / baseline["recall_5"]) * 100 if baseline["recall_5"] > 0 else 0
    
    print(f"\n🏆 Best: {best['strategy']} (Recall@5: {best['recall_5']:.3f})")
    if best['strategy'] != 'vanilla':
        print(f"📈 Improvement over vanilla: +{improvement:.1f}%")


def save_results(results, output_dir="results"):
    """Save results to JSON."""
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"benchmark_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run retrieval benchmarks")
    parser.add_argument("--strategies", nargs="+", default=None, help="Strategies to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit cases")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--list", action="store_true", help="List available strategies")
    args = parser.parse_args()
    
    all_strategies = get_all_strategies()
    
    if args.list:
        print("Available strategies:")
        print("  Experiments:", list(EXP_STRATEGIES.keys()))
        print("  Research:", [k for k in all_strategies.keys() if k not in EXP_STRATEGIES])
        return
    
    print("=" * 80)
    print("MEDMEM0 RETRIEVAL BENCHMARKS (Experiments + Research)")
    print("=" * 80)
    
    # Load gold dataset
    cases = load_gold_dataset()
    print(f"\nLoaded {len(cases)} evaluation cases")
    
    # Determine strategies
    if args.strategies:
        strategy_names = args.strategies
    else:
        # Default: all
        strategy_names = list(all_strategies.keys())
    
    print(f"Strategies: {strategy_names}")
    
    # Run benchmarks
    results = []
    for name in strategy_names:
        if name not in all_strategies:
            print(f"  WARNING: Unknown strategy '{name}', skipping")
            continue
        
        try:
            result = run_strategy(name, all_strategies[name], cases, args.limit)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {name} failed - {e}")
    
    # Print results
    if results:
        print_results_table(results)
        
        if args.save:
            save_results(results)
    else:
        print("\nNo results to show.")


if __name__ == "__main__":
    main()
