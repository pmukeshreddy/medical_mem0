#!/usr/bin/env python3
"""
Debug script to test retrieval strategies interactively.

Usage:
    python debug_retrieval.py                    # Interactive mode
    python debug_retrieval.py --case 0           # Test specific case
    python debug_retrieval.py --query "..."      # Custom query
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))


def load_case(case_idx: int):
    """Load a specific test case."""
    gold_path = Path(__file__).parent / "eval" / "gold_dataset" / "cases.json"
    with open(gold_path) as f:
        cases = json.load(f)["cases"]
    return cases[case_idx]


def test_strategy(strategy_name: str, query: str, patient_id: str, k: int = 5):
    """Test a single strategy and show detailed results."""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")
    
    # Import based on strategy
    if strategy_name == "vanilla":
        from experiments import get_strategy
        strategy = get_strategy("vanilla")
    elif strategy_name == "colbert":
        from research.colbert_retrieval import ColBERTRetriever
        strategy = ColBERTRetriever()
    elif strategy_name == "hybrid_colbert":
        from research.colbert_retrieval import HybridColBERTBM25
        strategy = HybridColBERTBM25()
    elif strategy_name == "fast_medical":
        from research.advanced_retrieval import FastMedicalRetriever
        strategy = FastMedicalRetriever()
    elif strategy_name == "balanced":
        from research.advanced_retrieval import BalancedMedicalRetriever
        strategy = BalancedMedicalRetriever()
    elif strategy_name == "advanced":
        from research.advanced_retrieval import AdvancedMedicalRetriever
        strategy = AdvancedMedicalRetriever()
    else:
        print(f"Unknown strategy: {strategy_name}")
        return None, 0
    
    results, latency = strategy.search(query, patient_id, k)
    
    print(f"Latency: {latency:.1f}ms")
    print(f"Results: {len(results)}")
    print()
    
    for i, r in enumerate(results):
        content = r.get("content", r.get("memory", ""))
        scores = []
        if "final_score" in r:
            scores.append(f"final={r['final_score']:.3f}")
        if "colbert_score" in r:
            scores.append(f"colbert={r['colbert_score']:.3f}")
        if "rrf_score" in r:
            scores.append(f"rrf={r['rrf_score']:.3f}")
        
        score_str = " | ".join(scores) if scores else ""
        print(f"[{i+1}] {score_str}")
        print(f"    {content[:150]}...")
        print()
    
    return results, latency


def compute_recall(contents, keywords):
    """Compute keyword recall."""
    combined = " ".join(contents).lower()
    found = [k for k in keywords if k.lower() in combined]
    missing = [k for k in keywords if k.lower() not in combined]
    return found, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=0, help="Test case index")
    parser.add_argument("--query", type=str, help="Custom query")
    parser.add_argument("--patient", type=str, help="Patient ID for custom query")
    parser.add_argument("--strategies", nargs="+", 
                        default=["vanilla", "colbert", "fast_medical"],
                        help="Strategies to compare")
    args = parser.parse_args()
    
    # Load test case or use custom query
    if args.query:
        query = args.query
        patient_id = args.patient or "test-patient"
        keywords = []
        print(f"\nCustom Query: {query}")
    else:
        case = load_case(args.case)
        query = case["query"]
        patient_id = case["patient_id"]
        keywords = case["expected_keywords"]
        
        print(f"\n{'='*60}")
        print(f"TEST CASE: {case['id']}")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Patient: {patient_id}")
        print(f"Expected keywords: {keywords}")
        print(f"Source: {case.get('source_content', '')[:100]}...")
    
    # Test each strategy
    all_results = {}
    for strategy in args.strategies:
        try:
            results, latency = test_strategy(strategy, query, patient_id)
            all_results[strategy] = results
        except Exception as e:
            print(f"Error with {strategy}: {e}")
    
    # Compare recall if we have keywords
    if keywords and all_results:
        print(f"\n{'='*60}")
        print("RECALL COMPARISON")
        print(f"{'='*60}")
        
        for strategy, results in all_results.items():
            contents = [r.get("content", "") for r in (results or [])]
            found, missing = compute_recall(contents, keywords)
            recall = len(found) / len(keywords) if keywords else 0
            
            print(f"\n{strategy}:")
            print(f"  Recall@5: {recall:.1%} ({len(found)}/{len(keywords)})")
            print(f"  Found: {found[:5]}{'...' if len(found) > 5 else ''}")
            print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")


if __name__ == "__main__":
    main()
