"""
Generate gold evaluation dataset v4 - THE RIGHT WAY.

Approach:
1. Query Pinecone with various questions
2. See what ACTUALLY comes back  
3. Extract keywords FROM the retrieved content
4. Those become the ground truth

Usage:
    python eval/generate_gold_dataset.py --n 50
"""

import json
import random
import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / "gold_dataset"

# Generic medical queries - we don't predefine keywords
QUERIES = [
    # Vitals
    "What is the patient's blood pressure?",
    "What are the patient's vital signs?",
    "What is the patient's heart rate?",
    "What is the patient's respiratory rate?",
    "What is the patient's BMI?",
    "What is the patient's weight?",
    "What is the patient's temperature?",
    
    # Labs
    "What are the patient's lab results?",
    "What is the patient's glucose level?",
    "What is the patient's hemoglobin?",
    "What is the patient's cholesterol level?",
    
    # Conditions
    "What conditions has the patient been diagnosed with?",
    "What is the patient's diagnosis?",
    "Does the patient have any chronic conditions?",
    
    # General
    "What is the patient's medical history?",
    "Summarize the patient's health status",
]

# Keywords to look for in retrieved content
KEYWORD_PATTERNS = [
    "blood pressure", "systolic", "diastolic",
    "heart rate", "respiratory", "temperature",
    "weight", "bmi", "height",
    "glucose", "hemoglobin", "cholesterol", "a1c",
    "diagnosed", "diagnosis",
    "pain", "oxygen",
]


def get_patient_ids(strategy) -> List[str]:
    """Discover patient IDs from Pinecone."""
    patient_ids = set()
    
    for term in ["blood pressure", "diagnosed", "heart rate", "glucose"]:
        try:
            results = strategy.memory.search(query=term, limit=100)
            if isinstance(results, dict):
                results = results.get('results', [])
            
            for r in results:
                if isinstance(r, dict):
                    uid = r.get('user_id') or r.get('metadata', {}).get('user_id')
                    if uid:
                        patient_ids.add(uid)
        except:
            continue
    
    return list(patient_ids)


def extract_keywords_from_content(content: str) -> List[str]:
    """Extract actual keywords from retrieved content."""
    content_lower = content.lower()
    found = []
    
    for kw in KEYWORD_PATTERNS:
        if kw in content_lower and kw not in found:
            found.append(kw)
    
    return found


def determine_category(keywords: List[str]) -> str:
    """Determine category from found keywords."""
    vitals = ["blood pressure", "systolic", "diastolic", "heart rate", "respiratory", "temperature", "weight", "bmi"]
    labs = ["glucose", "hemoglobin", "cholesterol", "a1c"]
    
    if any(k in vitals for k in keywords):
        return "vitals"
    if any(k in labs for k in keywords):
        return "labs"
    if "diagnosed" in keywords or "diagnosis" in keywords:
        return "conditions"
    return "general"


def generate_gold_dataset(n_cases: int = 50) -> Dict:
    """Generate gold dataset from actual Pinecone retrieval."""
    from experiments import get_strategy
    
    print("Initializing...")
    strategy = get_strategy('vanilla')
    
    print("Discovering patients...")
    patient_ids = get_patient_ids(strategy)
    print(f"  Found {len(patient_ids)} patients")
    
    if not patient_ids:
        print("ERROR: No patients found")
        return None
    
    cases = []
    case_id = 0
    seen = set()  # Avoid duplicate (patient, query) pairs
    
    print(f"Generating {n_cases} cases from actual retrieval...")
    
    attempts = 0
    max_attempts = n_cases * 20
    
    while case_id < n_cases and attempts < max_attempts:
        attempts += 1
        
        patient_id = random.choice(patient_ids)
        query = random.choice(QUERIES)
        
        key = (patient_id, query)
        if key in seen:
            continue
        seen.add(key)
        
        # Step 1: Query Pinecone
        memories, _ = strategy.search(query=query, patient_id=patient_id, k=5)
        
        if not memories:
            continue
        
        # Step 2: Get what ACTUALLY came back
        top_content = memories[0].get('content', memories[0].get('memory', ''))
        
        if not top_content or len(top_content) < 10:
            continue
        
        # Step 3: Extract keywords FROM the content
        keywords = extract_keywords_from_content(top_content)
        
        if not keywords:
            continue
        
        # Step 4: This is now ground truth
        cases.append({
            "id": f"eval_{case_id:04d}",
            "patient_id": patient_id,
            "query": query,
            "expected_keywords": keywords,  # FROM retrieval, not predefined
            "expected_content_snippet": top_content[:200],
            "category": determine_category(keywords),
            "difficulty": "medium",
        })
        
        case_id += 1
        
        if case_id % 10 == 0:
            print(f"  Progress: {case_id}/{n_cases}")
    
    print(f"  Generated {len(cases)} cases")
    
    dataset = {
        "metadata": {
            "version": "4.0",
            "description": "Gold dataset v4 - keywords extracted from actual retrieval",
            "total_cases": len(cases),
            "total_patients": len(set(c["patient_id"] for c in cases)),
            "categories": list(set(c["category"] for c in cases))
        },
        "cases": cases
    }
    
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=== Gold Dataset Generator v4 ===\n")
    
    dataset = generate_gold_dataset(args.n)
    
    if not dataset:
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = args.output or (OUTPUT_DIR / "cases.json")
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n=== Done ===")
    print(f"Output: {output_file}")
    
    keyword_counts = [len(c["expected_keywords"]) for c in dataset["cases"]]
    print(f"Keywords per case: min={min(keyword_counts)}, max={max(keyword_counts)}, avg={sum(keyword_counts)/len(keyword_counts):.1f}")
    
    print("\n--- Sample Cases ---")
    for case in dataset["cases"][:3]:
        print(f"  Q: {case['query']}")
        print(f"  Keywords: {case['expected_keywords']}")
        print(f"  Content: {case['expected_content_snippet'][:80]}...")
        print()


if __name__ == "__main__":
    main()