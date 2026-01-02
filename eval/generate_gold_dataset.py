"""
Gold Dataset Generator v5 - PROPER EVALUATION

Approach:
1. Load SOURCE memories (ground truth)
2. LLM generates HARD queries (no keyword overlap)
3. Keywords from SOURCE memory
4. Test if retrieval can find the source memory

No circular logic.
"""

import json
import random
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / "gold_dataset"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Keywords to extract from memories
KEYWORD_PATTERNS = [
    "blood pressure", "systolic", "diastolic",
    "heart rate", "respiratory", "temperature",
    "weight", "bmi", "height", "oxygen",
    "glucose", "hemoglobin", "cholesterol", "a1c",
    "diagnosed", "diabetes", "hypertension",
    "pain", "medication", "prescribed"
]


def load_source_memories() -> List[Dict]:
    """Load memories from source files."""
    memories = []
    
    # Try mem0_records.jsonl
    records_path = DATA_DIR / "mem0_records.jsonl"
    if records_path.exists():
        print(f"  Loading from: {records_path}")
        with open(records_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("content") and record.get("user_id"):
                        memories.append({
                            "id": record.get("id", ""),
                            "content": record["content"],
                            "patient_id": record["user_id"],
                            "metadata": record.get("metadata", {})
                        })
                except:
                    continue
    
    return memories


def extract_keywords(content: str) -> List[str]:
    """Extract keywords from memory content."""
    content_lower = content.lower()
    found = []
    for kw in KEYWORD_PATTERNS:
        if kw in content_lower and kw not in found:
            found.append(kw)
    return found


def generate_hard_query(memory_content: str, openai_client: OpenAI) -> str:
    """Use LLM to generate a hard query that doesn't use exact words."""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You generate medical questions for retrieval testing.

RULES:
1. Generate a question that a doctor would ask to find this information
2. DO NOT use the exact medical terms from the record
3. Use synonyms, related terms, or broader categories
4. Keep it natural and realistic

Examples:
- Record: "Systolic Blood Pressure: 120 mmHg" 
  BAD: "What is the systolic blood pressure?"  (uses exact terms)
  GOOD: "What are the cardiovascular readings?"

- Record: "Diagnosed with Type 2 Diabetes"
  BAD: "Does patient have diabetes?"  (uses exact term)
  GOOD: "Any metabolic or endocrine conditions?"

- Record: "Heart rate: 72 bpm"
  BAD: "What is heart rate?"
  GOOD: "What are the cardiac rhythm measurements?"
"""
            },
            {
                "role": "user", 
                "content": f"Generate a hard query for this record:\n{memory_content}"
            }
        ],
        temperature=0.7,
        max_tokens=50
    )
    
    return response.choices[0].message.content.strip().strip('"')


def determine_category(keywords: List[str]) -> str:
    """Determine category from keywords."""
    vitals = ["blood pressure", "systolic", "diastolic", "heart rate", "respiratory", "temperature", "weight", "bmi", "oxygen"]
    labs = ["glucose", "hemoglobin", "cholesterol", "a1c"]
    conditions = ["diagnosed", "diabetes", "hypertension"]
    
    if any(k in vitals for k in keywords):
        return "vitals"
    if any(k in labs for k in keywords):
        return "labs"
    if any(k in conditions for k in keywords):
        return "conditions"
    return "general"


def generate_gold_dataset(n_cases: int = 50) -> Dict:
    """Generate gold dataset with hard queries."""
    
    print("Loading source memories...")
    memories = load_source_memories()
    print(f"  Found {len(memories)} memories")
    
    if not memories:
        print("ERROR: No source memories found")
        return None
    
    # Filter memories that have extractable keywords
    valid_memories = []
    for m in memories:
        keywords = extract_keywords(m["content"])
        if keywords:  # Only keep memories with keywords
            m["keywords"] = keywords
            valid_memories.append(m)
    
    print(f"  Memories with keywords: {len(valid_memories)}")
    
    if len(valid_memories) < n_cases:
        print(f"WARNING: Only {len(valid_memories)} valid memories, reducing n_cases")
        n_cases = len(valid_memories)
    
    # Sample memories
    random.shuffle(valid_memories)
    selected = valid_memories[:n_cases]
    
    # Generate hard queries
    print(f"Generating {n_cases} hard queries with LLM...")
    openai_client = OpenAI()
    
    cases = []
    for i, memory in enumerate(selected):
        try:
            hard_query = generate_hard_query(memory["content"], openai_client)
            
            cases.append({
                "id": f"eval_{i:04d}",
                "patient_id": memory["patient_id"],
                "query": hard_query,
                "expected_keywords": memory["keywords"],
                "source_content": memory["content"][:200],
                "source_memory_id": memory["id"],
                "category": determine_category(memory["keywords"]),
                "difficulty": "hard"
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{n_cases}")
                
        except Exception as e:
            print(f"  Error on {i}: {e}")
            continue
    
    dataset = {
        "metadata": {
            "version": "5.0",
            "description": "Gold dataset v5 - LLM-generated hard queries, source-based keywords",
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
    
    print("=== Gold Dataset Generator v5 ===\n")
    
    dataset = generate_gold_dataset(args.n)
    
    if not dataset:
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = args.output or (OUTPUT_DIR / "cases.json")
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n=== Done ===")
    print(f"Output: {output_file}")
    
    # Show samples
    print("\n--- Sample Cases ---")
    for case in dataset["cases"][:3]:
        print(f"  Source: {case['source_content'][:60]}...")
        print(f"  Query:  {case['query']}")
        print(f"  Keywords: {case['expected_keywords']}")
        print()


if __name__ == "__main__":
    main()