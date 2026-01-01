"""
Generate gold dataset from ACTUAL memory content.

This creates evaluation cases where keywords ACTUALLY exist in the data,
giving realistic recall numbers.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent / "gold_dataset"


def load_memories():
    """Load memories from JSONL."""
    memories = []
    mem_file = DATA_DIR / "mem0_records.jsonl"
    
    if not mem_file.exists():
        print(f"ERROR: {mem_file} not found")
        return []
    
    with open(mem_file) as f:
        for line in f:
            memories.append(json.loads(line))
    
    return memories


def extract_meaningful_keywords(content: str) -> list:
    """Extract meaningful medical keywords from content."""
    
    # Medical terms to look for
    medical_terms = [
        # Conditions
        "diabetes", "prediabetes", "hypertension", "asthma", "copd", "anemia",
        "obesity", "depression", "anxiety", "arthritis", "stroke", "cardiac",
        "kidney", "renal", "liver", "cancer", "pneumonia", "bronchitis",
        
        # Medications
        "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin",
        "omeprazole", "albuterol", "prednisone", "aspirin", "ibuprofen",
        
        # Vitals/Labs
        "glucose", "cholesterol", "hemoglobin", "creatinine", "pressure",
        "weight", "bmi", "temperature", "pulse", "oxygen",
        
        # Actions
        "diagnosed", "prescribed", "visit", "encounter", "immunization",
        "vaccination", "screening", "assessment"
    ]
    
    content_lower = content.lower()
    found = []
    
    for term in medical_terms:
        if term in content_lower:
            found.append(term)
    
    return found


def create_query_for_keywords(keywords: list, content: str) -> str:
    """Create natural query that should retrieve this content."""
    
    if not keywords:
        return None
    
    keyword = keywords[0]
    
    # Query templates
    templates = [
        f"What is the patient's {keyword} status?",
        f"Does the patient have {keyword}?",
        f"Tell me about {keyword}",
        f"Any {keyword} in the records?",
        f"Show {keyword} history",
        f"Patient {keyword} information",
    ]
    
    # Specific templates for certain keywords
    if keyword in ["glucose", "cholesterol", "hemoglobin", "creatinine"]:
        templates = [
            f"What is the patient's {keyword} level?",
            f"Show {keyword} results",
            f"Any {keyword} tests?",
        ]
    elif keyword in ["diagnosed", "visit", "encounter"]:
        templates = [
            "What was the patient diagnosed with?",
            "Show recent visits",
            "Patient encounter history",
        ]
    elif keyword in ["weight", "bmi", "pressure"]:
        templates = [
            f"What is the patient's {keyword}?",
            f"Show {keyword} readings",
            "Patient vital signs",
        ]
    
    return random.choice(templates)


def determine_difficulty(keywords: list, content: str) -> str:
    """Determine query difficulty."""
    if len(keywords) >= 3:
        return "easy"
    elif len(keywords) == 2:
        return "medium"
    else:
        return "hard"


def generate_gold_dataset(n_cases: int = 50):
    """Generate gold dataset from actual data."""
    
    print("Loading memories...")
    memories = load_memories()
    
    if not memories:
        return None
    
    print(f"  Total memories: {len(memories)}")
    
    # Group by patient
    by_patient = defaultdict(list)
    for m in memories:
        by_patient[m["user_id"]].append(m)
    
    print(f"  Patients: {len(by_patient)}")
    
    # Generate cases
    cases = []
    patients = list(by_patient.keys())
    random.shuffle(patients)
    
    for patient_id in patients:
        if len(cases) >= n_cases:
            break
        
        mems = by_patient[patient_id]
        random.shuffle(mems)
        
        for mem in mems[:3]:  # Max 3 per patient
            if len(cases) >= n_cases:
                break
            
            content = mem["content"]
            keywords = extract_meaningful_keywords(content)
            
            if not keywords:
                continue
            
            query = create_query_for_keywords(keywords, content)
            if not query:
                continue
            
            cases.append({
                "id": f"eval_{len(cases):04d}",
                "patient_id": patient_id,
                "query": query,
                "expected_keywords": keywords[:3],  # Top 3 keywords
                "expected_content_snippet": content[:150],
                "category": keywords[0],
                "difficulty": determine_difficulty(keywords, content),
            })
    
    # Build dataset
    dataset = {
        "metadata": {
            "version": "2.0",
            "description": "Gold dataset generated from actual memory content",
            "total_cases": len(cases),
            "patients": len(set(c["patient_id"] for c in cases)),
        },
        "cases": cases
    }
    
    return dataset


def main():
    print("=== Better Gold Dataset Generator ===\n")
    
    dataset = generate_gold_dataset(n_cases=50)
    
    if not dataset:
        print("Failed to generate dataset")
        return
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "cases.json"
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n=== Done ===")
    print(f"Generated: {len(dataset['cases'])} cases")
    print(f"Output: {output_file}")
    
    # Show samples
    print("\n--- Sample Cases ---")
    for case in dataset["cases"][:3]:
        print(f"\nQuery: {case['query']}")
        print(f"Keywords: {case['expected_keywords']}")
        print(f"Content: {case['expected_content_snippet'][:80]}...")


if __name__ == "__main__":
    main()
