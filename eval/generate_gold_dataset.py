"""
Generate gold evaluation dataset from actual patient data.

Reads your patients.jsonl and mem0_records.jsonl,
creates evaluation cases with REAL data as ground truth.

Usage:
    python generate_gold_dataset.py
    python generate_gold_dataset.py --n 100  # 100 cases
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent / "gold_dataset"


def load_patients() -> List[Dict]:
    """Load patients from JSONL."""
    patients = []
    patients_file = DATA_DIR / "patients.jsonl"
    
    if not patients_file.exists():
        print(f"ERROR: {patients_file} not found")
        return []
    
    with open(patients_file) as f:
        for line in f:
            patients.append(json.loads(line))
    
    return patients


def load_memories() -> Dict[str, List[Dict]]:
    """Load memories grouped by patient_id."""
    memories_by_patient = defaultdict(list)
    memories_file = DATA_DIR / "mem0_records.jsonl"
    
    if not memories_file.exists():
        print(f"ERROR: {memories_file} not found")
        return {}
    
    with open(memories_file) as f:
        for line in f:
            record = json.loads(line)
            patient_id = record.get("user_id")
            if patient_id:
                memories_by_patient[patient_id].append(record)
    
    return dict(memories_by_patient)


def extract_keywords_from_memory(memory: Dict) -> List[str]:
    """Extract searchable keywords from a memory."""
    content = memory.get("content", "").lower()
    metadata = memory.get("metadata", {})
    
    keywords = []
    
    # Extract condition keywords
    condition_terms = [
        "diabetes", "hypertension", "copd", "asthma", "cardiac", "heart",
        "kidney", "renal", "liver", "cancer", "depression", "anxiety",
        "obesity", "stroke", "pneumonia", "bronchitis", "sinusitis",
        "arthritis", "anemia", "hypothyroid", "hyperlipidemia"
    ]
    for term in condition_terms:
        if term in content:
            keywords.append(term)
    
    # Extract medication keywords
    med_terms = [
        "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin",
        "omeprazole", "levothyroxine", "albuterol", "prednisone", "aspirin",
        "ibuprofen", "gabapentin", "sertraline", "metoprolol", "furosemide"
    ]
    for term in med_terms:
        if term in content:
            keywords.append(term)
    
    # Extract vital/lab keywords
    vital_terms = [
        "blood pressure", "heart rate", "temperature", "weight", "bmi",
        "glucose", "a1c", "cholesterol", "creatinine", "hemoglobin"
    ]
    for term in vital_terms:
        if term in content:
            keywords.append(term)
    
    # Extract from metadata
    if metadata.get("type"):
        keywords.append(metadata["type"])
    
    return list(set(keywords))


def generate_query_for_memory(memory: Dict, keywords: List[str]) -> str:
    """Generate a natural query that should retrieve this memory."""
    content = memory.get("content", "")
    metadata = memory.get("metadata", {})
    
    templates = []
    
    # Condition-based queries
    conditions = ["diabetes", "hypertension", "cardiac", "kidney", "copd", "asthma"]
    for cond in conditions:
        if cond in keywords:
            templates.extend([
                f"Does the patient have {cond}?",
                f"What is the patient's {cond} history?",
                f"Any {cond} related visits?",
            ])
    
    # Medication queries
    meds = ["metformin", "insulin", "lisinopril", "atorvastatin"]
    for med in meds:
        if med in keywords:
            templates.extend([
                f"Is the patient taking {med}?",
                f"What medications is the patient on?",
            ])
    
    # Vital queries
    if any(v in keywords for v in ["blood pressure", "heart rate", "weight", "bmi"]):
        templates.extend([
            "What are the patient's recent vitals?",
            "What is the patient's blood pressure?",
            "Has the patient's weight changed?",
        ])
    
    # Lab queries
    if any(l in keywords for l in ["glucose", "a1c", "cholesterol", "creatinine"]):
        templates.extend([
            "What are the patient's recent lab results?",
            "What is the patient's A1C level?",
            "Any abnormal lab values?",
        ])
    
    # Visit queries
    if "visit" in keywords or metadata.get("type") == "visit":
        templates.extend([
            "When was the patient's last visit?",
            "Summarize recent visits",
        ])
    
    # Fallback
    if not templates:
        templates = [
            "What is in the patient's medical history?",
            "Summarize the patient's records",
        ]
    
    return random.choice(templates)


def determine_difficulty(keywords: List[str], query: str) -> str:
    """Determine query difficulty."""
    # Easy: direct keyword match
    if len(keywords) >= 3:
        return "easy"
    
    # Hard: requires inference or multiple concepts
    complex_terms = ["trend", "history", "summarize", "changed", "abnormal"]
    if any(t in query.lower() for t in complex_terms):
        return "hard"
    
    return "medium"


def determine_category(keywords: List[str]) -> str:
    """Determine query category."""
    if any(k in keywords for k in ["diabetes", "glucose", "a1c", "insulin", "metformin"]):
        return "diabetes"
    if any(k in keywords for k in ["cardiac", "heart", "blood pressure", "hypertension"]):
        return "cardiac"
    if any(k in keywords for k in ["metformin", "lisinopril", "medication", "prescribed"]):
        return "medication"
    if any(k in keywords for k in ["weight", "bmi", "temperature", "heart rate"]):
        return "vitals"
    if any(k in keywords for k in ["glucose", "cholesterol", "creatinine", "hemoglobin"]):
        return "labs"
    if any(k in keywords for k in ["copd", "asthma", "respiratory"]):
        return "respiratory"
    
    return "general"


def generate_gold_dataset(n_cases: int = 50) -> Dict:
    """Generate gold dataset from actual data."""
    
    print("Loading data...")
    patients = load_patients()
    memories_by_patient = load_memories()
    
    if not patients or not memories_by_patient:
        print("ERROR: No data found")
        return None
    
    print(f"  Patients: {len(patients)}")
    print(f"  Patients with memories: {len(memories_by_patient)}")
    
    cases = []
    case_id = 0
    
    # Generate cases from actual memories
    for patient in patients:
        patient_id = patient["id"]
        
        if patient_id not in memories_by_patient:
            continue
        
        memories = memories_by_patient[patient_id]
        
        # Sample memories from this patient
        sampled = random.sample(memories, min(3, len(memories)))
        
        for memory in sampled:
            keywords = extract_keywords_from_memory(memory)
            
            if not keywords:
                continue
            
            query = generate_query_for_memory(memory, keywords)
            difficulty = determine_difficulty(keywords, query)
            category = determine_category(keywords)
            
            cases.append({
                "id": f"eval_{case_id:04d}",
                "patient_id": patient_id,
                "query": query,
                "expected_keywords": keywords,
                "expected_content_snippet": memory["content"][:200],
                "category": category,
                "difficulty": difficulty,
                "source_memory_id": memory.get("metadata", {}).get("encounter_id", "")
            })
            
            case_id += 1
            
            if case_id >= n_cases:
                break
        
        if case_id >= n_cases:
            break
    
    # Build final dataset
    dataset = {
        "metadata": {
            "version": "1.0",
            "description": "Gold evaluation dataset generated from actual patient data",
            "generated_from": str(DATA_DIR),
            "total_cases": len(cases),
            "total_patients": len(set(c["patient_id"] for c in cases)),
            "categories": list(set(c["category"] for c in cases))
        },
        "cases": cases
    }
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate gold dataset from patient data")
    parser.add_argument("--n", type=int, default=50, help="Number of cases to generate")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    args = parser.parse_args()
    
    print("=== Gold Dataset Generator ===\n")
    
    # Generate
    dataset = generate_gold_dataset(args.n)
    
    if not dataset:
        return
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = args.output or (OUTPUT_DIR / "cases.json")
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n=== Done ===")
    print(f"Generated {len(dataset['cases'])} cases")
    print(f"Categories: {dataset['metadata']['categories']}")
    print(f"Output: {output_file}")
    
    # Show sample
    print("\n--- Sample Case ---")
    sample = dataset["cases"][0]
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
