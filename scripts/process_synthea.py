"""
Process Synthea CSV output into MedMem0 format.

Converts raw Synthea data → structured patient records with:
- Demographics
- Conditions (with onset/resolution dates)
- Medications (with start/stop dates)
- Encounters/Visits (with notes)
- Observations (labs, vitals)

Output: data/processed/patients.jsonl
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
import argparse


DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "synthea" / "csv"
OUTPUT_DIR = DATA_DIR / "processed"


def load_csv(name: str) -> pd.DataFrame:
    """Load a Synthea CSV file."""
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def process_patients() -> Dict[str, Dict]:
    """Load patient demographics."""
    df = load_csv("patients")
    if df.empty:
        return {}
    
    patients = {}
    for _, row in df.iterrows():
        patients[row["Id"]] = {
            "id": row["Id"],
            "name": f"{row.get('FIRST', '')} {row.get('LAST', '')}".strip(),
            "birth_date": row.get("BIRTHDATE"),
            "gender": row.get("GENDER"),
            "race": row.get("RACE"),
            "city": row.get("CITY"),
            "state": row.get("STATE"),
            "conditions": [],
            "medications": [],
            "encounters": [],
            "observations": [],
        }
    return patients


def add_conditions(patients: Dict[str, Dict]):
    """Add conditions/diagnoses to patients."""
    df = load_csv("conditions")
    if df.empty:
        return
    
    for _, row in df.iterrows():
        pid = row.get("PATIENT")
        if pid not in patients:
            continue
        
        patients[pid]["conditions"].append({
            "code": row.get("CODE"),
            "description": row.get("DESCRIPTION"),
            "onset_date": row.get("START"),
            "resolution_date": row.get("STOP"),
            "encounter_id": row.get("ENCOUNTER"),
        })


def add_medications(patients: Dict[str, Dict]):
    """Add medications to patients."""
    df = load_csv("medications")
    if df.empty:
        return
    
    for _, row in df.iterrows():
        pid = row.get("PATIENT")
        if pid not in patients:
            continue
        
        patients[pid]["medications"].append({
            "code": row.get("CODE"),
            "description": row.get("DESCRIPTION"),
            "start_date": row.get("START"),
            "stop_date": row.get("STOP"),
            "reason_code": row.get("REASONCODE"),
            "reason_description": row.get("REASONDESCRIPTION"),
            "encounter_id": row.get("ENCOUNTER"),
        })


def add_encounters(patients: Dict[str, Dict]):
    """Add encounters/visits to patients."""
    df = load_csv("encounters")
    if df.empty:
        return
    
    for _, row in df.iterrows():
        pid = row.get("PATIENT")
        if pid not in patients:
            continue
        
        patients[pid]["encounters"].append({
            "id": row.get("Id"),
            "date": row.get("START"),
            "end_date": row.get("STOP"),
            "type": row.get("ENCOUNTERCLASS"),
            "code": row.get("CODE"),
            "description": row.get("DESCRIPTION"),
            "reason_code": row.get("REASONCODE"),
            "reason_description": row.get("REASONDESCRIPTION"),
        })


def add_observations(patients: Dict[str, Dict]):
    """Add observations (labs, vitals) to patients."""
    df = load_csv("observations")
    if df.empty:
        return
    
    for _, row in df.iterrows():
        pid = row.get("PATIENT")
        if pid not in patients:
            continue
        
        patients[pid]["observations"].append({
            "date": row.get("DATE"),
            "code": row.get("CODE"),
            "description": row.get("DESCRIPTION"),
            "value": row.get("VALUE"),
            "units": row.get("UNITS"),
            "type": row.get("TYPE"),
            "encounter_id": row.get("ENCOUNTER"),
        })


def generate_visit_narratives(patient: Dict) -> List[Dict]:
    """
    Generate natural language visit narratives for Mem0 ingestion.
    Groups data by encounter and creates readable summaries.
    """
    encounters_map = {e["id"]: e for e in patient.get("encounters", [])}
    
    # Group conditions, meds, observations by encounter
    encounter_data = defaultdict(lambda: {"conditions": [], "meds": [], "obs": []})
    
    for cond in patient.get("conditions", []):
        eid = cond.get("encounter_id")
        if eid:
            encounter_data[eid]["conditions"].append(cond)
    
    for med in patient.get("medications", []):
        eid = med.get("encounter_id")
        if eid:
            encounter_data[eid]["meds"].append(med)
    
    for obs in patient.get("observations", []):
        eid = obs.get("encounter_id")
        if eid:
            encounter_data[eid]["obs"].append(obs)
    
    narratives = []
    for eid, enc in encounters_map.items():
        data = encounter_data[eid]
        
        # Build narrative
        parts = []
        parts.append(f"Visit on {enc.get('date', 'unknown date')}")
        parts.append(f"Type: {enc.get('type', 'unknown')}")
        
        if enc.get("reason_description"):
            parts.append(f"Reason: {enc['reason_description']}")
        
        if data["conditions"]:
            conds = [c["description"] for c in data["conditions"] if c.get("description")]
            if conds:
                parts.append(f"Diagnoses: {', '.join(conds)}")
        
        if data["meds"]:
            meds = [m["description"] for m in data["meds"] if m.get("description")]
            if meds:
                parts.append(f"Medications: {', '.join(set(meds))}")
        
        if data["obs"]:
            # Just key vitals/labs
            key_obs = [
                f"{o['description']}: {o['value']} {o.get('units', '')}"
                for o in data["obs"]
                if o.get("description") and o.get("value")
            ][:10]  # Limit to 10
            if key_obs:
                parts.append(f"Observations: {'; '.join(key_obs)}")
        
        narrative = ". ".join(parts)
        narratives.append({
            "encounter_id": eid,
            "date": enc.get("date"),
            "type": enc.get("type"),
            "narrative": narrative,
        })
    
    # Sort by date
    narratives.sort(key=lambda x: x.get("date") or "")
    return narratives


def create_mem0_records(patient: Dict) -> List[Dict]:
    """
    Create Mem0-ready memory records for a patient.
    Each visit becomes a memory entry.
    """
    narratives = generate_visit_narratives(patient)
    
    records = []
    for narr in narratives:
        records.append({
            "user_id": patient["id"],
            "content": narr["narrative"],
            "metadata": {
                "type": "visit",
                "encounter_id": narr["encounter_id"],
                "date": narr["date"],
                "encounter_type": narr["type"],
                "patient_name": patient.get("name"),
            }
        })
    
    # Add patient summary as a memory
    active_conditions = [
        c["description"] for c in patient.get("conditions", [])
        if c.get("description") and not c.get("resolution_date")
    ]
    active_meds = [
        m["description"] for m in patient.get("medications", [])
        if m.get("description") and not m.get("stop_date")
    ]
    
    summary = f"Patient: {patient.get('name', 'Unknown')}. "
    summary += f"DOB: {patient.get('birth_date', 'Unknown')}. "
    summary += f"Gender: {patient.get('gender', 'Unknown')}. "
    if active_conditions:
        summary += f"Active conditions: {', '.join(set(active_conditions))}. "
    if active_meds:
        summary += f"Current medications: {', '.join(set(active_meds))}. "
    
    records.append({
        "user_id": patient["id"],
        "content": summary,
        "metadata": {
            "type": "patient_summary",
            "patient_name": patient.get("name"),
        }
    })
    
    return records


def main():
    parser = argparse.ArgumentParser(description="Process Synthea data for MedMem0")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of patients")
    args = parser.parse_args()
    
    print("=== Processing Synthea Data ===")
    
    # Check if data exists
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found")
        print("Run download_synthea.sh first")
        return
    
    # Load and process
    print("Loading patients...")
    patients = process_patients()
    print(f"Found {len(patients)} patients")
    
    if args.limit:
        patient_ids = list(patients.keys())[:args.limit]
        patients = {pid: patients[pid] for pid in patient_ids}
        print(f"Limited to {len(patients)} patients")
    
    print("Adding conditions...")
    add_conditions(patients)
    
    print("Adding medications...")
    add_medications(patients)
    
    print("Adding encounters...")
    add_encounters(patients)
    
    print("Adding observations...")
    add_observations(patients)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save full patient records
    patients_file = OUTPUT_DIR / "patients.jsonl"
    print(f"Saving patients to {patients_file}...")
    with open(patients_file, "w") as f:
        for patient in patients.values():
            f.write(json.dumps(patient) + "\n")
    
    # Save Mem0-ready records
    mem0_file = OUTPUT_DIR / "mem0_records.jsonl"
    print(f"Generating Mem0 records to {mem0_file}...")
    total_records = 0
    with open(mem0_file, "w") as f:
        for patient in patients.values():
            records = create_mem0_records(patient)
            for record in records:
                f.write(json.dumps(record) + "\n")
                total_records += 1
    
    print(f"\n=== Done ===")
    print(f"Patients: {len(patients)}")
    print(f"Mem0 records: {total_records}")
    print(f"\nFiles:")
    print(f"  {patients_file}")
    print(f"  {mem0_file}")
    print(f"\nNext: python seed_patients.py")


if __name__ == "__main__":
    main()
