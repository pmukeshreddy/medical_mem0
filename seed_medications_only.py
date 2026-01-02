"""Seed medications by matching patient NAMES between old and new data."""

import json
import pandas as pd
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from mem0 import Memory

load_dotenv("backend/.env")

OLD_PATIENTS = Path("data/processed/patients.jsonl")
NEW_PATIENTS_CSV = Path("data/raw/synthea/csv/patients.csv")
MEDS_CSV = Path("data/raw/synthea/csv/medications.csv")

def init_mem0():
    config = {
        "vector_store": {
            "provider": "pinecone",
            "config": {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "collection_name": os.getenv("PINECONE_INDEX_NAME", "medmem0"),
                "embedding_model_dims": 1536,
                "serverless_config": {"cloud": "aws", "region": "us-east-1"}
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
    return Memory.from_config(config)

def main():
    print("=== Seed Medications via Name Matching ===\n")
    
    # 1. Load old patients (name -> old_uuid)
    old_name_to_uuid = {}
    with open(OLD_PATIENTS) as f:
        for line in f:
            p = json.loads(line)
            # Use first name as key (more likely to match)
            first_name = p["name"].split()[0] if p.get("name") else None
            if first_name:
                old_name_to_uuid[first_name] = p["id"]
    print(f"Old patients: {len(old_name_to_uuid)}")
    
    # 2. Load new patients (new_uuid -> first_name)
    new_df = pd.read_csv(NEW_PATIENTS_CSV, low_memory=False)
    new_uuid_to_name = dict(zip(new_df["Id"], new_df["FIRST"]))
    new_name_to_uuid = {v: k for k, v in new_uuid_to_name.items()}
    print(f"New patients: {len(new_uuid_to_name)}")
    
    # 3. Find matches
    matched = set(old_name_to_uuid.keys()) & set(new_name_to_uuid.keys())
    print(f"Matched by name: {len(matched)}\n")
    
    if not matched:
        print("No matches found!")
        return
    
    # 4. Load medications (new_uuid -> meds list)
    meds_df = pd.read_csv(MEDS_CSV, low_memory=False)
    new_uuid_meds = meds_df.groupby("PATIENT")["DESCRIPTION"].apply(
        lambda x: list(set(x))
    ).to_dict()
    print(f"Patients with meds in CSV: {len(new_uuid_meds)}")
    
    # 5. Build final mapping: old_uuid -> meds
    to_seed = {}
    for first_name in matched:
        old_uuid = old_name_to_uuid[first_name]
        new_uuid = new_name_to_uuid[first_name]
        if new_uuid in new_uuid_meds:
            to_seed[old_uuid] = new_uuid_meds[new_uuid]
    
    print(f"Patients to seed with meds: {len(to_seed)}\n")
    
    if not to_seed:
        print("No medications to seed!")
        return
    
    # 6. Seed to Mem0
    memory = init_mem0()
    success, failed = 0, 0
    start = time.time()
    
    for i, (old_uuid, meds) in enumerate(to_seed.items()):
        try:
            content = f"Patient takes: {', '.join(meds)}"
            memory.add(
                messages=[{"role": "assistant", "content": content}],
                user_id=old_uuid,
                metadata={"type": "medications"}
            )
            success += 1
            
            if (i + 1) % 5 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 1
                eta = (len(to_seed) - i - 1) / rate
                print(f"Progress: {i+1}/{len(to_seed)} - ETA: {eta:.0f}s")
                
        except Exception as e:
            failed += 1
            print(f"ERROR: {e}")
    
    print(f"\n=== Done! ===")
    print(f"Success: {success}, Failed: {failed}")

if __name__ == "__main__":
    main()