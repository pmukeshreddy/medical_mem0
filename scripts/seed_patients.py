

import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Mem0 imports
try:
    from mem0 import Memory
except ImportError:
    print("ERROR: mem0 not installed. Run: pip install mem0ai")
    exit(1)


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MEM0_RECORDS_FILE = DATA_DIR / "mem0_records.jsonl"


def load_records(limit: int = None, patient_id: str = None) -> List[Dict]:
    """Load records from JSONL file."""
    if not MEM0_RECORDS_FILE.exists():
        print(f"ERROR: {MEM0_RECORDS_FILE} not found")
        print("Run process_synthea.py first")
        exit(1)
    
    records = []
    with open(MEM0_RECORDS_FILE, "r") as f:
        for line in f:
            record = json.loads(line)
            if patient_id and record["user_id"] != patient_id:
                continue
            records.append(record)
            if limit and len(records) >= limit:
                break
    
    return records


def init_mem0() -> Memory:
    """Initialize Mem0 client with Pinecone."""
    
    # Required env vars
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not pinecone_api_key:
        print("ERROR: PINECONE_API_KEY not found in .env")
        exit(1)
    
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        exit(1)
    
    print("Using Pinecone vector store...")
    
    config = {
        "vector_store": {
            "provider": "pinecone",
            "config": {
                "api_key": pinecone_api_key,
                "collection_name": os.getenv("PINECONE_INDEX_NAME", "medmem0"),
                "embedding_model_dims": 1536,
                "serverless_config": {
                    "cloud": "aws",
                    "region": "us-east-1",  # Free tier region
                }
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "api_key": openai_api_key,
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "api_key": openai_api_key,
            }
        }
    }
    
    return Memory.from_config(config)


def seed_records(memory: Memory, records: List[Dict], batch_size: int = 10):
    """Seed records into Mem0."""
    total = len(records)
    success = 0
    failed = 0
    
    print(f"\nSeeding {total} records...")
    start_time = time.time()
    
    for i, record in enumerate(records):
        try:
            # Add memory to Mem0
            memory.add(
                messages=[{"role": "assistant", "content": record["content"]}],
                user_id=record["user_id"],
                metadata=record.get("metadata", {})
            )
            success += 1
            
            # Progress
            if (i + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  Progress: {i + 1}/{total} ({success} ok, {failed} failed) - ETA: {eta:.0f}s")
                
        except Exception as e:
            failed += 1
            print(f"  ERROR on record {i}: {e}")
    
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    
    return success, failed


def verify_seed(memory: Memory, sample_patient_id: str):
    """Verify seeding by searching for a patient."""
    print(f"\nVerifying with patient: {sample_patient_id[:8]}...")
    
    try:
        results = memory.search(
            query="patient history",
            user_id=sample_patient_id,
            limit=3
        )
        
        if results:
            print(f"  Found {len(results)} memories:")
            for i, r in enumerate(results):
                if i >= 2:
                    break
                content = r.get("memory", r.get("content", ""))[:100]
                print(f"    - {content}...")
        else:
            print("  No memories found (may need time to index)")
            
    except Exception as e:
        print(f"  Verification error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Seed patient data into Mem0")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")
    parser.add_argument("--patient-id", type=str, default=None, help="Seed single patient")
    parser.add_argument("--verify", action="store_true", help="Verify after seeding")
    parser.add_argument("--dry-run", action="store_true", help="Load data but don't seed")
    args = parser.parse_args()
    
    print("=== MedMem0 Data Seeder ===\n")
    
    # Load records
    records = load_records(limit=args.limit, patient_id=args.patient_id)
    print(f"Loaded {len(records)} records")
    
    if not records:
        print("No records to seed")
        return
    
    # Get unique patients
    patient_ids = list(set(r["user_id"] for r in records))
    print(f"Unique patients: {len(patient_ids)}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would seed these records:")
        for r in records[:3]:
            print(f"  - {r['user_id'][:8]}... : {r['content'][:60]}...")
        if len(records) > 3:
            print(f"  ... and {len(records) - 3} more")
        return
    
    # Init Mem0
    memory = init_mem0()
    
    # Seed
    success, failed = seed_records(memory, records)
    
    # Verify
    if args.verify and success > 0:
        verify_seed(memory, patient_ids[0])
    
    print("\n=== Complete ===")
    print(f"Records seeded: {success}/{len(records)}")


if __name__ == "__main__":
    main()
