"""Patient endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
import json
from pathlib import Path

from models import Patient, PatientSummary, PatientCreate
from api.deps import get_memory
from core.memory_service import MemoryService

router = APIRouter(prefix="/patients", tags=["patients"])

# Load patients from JSONL (in production, use a real DB)
DATA_FILE = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "patients.jsonl"


def load_patients() -> List[dict]:
    """Load patients from JSONL file."""
    if not DATA_FILE.exists():
        return []
    
    patients = []
    with open(DATA_FILE) as f:
        for line in f:
            patients.append(json.loads(line))
    return patients


@router.get("/", response_model=List[PatientSummary])
async def list_patients(limit: int = 20, offset: int = 0):
    """List all patients."""
    patients = load_patients()
    
    summaries = []
    for p in patients[offset:offset + limit]:
        active_conditions = len([
            c for c in p.get("conditions", []) 
            if not c.get("resolution_date")
        ])
        active_meds = len([
            m for m in p.get("medications", []) 
            if not m.get("stop_date")
        ])
        
        summaries.append(PatientSummary(
            id=p["id"],
            name=p.get("name", "Unknown"),
            birth_date=p.get("birth_date", ""),
            gender=p.get("gender", ""),
            active_conditions=active_conditions,
            active_medications=active_meds
        ))
    
    return summaries


@router.get("/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    """Get patient by ID."""
    patients = load_patients()
    
    for p in patients:
        if p["id"] == patient_id:
            return Patient(**p)
    
    raise HTTPException(status_code=404, detail="Patient not found")


@router.get("/{patient_id}/memories")
async def get_patient_memories(
    patient_id: str,
    memory: MemoryService = Depends(get_memory)
):
    """Get all memories for a patient."""
    memories = memory.get_all(patient_id)
    return {"patient_id": patient_id, "memories": memories}


@router.post("/{patient_id}/memories")
async def add_patient_memory(
    patient_id: str,
    content: str,
    memory: MemoryService = Depends(get_memory)
):
    """Add a memory for a patient."""
    result = memory.add(patient_id, content)
    return {"status": "ok", "result": result}
