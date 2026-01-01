"""Patient models."""

from pydantic import BaseModel
from typing import Optional, List
from datetime import date


class Condition(BaseModel):
    """Patient condition/diagnosis."""
    code: str
    description: str
    onset_date: Optional[str] = None
    resolution_date: Optional[str] = None


class Medication(BaseModel):
    """Patient medication."""
    code: str
    description: str
    start_date: Optional[str] = None
    stop_date: Optional[str] = None


class PatientCreate(BaseModel):
    """Create patient request."""
    name: str
    birth_date: str
    gender: str
    city: Optional[str] = None
    state: Optional[str] = None


class Patient(BaseModel):
    """Full patient model."""
    id: str
    name: str
    birth_date: str
    gender: str
    city: Optional[str] = None
    state: Optional[str] = None
    conditions: List[Condition] = []
    medications: List[Medication] = []


class PatientSummary(BaseModel):
    """Patient summary for list view."""
    id: str
    name: str
    birth_date: str
    gender: str
    active_conditions: int = 0
    active_medications: int = 0
