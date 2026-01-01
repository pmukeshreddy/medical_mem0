"""Memory and chat models."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class Memory(BaseModel):
    """Memory item from Mem0."""
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None


class MemoryCreate(BaseModel):
    """Create memory request."""
    patient_id: str
    content: str
    metadata: Dict[str, Any] = {}


class MemorySearch(BaseModel):
    """Search memories request."""
    patient_id: str
    query: str
    limit: int = 5
    strategy: str = "vanilla"  # vanilla, hybrid, rerank, temporal, entity


class MemorySearchResponse(BaseModel):
    """Search response with metrics."""
    memories: List[Memory]
    latency_ms: float
    strategy: str


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str  # user, assistant
    content: str


class ChatRequest(BaseModel):
    """Chat request with patient context."""
    patient_id: str
    message: str
    history: List[ChatMessage] = []
    strategy: str = "vanilla"


class ChatResponse(BaseModel):
    """Chat response with context."""
    response: str
    memories_used: List[Memory]
    latency_ms: float
