"""Chat endpoints."""

import time
from fastapi import APIRouter, Depends

from models.memory import ChatRequest, ChatResponse, Memory
from api.deps import get_memory, get_llm
from core.memory_service import MemoryService
from core.llm_orchestrator import LLMOrchestrator

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    memory: MemoryService = Depends(get_memory),
    llm: LLMOrchestrator = Depends(get_llm)
):
    """
    Chat with a patient's memory context.
    
    1. Retrieves relevant memories using specified strategy
    2. Passes memories as context to LLM
    3. Returns response with used memories
    """
    start = time.perf_counter()
    
    # Retrieve relevant memories
    memories, search_latency = memory.search(
        patient_id=request.patient_id,
        query=request.message,
        limit=5,
        strategy=request.strategy
    )
    
    # Generate response with context
    response = llm.chat(
        message=request.message,
        memories=memories,
        history=[h.model_dump() for h in request.history]
    )
    
    total_latency = (time.perf_counter() - start) * 1000
    
    return ChatResponse(
        response=response,
        memories_used=[Memory(**m) for m in memories],
        latency_ms=total_latency
    )


@router.post("/summarize")
async def summarize_visit(
    visit_text: str,
    llm: LLMOrchestrator = Depends(get_llm)
):
    """Summarize a visit note."""
    summary = llm.summarize_visit(visit_text)
    return {"summary": summary}


@router.post("/extract-entities")
async def extract_entities(
    text: str,
    llm: LLMOrchestrator = Depends(get_llm)
):
    """Extract medical entities from text."""
    entities = llm.extract_entities(text)
    return {"entities": entities}
