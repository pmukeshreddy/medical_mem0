"""Search/retrieval endpoints."""

from fastapi import APIRouter, Depends
from typing import List

from models.memory import MemorySearch, MemorySearchResponse, Memory
from api.deps import get_memory
from core.memory_service import MemoryService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=MemorySearchResponse)
async def search_memories(
    request: MemorySearch,
    memory: MemoryService = Depends(get_memory)
):
    """
    Search patient memories with different retrieval strategies.
    
    Strategies:
    - vanilla: Basic dense vector search
    - hybrid: BM25 + dense fusion
    - temporal: Boost recent memories
    - entity: Medical entity filtering
    - rerank: With Cohere reranker (if configured)
    """
    memories, latency_ms = memory.search(
        patient_id=request.patient_id,
        query=request.query,
        limit=request.limit,
        strategy=request.strategy
    )
    
    return MemorySearchResponse(
        memories=[Memory(**m) for m in memories],
        latency_ms=latency_ms,
        strategy=request.strategy
    )


@router.post("/compare")
async def compare_strategies(
    patient_id: str,
    query: str,
    limit: int = 5,
    memory: MemoryService = Depends(get_memory)
):
    """
    Compare all retrieval strategies for the same query.
    Useful for benchmarking.
    """
    strategies = ["vanilla", "hybrid", "temporal", "entity"]
    results = {}
    
    for strategy in strategies:
        memories, latency_ms = memory.search(
            patient_id=patient_id,
            query=query,
            limit=limit,
            strategy=strategy
        )
        
        results[strategy] = {
            "memories": memories,
            "latency_ms": latency_ms,
            "count": len(memories)
        }
    
    return {
        "patient_id": patient_id,
        "query": query,
        "results": results
    }
