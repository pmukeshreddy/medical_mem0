"""Shared dependencies for API routes."""

from core.memory_service import MemoryService, get_memory_service
from core.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator


def get_memory() -> MemoryService:
    """Dependency: get memory service."""
    return get_memory_service()


def get_llm() -> LLMOrchestrator:
    """Dependency: get LLM orchestrator."""
    return get_llm_orchestrator()
