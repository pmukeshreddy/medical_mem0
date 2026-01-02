"""
Research Implementations

Advanced retrieval strategies based on recent papers:
- RAG-Fusion: Multi-query expansion with RRF
- Advanced Medical: Query2Doc + StepBack + SelfRAG
- ColBERT: Late interaction retrieval
"""

from .rag_fusion import RAGFusion, create_rag_fusion
from .advanced_retrieval import (
    AdvancedMedicalRetriever,
    FastMedicalRetriever,
    BalancedMedicalRetriever,
    LLMMedicalExpander,
    Query2Doc,
    StepBackRetrieval,
    MedicalEntityExpander,
    SelfRAGVerifier
)
from .colbert_retrieval import ColBERTRetriever, ColBERTApproximator, HybridColBERTBM25

__all__ = [
    "RAGFusion",
    "create_rag_fusion",
    "AdvancedMedicalRetriever",
    "FastMedicalRetriever",
    "BalancedMedicalRetriever",
    "LLMMedicalExpander",
    "Query2Doc",
    "StepBackRetrieval", 
    "MedicalEntityExpander",
    "SelfRAGVerifier",
    "ColBERTRetriever",
    "ColBERTApproximator",
    "HybridColBERTBM25"
]