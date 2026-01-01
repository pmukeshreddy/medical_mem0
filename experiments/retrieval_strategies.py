"""
Retrieval Strategies Registry

Central registry for all retrieval strategies.
"""

from typing import Dict, Type
from .base import RetrievalStrategy
from .vanilla_dense import VanillaDense
from .hybrid_bm25 import HybridBM25
from .with_reranker import WithReranker
from .temporal_decay import TemporalDecay
from .entity_filtered import EntityFiltered


# Strategy registry
STRATEGIES: Dict[str, Type[RetrievalStrategy]] = {
    "vanilla": VanillaDense,
    "hybrid": HybridBM25,
    "reranker": WithReranker,
    "temporal": TemporalDecay,
    "entity": EntityFiltered,
}


def get_strategy(name: str, config: Dict = None, **kwargs) -> RetrievalStrategy:
    """Get a strategy instance by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    
    return STRATEGIES[name](config=config, **kwargs)


def list_strategies() -> Dict[str, str]:
    """List all available strategies with descriptions."""
    return {
        name: cls.description 
        for name, cls in STRATEGIES.items()
    }


def compare_strategies(
    query: str, 
    patient_id: str, 
    k: int = 5,
    config: Dict = None
) -> Dict[str, Dict]:
    """Run all strategies on the same query for comparison."""
    results = {}
    
    for name in STRATEGIES:
        try:
            strategy = get_strategy(name, config)
            memories, latency = strategy.search(query, patient_id, k)
            
            results[name] = {
                "memories": memories,
                "latency_ms": latency,
                "count": len(memories)
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    
    return results
