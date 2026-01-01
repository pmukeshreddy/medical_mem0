"""
Base class for retrieval strategies.

All strategies inherit from this and implement search().
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    memories: List[Dict]
    latency_ms: float
    strategy: str
    metadata: Dict = None


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""
    
    name: str = "base"
    description: str = "Base retrieval strategy"
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        patient_id: str, 
        k: int = 5
    ) -> Tuple[List[Dict], float]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            patient_id: Patient/user ID
            k: Number of results to return
        
        Returns:
            (memories, latency_ms)
        """
        pass
    
    def get_info(self) -> Dict:
        """Get strategy info."""
        return {
            "name": self.name,
            "description": self.description
        }
