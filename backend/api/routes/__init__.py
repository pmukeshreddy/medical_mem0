from .patients import router as patients_router
from .search import router as search_router
from .chat import router as chat_router

__all__ = ["patients_router", "search_router", "chat_router"]
