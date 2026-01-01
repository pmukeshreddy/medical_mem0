"""Configuration settings for MedMem0 backend."""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache

#/Users/mukeshreddypochamreddy/medical_mem0
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API
    app_name: str = "MedMem0 API"
    debug: bool = False
    
    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "medmem0"
    pinecone_region: str = "us-east-1"
    pinecone_cloud: str = "aws"
    
    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    # Optional - Cohere for reranking
    cohere_api_key: str | None = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
