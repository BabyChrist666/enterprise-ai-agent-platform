"""
Configuration management for the Enterprise AI Agent Platform.
Supports multiple environments and secure credential handling.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Keys
    cohere_api_key: str = ""

    # Application
    app_name: str = "Enterprise AI Agent Platform"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Vector Store
    vector_store_path: str = "./data/vector_store"
    embedding_model: str = "embed-english-v3.0"
    rerank_model: str = "rerank-english-v3.0"
    generation_model: str = "command-r-plus"

    # Agent Settings
    max_agent_iterations: int = 10
    agent_timeout_seconds: int = 120

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
