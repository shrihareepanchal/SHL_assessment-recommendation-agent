
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Strongly-typed application settings, validated at process startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App metadata -----------------------------------------------------
    app_name: str = "SHL Assessment Recommendation Agent"
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # --- LLM provider -------------------------------------------------------
    # LLM_PROVIDER lets us swap providers without touching business logic.
    # Only "gemini" is implemented today; the interface (LLMClient) is
    # provider-agnostic so a second provider is a new adapter, not a rewrite.
    llm_provider: str = Field(default="gemini")
    google_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.5-flash")
    llm_temperature: float = Field(default=0.2)
    llm_max_output_tokens: int = Field(default=1024)
    llm_request_timeout_seconds: int = Field(default=20)

    # --- Embeddings ---------------------------------------------------------
    embedding_model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_device: str = Field(default="cpu")

    # --- Vector store ---------------------------------------------------------
    chroma_persist_dir: str = Field(default=str(PROJECT_ROOT / "data" / "chroma_db"))
    chroma_collection_name: str = Field(default="shl_catalog")

    # --- Catalog data ---------------------------------------------------------
    catalog_processed_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "catalog_processed.json")
    )

    # --- Retrieval tuning -----------------------------------------------------
    retrieval_top_k_semantic: int = Field(default=25)
    retrieval_top_k_keyword: int = Field(default=25)
    retrieval_final_k: int = Field(default=10)
    hybrid_semantic_weight: float = Field(default=0.6)
    hybrid_keyword_weight: float = Field(default=0.4)
    metadata_boost_weight: float = Field(default=0.15)

    # --- Conversation policy ---------------------------------------------------
    max_conversation_turns: int = Field(default=8)
    min_recommendations: int = Field(default=1)
    max_recommendations: int = Field(default=10)

    # --- API server -------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance."""
    return Settings()
