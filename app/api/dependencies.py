
from __future__ import annotations

from functools import lru_cache

from app.agent.orchestrator import AgentDependencies
from app.catalog.loader import CatalogRepository
from app.config.settings import Settings, get_settings
from app.retriever.embedder import get_embedder
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.keyword_search import KeywordIndex
from app.retriever.vector_store import VectorStore
from app.services.llm_service import build_llm_client
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_catalog_repository() -> CatalogRepository:
    settings = get_settings()
    return CatalogRepository(settings.catalog_processed_path)


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()
    catalog = get_catalog_repository()

    embedder = get_embedder(settings.embedding_model_name, settings.embedding_device)
    vector_store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedder=embedder,
    )
    if vector_store.is_empty():
        logger.info("Vector store empty on startup - building index now")
        vector_store.rebuild(catalog.all)

    keyword_index = KeywordIndex(catalog.all)

    return HybridRetriever(
        catalog=catalog,
        vector_store=vector_store,
        keyword_index=keyword_index,
        top_k_semantic=settings.retrieval_top_k_semantic,
        top_k_keyword=settings.retrieval_top_k_keyword,
        semantic_weight=settings.hybrid_semantic_weight,
        keyword_weight=settings.hybrid_keyword_weight,
        metadata_weight=settings.metadata_boost_weight,
    )


@lru_cache(maxsize=1)
def get_agent_dependencies() -> AgentDependencies:
    settings = get_settings()
    return AgentDependencies(
        llm=build_llm_client(settings),
        retriever=get_hybrid_retriever(),
        catalog=get_catalog_repository(),
        settings=settings,
    )


def reset_dependency_caches() -> None:
    """Test-only helper to force re-construction between test cases that
    monkeypatch settings/env vars."""
    get_catalog_repository.cache_clear()
    get_hybrid_retriever.cache_clear()
    get_agent_dependencies.cache_clear()
