#!/usr/bin/env python3

from __future__ import annotations

from app.catalog.loader import CatalogRepository
from app.config.settings import get_settings
from app.retriever.embedder import get_embedder
from app.retriever.vector_store import VectorStore
from app.utils.logging_config import configure_logging, get_logger


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    catalog = CatalogRepository(settings.catalog_processed_path)
    embedder = get_embedder(settings.embedding_model_name, settings.embedding_device)
    vector_store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedder=embedder,
    )
    vector_store.rebuild(catalog.all)
    logger.info(f"Indexed {len(catalog)} assessments into {settings.chroma_persist_dir}")


if __name__ == "__main__":
    main()
