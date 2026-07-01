
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.models.catalog_models import Assessment
from app.retriever.embedder import Embedder
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def _to_metadata(assessment: Assessment) -> dict[str, Any]:
    """Chroma metadata values must be scalars, so list fields are joined."""
    return {
        "name": assessment.name,
        "url": assessment.url,
        "job_levels": "|".join(assessment.job_levels),
        "languages": "|".join(assessment.languages),
        "duration_minutes": assessment.duration_minutes or -1,
        "remote_testing": assessment.remote_testing,
        "adaptive": assessment.adaptive,
        "test_type_codes": "|".join(assessment.test_type_codes),
        "primary_test_type": assessment.primary_test_type,
    }


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str, embedder: Embedder):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection_name = collection_name
        self._embedder = embedder
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def is_empty(self) -> bool:
        return self._collection.count() == 0

    def rebuild(self, assessments: list[Assessment]) -> None:
        """Full re-index. Idempotent: safe to re-run whenever the catalog changes."""
        existing_ids = self._collection.get()["ids"]
        if existing_ids:
            self._collection.delete(ids=existing_ids)

        texts = [a.to_retrieval_document() for a in assessments]
        embeddings = self._embedder.embed_documents(texts)

        self._collection.add(
            ids=[a.id for a in assessments],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[_to_metadata(a) for a in assessments],
        )
        logger.info(
            "Vector store rebuilt", extra={"context": {"count": len(assessments)}}
        )

    def query(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Returns [(assessment_id, similarity_score)], higher score = closer.

        Chroma returns cosine *distance* (0 = identical); we convert to a
        similarity in [0, 1] so it's directly comparable/fusable with the
        BM25 scores in the hybrid retriever.
        """
        query_embedding = self._embedder.embed_query(query_text)
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(doc_id, 1.0 - dist) for doc_id, dist in zip(ids, distances)]
