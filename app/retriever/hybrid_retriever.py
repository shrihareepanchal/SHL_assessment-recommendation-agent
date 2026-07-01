
from __future__ import annotations

from app.catalog.loader import CatalogRepository
from app.ranking.metadata_scorer import ConstraintProfile
from app.ranking.rank_fusion import ScoredAssessment, fuse
from app.retriever.keyword_search import KeywordIndex
from app.retriever.vector_store import VectorStore
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        catalog: CatalogRepository,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        top_k_semantic: int,
        top_k_keyword: int,
        semantic_weight: float,
        keyword_weight: float,
        metadata_weight: float,
    ):
        self._catalog = catalog
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._top_k_semantic = top_k_semantic
        self._top_k_keyword = top_k_keyword
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
        self._metadata_weight = metadata_weight
        self._by_id = {a.id: a for a in catalog.all}

    def retrieve(
        self,
        query_text: str,
        constraints: ConstraintProfile,
        final_k: int = 10,
    ) -> list[ScoredAssessment]:
        semantic_hits = self._vector_store.query(query_text, top_k=self._top_k_semantic)
        keyword_hits = self._keyword_index.query(query_text, top_k=self._top_k_keyword)

        results = fuse(
            assessments_by_id=self._by_id,
            semantic_hits=semantic_hits,
            keyword_hits=keyword_hits,
            constraints=constraints,
            semantic_weight=self._semantic_weight,
            keyword_weight=self._keyword_weight,
            metadata_weight=self._metadata_weight,
            final_k=final_k,
        )
        logger.info(
            "Retrieval complete",
            extra={
                "context": {
                    "query": query_text[:120],
                    "semantic_hits": len(semantic_hits),
                    "keyword_hits": len(keyword_hits),
                    "final_count": len(results),
                }
            },
        )
        return results
