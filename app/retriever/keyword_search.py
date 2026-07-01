
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.models.catalog_models import Assessment

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.\-]*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class KeywordIndex:
    def __init__(self, assessments: list[Assessment]):
        self._ids = [a.id for a in assessments]
        corpus = [_tokenize(a.to_retrieval_document()) for a in assessments]
        self._bm25 = BM25Okapi(corpus)

    def query(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Returns [(assessment_id, normalized_score)] sorted descending."""
        tokens = _tokenize(query_text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0
        ranked = sorted(zip(self._ids, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc_id, score / max_score) for doc_id, score in ranked[:top_k] if score > 0]
