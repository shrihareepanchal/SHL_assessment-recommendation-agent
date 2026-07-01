
from __future__ import annotations

from dataclasses import dataclass

from app.models.catalog_models import Assessment
from app.ranking.metadata_scorer import ConstraintProfile, score_metadata_fit


@dataclass
class ScoredAssessment:
    assessment: Assessment
    semantic_score: float
    keyword_score: float
    metadata_score: float
    final_score: float


def fuse(
    assessments_by_id: dict[str, Assessment],
    semantic_hits: list[tuple[str, float]],
    keyword_hits: list[tuple[str, float]],
    constraints: ConstraintProfile,
    semantic_weight: float,
    keyword_weight: float,
    metadata_weight: float,
    final_k: int,
) -> list[ScoredAssessment]:
    """Merge two ranked hit lists + metadata scoring into one ranking."""
    semantic_map = dict(semantic_hits)
    keyword_map = dict(keyword_hits)

    candidate_ids = set(semantic_map) | set(keyword_map)
    scored: list[ScoredAssessment] = []

    for assessment_id in candidate_ids:
        assessment = assessments_by_id.get(assessment_id)
        if assessment is None:
            continue  # defensive: index and catalog out of sync

        sem = semantic_map.get(assessment_id, 0.0)
        kw = keyword_map.get(assessment_id, 0.0)
        meta = score_metadata_fit(assessment, constraints)

        final = (semantic_weight * sem) + (keyword_weight * kw) + (metadata_weight * meta)
        scored.append(
            ScoredAssessment(
                assessment=assessment,
                semantic_score=sem,
                keyword_score=kw,
                metadata_score=meta,
                final_score=final,
            )
        )

    scored.sort(key=lambda s: s.final_score, reverse=True)
    return scored[:final_k]
