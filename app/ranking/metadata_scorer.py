
from __future__ import annotations

from dataclasses import dataclass, field

from app.models.catalog_models import Assessment


@dataclass
class ConstraintProfile:
    """Structured constraints extracted from the conversation so far.

    Populated by `app.agent.slot_filling`. Every field is optional -
    absence means "the user hasn't told us", not "the user doesn't want it".
    """

    job_levels: list[str] = field(default_factory=list)
    test_type_codes: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    max_duration_minutes: int | None = None
    adaptive_required: bool | None = None
    remote_required: bool | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.job_levels,
                self.test_type_codes,
                self.languages,
                self.max_duration_minutes,
                self.adaptive_required is not None,
                self.remote_required is not None,
            ]
        )


def score_metadata_fit(assessment: Assessment, constraints: ConstraintProfile) -> float:
    """Returns a fit score roughly in [0, 1]. 0 constraints stated -> 0.0
    (neutral - contributes nothing to fusion, semantic/keyword scores decide)."""
    if constraints.is_empty():
        return 0.0

    checks: list[float] = []

    if constraints.job_levels:
        overlap = set(l.lower() for l in constraints.job_levels) & {
            l.lower() for l in assessment.job_levels
        }
        checks.append(1.0 if overlap else 0.0)

    if constraints.test_type_codes:
        overlap = set(constraints.test_type_codes) & set(assessment.test_type_codes)
        checks.append(1.0 if overlap else 0.0)

    if constraints.languages:
        overlap = set(l.lower() for l in constraints.languages) & {
            l.lower() for l in assessment.languages
        }
        checks.append(1.0 if overlap else 0.0)

    if constraints.max_duration_minutes is not None:
        if assessment.duration_minutes is None:
            checks.append(0.5)  # unknown duration - don't penalize outright
        else:
            checks.append(1.0 if assessment.duration_minutes <= constraints.max_duration_minutes else 0.0)

    if constraints.adaptive_required is not None:
        checks.append(1.0 if assessment.adaptive == constraints.adaptive_required else 0.0)

    if constraints.remote_required is not None:
        checks.append(1.0 if assessment.remote_testing == constraints.remote_required else 0.0)

    return sum(checks) / len(checks) if checks else 0.0
