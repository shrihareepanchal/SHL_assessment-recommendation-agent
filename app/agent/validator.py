
from __future__ import annotations

from app.catalog.loader import CatalogRepository
from app.models.api_models import Recommendation
from app.models.catalog_models import Assessment
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def assessments_to_validated_recommendations(
    assessments: list[Assessment],
    catalog: CatalogRepository,
    min_count: int,
    max_count: int,
) -> list[Recommendation]:
    """Converts ranked `Assessment` objects into API `Recommendation`
    objects, dropping anything that isn't verifiably in the catalog and
    clamping to the assignment's [min, max] size bound."""
    validated: list[Recommendation] = []
    for assessment in assessments:
        if not catalog.is_valid_url(assessment.url):
            logger.warning(
                "Dropped recommendation with unverified URL",
                extra={"context": {"name": assessment.name, "url": assessment.url}},
            )
            continue
        validated.append(
            Recommendation(
                name=assessment.name,
                url=assessment.url,
                test_type=assessment.primary_test_type,
            )
        )
        if len(validated) >= max_count:
            break

    if len(validated) < min_count and len(validated) < len(assessments):
        # Shouldn't normally happen (retrieval only returns catalog items),
        # but guarantees we never silently under-deliver due to a bug.
        logger.warning("Fewer valid recommendations than the minimum after validation")

    return validated
