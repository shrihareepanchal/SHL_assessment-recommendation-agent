
from __future__ import annotations

import json
from pathlib import Path

from app.models.catalog_models import Assessment
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class CatalogLoadError(RuntimeError):
    """Raised when the processed catalog file is missing or malformed."""


class CatalogRepository:
    """In-memory repository over the full SHL Individual Test Solutions catalog."""

    def __init__(self, catalog_path: str | Path):
        self._path = Path(catalog_path)
        self._assessments: list[Assessment] = []
        self._by_id: dict[str, Assessment] = {}
        self._by_name_lower: dict[str, Assessment] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            raise CatalogLoadError(
                f"Catalog file not found at {self._path}. "
                "Run `python scripts/build_catalog.py` first."
            )
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CatalogLoadError(f"Catalog file is not valid JSON: {exc}") from exc

        assessments: list[Assessment] = []
        for entry in raw:
            try:
                assessments.append(Assessment(**entry))
            except Exception as exc:  # noqa: BLE001 - log and skip malformed rows
                logger.warning(
                    "Skipping malformed catalog entry",
                    extra={"context": {"entry_id": entry.get("id"), "error": str(exc)}},
                )

        if not assessments:
            raise CatalogLoadError("Catalog loaded but contains zero valid assessments.")

        self._assessments = assessments
        self._by_id = {a.id: a for a in assessments}
        self._by_name_lower = {a.name.lower(): a for a in assessments}

        logger.info(
            "Catalog loaded", extra={"context": {"count": len(assessments), "path": str(self._path)}}
        )

    @property
    def all(self) -> list[Assessment]:
        return list(self._assessments)

    def __len__(self) -> int:
        return len(self._assessments)

    def get_by_id(self, assessment_id: str) -> Assessment | None:
        return self._by_id.get(assessment_id)

    def get_by_name(self, name: str) -> Assessment | None:
        return self._by_name_lower.get(name.lower())

    def is_valid_url(self, url: str) -> bool:
        """Used by the response Validator to guarantee zero hallucinated URLs."""
        return any(a.url == url for a in self._assessments)

    def fuzzy_find_by_name_fragment(self, fragment: str, limit: int = 5) -> list[Assessment]:
        """Cheap substring search, used for comparison-intent name resolution
        (e.g. user says "OPQ" and means "Occupational Personality
        Questionnaire OPQ32r")."""
        fragment_lower = fragment.lower().strip()
        if not fragment_lower:
            return []
        matches = [a for a in self._assessments if fragment_lower in a.name.lower()]
        matches.sort(key=lambda a: len(a.name))
        return matches[:limit]
