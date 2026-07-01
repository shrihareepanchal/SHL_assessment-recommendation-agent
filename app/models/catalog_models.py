
from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

# SHL's public catalog uses single-letter codes for assessment categories.
# Reconstructed from the scraped catalog's `keys` field (see
# scripts/build_catalog.py) - this mapping matches SHL's published
# convention (A/B/C/D/E/K/P/S) used throughout their product catalog UI.
CATEGORY_TO_TEST_TYPE_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


class Assessment(BaseModel):
    """A single Individual Test Solution from the SHL product catalog."""

    id: str
    name: str
    url: str
    description: str = ""
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration_raw: str = ""
    duration_minutes: int | None = None
    remote_testing: bool = False
    adaptive: bool = False
    categories: list[str] = Field(default_factory=list)
    test_type_codes: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def primary_test_type(self) -> str:
        """The single test-type code shown to the user (API requires one)."""
        return self.test_type_codes[0] if self.test_type_codes else "-"

    def to_retrieval_document(self) -> str:
        """Flatten the assessment into a single text blob for embedding/BM25.

        Name is repeated because short, high-signal fields tend to be
        under-weighted by both dense embeddings (diluted by long descriptions)
        and BM25 (their raw term frequency is low). Duplicating it cheaply
        boosts its contribution to both signals without a custom weighting
        scheme.
        """
        parts = [
            self.name,
            self.name,
            self.description,
            " ".join(self.categories),
            " ".join(self.job_levels),
            "adaptive" if self.adaptive else "non-adaptive",
            "remote testing" if self.remote_testing else "",
        ]
        return " | ".join(p for p in parts if p)
