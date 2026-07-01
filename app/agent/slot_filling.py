
from __future__ import annotations

import json
import re

from app.prompts.templates import EXTRACTION_PROMPT
from app.ranking.metadata_scorer import ConstraintProfile
from app.services.llm_service import LLMClient
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_VALID_TEST_TYPE_CODES = {"A", "B", "C", "D", "E", "K", "P", "S"}

# A request this short/generic, with no nouns beyond filler words, cannot be
# retrieved against meaningfully - this is the "I need an assessment" case
# from the assignment.
_VAGUE_FILLER_WORDS = {
    "i", "need", "an", "a", "assessment", "test", "want", "some", "help",
    "hi", "hello", "hey", "please", "with", "for", "we", "are", "looking",
}


def _looks_vague(latest_user_message: str) -> bool:
    words = re.findall(r"[a-z']+", latest_user_message.lower())
    content_words = [w for w in words if w not in _VAGUE_FILLER_WORDS]
    return len(content_words) == 0


def extract_constraints(llm: LLMClient, conversation_history_text: str) -> tuple[ConstraintProfile, str]:
    """Returns (constraints, core_need). Falls back to an empty profile with
    a best-effort core_need if the LLM output can't be parsed - never raises,
    since a parsing hiccup here must not take the whole request down."""
    raw = llm.generate(EXTRACTION_PROMPT, conversation_history=conversation_history_text)
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        data = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Constraint extraction returned non-JSON output; using empty profile")
        return ConstraintProfile(), conversation_history_text[-200:]

    codes = [c for c in (data.get("test_type_codes") or []) if c in _VALID_TEST_TYPE_CODES]
    profile = ConstraintProfile(
        job_levels=list(data.get("job_levels") or []),
        test_type_codes=codes,
        languages=list(data.get("languages") or []),
        max_duration_minutes=data.get("max_duration_minutes"),
        adaptive_required=data.get("adaptive_required"),
        remote_required=data.get("remote_required"),
    )
    core_need = data.get("core_need") or ""
    return profile, core_need


def missing_info_summary(constraints: ConstraintProfile, core_need: str, latest_user_message: str) -> list[str]:
    """Returns a list of human-readable gaps. Empty list = ready to recommend."""
    if _looks_vague(latest_user_message) and not core_need and constraints.is_empty():
        return ["what role or skill is being assessed"]

    gaps: list[str] = []
    if not core_need:
        gaps.append("what role or skill is being assessed")
    if not constraints.job_levels:
        gaps.append("the seniority / job level")
    return gaps
