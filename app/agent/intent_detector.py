
from __future__ import annotations

import re
from enum import Enum

from app.catalog.loader import CatalogRepository


class Intent(str, Enum):
    REFUSE = "refuse"
    COMPARE = "compare"
    REFINE = "refine"
    RECOMMEND_OR_CLARIFY = "recommend_or_clarify"


_OFF_TOPIC_PATTERNS = [
    r"\blegal advice\b",
    r"\bsue\b|\blawsuit\b",
    r"\bsalary\b|\bcompensation\b|\bpay\s+range\b",
    r"\bwrite (my |a )?(job description|offer letter|termination letter)\b",
    r"\bhow (do|should) i fire\b",
    r"\bperformance improvement plan\b",
    r"\bimmigration\b|\bvisa sponsorship\b",
    r"\bwhat stock\b|\binvest\b",
    r"\bwrite (me )?(a poem|code|an essay)\b",
    r"\bweather\b",
]

_INJECTION_PATTERNS = [
    r"ignore (all |your )?(previous|prior|above) instructions",
    r"system prompt",
    r"you are now",
    r"disregard (all|your) (rules|instructions|guidelines)",
    r"act as (an? )?(unrestricted|jailbroken|dan)\b",
    r"reveal your (prompt|instructions)",
    r"pretend (you|to) (have no|ignore)",
]

_COMPARE_PATTERNS = [
    r"\bcompare\b",
    r"\bdifference between\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"which (one|is better)\b",
]

_REFINE_PATTERNS = [
    r"^actually\b",
    r"\binstead\b",
    r"\balso add\b",
    r"\bremove\b",
    r"\bcan you (also|instead)\b",
    r"\bchange (that|it|to)\b",
    r"\bnarrow (it |that )?down\b",
    r"\bwhat about\b",
    r"\bswap\b",
]


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def is_prompt_injection(text: str) -> bool:
    return _matches_any(_INJECTION_PATTERNS, text)


def is_off_topic(text: str) -> bool:
    return _matches_any(_OFF_TOPIC_PATTERNS, text)


def mentions_two_or_more_known_assessments(text: str, catalog: CatalogRepository) -> bool:
    """Comparison intent is strongest when the user names specific assessments
    (not just says "compare") - used to build a grounded catalog_context."""
    hits = 0
    lowered = text.lower()
    for assessment in catalog.all:
        # Cheap guard: only check reasonably short/specific names to avoid
        # matching on generic substrings.
        name = assessment.name.lower()
        if len(name) >= 4 and name in lowered:
            hits += 1
        if hits >= 2:
            return True
    return False


def detect_intent(
    latest_user_message: str,
    catalog: CatalogRepository,
    has_prior_assistant_turns: bool,
) -> Intent:
    if is_prompt_injection(latest_user_message) or is_off_topic(latest_user_message):
        return Intent.REFUSE

    if _matches_any(_COMPARE_PATTERNS, latest_user_message) or mentions_two_or_more_known_assessments(
        latest_user_message, catalog
    ):
        return Intent.COMPARE

    if has_prior_assistant_turns and _matches_any(_REFINE_PATTERNS, latest_user_message):
        return Intent.REFINE

    return Intent.RECOMMEND_OR_CLARIFY
