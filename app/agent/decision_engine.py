
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.agent.intent_detector import Intent


class Action(str, Enum):
    REFUSE = "refuse"
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"


_CONFIRMATION_PATTERNS = [
    r"\bperfect\b",
    r"\bgreat,?\s*thanks\b",
    r"\bthat('s| is) (what we need|it|great|good|correct)\b",
    r"\bsounds good\b",
    r"\blooks good\b",
    r"\bthat works\b",
    r"\byes,? (that|exactly)\b",
    r"\bconfirmed?\b",
]


def _is_confirmation(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _CONFIRMATION_PATTERNS)


@dataclass
class Decision:
    action: Action
    end_of_conversation: bool
    reason: str


def decide(
    intent: Intent,
    missing_info_gaps: list[str],
    latest_user_message: str,
    has_prior_recommendations: bool,
    turn_count: int,
    max_turns: int,
) -> Decision:
    """`turn_count` is the number of messages already in the request
    (i.e. including the latest user turn, but NOT the reply we're about to
    produce). If adding our reply would hit `max_turns`, we must commit to a
    shortlist now rather than asking another clarifying question."""
    is_last_turn_available = (turn_count + 1) >= max_turns

    if intent is Intent.REFUSE:
        return Decision(Action.REFUSE, end_of_conversation=False, reason="off_topic_or_injection")

    if intent is Intent.COMPARE:
        return Decision(Action.COMPARE, end_of_conversation=False, reason="comparison_requested")

    if intent is Intent.REFINE:
        end = has_prior_recommendations and _is_confirmation(latest_user_message)
        return Decision(Action.REFINE, end_of_conversation=end, reason="refinement_turn")

    # Intent.RECOMMEND_OR_CLARIFY
    if missing_info_gaps and not is_last_turn_available and not has_prior_recommendations:
        return Decision(Action.CLARIFY, end_of_conversation=False, reason="insufficient_context")

    end = is_last_turn_available or _is_confirmation(latest_user_message)
    return Decision(Action.RECOMMEND, end_of_conversation=end, reason="sufficient_context")
