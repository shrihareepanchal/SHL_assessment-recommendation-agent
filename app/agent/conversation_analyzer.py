
from __future__ import annotations

from dataclasses import dataclass

from app.models.api_models import ChatMessage, Role


@dataclass
class ConversationState:
    messages: list[ChatMessage]
    turn_count: int
    latest_user_message: str
    conversation_history_text: str
    has_prior_assistant_turns: bool
    is_final_allowed_turn: bool


def analyze(messages: list[ChatMessage], max_turns: int) -> ConversationState:
    turn_count = len(messages)
    latest_user_message = messages[-1].content
    history_lines = [f"{m.role.value}: {m.content}" for m in messages]

    return ConversationState(
        messages=messages,
        turn_count=turn_count,
        latest_user_message=latest_user_message,
        conversation_history_text="\n".join(history_lines),
        has_prior_assistant_turns=any(m.role == Role.assistant for m in messages),
        is_final_allowed_turn=turn_count >= max_turns,
    )
