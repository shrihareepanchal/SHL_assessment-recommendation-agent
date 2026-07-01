from __future__ import annotations

from app.agent.conversation_analyzer import ConversationState
from app.models.api_models import ChatResponse
from app.prompts.templates import CLARIFICATION_PROMPT
from app.services.llm_service import LLMClient


def handle_clarify(llm: LLMClient, state: ConversationState, missing_info_gaps: list[str]) -> ChatResponse:
    reply = llm.generate(
        CLARIFICATION_PROMPT,
        conversation_history=state.conversation_history_text,
        missing_info=", ".join(missing_info_gaps),
    )
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
