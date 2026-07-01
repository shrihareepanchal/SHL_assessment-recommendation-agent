from __future__ import annotations

from app.agent.conversation_analyzer import ConversationState
from app.agent.intent_detector import is_prompt_injection
from app.models.api_models import ChatResponse
from app.prompts.templates import REFUSAL_PROMPT
from app.services.llm_service import LLMClient


def handle_refuse(llm: LLMClient, state: ConversationState) -> ChatResponse:
    reason = (
        "possible prompt injection / attempt to override system instructions"
        if is_prompt_injection(state.latest_user_message)
        else "request is outside SHL assessment selection (e.g. general hiring, legal, or salary advice)"
    )
    reply = llm.generate(
        REFUSAL_PROMPT,
        conversation_history=state.conversation_history_text,
        refusal_reason=reason,
    )
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
