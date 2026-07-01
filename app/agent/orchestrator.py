
from __future__ import annotations

from dataclasses import dataclass

from app.agent import conversation_analyzer, intent_detector, slot_filling
from app.agent.decision_engine import Action, decide
from app.catalog.loader import CatalogRepository
from app.config.settings import Settings
from app.models.api_models import ChatMessage, ChatResponse
from app.retriever.hybrid_retriever import HybridRetriever
from app.services import comparison_service, recommendation_service, refusal_service
from app.services.clarification_service import handle_clarify
from app.services.llm_service import LLMClient
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AgentDependencies:
    llm: LLMClient
    retriever: HybridRetriever
    catalog: CatalogRepository
    settings: Settings


def handle_chat(deps: AgentDependencies, messages: list[ChatMessage]) -> ChatResponse:
    settings = deps.settings

    # 1. Conversation Analyzer
    state = conversation_analyzer.analyze(messages, max_turns=settings.max_conversation_turns)

    # 2. Intent Detection
    intent = intent_detector.detect_intent(
        state.latest_user_message, deps.catalog, state.has_prior_assistant_turns
    )

    # 3. Missing Information Detector (skip the LLM call entirely for
    #    REFUSE/COMPARE - constraint extraction is irrelevant there, and
    #    skipping it keeps latency down against the 30s per-call budget).
    if intent in (intent_detector.Intent.REFUSE, intent_detector.Intent.COMPARE):
        constraints, core_need, gaps = None, "", []
    else:
        constraints, core_need = slot_filling.extract_constraints(deps.llm, state.conversation_history_text)
        gaps = slot_filling.missing_info_summary(constraints, core_need, state.latest_user_message)

    # 4. Decision Engine
    decision = decide(
        intent=intent,
        missing_info_gaps=gaps,
        latest_user_message=state.latest_user_message,
        has_prior_recommendations=state.has_prior_assistant_turns,
        turn_count=state.turn_count,
        max_turns=settings.max_conversation_turns,
    )
    logger.info(
        "Decision made",
        extra={"context": {"intent": intent.value, "action": decision.action.value, "reason": decision.reason}},
    )

    # 5-7. Retriever -> Recommendation Generator -> Validator (delegated to
    # the relevant service; each service already returns a fully-formed,
    # schema-valid ChatResponse - see app/services/*).
    if decision.action is Action.REFUSE:
        response = refusal_service.handle_refuse(deps.llm, state)
    elif decision.action is Action.COMPARE:
        response = comparison_service.handle_compare(deps.llm, deps.catalog, state)
    elif decision.action is Action.CLARIFY:
        response = handle_clarify(deps.llm, state, gaps)
    else:  # RECOMMEND or REFINE
        response = recommendation_service.handle_recommend_or_refine(
            llm=deps.llm,
            retriever=deps.retriever,
            catalog=deps.catalog,
            settings=settings,
            state=state,
            constraints=constraints,
            core_need=core_need,
            is_refinement=(decision.action is Action.REFINE),
        )

    # 8. JSON Formatter - the decision engine is the single source of truth
    #    for end_of_conversation, overriding whatever the service defaulted to.
    return response.model_copy(update={"end_of_conversation": decision.end_of_conversation})
