from __future__ import annotations

from app.agent.conversation_analyzer import ConversationState
from app.agent.validator import assessments_to_validated_recommendations
from app.catalog.loader import CatalogRepository
from app.config.settings import Settings
from app.models.api_models import ChatResponse
from app.prompts.templates import RECOMMENDATION_PROMPT, REFINEMENT_PROMPT
from app.ranking.metadata_scorer import ConstraintProfile
from app.retriever.hybrid_retriever import HybridRetriever
from app.services.llm_service import LLMClient


def _build_catalog_context(scored_assessments) -> str:
    """Compact, LLM-friendly rendering of retrieved assessments - this is
    the ONLY source of truth the generation prompts are allowed to draw
    assessment facts from."""
    lines = []
    for s in scored_assessments:
        a = s.assessment
        lines.append(
            f"- {a.name} | test_type={','.join(a.test_type_codes) or '-'} | "
            f"duration={a.duration_raw or 'n/a'} | adaptive={a.adaptive} | "
            f"remote={a.remote_testing} | job_levels={', '.join(a.job_levels) or 'n/a'} | "
            f"url={a.url} | description={a.description[:220]}"
        )
    return "\n".join(lines) if lines else "(no matching assessments found)"


def handle_recommend_or_refine(
    llm: LLMClient,
    retriever: HybridRetriever,
    catalog: CatalogRepository,
    settings: Settings,
    state: ConversationState,
    constraints: ConstraintProfile,
    core_need: str,
    is_refinement: bool,
) -> ChatResponse:
    query_text = core_need or state.latest_user_message
    scored = retriever.retrieve(
        query_text=query_text,
        constraints=constraints,
        final_k=settings.retrieval_final_k,
    )

    recommendations = assessments_to_validated_recommendations(
        assessments=[s.assessment for s in scored],
        catalog=catalog,
        min_count=settings.min_recommendations,
        max_count=settings.max_recommendations,
    )

    catalog_context = _build_catalog_context(scored)

    if not recommendations:
        # No confident match - be honest instead of forcing a bad shortlist.
        reply = (
            "I couldn't find an SHL assessment in the catalog that confidently matches "
            "what you've described. Could you share more about the specific role or skills "
            "you want to assess?"
        )
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    if is_refinement:
        reply = llm.generate(
            REFINEMENT_PROMPT,
            conversation_history=state.conversation_history_text,
            catalog_context=catalog_context,
            change_summary=state.latest_user_message,
        )
    else:
        reply = llm.generate(
            RECOMMENDATION_PROMPT,
            conversation_history=state.conversation_history_text,
            catalog_context=catalog_context,
            constraints_summary=str(constraints),
        )

    return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=False)
