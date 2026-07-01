from __future__ import annotations

from app.agent.conversation_analyzer import ConversationState
from app.catalog.loader import CatalogRepository
from app.models.api_models import ChatResponse
from app.models.catalog_models import Assessment
from app.prompts.templates import COMPARISON_PROMPT
from app.services.llm_service import LLMClient


def _resolve_named_assessments(text: str, catalog: CatalogRepository) -> list[Assessment]:
    
    found: dict[str, Assessment] = {}
    lowered = text.lower()

    for assessment in catalog.all:
        if assessment.name.lower() in lowered:
            found[assessment.id] = assessment

    if len(found) < 2:
        # Try shorter fragments split on common separators (vs, and, comma).
        import re

        fragments = re.split(r"\bvs\.?\b|\bversus\b|,| and ", text, flags=re.IGNORECASE)
        for frag in fragments:
            frag = frag.strip()
            if len(frag) < 2:
                continue
            for match in catalog.fuzzy_find_by_name_fragment(frag, limit=1):
                found[match.id] = match

    return list(found.values())


def handle_compare(llm: LLMClient, catalog: CatalogRepository, state: ConversationState) -> ChatResponse:
    named = _resolve_named_assessments(state.latest_user_message, catalog)

    if len(named) < 2:
        reply = (
            "I can compare specific assessments, but I need at least two names to work with - "
            "which SHL assessments would you like compared?"
        )
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    catalog_context = "\n".join(
        f"- {a.name} | test_type={','.join(a.test_type_codes) or '-'} | duration={a.duration_raw or 'n/a'} | "
        f"adaptive={a.adaptive} | remote={a.remote_testing} | job_levels={', '.join(a.job_levels) or 'n/a'} | "
        f"url={a.url} | description={a.description}"
        for a in named
    )

    reply = llm.generate(
        COMPARISON_PROMPT,
        conversation_history=state.conversation_history_text,
        catalog_context=catalog_context,
        assessment_names=", ".join(a.name for a in named),
    )
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
