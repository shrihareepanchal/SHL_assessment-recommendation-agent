from __future__ import annotations

from app.prompts import templates


def test_clarification_prompt_renders_with_required_vars():
    rendered = templates.CLARIFICATION_PROMPT.format(
        conversation_history="user: We need a solution for senior leadership.",
        missing_info="seniority level",
    )
    assert "seniority level" in rendered
    assert "ONE" in rendered  # enforces the "one high-value question" rule


def test_recommendation_prompt_includes_grounding_rules():
    rendered = templates.RECOMMENDATION_PROMPT.format(
        conversation_history="...",
        catalog_context="- Java 8 (New) | test_type=K | url=https://www.shl.com/x",
        constraints_summary="job_levels=Mid-Professional",
    )
    assert "Never invent" in rendered or "never invent" in rendered.lower()
    assert "url" in rendered.lower()


def test_comparison_prompt_forbids_fabricated_differences():
    rendered = templates.COMPARISON_PROMPT.format(
        conversation_history="...",
        catalog_context="- OPQ32r | ...",
        assessment_names="OPQ32r, GSA",
    )
    assert "never invent" in rendered.lower()


def test_refusal_prompt_asks_for_single_sentence_redirect():
    rendered = templates.REFUSAL_PROMPT.format(
        conversation_history="...",
        refusal_reason="salary advice is out of scope",
    )
    assert "ONE sentence" in rendered


def test_extraction_prompt_requires_strict_json_keys():
    rendered = templates.EXTRACTION_PROMPT.format(conversation_history="user: hiring a java dev")
    for key in ("job_levels", "test_type_codes", "languages", "max_duration_minutes", "core_need"):
        assert key in rendered
