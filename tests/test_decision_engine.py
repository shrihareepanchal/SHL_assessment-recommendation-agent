from __future__ import annotations

from app.agent.decision_engine import Action, decide
from app.agent.intent_detector import Intent


def test_vague_first_turn_triggers_clarify():
    decision = decide(
        intent=Intent.RECOMMEND_OR_CLARIFY,
        missing_info_gaps=["what role or skill is being assessed"],
        latest_user_message="I need an assessment",
        has_prior_recommendations=False,
        turn_count=1,
        max_turns=8,
    )
    assert decision.action == Action.CLARIFY
    assert decision.end_of_conversation is False


def test_sufficient_context_triggers_recommend_without_ending():
    decision = decide(
        intent=Intent.RECOMMEND_OR_CLARIFY,
        missing_info_gaps=[],
        latest_user_message="Mid-level Java developer who works with stakeholders",
        has_prior_recommendations=False,
        turn_count=2,
        max_turns=8,
    )
    assert decision.action == Action.RECOMMEND
    assert decision.end_of_conversation is False


def test_confirmation_after_shortlist_ends_conversation():
    decision = decide(
        intent=Intent.REFINE,
        missing_info_gaps=[],
        latest_user_message="Perfect, that's what we need.",
        has_prior_recommendations=True,
        turn_count=4,
        max_turns=8,
    )
    assert decision.action == Action.REFINE
    assert decision.end_of_conversation is True


def test_turn_cap_forces_recommendation_over_clarification():
    decision = decide(
        intent=Intent.RECOMMEND_OR_CLARIFY,
        missing_info_gaps=["seniority"],
        latest_user_message="Still fairly vague request",
        has_prior_recommendations=False,
        turn_count=7,
        max_turns=8,
    )
    assert decision.action == Action.RECOMMEND
    assert decision.end_of_conversation is True


def test_off_topic_triggers_refuse():
    decision = decide(
        intent=Intent.REFUSE,
        missing_info_gaps=[],
        latest_user_message="What salary should I offer?",
        has_prior_recommendations=False,
        turn_count=1,
        max_turns=8,
    )
    assert decision.action == Action.REFUSE


def test_comparison_intent_triggers_compare():
    decision = decide(
        intent=Intent.COMPARE,
        missing_info_gaps=[],
        latest_user_message="What is the difference between OPQ and GSA?",
        has_prior_recommendations=False,
        turn_count=1,
        max_turns=8,
    )
    assert decision.action == Action.COMPARE
