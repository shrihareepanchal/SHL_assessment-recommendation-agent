
from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from app.evaluation.recall import recall_at_k
from app.evaluation.trace_parser import ConversationTrace


@dataclass
class TraceReplayResult:
    trace_id: str
    recall_at_10: float
    schema_violations: list[str] = field(default_factory=list)
    exceeded_turn_cap: bool = False
    final_recommendation_urls: list[str] = field(default_factory=list)


def _validate_schema(body: dict) -> list[str]:
    violations = []
    if not isinstance(body.get("reply"), str):
        violations.append("reply is not a string")
    recs = body.get("recommendations")
    if not isinstance(recs, list):
        violations.append("recommendations is not a list")
    elif not (0 <= len(recs) <= 10):
        violations.append("recommendations length outside [0, 10]")
    else:
        for r in recs:
            if set(r.keys()) - {"name", "url", "test_type"}:
                violations.append(f"recommendation has extra fields: {r.keys()}")
            if not all(k in r for k in ("name", "url", "test_type")):
                violations.append("recommendation missing required field")
    if not isinstance(body.get("end_of_conversation"), bool):
        violations.append("end_of_conversation is not a boolean")
    return violations


def replay_trace(client: httpx.Client, base_url: str, trace: ConversationTrace, max_turns: int = 8) -> TraceReplayResult:
    messages: list[dict] = []
    schema_violations: list[str] = []
    last_urls: list[str] = []
    exceeded = False

    for turn in trace.turns:
        messages.append({"role": "user", "content": turn.user_message})
        if len(messages) > max_turns:
            exceeded = True
            break

        response = client.post(f"{base_url}/chat", json={"messages": messages}, timeout=30)
        response.raise_for_status()
        body = response.json()

        schema_violations.extend(_validate_schema(body))
        messages.append({"role": "assistant", "content": body["reply"]})
        urls = [r["url"] for r in body.get("recommendations", [])]
        if urls:
            last_urls = urls

        if body.get("end_of_conversation"):
            break

    recall = recall_at_k(last_urls, trace.expected_shortlist_urls, k=10)
    return TraceReplayResult(
        trace_id=trace.trace_id,
        recall_at_10=recall,
        schema_violations=schema_violations,
        exceeded_turn_cap=exceeded,
        final_recommendation_urls=last_urls,
    )
