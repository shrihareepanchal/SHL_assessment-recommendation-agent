from __future__ import annotations


def test_full_conversation_stays_under_turn_cap(app_client):
    messages = [{"role": "user", "content": "We need a solution for senior leadership."}]
    turns = 0
    max_turns = 8

    follow_ups = [
        "The pool is CXOs and directors with 15+ years experience.",
        "Selection, comparing candidates against a leadership benchmark.",
        "Perfect, that's what we need.",
    ]

    for follow_up in follow_ups:
        response = app_client.post("/chat", json={"messages": messages})
        assert response.status_code == 200
        body = response.json()
        turns += 1
        messages.append({"role": "assistant", "content": body["reply"]})
        assert len(messages) <= max_turns
        if body["end_of_conversation"]:
            break
        messages.append({"role": "user", "content": follow_up})

    assert turns <= max_turns


def test_refinement_does_not_restart_conversation(app_client):
    messages = [
        {"role": "user", "content": "Hiring a mid-level Java developer who collaborates with stakeholders."}
    ]
    first = app_client.post("/chat", json={"messages": messages}).json()
    assert first["recommendations"], "expected an initial shortlist"

    messages.append({"role": "assistant", "content": first["reply"]})
    messages.append({"role": "user", "content": "Actually, also add a personality assessment."})
    second = app_client.post("/chat", json={"messages": messages}).json()

    # Refinement should still produce a shortlist (not fall back to
    # clarification) since context was already established.
    assert second["recommendations"]
