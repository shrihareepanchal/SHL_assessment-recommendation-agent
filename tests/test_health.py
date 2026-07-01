from __future__ import annotations


def test_health_returns_ok(app_client):
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
