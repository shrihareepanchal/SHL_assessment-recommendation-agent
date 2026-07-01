from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Point settings at the real bundled catalog for every test run, and force a
# deterministic, no-network LLM client (EchoFallbackLLMClient) unless a test
# explicitly overrides GOOGLE_API_KEY.
os.environ.setdefault("CATALOG_PROCESSED_PATH", str(PROJECT_ROOT / "data" / "catalog_processed.json"))
os.environ.setdefault("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db_test"))
os.environ.setdefault("GOOGLE_API_KEY", "")


@pytest.fixture(scope="session")
def catalog_path() -> str:
    return str(PROJECT_ROOT / "data" / "catalog_processed.json")


@pytest.fixture()
def app_client():
    """FastAPI TestClient wired against the real app + real catalog, but the
    fallback (non-network) LLM client - fast, deterministic, CI-safe."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client
