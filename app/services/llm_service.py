
from __future__ import annotations

from typing import Protocol

from langchain_core.prompts import PromptTemplate

from app.config.settings import Settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class LLMClient(Protocol):
    def generate(self, prompt_template: PromptTemplate, **kwargs: str) -> str:
        """Render `prompt_template` with `kwargs` and return the raw text completion."""
        ...


class GeminiLLMClient:
    """Google Gemini adapter, via LangChain's `ChatGoogleGenerativeAI`."""

    def __init__(self, settings: Settings):
        # Imported lazily so environments that only run the retrieval/agent
        # unit tests (no GOOGLE_API_KEY) don't need the google SDK installed.
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            timeout=settings.llm_request_timeout_seconds,
        )

    def generate(self, prompt_template: PromptTemplate, **kwargs: str) -> str:
        chain = prompt_template | self._model
        result = chain.invoke(kwargs)
        return (result.content or "").strip()


class EchoFallbackLLMClient:
    """Deterministic, no-network fallback used when no LLM provider is
    configured (e.g. local dev without an API key, or CI). Lets the
    retrieval/decision/API layers be tested end-to-end without a live LLM
    call, and keeps the service *available* (never 500s) if the LLM
    provider has an outage - degraded, not down."""

    def generate(self, prompt_template: PromptTemplate, **kwargs: str) -> str:
        logger.warning("Using EchoFallbackLLMClient - no LLM provider configured")
        if "missing_info" in kwargs:
            return "Could you tell me a bit more about the role - what level and what skills matter most?"
        if "assessment_names" in kwargs:
            return f"Here's what the catalog shows for {kwargs.get('assessment_names', 'those assessments')}."
        if "refusal_reason" in kwargs:
            return "I can only help with selecting SHL assessments - what role are you hiring for?"
        return "Here are assessments that match what you've described so far."


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "gemini" and settings.google_api_key:
        try:
            return GeminiLLMClient(settings)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to initialize Gemini client, falling back")
    return EchoFallbackLLMClient()
