
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.agent.orchestrator import AgentDependencies, handle_chat
from app.api.dependencies import get_agent_dependencies
from app.models.api_models import ChatRequest, ChatResponse, HealthResponse
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    deps: AgentDependencies = Depends(get_agent_dependencies),
) -> ChatResponse:
    try:
        return handle_chat(deps, request.messages)
    except Exception:
        logger.exception("Unhandled error in /chat")
        # Never leak a stack trace or violate the response schema - degrade
        # to a safe, schema-compliant reply instead of a raw 500 where
        # possible, but a genuine 500 is still preferable to a malformed body.
        raise HTTPException(status_code=500, detail="Internal error processing chat request.")
