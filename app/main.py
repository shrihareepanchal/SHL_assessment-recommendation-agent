"""FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload

Run in a container with:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import get_agent_dependencies
from app.api.routes import router
from app.config.settings import get_settings
from app.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting up - warming catalog, retriever, and LLM client")
    # Building this eagerly at startup (rather than lazily on first request)
    # avoids paying the embedding-model load + index-build cost inside a
    # user-facing request's 30s timeout budget.
    get_agent_dependencies()
    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
