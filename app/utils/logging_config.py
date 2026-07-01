
from __future__ import annotations

import logging
import sys
from typing import Any


class KeyValueFormatter(logging.Formatter):
    """Renders log records as key=value pairs for easy grepping."""

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Allow callers to pass structured context via `extra={"context": {...}}`
        context = getattr(record, "context", None)
        if context:
            base.update(context)
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return " ".join(f"{k}={v!r}" for k, v in base.items())


def configure_logging(level: str = "INFO") -> None:
    """Idempotently configure the root logger for the process."""
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. re-imported in tests) - just update level.
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(KeyValueFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party libraries unless we're debugging.
    for noisy in ("httpx", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
