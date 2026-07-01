
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class Embedder:

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        logger.info("Loading embedding model", extra={"context": {"model": model_name}})
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """bge models recommend no special prefix for the corpus/passage side."""
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )

    def embed_query(self, text: str) -> np.ndarray:
        """bge models recommend an instruction prefix on the query side for
        asymmetric search - this measurably improves retrieval quality over
        embedding the query "bare"."""
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        return self._model.encode(
            [prefixed], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )[0]


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu") -> Embedder:
    """Process-wide singleton - loading the model is the expensive part."""
    return Embedder(model_name=model_name, device=device)
