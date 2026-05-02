"""Embedding provider selection.

The OpenAI embedding model is best for production quality. The local hashing
embedding is deterministic and quota-free, which is useful for demos or when an
OpenAI account has no embedding quota.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from langchain_core.embeddings import Embeddings

from app.config import Settings


class HashingEmbeddings(Embeddings):
    """Small deterministic embedding model with no external API calls."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "local":
        return HashingEmbeddings()
    if settings.embedding_provider != "openai":
        raise ValueError(
            "EMBEDDING_PROVIDER must be either 'openai' or 'local', "
            f"got {settings.embedding_provider!r}"
        )

    settings.require_openai("embedding Debales website content")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
