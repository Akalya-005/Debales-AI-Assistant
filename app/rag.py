"""RAG retrieval over the Debales Chroma collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma

from app.config import Settings
from app.embeddings import get_embeddings


@dataclass(frozen=True)
class RagEvidence:
    content: str
    score: float
    title: str
    url: str
    source_type: str = "debales_website"

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type,
        }


class DebalesRetriever:
    """Small wrapper around Chroma retrieval with relevance filtering."""

    def __init__(self, settings: Settings):
        self.settings = settings
        embeddings = get_embeddings(settings)
        self.vector_store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_dir),
        )

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=self.settings.retrieval_top_k
            )
        except Exception as exc:
            raise RuntimeError(
                "Debales vector store retrieval failed. "
                "Run `python scripts/ingest.py` after configuring .env."
            ) from exc

        evidence: list[dict[str, Any]] = []
        for document, score in results:
            if score < self.settings.min_relevance_score:
                continue
            metadata = document.metadata or {}
            item = RagEvidence(
                content=document.page_content,
                score=float(score),
                title=str(metadata.get("title") or "Debales AI"),
                url=str(metadata.get("source_url") or ""),
            )
            evidence.append(item.as_dict())
        return evidence
