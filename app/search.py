"""SerpAPI integration for external questions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_community.utilities import SerpAPIWrapper

from app.config import Settings


@dataclass(frozen=True)
class SearchEvidence:
    title: str
    url: str
    snippet: str
    source_type: str = "web_search"

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source_type": self.source_type,
        }


class SerpSearchClient:
    """Thin SerpAPI wrapper returning normalized evidence dictionaries."""

    def __init__(self, settings: Settings):
        settings.require_serpapi()
        self.wrapper = SerpAPIWrapper(serpapi_api_key=settings.serpapi_api_key)

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        try:
            raw = self.wrapper.results(query)
        except Exception as exc:
            raise RuntimeError(f"SerpAPI search failed: {exc}") from exc

        organic = raw.get("organic_results") or []
        normalized: list[dict[str, Any]] = []
        for result in organic[:max_results]:
            title = result.get("title") or result.get("source") or "Search result"
            url = result.get("link") or result.get("url") or ""
            snippet = result.get("snippet") or result.get("rich_snippet", "")
            if isinstance(snippet, dict):
                snippet = " ".join(str(v) for v in snippet.values())
            if url or snippet:
                normalized.append(
                    SearchEvidence(
                        title=str(title),
                        url=str(url),
                        snippet=str(snippet),
                    ).as_dict()
                )
        return normalized
