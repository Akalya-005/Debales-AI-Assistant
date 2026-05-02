"""Configuration loading for the Debales AI assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


def _str_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int_env(name: str, default: int) -> int:
    value = _str_env(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _float_env(name: str, default: float) -> float:
    value = _str_env(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    openai_api_key: str
    serpapi_api_key: str
    openai_chat_model: str
    openai_embedding_model: str
    chat_provider: str
    embedding_provider: str
    groq_api_key: str
    google_api_key: str
    debales_base_url: str
    chroma_dir: Path
    chroma_collection: str
    crawl_max_pages: int
    retrieval_top_k: int
    min_relevance_score: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            openai_api_key=_str_env("OPENAI_API_KEY"),
            serpapi_api_key=_str_env("SERPAPI_API_KEY"),
            openai_chat_model=_str_env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            openai_embedding_model=_str_env(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            chat_provider=_str_env("CHAT_PROVIDER", "openai").lower(),
            embedding_provider=_str_env("EMBEDDING_PROVIDER", "openai").lower(),
            groq_api_key=_str_env("GROQ_API_KEY"),
            google_api_key=_str_env("GOOGLE_API_KEY"),
            debales_base_url=_str_env("DEBALES_BASE_URL", "https://debales.ai/"),
            chroma_dir=Path(_str_env("CHROMA_DIR", "./data/chroma")),
            chroma_collection=_str_env("CHROMA_COLLECTION", "debales_ai_docs"),
            crawl_max_pages=_int_env("CRAWL_MAX_PAGES", 50),
            retrieval_top_k=_int_env("RETRIEVAL_TOP_K", 5),
            min_relevance_score=_float_env("MIN_RELEVANCE_SCORE", 0.20),
        )

    def require_openai(self, purpose: str) -> None:
        if self.embedding_provider == "local" and "embedding" in purpose.lower():
            return
        if not self.openai_api_key:
            raise RuntimeError(
                f"OPENAI_API_KEY is required for {purpose}. "
                "Copy .env.example to .env and set OPENAI_API_KEY."
            )

    def require_serpapi(self) -> None:
        if not self.serpapi_api_key:
            raise RuntimeError(
                "SERPAPI_API_KEY is required for web search. "
                "Copy .env.example to .env and set SERPAPI_API_KEY."
            )
