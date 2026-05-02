"""Query intent classification for routing the assistant workflow."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.config import Settings


class Intent(str, Enum):
    DEBALES = "debales"
    EXTERNAL = "external"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Classification(BaseModel):
    intent: Intent = Field(description="Routing intent for the user query.")
    rationale: str = Field(description="Short reason for the routing decision.")


DEBALES_KEYWORDS = {
    "debales",
    "debales ai",
    "ai readiness score",
    "messometer",
    "email ai agent",
    "support ai agent",
    "sms ai agent",
    "phone ai agent",
    "crm agent",
    "freight broker",
    "3pl",
    "carrier",
    "logistics team",
    "tai tms",
    "warehouse automation",
}

COMPARISON_MARKERS = {
    "compare",
    "comparison",
    "versus",
    " vs ",
    "better than",
    "different from",
    "alternative",
    "competitor",
}

EXTERNAL_MARKERS = {
    "latest",
    "today",
    "current",
    "news",
    "market",
    "weather",
    "who is",
    "what is",
    "when did",
    "where is",
    "how many",
    "definition",
    "explain",
}

VAGUE_QUERIES = {
    "",
    "hi",
    "hello",
    "hey",
    "help",
    "ok",
    "thanks",
    "thank you",
}


def _normalized(query: str) -> str:
    return re.sub(r"\s+", " ", query or "").strip().lower()


def _contains_any(text: str, markers: set[str]) -> bool:
    return any(marker in text for marker in markers)


def deterministic_classify(query: str) -> Optional[Classification]:
    """Fast, explainable routing for common cases."""

    text = _normalized(query)
    if text in VAGUE_QUERIES or len(text) < 3:
        return Classification(
            intent=Intent.UNKNOWN,
            rationale="The query is empty or too vague to answer safely.",
        )

    has_debales = _contains_any(text, DEBALES_KEYWORDS)
    has_comparison = _contains_any(text, COMPARISON_MARKERS)
    has_external = _contains_any(text, EXTERNAL_MARKERS)

    if has_debales and (has_comparison or has_external):
        return Classification(
            intent=Intent.MIXED,
            rationale="The query mentions Debales and asks for broader context.",
        )

    if has_debales:
        return Classification(
            intent=Intent.DEBALES,
            rationale="The query is about Debales AI or its logistics workflows.",
        )

    if has_external or len(text.split()) >= 3:
        return Classification(
            intent=Intent.EXTERNAL,
            rationale="The query asks for general or external information.",
        )

    return None


def llm_classify(query: str, settings: Settings) -> Classification:
    """Use an LLM fallback when deterministic routing is inconclusive."""

    from app.llm import get_chat_model

    model = get_chat_model(settings)
    if model is None:
        return Classification(
            intent=Intent.UNKNOWN,
            rationale="No API key or chat provider is configured for LLM classification.",
        )
    prompt = (
        "Classify the user query for a Debales AI assistant. "
        "Return only JSON with keys intent and rationale. "
        "intent must be one of: debales, external, mixed, unknown.\n\n"
        "Definitions:\n"
        "- debales: answer from Debales AI website/product/blog evidence.\n"
        "- external: answer from general web search.\n"
        "- mixed: needs both Debales evidence and external context.\n"
        "- unknown: too vague, unsafe, or unanswerable.\n\n"
        f"Query: {query}"
    )
    response = model.invoke(prompt)
    content = getattr(response, "content", "")
    try:
        parsed = json.loads(content)
        return Classification.model_validate(parsed)
    except Exception:
        return Classification(
            intent=Intent.UNKNOWN,
            rationale="LLM classification did not return valid JSON.",
        )


def classify_query(query: str, settings: Optional[Settings] = None) -> Classification:
    """Classify a query using deterministic routing, then optional LLM fallback."""

    deterministic = deterministic_classify(query)
    if deterministic is not None:
        return deterministic

    if settings is None:
        settings = Settings.from_env()

    try:
        return llm_classify(query, settings)
    except Exception as exc:
        return Classification(
            intent=Intent.UNKNOWN,
            rationale=f"Classification failed safely: {exc}",
        )
