"""LangGraph workflow for the Debales AI assistant."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.classifier import Intent, classify_query
from app.config import Settings
from app.rag import DebalesRetriever
from app.search import SerpSearchClient


class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    rationale: str
    rag_docs: list[dict[str, Any]]
    search_results: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, str]]
    errors: list[str]


def _append_error(state: AssistantState, message: str) -> list[str]:
    return [*state.get("errors", []), message]


def _unique_sources(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    sources: list[dict[str, str]] = []
    for item in items:
        url = str(item.get("url") or "")
        title = str(item.get("title") or url or "Source")
        source_type = str(item.get("source_type") or "source")
        key = url or title
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append({"title": title, "url": url, "source_type": source_type})
    return sources


def _format_evidence(state: AssistantState) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(state.get("rag_docs", []), start=1):
        lines.append(
            f"[Debales {idx}] {doc.get('title')} ({doc.get('url')})\n"
            f"{doc.get('content')}"
        )
    for idx, result in enumerate(state.get("search_results", []), start=1):
        lines.append(
            f"[Web {idx}] {result.get('title')} ({result.get('url')})\n"
            f"{result.get('snippet')}"
        )
    return "\n\n".join(lines)


def _fallback_answer(state: AssistantState) -> str:
    evidence = state.get("rag_docs", []) + state.get("search_results", [])
    if not evidence:
        return "I don't have enough information to answer this."

    snippets: list[str] = []
    for item in evidence[:3]:
        text = item.get("content") or item.get("snippet") or ""
        if text:
            snippets.append(" ".join(str(text).split())[:350])
    if not snippets:
        return "I don't have enough information to answer this."
    return " ".join(snippets)


class DebalesAssistant:
    """Builds and runs the LangGraph assistant."""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: Any | None = None,
        search_client: Any | None = None,
        llm: Any | None = None,
    ):
        self.settings = settings or Settings.from_env()
        self._retriever = retriever
        self._search_client = search_client
        self._llm = llm
        self.graph = self._build_graph()

    @property
    def retriever(self) -> DebalesRetriever:
        if self._retriever is None:
            self._retriever = DebalesRetriever(self.settings)
        return self._retriever

    @property
    def search_client(self) -> SerpSearchClient:
        if self._search_client is None:
            self._search_client = SerpSearchClient(self.settings)
        return self._search_client

    @property
    def llm(self) -> Any | None:
        if self._llm is not None:
            return self._llm
        from app.llm import get_chat_model

        self._llm = get_chat_model(self.settings)
        return self._llm

    def _build_graph(self):
        builder = StateGraph(AssistantState)
        builder.add_node("input_node", self.input_node)
        builder.add_node("classify_intent", self.classify_intent)
        builder.add_node("rag_retrieval", self.rag_retrieval)
        builder.add_node("serp_search", self.serp_search)
        builder.add_node("response_generator", self.response_generator)
        builder.add_node("final_answer", self.final_answer)

        builder.add_edge(START, "input_node")
        builder.add_edge("input_node", "classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            self.route_after_classification,
            {
                "rag_retrieval": "rag_retrieval",
                "serp_search": "serp_search",
                "final_answer": "final_answer",
            },
        )
        builder.add_conditional_edges(
            "rag_retrieval",
            self.route_after_rag,
            {
                "serp_search": "serp_search",
                "response_generator": "response_generator",
            },
        )
        builder.add_edge("serp_search", "response_generator")
        builder.add_edge("response_generator", "final_answer")
        builder.add_edge("final_answer", END)
        return builder.compile()

    def input_node(self, state: AssistantState) -> AssistantState:
        query = " ".join(str(state.get("query", "")).split())
        return {
            "query": query,
            "rag_docs": state.get("rag_docs", []),
            "search_results": state.get("search_results", []),
            "errors": state.get("errors", []),
        }

    def classify_intent(self, state: AssistantState) -> AssistantState:
        classification = classify_query(state.get("query", ""), self.settings)
        return {
            "intent": classification.intent.value,
            "rationale": classification.rationale,
        }

    def route_after_classification(
        self, state: AssistantState
    ) -> Literal["rag_retrieval", "serp_search", "final_answer"]:
        intent = state.get("intent")
        if intent in {Intent.DEBALES.value, Intent.MIXED.value}:
            return "rag_retrieval"
        if intent == Intent.EXTERNAL.value:
            return "serp_search"
        return "final_answer"

    def rag_retrieval(self, state: AssistantState) -> AssistantState:
        try:
            docs = self.retriever.retrieve(state["query"])
            return {"rag_docs": docs}
        except Exception as exc:
            return {"rag_docs": [], "errors": _append_error(state, str(exc))}

    def route_after_rag(
        self, state: AssistantState
    ) -> Literal["serp_search", "response_generator"]:
        if state.get("intent") == Intent.MIXED.value:
            return "serp_search"
        return "response_generator"

    def serp_search(self, state: AssistantState) -> AssistantState:
        try:
            results = self.search_client.search(state["query"])
            return {"search_results": results}
        except Exception as exc:
            return {"search_results": [], "errors": _append_error(state, str(exc))}

    def response_generator(self, state: AssistantState) -> AssistantState:
        evidence_items = state.get("rag_docs", []) + state.get("search_results", [])
        sources = _unique_sources(evidence_items)
        if not evidence_items:
            return {
                "answer": "I don't have enough information to answer this.",
                "sources": sources,
            }

        model = self.llm
        if model is None:
            return {"answer": _fallback_answer(state), "sources": sources}

        evidence = _format_evidence(state)
        intent = state.get('intent', 'unknown')
        
        prompt = (
            "You are a senior-level Debales AI assistant. Your goal is to provide highly confident, structured, and accurate answers.\n\n"
            "### STRICT NO-HALLUCINATION GUARD:\n"
            "If the provided evidence does not contain sufficient information to answer the question, you MUST return exactly and only this string:\n"
            "'I don't have enough information to answer this.'\n"
            "Do not invent, guess, or extrapolate Debales claims.\n\n"
            "### RESPONSE QUALITY RULES:\n"
            "- DO NOT use weak phrases like 'Based on available evidence' or 'The evidence suggests'. State facts confidently.\n"
            "- Keep answers concise, clear, and professional.\n"
            "- Cite source URLs inline when making factual claims.\n\n"
            "### MIXED QUERY HANDLING:\n"
            f"If the query intent is '{Intent.MIXED.value}', structure your answer exactly with these three headings:\n"
            "### Debales AI\n<Synthesize Debales capabilities from evidence>\n"
            "### Traditional/External\n<Synthesize External context from evidence>\n"
            "### Conclusion\n<Provide a concise summary linking both>\n\n"
            f"Intent: {intent}\n"
            f"User question: {state['query']}\n\n"
            f"Evidence:\n{evidence}"
        )
        try:
            response = model.invoke(prompt)
            answer = str(getattr(response, "content", "")).strip()
        except Exception as exc:
            answer = _fallback_answer(state)
            return {
                "answer": answer,
                "sources": sources,
                "errors": _append_error(state, f"LLM generation failed: {exc}"),
            }
        if not answer:
            answer = "I don't have enough information to answer this."
        return {"answer": answer, "sources": sources}

    def final_answer(self, state: AssistantState) -> AssistantState:
        if state.get("intent") == Intent.UNKNOWN.value:
            return {
                "answer": "I don't know. Please ask a specific Debales AI or general question.",
                "sources": [],
            }
        if not state.get("answer"):
            return {
                "answer": "I don't have enough information to answer this.",
                "sources": state.get("sources", []),
            }
        return state

    def ask(self, query: str) -> AssistantState:
        return self.graph.invoke({"query": query})
