from app.config import Settings
from app.graph import DebalesAssistant


class FakeRetriever:
    def retrieve(self, query):
        return [
            {
                "content": "Debales AI automates customer communication and logistics workflows.",
                "score": 0.9,
                "title": "Debales AI",
                "url": "https://debales.ai/",
                "source_type": "debales_website",
            }
        ]


class FakeSearch:
    def search(self, query):
        return [
            {
                "title": "AI in logistics",
                "url": "https://example.com/ai-logistics",
                "snippet": "AI in logistics helps automate routing, forecasting, and support.",
                "source_type": "web_search",
            }
        ]


class FakeMessage:
    content = "Grounded answer with citations."


class FakeLLM:
    def invoke(self, prompt):
        return FakeMessage()


def _settings(tmp_path):
    return Settings(
        openai_api_key="",
        serpapi_api_key="",
        openai_chat_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        embedding_provider="local",
        debales_base_url="https://debales.ai/",
        chroma_dir=tmp_path,
        chroma_collection="test",
        crawl_max_pages=1,
        retrieval_top_k=1,
        min_relevance_score=0.2,
    )


def test_graph_debales_route(tmp_path):
    assistant = DebalesAssistant(
        settings=_settings(tmp_path),
        retriever=FakeRetriever(),
        search_client=FakeSearch(),
        llm=FakeLLM(),
    )
    result = assistant.ask("What does Debales AI automate?")
    assert result["intent"] == "debales"
    assert result["rag_docs"]
    assert result["answer"] == "Grounded answer with citations."


def test_graph_mixed_route(tmp_path):
    assistant = DebalesAssistant(
        settings=_settings(tmp_path),
        retriever=FakeRetriever(),
        search_client=FakeSearch(),
        llm=FakeLLM(),
    )
    result = assistant.ask("Compare Debales AI with general logistics automation.")
    assert result["intent"] == "mixed"
    assert result["rag_docs"]
    assert result["search_results"]


def test_graph_unknown_route(tmp_path):
    assistant = DebalesAssistant(
        settings=_settings(tmp_path),
        retriever=FakeRetriever(),
        search_client=FakeSearch(),
        llm=FakeLLM(),
    )
    result = assistant.ask("")
    assert result["intent"] == "unknown"
    assert "I don't know" in result["answer"]
