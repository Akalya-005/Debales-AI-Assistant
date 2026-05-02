import pytest

from app.config import Settings
from app.search import SerpSearchClient


def test_serpapi_missing_key_raises(tmp_path):
    settings = Settings(
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
    with pytest.raises(RuntimeError):
        SerpSearchClient(settings)
