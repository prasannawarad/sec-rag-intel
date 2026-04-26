import pytest

from src.retrieval.retriever import build_retriever


@pytest.mark.integration
def test_build_retriever_returns_object():
    """Smoke test: retriever can be constructed (requires OPENAI_API_KEY + a populated store)."""
    r = build_retriever(ticker="AAPL", year=2024, filing_type="10-K")
    assert r is not None
