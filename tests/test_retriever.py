import pytest

from src.retrieval.retriever import build_retriever


@pytest.mark.integration
def test_build_retriever_returns_object():
    """Smoke test: retriever can be constructed (loads the local BGE model + a populated store)."""
    r = build_retriever(ticker="AAPL", year=2024, filing_type="10-K")
    assert r is not None
