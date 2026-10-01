import pytest
from langchain_core.documents import Document

from src.retrieval.retriever import (
    COMPANY_ALIASES,
    MAX_ROUTED_CHUNKS,
    CompanyRoutingRetriever,
    build_retriever,
    detect_tickers,
)


@pytest.mark.integration
def test_build_retriever_returns_object():
    """Smoke test: retriever can be constructed (loads the local BGE model + a populated store)."""
    r = build_retriever(ticker="AAPL", year=2024, filing_type="10-K")
    assert r is not None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Compare the R&D spending of Apple and Microsoft.", ["AAPL", "MSFT"]),
        ("Compare Tesla's and NVIDIA's gross margin trends.", ["TSLA", "NVDA"]),
        ("What does Alphabet say about ad revenue?", ["GOOGL"]),
        ("How fast is AWS growing at Amazon?", ["AMZN"]),
        ("Which companies in the dataset call out generative AI?", list(COMPANY_ALIASES)),
        (
            "What stock repurchase programmes are active across the eight companies?",
            list(COMPANY_ALIASES),
        ),
        ("What cybersecurity incident disclosures appear in the dataset?", list(COMPANY_ALIASES)),
        ("How is metadata filtering used in retrieval?", []),  # "meta" must be a whole word
    ],
)
def test_detect_tickers(question, expected):
    assert detect_tickers(question) == expected


class _FakeStore:
    def __init__(self):
        self.calls = []

    def max_marginal_relevance_search(self, query, k, fetch_k, filter=None):
        self.calls.append((k, filter))
        return [Document(page_content=f"{filter}-{i}") for i in range(k)]


def _routing_retriever(monkeypatch, base_filter=None):
    store = _FakeStore()
    monkeypatch.setattr("src.retrieval.retriever.get_vectorstore", lambda: store)
    return CompanyRoutingRetriever(base_filter=base_filter or {}, k=5, fetch_k=20), store


def test_routing_two_companies_searches_each(monkeypatch):
    r, store = _routing_retriever(monkeypatch, {"filing_type": "10-K"})
    docs = r.invoke("Compare Apple and Microsoft R&D")
    assert len(docs) == 6  # 3 per company
    assert [c[1] for c in store.calls] == [
        {"$and": [{"filing_type": {"$eq": "10-K"}}, {"ticker": {"$eq": t}}]}
        for t in ("AAPL", "MSFT")
    ]


def test_routing_all_companies_caps_total_chunks(monkeypatch):
    r, store = _routing_retriever(monkeypatch)
    docs = r.invoke("Which companies disclose China risk?")
    assert len(store.calls) == 8 and len(docs) == MAX_ROUTED_CHUNKS


def test_routing_no_company_is_plain_mmr(monkeypatch):
    r, store = _routing_retriever(monkeypatch)
    assert len(r.invoke("What are common risk factors?")) == 5
    assert store.calls == [(5, None)]
