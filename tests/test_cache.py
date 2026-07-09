"""Unit tests for the disk answer cache (no network, no model load)."""

from __future__ import annotations

from src.chain.cache import AnswerCache

MODEL = "llama-3.3-70b-versatile"
FILTERS = {"ticker": "AAPL", "year": 2025, "filing_type": "10-K"}
SOURCES = [{"ticker": "AAPL", "year": 2025, "filing_type": "10-K", "section_label": "Item 1A"}]


def make_cache(tmp_path, **kwargs) -> AnswerCache:
    return AnswerCache(tmp_path / "answers.json", **kwargs)


def test_miss_returns_none(tmp_path):
    assert make_cache(tmp_path).get("What are the risks?", FILTERS, MODEL) is None


def test_set_then_get_roundtrip(tmp_path):
    cache = make_cache(tmp_path)
    cache.set("What are the risks?", FILTERS, MODEL, answer="Supply chain.", sources=SOURCES)
    hit = cache.get("What are the risks?", FILTERS, MODEL)
    assert hit == {"answer": "Supply chain.", "sources": SOURCES}


def test_question_normalisation_hits_on_case_and_whitespace(tmp_path):
    cache = make_cache(tmp_path)
    cache.set("What are the risks?", FILTERS, MODEL, answer="A.", sources=[])
    assert cache.get("  what ARE the\nrisks?  ", FILTERS, MODEL) is not None


def test_different_filters_are_different_entries(tmp_path):
    cache = make_cache(tmp_path)
    cache.set("What are the risks?", FILTERS, MODEL, answer="Apple.", sources=[])
    assert cache.get("What are the risks?", {**FILTERS, "ticker": "MSFT"}, MODEL) is None


def test_none_filters_equal_missing_filters(tmp_path):
    cache = make_cache(tmp_path)
    cache.set(
        "Q?", {"ticker": None, "year": 2025, "filing_type": None}, MODEL, answer="A.", sources=[]
    )
    assert cache.get("Q?", {"year": 2025}, MODEL) is not None


def test_model_swap_invalidates(tmp_path):
    cache = make_cache(tmp_path)
    cache.set("Q?", FILTERS, MODEL, answer="A.", sources=[])
    assert cache.get("Q?", FILTERS, "other-model") is None


def test_eviction_drops_oldest(tmp_path):
    cache = make_cache(tmp_path, max_entries=2)
    cache.set("q1", {}, MODEL, answer="a1", sources=[])
    cache.set("q2", {}, MODEL, answer="a2", sources=[])
    cache.set("q3", {}, MODEL, answer="a3", sources=[])
    assert cache.get("q1", {}, MODEL) is None
    assert cache.get("q2", {}, MODEL) is not None
    assert cache.get("q3", {}, MODEL) is not None


def test_persists_across_instances(tmp_path):
    make_cache(tmp_path).set("Q?", FILTERS, MODEL, answer="A.", sources=SOURCES)
    assert make_cache(tmp_path).get("Q?", FILTERS, MODEL) is not None
