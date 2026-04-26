import polars as pl
import pytest

from src.ingest.chunker import CHUNK_SCHEMA
from src.ingest.quality import DataQualityError, assert_filing_quality


def _make_df(n_sections: int = 5, chars_per_chunk: int = 400, n_chunks: int = 100) -> pl.DataFrame:
    items = [f"Item {i + 1}" for i in range(n_sections)]
    rows = [
        {
            "chunk_id": f"ACC::Item{i % n_sections}::{i:04d}",
            "accession_number": "0001234567-25-000001",
            "ticker": "TEST",
            "fiscal_year": 2025,
            "filing_type": "10-K",
            "item_code": items[i % n_sections],
            "section_label": f"Section {i % n_sections}",
            "text": "x " * (chars_per_chunk // 2),
            "char_count": chars_per_chunk,
            "content_hash": f"hash{i}",
        }
        for i in range(n_chunks)
    ]
    return pl.DataFrame(rows, schema=CHUNK_SCHEMA)


def test_quality_passes_valid_filing():
    df = _make_df()  # defaults: 5 sections, 400 chars, 100 chunks = 40k total chars
    assert_filing_quality(df, "TEST-ACC")  # should not raise


def test_quality_fails_too_few_sections():
    df = _make_df(n_sections=2, n_chunks=20)
    with pytest.raises(DataQualityError, match="section"):
        assert_filing_quality(df, "TEST-ACC")


def test_quality_fails_too_few_total_chars():
    df = _make_df(n_sections=4, chars_per_chunk=10, n_chunks=10)
    with pytest.raises(DataQualityError, match="char count"):
        assert_filing_quality(df, "TEST-ACC")


def test_quality_fails_tiny_chunks():
    rows = [
        {
            "chunk_id": f"x::{i}",
            "accession_number": "A",
            "ticker": "T",
            "fiscal_year": 2025,
            "filing_type": "10-K",
            "item_code": f"Item {(i % 5) + 1}",
            "section_label": "S",
            "text": "hi",
            "char_count": 2,
            "content_hash": f"h{i}",
        }
        for i in range(100)
    ]
    df = pl.DataFrame(rows, schema=CHUNK_SCHEMA)
    with pytest.raises(DataQualityError, match="shorter than"):
        assert_filing_quality(df, "TEST-ACC")
