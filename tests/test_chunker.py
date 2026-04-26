from src.ingest.chunker import chunk_text


def test_chunker_attaches_metadata():
    text = "First paragraph. " * 200
    chunks = chunk_text(text, ticker="AAPL", year=2024, filing_type="10-K", section="risk")
    assert len(chunks) > 1
    first = chunks[0]
    assert first.metadata["ticker"] == "AAPL"
    assert first.metadata["year"] == 2024
    assert first.metadata["filing_type"] == "10-K"
    assert first.metadata["section"] == "risk"
    assert first.metadata["chunk_id"].startswith("AAPL-2024-10-K-")


def test_chunker_empty_text_returns_no_chunks():
    chunks = chunk_text("", ticker="AAPL", year=2024, filing_type="10-K")
    assert chunks == []
