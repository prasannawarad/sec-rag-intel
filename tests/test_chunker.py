from src.ingest.chunker import chunk_sections, chunk_text
from src.ingest.parser import Section


def test_chunk_text_attaches_flat_metadata():
    """The legacy single-string chunker still works for the LCEL chain."""
    text = "First paragraph. " * 200
    chunks = chunk_text(text, ticker="AAPL", year=2024, filing_type="10-K", section="risk")
    assert len(chunks) > 1
    first = chunks[0]
    assert first.metadata["ticker"] == "AAPL"
    assert first.metadata["year"] == 2024
    assert first.metadata["filing_type"] == "10-K"
    assert first.metadata["section"] == "risk"
    assert first.metadata["chunk_id"].startswith("AAPL-2024-10-K-")


def test_chunk_text_empty_returns_no_chunks():
    assert chunk_text("", ticker="AAPL", year=2024, filing_type="10-K") == []


def test_chunk_sections_attaches_section_metadata_and_hashes():
    sections = [
        Section(item_code="Item 1A", section_label="Risk Factors", text="risk text. " * 200),
        Section(item_code="Item 7", section_label="MD&A", text="discussion text. " * 200),
    ]
    df = chunk_sections(
        sections,
        accession_number="0001234567-25-000001",
        ticker="AAPL",
        fiscal_year=2025,
        filing_type="10-K",
    )
    assert df["item_code"].n_unique() == 2
    assert set(df["item_code"].to_list()) == {"Item 1A", "Item 7"}
    # Each chunk has a unique deterministic id and a hash
    assert df["chunk_id"].n_unique() == len(df)
    assert df["content_hash"].null_count() == 0
    # All chunks belong to the same filing
    assert df["accession_number"].unique().to_list() == ["0001234567-25-000001"]
    assert df["fiscal_year"].unique().to_list() == [2025]


def test_chunk_sections_content_hash_is_deterministic():
    sec = [Section(item_code="Item 1A", section_label="Risk Factors", text="same text " * 200)]
    df1 = chunk_sections(
        sec, accession_number="X", ticker="T", fiscal_year=2025, filing_type="10-K"
    )
    df2 = chunk_sections(
        sec, accession_number="X", ticker="T", fiscal_year=2025, filing_type="10-K"
    )
    assert df1["content_hash"].to_list() == df2["content_hash"].to_list()
