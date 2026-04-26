"""End-to-end indexing pipeline: download → parse → chunk → embed → store."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import RAW_DIR
from src.embeddings.vectorstore import index_chunks
from src.ingest.chunker import chunk_text
from src.ingest.downloader import DEFAULT_TICKERS, download_filings
from src.ingest.parser import parse_filing
from src.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def discover_filings(ticker: str) -> list[Path]:
    """Find all downloaded HTML filings for a ticker under data/raw."""
    base = RAW_DIR / "sec-edgar-filings" / ticker
    if not base.exists():
        return []
    return list(base.rglob("*.htm")) + list(base.rglob("*.html")) + list(base.rglob("*.txt"))


def main() -> None:
    download_filings(DEFAULT_TICKERS, filing_type="10-K", limit=1)
    total = 0
    for ticker in DEFAULT_TICKERS:
        for path in discover_filings(ticker):
            try:
                year = int(path.parent.name.split("-")[1]) + 2000  # crude; adjust per layout
            except (IndexError, ValueError):
                year = 0
            text = parse_filing(path)
            chunks = chunk_text(text, ticker=ticker, year=year, filing_type="10-K")
            total += index_chunks(chunks)
    logger.info("Done. Indexed %d total chunks.", total)


if __name__ == "__main__":
    main()
