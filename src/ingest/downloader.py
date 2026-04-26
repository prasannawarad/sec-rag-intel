"""Wrapper around sec-edgar-downloader for fetching 10-K / 10-Q filings."""
from __future__ import annotations

import logging
from pathlib import Path

from src.config import RAW_DIR, get_settings

logger = logging.getLogger(__name__)

DEFAULT_TICKERS: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "JPM", "NVDA", "META",
]


def download_filings(
    tickers: list[str] | None = None,
    filing_type: str = "10-K",
    limit: int = 1,
    download_dir: Path = RAW_DIR,
) -> dict[str, int]:
    """Download SEC filings for the given tickers.

    Returns a mapping of ticker -> number of filings successfully downloaded.
    """
    from sec_edgar_downloader import Downloader

    settings = get_settings()
    download_dir.mkdir(parents=True, exist_ok=True)

    name, _, email = settings.sec_user_agent.partition(" ")
    dl = Downloader(name or "sec-rag-intel", email or "noreply@example.com", str(download_dir))

    results: dict[str, int] = {}
    for ticker in tickers or DEFAULT_TICKERS:
        try:
            count = dl.get(filing_type, ticker, limit=limit)
            results[ticker] = count
            logger.info("Downloaded %d %s filings for %s", count, filing_type, ticker)
        except Exception:
            logger.exception("Failed to download %s for %s", filing_type, ticker)
            results[ticker] = 0
    return results
