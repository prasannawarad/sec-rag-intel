"""Idempotent SEC EDGAR downloader.

Wraps sec-edgar-downloader to fetch 10-K / 10-Q filings, then walks the
on-disk layout (data/raw/sec-edgar-filings/{TICKER}/{FORM}/{ACCESSION}/) and
upserts each new accession into the Polars manifest with metadata parsed
from the SGML header. Re-runs are cheap: filings already in the manifest
are skipped without re-downloading.

The library handles the 10-req/sec SEC EDGAR rate limit and User-Agent
header internally; we only have to pass identity into the constructor.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import RAW_DIR, get_settings
from src.ingest.manifest import Manifest
from src.ingest.sgml import parse_header

logger = logging.getLogger(__name__)

DEFAULT_TICKERS: list[str] = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "TSLA",
    "JPM",
    "NVDA",
    "META",
]


def _filings_root(download_dir: Path) -> Path:
    return download_dir / "sec-edgar-filings"


def _discover_filings(download_dir: Path, ticker: str, form_type: str) -> list[Path]:
    """All full-submission.txt files for a ticker/form on disk."""
    base = _filings_root(download_dir) / ticker / form_type
    if not base.exists():
        return []
    return sorted(base.glob("*/full-submission.txt"))


def _split_user_agent(user_agent: str) -> tuple[str, str]:
    """SEC requires "<name> <email>" — the library wants them split."""
    parts = user_agent.strip().split()
    if len(parts) < 2 or "@" not in parts[-1]:
        raise ValueError(
            f"SEC_USER_AGENT must be 'Your Name your.email@example.com', got: {user_agent!r}"
        )
    email = parts[-1]
    name = " ".join(parts[:-1]) or "sec-rag-intel"
    return name, email


def download_filings(
    tickers: list[str] | None = None,
    filing_type: str = "10-K",
    limit: int = 1,
    download_dir: Path = RAW_DIR,
    manifest: Manifest | None = None,
) -> dict[str, int]:
    """Download filings, then sync new accessions into the manifest.

    Returns: ticker -> number of NEW filings added to the manifest.
    """
    from sec_edgar_downloader import Downloader

    settings = get_settings()
    download_dir.mkdir(parents=True, exist_ok=True)
    name, email = _split_user_agent(settings.sec_user_agent)
    dl = Downloader(name, email, str(download_dir))
    mf = manifest or Manifest()

    new_counts: dict[str, int] = {}
    for ticker in tickers or DEFAULT_TICKERS:
        try:
            dl.get(filing_type, ticker, limit=limit)
        except Exception:
            logger.exception("Download failed for %s %s", ticker, filing_type)
            new_counts[ticker] = 0
            continue

        added = 0
        for path in _discover_filings(download_dir, ticker, filing_type):
            try:
                header = parse_header(path)
            except Exception:
                logger.exception("Could not parse SGML header for %s", path)
                continue

            if mf.has(header.accession_number):
                continue
            mf.upsert_filing(header, ticker=ticker, raw_path=path)
            added += 1

        new_counts[ticker] = added
        logger.info("%s: %d new filing(s) added to manifest", ticker, added)

    return new_counts
