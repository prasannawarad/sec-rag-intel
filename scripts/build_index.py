"""End-to-end indexing pipeline: download -> parse -> chunk -> embed.

Usage:
    python scripts/build_index.py                    # all default tickers
    python scripts/build_index.py --tickers AAPL     # single ticker
    python scripts/build_index.py --tickers AAPL MSFT --limit 2

Each stage is idempotent:
  - Download  : skips accessions already in the manifest
  - Parse     : skips accessions already marked parsed_at in the manifest
  - Embed     : skips chunks whose content_hash is already in the vector store
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from src.config import RAW_DIR
from src.embeddings.vectorstore import index_parquet
from src.ingest.chunker import chunks_path, parse_and_persist
from src.ingest.downloader import DEFAULT_TICKERS, download_filings
from src.ingest.manifest import Manifest
from src.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def run(tickers: list[str], limit: int = 1) -> None:
    mf = Manifest()

    # Stage 1: Download
    logger.info("=== Stage 1: Download ===")
    download_filings(tickers, filing_type="10-K", limit=limit, manifest=mf)

    # Stage 2: Parse -> Parquet (only filings not yet parsed)
    logger.info("=== Stage 2: Parse + Chunk -> Parquet ===")
    pending = mf.pending("parse")
    if pending.is_empty():
        logger.info("Nothing pending for parse stage.")
    for row in pending.iter_rows(named=True):
        raw_path = Path(row["raw_path"])
        ticker = row["ticker"]
        try:
            _, n = parse_and_persist(raw_path, ticker)
            mf.mark_parsed(row["accession_number"], num_chunks=n)
        except Exception:
            logger.exception("Parse failed for %s", raw_path)

    # Stage 3: Embed (reads Parquet for each un-embedded filing)
    logger.info("=== Stage 3: Embed -> Vector Store ===")
    settings_model = None
    pending_embed = mf.pending("embed").filter(pl.col("parsed_at").is_not_null())
    if pending_embed.is_empty():
        logger.info("Nothing pending for embed stage.")
    for row in pending_embed.iter_rows(named=True):
        path = chunks_path(row["ticker"], row["fiscal_year"], row["accession_number"])
        if not path.exists():
            logger.warning("Chunk file missing for %s — re-run parse stage.", row["accession_number"])
            continue
        try:
            df = pl.read_parquet(path)
            n_new = index_parquet(df)
            from src.config import get_settings
            settings_model = get_settings().embedding_model_name
            mf.mark_embedded(row["accession_number"], embed_model=settings_model)
            logger.info("%s: %d new chunks embedded", row["accession_number"], n_new)
        except Exception:
            logger.exception("Embed failed for %s", row["accession_number"])

    # Summary
    logger.info("=== Pipeline complete ===")
    df_all = mf.df()
    total = len(df_all)
    embedded = df_all.filter(pl.col("embedded_at").is_not_null())
    logger.info(
        "Manifest: %d total filing(s), %d fully embedded",
        total,
        len(embedded),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC RAG ingestion pipeline")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--limit", type=int, default=1, help="Max filings per ticker")
    args = parser.parse_args()
    run(args.tickers, limit=args.limit)


if __name__ == "__main__":
    main()
