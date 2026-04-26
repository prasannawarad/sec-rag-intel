"""End-to-end indexing pipeline: download -> parse -> chunk -> embed.

Usage:
    python -m scripts.build_index                    # all default tickers
    python -m scripts.build_index --tickers AAPL     # single ticker
    python -m scripts.build_index --tickers AAPL MSFT --limit 2

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

from src.config import RAW_DIR, get_settings
from src.embeddings.vectorstore import index_parquet
from src.ingest.chunker import chunks_path, parse_and_persist
from src.ingest.downloader import DEFAULT_TICKERS, download_filings
from src.ingest.manifest import Manifest, Stage
from src.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def _run_parse_stage(mf: Manifest) -> None:
    pending = mf.pending(Stage.PARSE)
    if pending.is_empty():
        logger.info("Nothing pending for parse stage.")
        return
    parsed: list[tuple[str, int]] = []
    for row in pending.iter_rows(named=True):
        try:
            _, n = parse_and_persist(Path(row["raw_path"]), row["ticker"])
            parsed.append((row["accession_number"], n))
        except Exception:
            logger.exception("Parse failed for %s", row["raw_path"])
    if parsed:
        mf.mark_parsed_batch(parsed)


def _run_embed_stage(mf: Manifest) -> None:
    embed_model = get_settings().embedding_model_name
    pending = mf.pending(Stage.EMBED).filter(pl.col("parsed_at").is_not_null())
    if pending.is_empty():
        logger.info("Nothing pending for embed stage.")
        return
    embedded: list[str] = []
    for row in pending.iter_rows(named=True):
        path = chunks_path(row["ticker"], row["fiscal_year"], row["accession_number"])
        if not path.exists():
            logger.warning("Chunk file missing for %s — re-run parse stage.", row["accession_number"])
            continue
        try:
            n_new = index_parquet(pl.read_parquet(path))
            embedded.append(row["accession_number"])
            logger.info("%s: %d new chunks embedded", row["accession_number"], n_new)
        except Exception:
            logger.exception("Embed failed for %s", row["accession_number"])
    if embedded:
        mf.mark_embedded_batch(embedded, embed_model=embed_model)


def run(tickers: list[str], limit: int = 1) -> None:
    mf = Manifest()

    logger.info("=== Stage 1: Download ===")
    download_filings(tickers, filing_type="10-K", limit=limit, manifest=mf)

    logger.info("=== Stage 2: Parse + Chunk -> Parquet ===")
    _run_parse_stage(mf)

    logger.info("=== Stage 3: Embed -> Vector Store ===")
    _run_embed_stage(mf)

    logger.info("=== Pipeline complete ===")
    df_all = mf.df()
    logger.info(
        "Manifest: %d total filing(s), %d fully embedded",
        len(df_all),
        df_all.filter(pl.col("embedded_at").is_not_null()).height,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC RAG ingestion pipeline")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--limit", type=int, default=1, help="Max filings per ticker")
    args = parser.parse_args()
    run(args.tickers, limit=args.limit)


if __name__ == "__main__":
    main()
