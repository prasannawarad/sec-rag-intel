"""Ingestion state manifest backed by a single Parquet file.

One row per (accession_number) — the unique SEC filing identifier. The
manifest is the source of truth for what has been downloaded, parsed, and
embedded, so each pipeline stage can be re-run idempotently.

Stages set their own timestamp + result columns on success:
- download: raw_path, downloaded_at
- parse:    num_chunks, parsed_at
- embed:    embed_model, embedded_at
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import polars as pl

from src.config import PROCESSED_DIR
from src.ingest.sgml import FilingHeader

logger = logging.getLogger(__name__)

MANIFEST_PATH = PROCESSED_DIR / "manifest.parquet"

Stage = Literal["download", "parse", "embed"]

_SCHEMA = {
    "accession_number": pl.Utf8,
    "ticker": pl.Utf8,
    "cik": pl.Utf8,
    "company_name": pl.Utf8,
    "form_type": pl.Utf8,
    "filed_date": pl.Date,
    "period_of_report": pl.Date,
    "fiscal_year": pl.Int32,
    "raw_path": pl.Utf8,
    "downloaded_at": pl.Datetime,
    "num_chunks": pl.Int32,
    "parsed_at": pl.Datetime,
    "embed_model": pl.Utf8,
    "embedded_at": pl.Datetime,
}


class Manifest:
    def __init__(self, path: Path = MANIFEST_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._df = pl.read_parquet(self.path)
        else:
            self._df = pl.DataFrame(schema=_SCHEMA)

    def _save(self) -> None:
        self._df.write_parquet(self.path)

    def has(self, accession_number: str) -> bool:
        if self._df.is_empty():
            return False
        return (self._df["accession_number"] == accession_number).any()

    def upsert_filing(self, header: FilingHeader, ticker: str, raw_path: Path) -> None:
        """Insert or replace a row for this filing with download metadata."""
        row = {
            "accession_number": header.accession_number,
            "ticker": ticker,
            "cik": header.cik,
            "company_name": header.company_name,
            "form_type": header.form_type,
            "filed_date": header.filed_date,
            "period_of_report": header.period_of_report,
            "fiscal_year": header.fiscal_year,
            "raw_path": str(raw_path),
            "downloaded_at": datetime.utcnow(),
            "num_chunks": None,
            "parsed_at": None,
            "embed_model": None,
            "embedded_at": None,
        }
        new_row = pl.DataFrame([row], schema=_SCHEMA)
        self._df = pl.concat(
            [
                self._df.filter(pl.col("accession_number") != header.accession_number),
                new_row,
            ]
        )
        self._save()
        logger.info("manifest: upserted %s (%s %s)", header.accession_number, ticker, header.form_type)

    def mark_parsed(self, accession_number: str, num_chunks: int) -> None:
        self._df = self._df.with_columns(
            num_chunks=pl.when(pl.col("accession_number") == accession_number)
            .then(num_chunks)
            .otherwise(pl.col("num_chunks")),
            parsed_at=pl.when(pl.col("accession_number") == accession_number)
            .then(datetime.utcnow())
            .otherwise(pl.col("parsed_at")),
        )
        self._save()

    def mark_embedded(self, accession_number: str, embed_model: str) -> None:
        self._df = self._df.with_columns(
            embed_model=pl.when(pl.col("accession_number") == accession_number)
            .then(pl.lit(embed_model))
            .otherwise(pl.col("embed_model")),
            embedded_at=pl.when(pl.col("accession_number") == accession_number)
            .then(datetime.utcnow())
            .otherwise(pl.col("embedded_at")),
        )
        self._save()

    def pending(self, stage: Stage) -> pl.DataFrame:
        """Return rows that have not yet completed the given stage."""
        col = {"download": "downloaded_at", "parse": "parsed_at", "embed": "embedded_at"}[stage]
        return self._df.filter(pl.col(col).is_null())

    def df(self) -> pl.DataFrame:
        return self._df
