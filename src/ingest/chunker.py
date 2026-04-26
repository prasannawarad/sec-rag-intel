"""Token-aware chunking that writes one Parquet per filing.

Pipeline:
  full-submission.txt -> parse_filing -> list[Section]
                                              |
                                              v
                  RecursiveCharacterTextSplitter (512 / 50)
                                              |
                                              v
            Parquet at data/processed/chunks/ticker=X/fiscal_year=Y/
                       chunks-<accession>.parquet

Each chunk row carries enough metadata for citations + dedup:
  chunk_id, accession_number, ticker, fiscal_year, filing_type,
  item_code, section_label, text, char_count, content_hash

Persisting before embedding means: re-embedding (e.g. after switching
embedding models or chunk size) does NOT require re-downloading or
re-parsing. content_hash lets the embedding stage skip chunks whose
text is unchanged across runs.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PROCESSED_DIR, get_settings
from src.ingest.parser import Section, parse_filing
from src.ingest.sgml import parse_header

logger = logging.getLogger(__name__)

CHUNKS_DIR = PROCESSED_DIR / "chunks"

CHUNK_SCHEMA = {
    "chunk_id": pl.Utf8,
    "accession_number": pl.Utf8,
    "ticker": pl.Utf8,
    "fiscal_year": pl.Int32,
    "filing_type": pl.Utf8,
    "item_code": pl.Utf8,
    "section_label": pl.Utf8,
    "text": pl.Utf8,
    "char_count": pl.Int32,
    "content_hash": pl.Utf8,
}


@dataclass
class Chunk:
    text: str
    metadata: dict[str, str | int]


def _splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(
    text: str,
    *,
    ticker: str,
    year: int,
    filing_type: str,
    section: str = "full",
) -> list[Chunk]:
    """Backwards-compatible helper: chunk a raw string with flat metadata."""
    pieces = _splitter().split_text(text)
    return [
        Chunk(
            text=piece,
            metadata={
                "ticker": ticker,
                "year": year,
                "filing_type": filing_type,
                "section": section,
                "chunk_id": f"{ticker}-{year}-{filing_type}-{i:05d}",
            },
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_sections(
    sections: list[Section],
    *,
    accession_number: str,
    ticker: str,
    fiscal_year: int,
    filing_type: str,
) -> pl.DataFrame:
    """Split each section into chunks and return one DataFrame for the filing."""
    splitter = _splitter()
    rows: list[dict] = []
    for s in sections:
        for i, piece in enumerate(splitter.split_text(s.text)):
            chunk_id = f"{accession_number}::{s.item_code.replace(' ', '')}::{i:04d}"
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "accession_number": accession_number,
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "filing_type": filing_type,
                    "item_code": s.item_code,
                    "section_label": s.section_label,
                    "text": piece,
                    "char_count": len(piece),
                    "content_hash": _hash(piece),
                }
            )
    return pl.DataFrame(rows, schema=CHUNK_SCHEMA)


def chunks_path(ticker: str, fiscal_year: int, accession_number: str) -> Path:
    out = CHUNKS_DIR / f"ticker={ticker}" / f"fiscal_year={fiscal_year}"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"chunks-{accession_number}.parquet"


def parse_and_persist(raw_path: Path, ticker: str) -> tuple[Path, int]:
    """Read filing -> sections -> chunks -> write Parquet. Returns (path, n)."""
    header = parse_header(raw_path)
    sections = parse_filing(raw_path, form_type=header.form_type)
    df = chunk_sections(
        sections,
        accession_number=header.accession_number,
        ticker=ticker,
        fiscal_year=header.fiscal_year,
        filing_type=header.form_type,
    )
    out = chunks_path(ticker, header.fiscal_year, header.accession_number)
    df.write_parquet(out)
    logger.info(
        "persisted %d chunks across %d sections for %s -> %s",
        len(df),
        df["item_code"].n_unique(),
        header.accession_number,
        out,
    )
    return out, len(df)
