"""Data-quality assertions run after parsing each filing.

Raises DataQualityError if a filing looks broken — catches parse regressions
before they pollute the vector store. Rules are intentionally conservative
(lower bounds, not exact expectations) so they survive across companies.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

MIN_SECTIONS = 3          # a 10-K with <3 detected sections is almost certainly mis-parsed
MIN_TOTAL_CHARS = 30_000  # a full 10-K is >200k chars after cleaning; 30k catches empty parses
MIN_CHUNK_SIZE = 50       # chunks shorter than this are noise
MAX_CHUNK_MULTIPLE = 5    # no chunk should be >5x the configured chunk_size


class DataQualityError(ValueError):
    pass


def assert_filing_quality(df: pl.DataFrame, accession_number: str) -> None:
    """Raise DataQualityError if the chunk DataFrame fails any quality check.

    Designed to be called immediately after chunk_sections() before persisting
    to Parquet, so bad data never reaches the vector store.
    """
    errors: list[str] = []

    n_sections = df["item_code"].n_unique()
    if n_sections < MIN_SECTIONS:
        errors.append(
            f"only {n_sections} section(s) detected (min {MIN_SECTIONS}) — "
            "parser likely failed to find Item headings"
        )

    total_chars = df["char_count"].sum()
    if total_chars < MIN_TOTAL_CHARS:
        errors.append(
            f"total char count {total_chars:,} below {MIN_TOTAL_CHARS:,} — "
            "filing body may be empty or unparsed"
        )

    tiny = (df["char_count"] < MIN_CHUNK_SIZE).sum()
    if tiny > 0:
        errors.append(f"{tiny} chunk(s) shorter than {MIN_CHUNK_SIZE} chars")

    if errors:
        msg = f"Quality check failed for {accession_number}:\n" + "\n".join(
            f"  • {e}" for e in errors
        )
        raise DataQualityError(msg)

    logger.info(
        "Quality OK — %s: %d sections, %d chunks, %s total chars",
        accession_number,
        n_sections,
        len(df),
        f"{total_chars:,}",
    )
