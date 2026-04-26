"""Token-aware text chunking with metadata for RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: dict[str, str | int]


def chunk_text(
    text: str,
    *,
    ticker: str,
    year: int,
    filing_type: str,
    section: str = "full",
) -> list[Chunk]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
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
