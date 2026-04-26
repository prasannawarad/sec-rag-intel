"""MMR retriever with optional metadata filters."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.retrievers import BaseRetriever

from src.config import get_settings
from src.embeddings.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def build_retriever(
    *,
    ticker: str | None = None,
    year: int | None = None,
    filing_type: str | None = None,
) -> BaseRetriever:
    """Return a MMR retriever with metadata filters applied."""
    settings = get_settings()
    store = get_vectorstore()

    metadata_filter: dict[str, Any] = {}
    if ticker:
        metadata_filter["ticker"] = ticker
    if year:
        metadata_filter["year"] = year
    if filing_type:
        metadata_filter["filing_type"] = filing_type

    search_kwargs: dict[str, Any] = {
        "k": settings.retriever_k,
        "fetch_k": settings.retriever_fetch_k,
    }
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    return store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
