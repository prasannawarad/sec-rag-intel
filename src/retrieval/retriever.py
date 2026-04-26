"""MMR retriever with optional metadata filters.

Chroma requires $and to combine multiple filter conditions. Pinecone and
other backends accept the plain dict form. We normalise here so callers
don't need to care about the backend.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.retrievers import BaseRetriever

from src.config import get_settings
from src.embeddings.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def _build_filter(conditions: dict[str, Any]) -> dict[str, Any] | None:
    """Return a Chroma-compatible filter from a flat {field: value} dict."""
    if not conditions:
        return None
    if len(conditions) == 1:
        field, value = next(iter(conditions.items()))
        return {field: {"$eq": value}}
    return {"$and": [{field: {"$eq": value}} for field, value in conditions.items()]}


def build_retriever(
    *,
    ticker: str | None = None,
    year: int | None = None,
    filing_type: str | None = None,
) -> BaseRetriever:
    """Return a MMR retriever with metadata filters applied."""
    settings = get_settings()
    store = get_vectorstore()

    raw: dict[str, Any] = {}
    if ticker:
        raw["ticker"] = ticker
    if year:
        raw["year"] = year
    if filing_type:
        raw["filing_type"] = filing_type

    search_kwargs: dict[str, Any] = {
        "k": settings.retriever_k,
        "fetch_k": settings.retriever_fetch_k,
    }
    chroma_filter = _build_filter(raw)
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter

    return store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
