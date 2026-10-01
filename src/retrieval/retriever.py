"""MMR retriever with optional metadata filters and company routing.

Chroma requires $and to combine multiple filter conditions. Pinecone and
other backends accept the plain dict form. We normalise here so callers
don't need to care about the backend.

Company routing: a question with no ticker filter that names several companies
("Compare Apple and Microsoft…", "across the eight companies") gets one MMR
search per company, merged. A single top-5 search over the whole corpus
otherwise returns chunks from whichever companies happen to rank highest —
the RAGAS baseline scored q03 (Apple vs Microsoft R&D) 0.0 relevancy that way.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import get_settings
from src.embeddings.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

# Names a question might use for each indexed company (keep in sync with
# DEFAULT_TICKERS in src/ingest/downloader.py). Matched as whole words.
COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple",),
    "MSFT": ("microsoft",),
    "AMZN": ("amazon", "aws"),
    "GOOGL": ("alphabet", "google"),
    "TSLA": ("tesla",),
    "JPM": ("jpmorgan", "jp morgan"),
    "NVDA": ("nvidia",),
    "META": ("meta", "facebook"),
}
# "Which companies…", "across the eight companies", "in the dataset" → all of them.
_ALL_COMPANIES_RE = re.compile(
    r"\b(?:across|all|each|every|which)\b[^?]*\bcompan(?:y|ies)\b|\bin the dataset\b", re.I
)
# Routed chunks per question, total. Groq's free tier allows qwen 8k tokens/min,
# so 8 × ~420-token chunks keeps one answer call comfortably under it.
MAX_ROUTED_CHUNKS = 8
MAX_CHUNKS_PER_COMPANY = 3


def detect_tickers(question: str) -> list[str]:
    """Tickers of the companies a question names (all of them for dataset-wide questions)."""
    q = question.lower()
    found = [
        ticker
        for ticker, names in COMPANY_ALIASES.items()
        if any(re.search(rf"\b{re.escape(n)}\b", q) for n in names)
    ]
    if len(found) < 2 and _ALL_COMPANIES_RE.search(question):
        return list(COMPANY_ALIASES)
    return found


def _build_filter(conditions: dict[str, Any]) -> dict[str, Any] | None:
    """Return a Chroma-compatible filter from a flat {field: value} dict."""
    if not conditions:
        return None
    if len(conditions) == 1:
        field, value = next(iter(conditions.items()))
        return {field: {"$eq": value}}
    return {"$and": [{field: {"$eq": value}} for field, value in conditions.items()]}


class CompanyRoutingRetriever(BaseRetriever):
    """MMR per company named in the question; plain MMR when none is named."""

    base_filter: dict[str, Any]
    k: int
    fetch_k: int

    def _search(self, query: str, ticker: str | None, k: int) -> list[Document]:
        conditions = {**self.base_filter, **({"ticker": ticker} if ticker else {})}
        kwargs: dict[str, Any] = {"k": k, "fetch_k": self.fetch_k}
        if flt := _build_filter(conditions):
            kwargs["filter"] = flt
        return get_vectorstore().max_marginal_relevance_search(query, **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        tickers = detect_tickers(query)
        if len(tickers) <= 1:
            return self._search(query, tickers[0] if tickers else None, self.k)
        per_company = max(1, min(MAX_CHUNKS_PER_COMPANY, MAX_ROUTED_CHUNKS // len(tickers)))
        logger.debug("Routing to %s (%d chunks each)", tickers, per_company)
        return [doc for t in tickers for doc in self._search(query, t, per_company)]


def build_retriever(
    *,
    ticker: str | None = None,
    year: int | None = None,
    filing_type: str | None = None,
) -> BaseRetriever:
    """Return a MMR retriever with metadata filters applied.

    Without an explicit ticker (and with settings.company_routing on), the
    retriever routes by the companies the question names.
    """
    settings = get_settings()
    store = get_vectorstore()

    raw: dict[str, Any] = {}
    if year:
        raw["year"] = year
    if filing_type:
        raw["filing_type"] = filing_type
    if not ticker and settings.company_routing:
        return CompanyRoutingRetriever(
            base_filter=raw, k=settings.retriever_k, fetch_k=settings.retriever_fetch_k
        )
    if ticker:
        raw["ticker"] = ticker

    search_kwargs: dict[str, Any] = {
        "k": settings.retriever_k,
        "fetch_k": settings.retriever_fetch_k,
    }
    chroma_filter = _build_filter(raw)
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter

    return store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
