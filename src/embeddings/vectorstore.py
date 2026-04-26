"""Vector store factory: switches between local Chroma and prod Pinecone.

Embeddings are local (BAAI/bge-small-en-v1.5 via sentence-transformers) — no
API key required. The same model runs in dev and on HF Spaces.

get_embeddings() and get_vectorstore() are module-level singletons (lru_cache)
so the 130MB model is loaded exactly once per process.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache

import polars as pl
import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings
from src.ingest.chunker import Chunk

logger = logging.getLogger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(texts: list[str]) -> int:
    return sum(len(_TOKENIZER.encode(t)) for t in texts)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Singleton — loads the 130MB BGE model once per process."""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=4)
def get_vectorstore() -> VectorStore:
    """Singleton per (mode, persist_dir/index) — avoids redundant connections."""
    settings = get_settings()
    embeddings = get_embeddings()

    if settings.vector_store_mode == "pinecone":
        from langchain_pinecone import PineconeVectorStore

        return PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=embeddings,
        )

    from langchain_chroma import Chroma

    return Chroma(
        collection_name="sec-filings",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def _row_to_metadata(row: dict) -> dict:
    return {
        "ticker": row["ticker"],
        "year": int(row["fiscal_year"]),
        "filing_type": row["filing_type"],
        "item_code": row["item_code"],
        "section_label": row["section_label"],
        "chunk_id": row["chunk_id"],
        "accession_number": row["accession_number"],
        "content_hash": row["content_hash"],
    }


def _existing_hashes(store: VectorStore, candidate_hashes: list[str]) -> set[str]:
    """Return which of candidate_hashes are already in the store.

    Queries only the candidate set (not the full collection) to keep this
    O(candidates) instead of O(collection).
    """
    if not candidate_hashes:
        return set()
    try:
        from langchain_chroma import Chroma

        if isinstance(store, Chroma):
            col = store._collection  # type: ignore[attr-defined]
            results = col.get(
                where={"content_hash": {"$in": candidate_hashes}},
                include=["metadatas"],
            )
            return {m.get("content_hash", "") for m in (results["metadatas"] or [])}
    except Exception:
        pass
    return set()


def index_chunks(chunks: Iterable[Chunk]) -> int:
    """Embed and store legacy Chunk objects. Returns number of new docs indexed."""
    store = get_vectorstore()
    docs = [Document(page_content=c.text, metadata=c.metadata) for c in chunks]
    if not docs:
        return 0
    store.add_documents(docs)
    logger.info("Indexed %d chunks", len(docs))
    return len(docs)


def index_parquet(df: pl.DataFrame) -> int:
    """Embed new chunks from a Parquet DataFrame, skipping already-indexed hashes.

    Returns the number of NEW chunks embedded.
    """
    if df.is_empty():
        return 0

    store = get_vectorstore()
    candidates = df["content_hash"].to_list()
    existing = _existing_hashes(store, candidates)

    new_df = df.filter(~pl.col("content_hash").is_in(existing))
    if new_df.is_empty():
        logger.info("All %d chunks already indexed — nothing to do", len(df))
        return 0

    skipped = len(df) - len(new_df)
    if skipped:
        logger.info("Skipping %d already-indexed chunks, embedding %d new", skipped, len(new_df))

    texts = new_df["text"].to_list()
    logger.info(
        "Pre-flight: %d chunks | %d tokens | model=%s",
        len(texts),
        estimate_tokens(texts),
        get_settings().embedding_model_name,
    )

    docs = [
        Document(page_content=row["text"], metadata=_row_to_metadata(row))
        for row in new_df.iter_rows(named=True)
    ]

    store.add_documents(docs)
    logger.info("Embedded and stored %d new chunks", len(docs))
    return len(docs)
