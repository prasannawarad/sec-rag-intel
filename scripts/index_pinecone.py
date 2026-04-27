"""One-time script to populate the Pinecone index from local Parquet chunks.

Run this once (or whenever you add new filings) before deploying to HF Spaces.
Requires PINECONE_API_KEY and PINECONE_INDEX_NAME to be set in .env.

Usage:
    VECTOR_STORE_MODE=pinecone python -m scripts.index_pinecone

The script:
  1. Creates the Pinecone serverless index if it does not exist
     (us-east-1, cosine similarity, 384-dim for bge-small-en-v1.5)
  2. Reads all Parquet chunk files from data/processed/chunks/
  3. Embeds and uploads any chunks not already in the index (content-hash dedup)
"""

from __future__ import annotations

import logging
import os

import polars as pl

from src.embeddings.vectorstore import get_embeddings, index_parquet
from src.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

CHUNKS_GLOB = "data/processed/chunks/**/*.parquet"
EMBEDDING_DIM = 384  # bge-small-en-v1.5


def _ensure_pinecone_index(index_name: str) -> None:
    """Create the Pinecone serverless index if it does not already exist."""
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        logger.info("Pinecone index '%s' already exists", index_name)
        return

    logger.info("Creating Pinecone serverless index '%s' (dim=%d, cosine)", index_name, EMBEDDING_DIM)
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    logger.info("Index created.")


def main() -> None:
    from src.config import get_settings

    settings = get_settings()
    if settings.vector_store_mode != "pinecone":
        raise RuntimeError(
            "Set VECTOR_STORE_MODE=pinecone (in .env or env var) before running this script."
        )

    _ensure_pinecone_index(settings.pinecone_index_name)

    # Warm the embedding model (one-time 130MB load)
    get_embeddings()

    df = pl.read_parquet(CHUNKS_GLOB, hive_partitioning=True)
    logger.info("Loaded %d total chunks from %s", len(df), CHUNKS_GLOB)

    n_indexed = index_parquet(df)
    logger.info("Done. %d new chunks uploaded to Pinecone.", n_indexed)


if __name__ == "__main__":
    main()
