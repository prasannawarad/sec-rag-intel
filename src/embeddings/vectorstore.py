"""Vector store factory: switches between local Chroma and prod Pinecone.

Embeddings are local (BAAI/bge-small-en-v1.5 via sentence-transformers) — no
API key required. The same model runs in dev and on HF Spaces.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings
from src.ingest.chunker import Chunk

logger = logging.getLogger(__name__)


def get_embeddings() -> Embeddings:
    """Local sentence-transformers model. First call downloads ~130MB to ~/.cache."""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> VectorStore:
    """Return a configured vector store based on VECTOR_STORE_MODE."""
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


def index_chunks(chunks: Iterable[Chunk]) -> int:
    """Embed and store chunks. Returns number of documents indexed."""
    store = get_vectorstore()
    docs = [Document(page_content=c.text, metadata=c.metadata) for c in chunks]
    if not docs:
        return 0
    store.add_documents(docs)
    logger.info("Indexed %d chunks", len(docs))
    return len(docs)
