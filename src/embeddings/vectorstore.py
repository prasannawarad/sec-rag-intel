"""Vector store factory: switches between local Chroma and prod Pinecone."""
from __future__ import annotations

import logging
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

from src.config import get_settings
from src.ingest.chunker import Chunk

logger = logging.getLogger(__name__)


def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


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
