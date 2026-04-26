"""Shared types for the RAG chain layer.

Importing these in both rag_chain.py and api/main.py keeps the dict
keys the chain returns in sync with the Pydantic model the API validates.
"""

from __future__ import annotations

from typing import TypedDict


class RAGSource(TypedDict):
    ticker: str
    year: int | str
    filing_type: str
    section_label: str


class RAGAnswer(TypedDict):
    answer: str
    sources: list[RAGSource]
