"""LCEL RAG chain: retriever → prompt → Groq LLM → parser, with source attribution."""
from __future__ import annotations

import logging
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from src.chain.prompts import build_prompt
from src.config import get_settings
from src.retrieval.retriever import build_retriever

logger = logging.getLogger(__name__)


class RAGAnswer(TypedDict):
    answer: str
    sources: list[dict[str, str | int]]


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('ticker')} {d.metadata.get('year')} "
        f"{d.metadata.get('filing_type')} - {d.metadata.get('section')}]\n{d.page_content}"
        for d in docs
    )


def _docs_to_sources(docs: list[Document]) -> list[dict[str, str | int]]:
    return [
        {
            "ticker": d.metadata.get("ticker", ""),
            "year": d.metadata.get("year", ""),
            "filing_type": d.metadata.get("filing_type", ""),
            "section": d.metadata.get("section", ""),
        }
        for d in docs
    ]


def build_rag_chain(
    *,
    ticker: str | None = None,
    year: int | None = None,
    filing_type: str | None = None,
):
    settings = get_settings()
    retriever = build_retriever(ticker=ticker, year=year, filing_type=filing_type)
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    prompt = build_prompt()

    answer_chain = (
        {
            "context": lambda x: _format_docs(x["docs"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    def _run(question: str) -> RAGAnswer:
        docs = retriever.invoke(question)
        answer = answer_chain.invoke({"docs": docs, "question": question})
        return RAGAnswer(answer=answer, sources=_docs_to_sources(docs))

    return RunnableLambda(_run)
