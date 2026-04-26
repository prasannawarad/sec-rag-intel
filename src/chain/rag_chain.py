"""LCEL RAG chain with last-4-turns conversational memory.

Session state lives in-process (resets on server restart). For the live
demo this is fine — a persistent store (Redis, DynamoDB) would be the
prod upgrade path.

Public API:
  build_rag_chain(ticker, year, filing_type) -> Runnable
    Input:  {"question": str, "session_id": str}  (session_id defaults to "default")
    Output: RAGAnswer {"answer": str, "sources": list[dict]}
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

from src.chain.prompts import build_prompt
from src.config import get_settings
from src.retrieval.retriever import build_retriever

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 4  # last N human/AI pairs sent to the LLM

# In-process session store: session_id -> [(human_msg, ai_msg), ...]
_SESSION_STORE: dict[str, list[tuple[str, str]]] = {}


class RAGAnswer(TypedDict):
    answer: str
    sources: list[dict[str, str | int]]


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('ticker')} {d.metadata.get('year')} "
        f"{d.metadata.get('filing_type')} - {d.metadata.get('section', d.metadata.get('section_label', ''))}]\n"
        f"{d.page_content}"
        for d in docs
    )


def _docs_to_sources(docs: list[Document]) -> list[dict[str, str | int]]:
    return [
        {
            "ticker": d.metadata.get("ticker", ""),
            "year": d.metadata.get("year", d.metadata.get("fiscal_year", "")),
            "filing_type": d.metadata.get("filing_type", ""),
            "section": d.metadata.get("section", d.metadata.get("section_label", "")),
            "section_label": d.metadata.get("section_label", d.metadata.get("section", "")),
        }
        for d in docs
    ]


def get_session_history(session_id: str) -> list[tuple[str, str]]:
    return _SESSION_STORE.get(session_id, [])


def clear_session(session_id: str) -> None:
    _SESSION_STORE.pop(session_id, None)


def build_rag_chain(
    *,
    ticker: str | None = None,
    year: int | None = None,
    filing_type: str | None = None,
) -> RunnableLambda:
    """Return a session-aware RAG chain.

    The returned runnable accepts:
      - a plain string (backwards-compatible, uses session "default")
      - a dict {"question": str, "session_id": str}
    """
    settings = get_settings()
    retriever = build_retriever(ticker=ticker, year=year, filing_type=filing_type)
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    prompt = build_prompt()
    answer_chain = prompt | llm | StrOutputParser()

    def _run(inputs: str | dict) -> RAGAnswer:
        if isinstance(inputs, str):
            question, session_id = inputs, "default"
        else:
            question = inputs["question"]
            session_id = inputs.get("session_id", "default")

        # Retrieve with the current question only (not augmented by history)
        docs = retriever.invoke(question)
        context = _format_docs(docs)

        # Build chat_history messages from last N turns
        history = get_session_history(session_id)[-MAX_HISTORY_TURNS:]
        history_messages = []
        for human, ai in history:
            history_messages.append(HumanMessage(content=human))
            history_messages.append(AIMessage(content=ai))

        answer = answer_chain.invoke(
            {
                "context": context,
                "question": question,
                "chat_history": history_messages,
            }
        )

        # Persist turn to session store
        if session_id not in _SESSION_STORE:
            _SESSION_STORE[session_id] = []
        _SESSION_STORE[session_id].append((question, answer))

        logger.debug(
            "session=%s turn=%d question=%r",
            session_id,
            len(_SESSION_STORE[session_id]),
            question[:60],
        )
        return RAGAnswer(answer=answer, sources=_docs_to_sources(docs))

    return RunnableLambda(_run)
