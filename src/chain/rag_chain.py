"""LCEL RAG chain with last-4-turns conversational memory and free-tier guardrails.

Session state lives in-process (resets on server restart). For the live
demo this is fine — a persistent store (Redis, DynamoDB) would be the
prod upgrade path.

Groq free-tier protection (see quota.py / cache.py):
  1. First-turn answers are served from a disk cache when possible (0 tokens).
  2. Every LLM call is pre-checked against a persisted daily token/request
     budget and throttled to stay under the requests-per-minute limit.
  3. When the daily budget is spent, the chain degrades gracefully to a
     retrieval-only answer (top excerpts + sources) instead of erroring.

Public API:
  build_rag_chain(ticker, year, filing_type, use_cache) -> Runnable
    Input:  {"question": str, "session_id": str}  (session_id defaults to "default")
    Output: RAGAnswer {"answer", "sources", "cached", "degraded"}
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from pydantic import SecretStr

from src.chain.cache import get_answer_cache
from src.chain.prompts import build_prompt
from src.chain.quota import QuotaExceededError, estimate_tokens, get_quota_guard
from src.chain.types import RAGAnswer, RAGSource
from src.config import get_settings
from src.retrieval.retriever import build_retriever

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 4
_MAX_SESSIONS = 200  # evict oldest sessions beyond this to prevent unbounded growth
_PROMPT_OVERHEAD_TOKENS = 250  # system prompt + message scaffolding, conservative
_DEGRADED_EXCERPTS = 3
_DEGRADED_EXCERPT_CHARS = 400

_SESSION_STORE: dict[str, list[tuple[str, str]]] = {}


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('ticker')} {d.metadata.get('year')} "
        f"{d.metadata.get('filing_type')} - {d.metadata.get('section_label', '')}]\n"
        f"{d.page_content}"
        for d in docs
    )


def _docs_to_sources(docs: list[Document]) -> list[RAGSource]:
    return [
        RAGSource(
            ticker=d.metadata.get("ticker", ""),
            year=int(d.metadata.get("year", 0)),
            filing_type=d.metadata.get("filing_type", ""),
            section_label=d.metadata.get("section_label", ""),
        )
        for d in docs
    ]


def _degraded_answer(docs: list[Document], reason: str) -> RAGAnswer:
    """Retrieval-only fallback: the daily Groq budget is spent, but retrieval
    is local and free — return the most relevant excerpts instead of an error."""
    excerpts = "\n\n".join(
        f"**{d.metadata.get('ticker')} {d.metadata.get('year')} "
        f"{d.metadata.get('filing_type')}** — {d.metadata.get('section_label', '')}\n"
        f"> {d.page_content[:_DEGRADED_EXCERPT_CHARS].strip()}…"
        for d in docs[:_DEGRADED_EXCERPTS]
    )
    answer = (
        f"⚠️ {reason}\n\n"
        "LLM synthesis is paused, but here are the most relevant filing "
        f"excerpts for your question:\n\n{excerpts}"
    )
    return RAGAnswer(answer=answer, sources=_docs_to_sources(docs), cached=False, degraded=True)


def _evict_sessions_if_needed() -> None:
    # Dict preserves insertion order (CPython 3.7+), so keys()[0] is the oldest session.
    if len(_SESSION_STORE) > _MAX_SESSIONS:
        for k in list(_SESSION_STORE.keys())[: len(_SESSION_STORE) - _MAX_SESSIONS]:
            del _SESSION_STORE[k]


def get_session_history(session_id: str) -> list[tuple[str, str]]:
    return _SESSION_STORE.get(session_id, [])


def clear_session(session_id: str) -> None:
    _SESSION_STORE.pop(session_id, None)


def _append_history(session_id: str, question: str, answer: str) -> None:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = []
        _evict_sessions_if_needed()
    _SESSION_STORE[session_id].append((question, answer))


def build_rag_chain(
    *,
    ticker: str | None = None,
    year: int | None = None,
    filing_type: str | None = None,
    use_cache: bool = True,
) -> RunnableLambda:
    """Return a session-aware RAG chain with free-tier guardrails.

    Accepts a plain string (session "default") or {"question": str, "session_id": str}.
    Pass use_cache=False when fresh generations are required (e.g. evaluation).
    """
    settings = get_settings()
    retriever = build_retriever(ticker=ticker, year=year, filing_type=filing_type)
    llm = ChatGroq(
        model=settings.llm_model,
        api_key=SecretStr(settings.groq_api_key),
        temperature=0,
        max_tokens=settings.llm_max_tokens,
        max_retries=2,
    )
    prompt = build_prompt()
    answer_chain = prompt | llm  # keep the AIMessage — usage_metadata feeds the quota guard

    guard = get_quota_guard()
    cache = get_answer_cache()
    cache_filters = {"ticker": ticker, "year": year, "filing_type": filing_type}
    caching_on = use_cache and settings.answer_cache_enabled

    def _run(inputs: str | dict) -> RAGAnswer:
        if isinstance(inputs, str):
            question, session_id = inputs, "default"
        else:
            question = inputs["question"]
            session_id = inputs.get("session_id", "default")

        history = get_session_history(session_id)[-MAX_HISTORY_TURNS:]
        first_turn = not history

        # Cache only first-turn questions — follow-ups depend on chat history.
        if caching_on and first_turn:
            hit = cache.get(question, cache_filters, settings.llm_model)
            if hit:
                _append_history(session_id, question, hit["answer"])
                return RAGAnswer(
                    answer=hit["answer"],
                    sources=hit["sources"],
                    cached=True,
                    degraded=False,
                )

        docs = retriever.invoke(question)
        context = _format_docs(docs)

        history_messages = [
            msg
            for human, ai in history
            for msg in (HumanMessage(content=human), AIMessage(content=ai))
        ]
        history_chars = sum(len(h) + len(a) for h, a in history)
        est_tokens = (
            estimate_tokens(context)
            + estimate_tokens(question)
            + history_chars // 4
            + _PROMPT_OVERHEAD_TOKENS
            + settings.llm_max_tokens
        )

        try:
            guard.check(est_tokens)
            guard.throttle()
            msg = answer_chain.invoke(
                {"context": context, "question": question, "chat_history": history_messages}
            )
        except QuotaExceededError as exc:
            logger.warning("Groq quota guard tripped: %s", exc)
            return _degraded_answer(docs, str(exc))

        answer = str(msg.content)
        usage = getattr(msg, "usage_metadata", None) or {}
        guard.record(
            usage.get("input_tokens", est_tokens - settings.llm_max_tokens),
            usage.get("output_tokens", estimate_tokens(answer)),
        )

        _append_history(session_id, question, answer)
        sources = _docs_to_sources(docs)
        if caching_on and first_turn:
            cache.set(
                question,
                cache_filters,
                settings.llm_model,
                answer=answer,
                sources=[dict(s) for s in sources],
            )

        logger.debug("session=%s turn=%d", session_id, len(_SESSION_STORE[session_id]))
        return RAGAnswer(answer=answer, sources=sources, cached=False, degraded=False)

    return RunnableLambda(_run)
