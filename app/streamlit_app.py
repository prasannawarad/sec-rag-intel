"""Streamlit chat UI for SEC RAG Intel.

Pure UI layer — no business logic. Calls build_rag_chain() directly
(same process) so we get conversational memory for free without
needing the FastAPI server running.

Layout:
  Sidebar  — company / year / filing-type filters + New Chat button
  Main     — st.chat_message history + st.chat_input at the bottom
"""

from __future__ import annotations

import uuid

import streamlit as st

from src.chain.rag_chain import build_rag_chain, clear_session
from src.ingest.downloader import DEFAULT_TICKERS


def _render_sources(sources: list[dict]) -> None:
    """Render a source list inside the current st context."""
    with st.expander("Sources", expanded=False):
        for s in sources:
            st.markdown(
                f"- **{s['ticker']}** {s['year']} {s['filing_type']}"
                f" — *{s.get('section_label', '')}*"
            )

st.set_page_config(page_title="SEC RAG Intel", page_icon="📊", layout="wide")

# ── Session bootstrap ──────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str, "sources": list}
    st.session_state.messages = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 SEC RAG Intel")
    st.caption("Natural-language queries over 10-K / 10-Q filings")
    st.divider()

    st.subheader("Filters")
    ticker = st.selectbox("Company", options=["(any)"] + DEFAULT_TICKERS)
    year = st.number_input("Year", min_value=2015, max_value=2030, value=2025, step=1)
    filing_type = st.selectbox("Filing type", options=["(any)", "10-K", "10-Q"])

    st.divider()
    if st.button("🗑️ New chat", use_container_width=True):
        clear_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "**Stack:** BGE-small embeddings · ChromaDB · "
        "Groq Llama 3.3 70B · LangChain LCEL · RAGAS eval"
    )

# ── Main area ──────────────────────────────────────────────────────────────────
st.header("SEC Filing Intelligence")
st.caption(
    "Ask questions about the latest 10-K filings for AAPL, MSFT, AMZN, "
    "GOOGL, TSLA, JPM, NVDA, and META. Answers are grounded and cited."
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])

# Chat input (pinned to bottom by Streamlit)
if question := st.chat_input("e.g. What are Apple's main supply chain risks?"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching filings and generating answer…"):
            chain = build_rag_chain(
                ticker=None if ticker == "(any)" else ticker,
                year=int(year),
                filing_type=None if filing_type == "(any)" else filing_type,
            )
            result = chain.invoke(
                {
                    "question": question,
                    "session_id": st.session_state.session_id,
                }
            )

        st.write(result["answer"])
        if result["sources"]:
            _render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
