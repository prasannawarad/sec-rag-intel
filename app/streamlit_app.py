"""Streamlit chat UI for SEC RAG Intel — "editorial ledger" theme.

Pure UI layer — no business logic. Calls build_rag_chain() directly
(same process) so we get conversational memory for free without
needing the FastAPI server running.

Layout:
  Sidebar  — filters, live Groq free-tier budget meter, New Chat
  Main     — filing-cover masthead, example-question pills (empty state),
             chat history with source chips + cache/fallback badges

Design: financial print. Paper + ink + ledger green, Fraunces serif
display over IBM Plex Mono data. Theme colors live in .streamlit/config.toml.
"""

from __future__ import annotations

import html
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

# `streamlit run` puts app/ (the script's own dir) on sys.path, not the repo
# root — so add the root here, before the `src` imports, to resolve them both
# locally (`make ui`) and on HuggingFace Spaces.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chain.quota import get_quota_guard  # noqa: E402
from src.chain.rag_chain import build_rag_chain, clear_session  # noqa: E402
from src.ingest.downloader import DEFAULT_TICKERS  # noqa: E402

EXAMPLE_QUESTIONS = [
    "What are Apple's main supply chain risks?",
    "How does Microsoft describe its AI strategy?",
    "What drives Amazon's revenue growth?",
    "What legal proceedings does JPMorgan disclose?",
]

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..700;1,9..144,300..700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #1C1B17;
    --paper: #F7F2E9;
    --paper-deep: #EFE7D6;
    --ledger: #0E5A43;
    --oxblood: #7A2E2E;
    --hairline: rgba(28, 27, 23, 0.35);
}

html, body, .stApp, p, li, label, input, textarea, button {
    font-family: 'Fraunces', Georgia, serif;
}
.stApp { background: var(--paper); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---- masthead: SEC filing cover page ---- */
.masthead { margin: 0 0 1.6rem 0; }
.mast-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem; letter-spacing: 0.22em; color: var(--ledger);
    border-top: 3px solid var(--ink); border-bottom: 1px solid var(--hairline);
    padding: 0.45rem 0; text-transform: uppercase;
}
.mast-title {
    font-family: 'Fraunces', serif; font-weight: 350;
    font-size: clamp(2.2rem, 5vw, 3.4rem); line-height: 1.05;
    color: var(--ink); margin: 1.1rem 0 0.9rem 0;
}
.mast-title em { font-style: italic; color: var(--ledger); }
.mast-sub {
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
    letter-spacing: 0.14em; color: var(--ink);
    border-top: 1px solid var(--hairline); border-bottom: 3px double var(--ink);
    padding: 0.4rem 0 0.5rem 0; text-transform: uppercase;
}

/* ---- chat ---- */
[data-testid="stChatMessage"] {
    background: #FFFDF7; border: 1px solid var(--hairline);
    border-left: 4px solid var(--ledger);
    border-radius: 2px; padding: 1rem 1.2rem; margin-bottom: 0.4rem;
    box-shadow: 2px 3px 0 rgba(28, 27, 23, 0.08);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: transparent; border-left: 4px solid var(--ink);
    box-shadow: none;
}
[data-testid="stChatInput"] textarea { font-family: 'Fraunces', serif; }

/* ---- source chips ---- */
.src-block { margin-top: 0.9rem; border-top: 1px solid var(--hairline); padding-top: 0.6rem; }
.src-head {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.2em; color: var(--ledger); text-transform: uppercase;
}
.src-chip {
    display: inline-block; margin: 0.35rem 0.4rem 0 0; padding: 0.22rem 0.55rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    border: 1px solid var(--hairline); border-radius: 2px;
    background: var(--paper-deep); color: var(--ink);
}
.src-chip b { color: var(--ledger); font-weight: 600; }
.src-chip i { font-style: normal; opacity: 0.65; }

/* ---- badges ---- */
.badge {
    display: inline-block; margin-bottom: 0.5rem; padding: 0.15rem 0.5rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.16em; text-transform: uppercase; border-radius: 2px;
}
.badge-cache { background: var(--ledger); color: var(--paper); }
.badge-degraded { background: var(--oxblood); color: var(--paper); }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { border-right: 1px solid var(--hairline); }
.side-brand {
    font-family: 'Fraunces', serif; font-weight: 400; font-size: 1.35rem;
    color: var(--ink); border-bottom: 3px double var(--ink); padding-bottom: 0.6rem;
}
.side-brand em { font-style: italic; color: var(--ledger); }
.side-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.2em; color: var(--ledger); text-transform: uppercase;
    margin: 1rem 0 0.2rem 0;
}

/* ---- budget meter ---- */
.meter-row {
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: var(--ink);
    margin: 0.45rem 0 0.15rem 0;
}
.meter { height: 7px; background: var(--paper); border: 1px solid var(--hairline); }
.meter-fill { height: 100%; transition: width 0.4s ease; }

/* ---- buttons (example pills, new chat) ---- */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.04em; color: var(--ink);
    background: transparent; border: 1px solid var(--hairline); border-radius: 2px;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: var(--ledger); color: var(--paper); border-color: var(--ledger);
}
</style>
"""


def _meter_html(label: str, used: int, budget: int, unit: str) -> str:
    pct = min(used / budget * 100, 100) if budget else 0
    color = "#0E5A43" if pct < 60 else ("#B07A1E" if pct < 85 else "#7A2E2E")
    return (
        f'<div class="meter-row"><span>{label}</span>'
        f"<span>{used:,} / {budget:,} {unit}</span></div>"
        f'<div class="meter"><div class="meter-fill" '
        f'style="width:{pct:.0f}%;background:{color}"></div></div>'
    )


def _render_sources(sources: list[dict]) -> None:
    """Deduplicated source chips (5 chunks often cite the same section)."""
    seen: list[tuple] = []
    for s in sources:
        key = (s["ticker"], s["year"], s["filing_type"], s.get("section_label", ""))
        if key not in seen:
            seen.append(key)
    chips = "".join(
        f'<span class="src-chip"><b>{html.escape(str(t))}</b> {html.escape(str(y))} '
        f"{html.escape(str(ft))} <i>· {html.escape(str(sec))}</i></span>"
        for t, y, ft, sec in seen
    )
    st.markdown(
        f'<div class="src-block"><span class="src-head">Cited filings</span><br>{chips}</div>',
        unsafe_allow_html=True,
    )


def _render_badge(msg: dict) -> None:
    if msg.get("cached"):
        st.markdown(
            '<span class="badge badge-cache">Served from cache · 0 tokens</span>',
            unsafe_allow_html=True,
        )
    elif msg.get("degraded"):
        st.markdown(
            '<span class="badge badge-degraded">Budget fallback · excerpts only</span>',
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="SEC Filing Intelligence", page_icon="📜", layout="wide")
st.markdown(_CSS, unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- sidebar ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="side-brand">SEC Filing <em>Intelligence</em></div>', unsafe_allow_html=True
    )

    st.markdown('<div class="side-label">Filters</div>', unsafe_allow_html=True)
    ticker = st.selectbox("Company", options=["(any)"] + DEFAULT_TICKERS)
    year = st.number_input("Year", min_value=2015, max_value=2030, value=2025, step=1)
    filing_type = st.selectbox("Filing type", options=["(any)", "10-K", "10-Q"])

    st.markdown('<div class="side-label">Groq free tier · today</div>', unsafe_allow_html=True)
    quota = get_quota_guard().status()
    st.markdown(
        _meter_html("TOKENS", quota["tokens_used"], quota["tokens_budget"], "tok")
        + _meter_html("REQUESTS", quota["requests_used"], quota["requests_budget"], "req"),
        unsafe_allow_html=True,
    )
    st.caption("Resets at midnight UTC. Repeat questions are served from cache at zero cost.")

    st.divider()
    if st.button("✳ New chat", use_container_width=True):
        clear_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "**Stack:** BGE-small embeddings · ChromaDB · "
        "Groq Llama 3.3 70B · LangChain LCEL · RAGAS eval"
    )

# ---- masthead --------------------------------------------------------
st.markdown(
    f"""
<div class="masthead">
  <div class="mast-kicker">United States Securities and Exchange Commission · EDGAR corpus</div>
  <div class="mast-title">SEC Filing <em>Intelligence</em></div>
  <div class="mast-sub">
    <span>Form 10-K / 10-Q · Natural-language analysis · Grounded &amp; cited</span>
    <span>{datetime.now(tz=UTC).strftime("%d %b %Y").upper()}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---- empty state: example-question pills -----------------------------
queued_question: str | None = None
if not st.session_state.messages:
    st.caption(
        f"Ask about the latest filings for {', '.join(DEFAULT_TICKERS)} — or start from one of these:"
    )
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(example, use_container_width=True):
            queued_question = example

# ---- history ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        _render_badge(msg)
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])

# ---- input + answer --------------------------------------------------
question = st.chat_input("e.g. What are Apple's main supply chain risks?") or queued_question

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Consulting the filings…"):
                chain = build_rag_chain(
                    ticker=None if ticker == "(any)" else ticker,
                    year=int(year),
                    filing_type=None if filing_type == "(any)" else filing_type,
                )
                result = chain.invoke(
                    {"question": question, "session_id": st.session_state.session_id}
                )
        except Exception:
            st.error(
                "Something went wrong while answering. The free-tier LLM may be "
                "rate-limited — please try again in a minute."
            )
            st.stop()

        _render_badge(result)
        st.write(result["answer"])
        if result["sources"]:
            _render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "cached": result["cached"],
            "degraded": result["degraded"],
        }
    )
    st.rerun()  # refresh the sidebar budget meter with this turn's spend
