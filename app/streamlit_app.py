"""Streamlit UI for SEC RAG Intel.

Pure UI: it talks to the FastAPI backend or imports the chain directly.
No business logic lives in this file.
"""

from __future__ import annotations

import streamlit as st

from src.chain.rag_chain import build_rag_chain
from src.ingest.downloader import DEFAULT_TICKERS

st.set_page_config(page_title="SEC RAG Intel", layout="wide")
st.title("SEC Filing Intelligence")
st.caption(
    "Ask natural-language questions about 10-K / 10-Q filings — answers are grounded and cited."
)

with st.sidebar:
    st.header("Filters")
    ticker = st.selectbox("Company", options=["(any)"] + DEFAULT_TICKERS)
    year = st.number_input("Year", min_value=2015, max_value=2030, value=2024, step=1)
    filing_type = st.selectbox("Filing type", options=["(any)", "10-K", "10-Q"])

question = st.text_area(
    "Your question",
    placeholder="e.g. What are Apple's main risk factors related to supply chain?",
    height=120,
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    chain = build_rag_chain(
        ticker=None if ticker == "(any)" else ticker,
        year=int(year),
        filing_type=None if filing_type == "(any)" else filing_type,
    )
    with st.spinner("Retrieving and generating answer..."):
        result = chain.invoke(question.strip())

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    for s in result["sources"]:
        st.markdown(f"- **{s['ticker']}** {s['year']} {s['filing_type']} — _{s['section']}_")
