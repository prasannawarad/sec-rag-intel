"""Prompt templates for the RAG chain."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about SEC filings.

Strict rules:
1. Answer ONLY using facts present in the provided context. If the context does not contain
   the answer, reply: "I cannot find this information in the provided filings."
2. Always cite the filing using the format: [TICKER YEAR FILING_TYPE - SECTION].
3. Quote exact dollar amounts, percentages, and dates verbatim.
4. Do not speculate, infer trends, or use outside knowledge.
"""

USER_TEMPLATE = """Context from SEC filings:
---
{context}
---

Question: {question}

Answer (with citations):"""


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_TEMPLATE),
        ]
    )
