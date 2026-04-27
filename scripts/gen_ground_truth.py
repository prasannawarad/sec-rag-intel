"""Generate ground-truth answers for golden_dataset.json from actual filing chunks.

Pulls text directly from Parquet (not the vector retriever) to avoid circular
evaluation, then calls Groq to produce a concise reference answer.

Usage:
    python -m scripts.gen_ground_truth
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
from langchain_groq import ChatGroq

from src.config import get_settings
from src.logging_setup import configure_logging

configure_logging()

GOLDEN_PATH = Path("src/evaluation/golden_dataset.json")
CHUNKS_GLOB = "data/processed/chunks/**/*.parquet"
MAX_CHARS_PER_QUESTION = 14_000  # ~3.5k tokens; llama-3.3-70b has 128k ctx window

# Per-question: which tickers and item_codes supply the reference context.
# Optional "keywords": prefer chunks containing these patterns (case-insensitive OR).
# JPM has no Item 7/8 in the index; use Item 1A + Item 5 (financial highlights).
QUESTION_META: dict[str, dict] = {
    "q01": {"tickers": ["AAPL"], "item_codes": ["Item 1A"]},
    "q02": {
        "tickers": ["MSFT"],
        "item_codes": ["Item 7", "Item 1A", "Item 1"],
        "keywords": ["cloud", "Azure", "intelligent cloud", "revenue"],
    },
    "q03": {
        "tickers": ["AAPL", "MSFT"],
        "item_codes": ["Item 7", "Item 1"],
        "keywords": ["research and development", "R&D"],
    },
    "q04": {
        "tickers": ["AMZN"],
        "item_codes": ["Item 7", "Item 8"],
        "keywords": ["AWS", "North America", "segment", "revenue"],
    },
    "q05": {"tickers": ["TSLA"], "item_codes": ["Item 1A"]},
    "q06": {"tickers": ["NVDA"], "item_codes": ["Item 7"]},
    "q07": {"tickers": ["JPM"], "item_codes": ["Item 1A", "Item 5"]},
    "q08": {"tickers": ["META"], "item_codes": ["Item 3", "Item 1A"]},
    "q09": {"tickers": ["GOOGL"], "item_codes": ["Item 1A", "Item 7"]},
    "q10": {
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "META"],
        "item_codes": ["Item 1A"],
        "keywords": ["generative AI", "artificial intelligence", "gen AI", "AI"],
    },
    "q11": {
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "JPM", "NVDA", "META"],
        "item_codes": ["Item 7", "Item 5"],
        "keywords": ["repurchase", "buyback", "share repurchase"],
    },
    "q12": {
        "tickers": ["AAPL"],
        "item_codes": ["Item 1A", "Item 1"],
        "keywords": ["manufacturer", "outsource", "contract manufacturer", "third-party"],
    },
    "q13": {
        "tickers": ["MSFT"],
        "item_codes": ["Item 1A", "Item 7"],
        "keywords": ["AI safety", "responsible AI", "artificial intelligence", "safety"],
    },
    "q14": {
        "tickers": ["TSLA", "NVDA"],
        "item_codes": ["Item 7", "Item 8"],
        "keywords": ["gross margin", "gross profit"],
    },
    "q15": {
        "tickers": ["AMZN"],
        "item_codes": ["Item 1A", "Item 7"],
        "keywords": ["climate", "weather", "environmental"],
    },
    "q16": {
        "tickers": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
        "item_codes": ["Item 1A"],
        "keywords": ["China", "Chinese", "PRC"],
    },
    "q17": {
        "tickers": ["JPM"],
        "item_codes": ["Item 9B"],
        "keywords": ["Tier 1", "CET1", "capital ratio"],
    },
    "q18": {"tickers": ["META"], "item_codes": ["Item 7"]},
    "q19": {
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "JPM", "NVDA", "META"],
        "item_codes": ["Item 1A", "Item 9A"],
        "keywords": ["cybersecurity", "cyber", "security incident", "data breach"],
    },
    "q20": {"tickers": ["GOOGL"], "item_codes": ["Item 7", "Item 1"]},
}


def _load_chunks() -> pl.DataFrame:
    return pl.read_parquet(CHUNKS_GLOB, hive_partitioning=True)


def _get_context(df: pl.DataFrame, qid: str) -> str:
    meta = QUESTION_META.get(qid)
    if not meta:
        return ""
    tickers = meta["tickers"]
    item_codes = meta["item_codes"]
    keywords: list[str] = meta.get("keywords", [])
    per_ticker = MAX_CHARS_PER_QUESTION // max(len(tickers), 1)

    parts: list[str] = []
    for ticker in tickers:
        rows = df.filter(
            (pl.col("ticker") == ticker) & pl.col("item_code").is_in(item_codes)
        )
        if rows.is_empty():
            continue
        # Prefer chunks that contain any keyword; fall back to all chunks
        if keywords:
            pattern = "|".join(keywords)
            keyword_rows = rows.filter(pl.col("text").str.contains(f"(?i){pattern}"))
            rows = keyword_rows if not keyword_rows.is_empty() else rows
        rows = rows.sort(["item_code", "chunk_id"])

        budget = per_ticker
        for row in rows.iter_rows(named=True):
            snippet = row["text"][:budget]
            parts.append(f"[{ticker} | {row['section_label']}]\n{snippet}")
            budget -= len(snippet)
            if budget <= 0:
                break
    return "\n\n---\n\n".join(parts)


def _generate_answer(llm: ChatGroq, question: str, context: str) -> str:
    prompt = (
        "You are a financial analyst. Based ONLY on the excerpts below from SEC filings, "
        "write a factual, concise ground-truth answer (2–5 sentences) to the question. "
        "Do not add information not present in the excerpts.\n\n"
        f"EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def main() -> None:
    settings = get_settings()
    llm = ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=512,
    )

    df = _load_chunks()
    data = json.loads(GOLDEN_PATH.read_text())
    questions = data["questions"]

    for i, q in enumerate(questions):
        if q["ground_truth"] != "TODO":
            print(f"  skip {q['id']} (already filled)")
            continue

        context = _get_context(df, q["id"])
        if not context:
            print(f"  skip {q['id']} (no context found)")
            continue

        print(f"  generating {q['id']}: {q['question'][:60]}…")
        answer = _generate_answer(llm, q["question"], context)
        q["ground_truth"] = answer
        print(f"    → {answer[:120]}…")

        # Respect Groq free-tier rate limits (~30 req/min)
        if i < len(questions) - 1:
            time.sleep(2)

    GOLDEN_PATH.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
