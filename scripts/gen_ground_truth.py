"""Generate ground-truth answers for golden_dataset.json from actual filing chunks.

Pulls text directly from Parquet (not the vector retriever) to avoid circular
evaluation, then asks the JUDGE model (not the generator under test) for a
concise reference answer. If the excerpts don't contain the answer the model
replies INSUFFICIENT and the entry is left for manual attention — a hedged
"not in the excerpts" reference would silently zero out RAGAS context_recall.

LLM-drafted references are a starting point: review them before citing scores.

Usage:
    python -m scripts.gen_ground_truth                  # fill TODO entries only
    python -m scripts.gen_ground_truth --ids q02 q17    # regenerate specific ids
    python -m scripts.gen_ground_truth --all            # regenerate everything
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import polars as pl
from langchain_groq import ChatGroq
from pydantic import SecretStr

from src.chain.quota import get_quota_guard
from src.config import get_settings
from src.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

GOLDEN_PATH = Path("src/evaluation/golden_dataset.json")
CHUNKS_GLOB = "data/processed/chunks/**/*.parquet"
MAX_CHARS_PER_QUESTION = 16_000  # ~4k tokens — the free-tier daily budget is the limit
MIN_CHARS_PER_TICKER = 2_500  # ≥ one full chunk per company on multi-company questions
INSUFFICIENT = "INSUFFICIENT"

# Per-question: which tickers and item_codes supply the reference context.
# Optional "keywords": prefer chunks containing these patterns (case-insensitive OR).
# JPM files a "wrapper" 10-K: its MD&A and financials live in the "Annual Report"
# pseudo-section (see src/ingest/parser.py), not Items 7/8.
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
    "q05": {"tickers": ["TSLA"], "item_codes": ["Item 1A"], "keywords": ["supplier", "supply"]},
    "q06": {"tickers": ["NVDA"], "item_codes": ["Item 7"], "keywords": ["Data Center"]},
    "q07": {
        "tickers": ["JPM"],
        "item_codes": ["Item 1A", "Annual Report"],
        "keywords": ["credit risk", "credit exposure", "wholesale credit", "consumer credit"],
    },
    "q08": {
        "tickers": ["META"],
        "item_codes": ["Item 3", "Item 1A", "Item 8"],
        "keywords": ["privacy", "FTC", "litigation"],
    },
    "q09": {
        "tickers": ["GOOGL"],
        "item_codes": ["Item 1A", "Item 7"],
        "keywords": ["advertising", "advertisers"],
    },
    "q10": {
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "JPM", "NVDA", "META"],
        "item_codes": ["Item 1A"],
        # Only the specific phrase: a bare "AI" outranks it and the reference misses companies.
        "keywords": ["generative AI", "gen AI", "GenAI"],
    },
    "q11": {
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "JPM", "NVDA", "META"],
        "item_codes": ["Item 7", "Item 5", "Annual Report"],
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
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NVDA"],
        "item_codes": ["Item 1A"],
        "keywords": ["China", "Chinese", "PRC"],
    },
    "q17": {
        "tickers": ["JPM"],
        "item_codes": ["Annual Report"],
        "keywords": ["Tier 1 capital ratio", "CET1", "capital ratio"],
    },
    "q18": {"tickers": ["META"], "item_codes": ["Item 7"], "keywords": ["Reality Labs"]},
    "q19": {
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "JPM", "NVDA", "META"],
        "item_codes": ["Item 1C", "Item 1A", "Annual Report"],
        "keywords": ["cybersecurity incident", "cyberattack", "security incident", "data breach"],
    },
    "q20": {
        "tickers": ["GOOGL"],
        "item_codes": ["Item 7", "Item 1"],
        "keywords": ["Other Bets"],
    },
}


def _load_chunks() -> pl.DataFrame:
    return pl.read_parquet(CHUNKS_GLOB, hive_partitioning=True)


def _get_context(df: pl.DataFrame, qid: str) -> str:
    meta = QUESTION_META.get(qid)
    if not meta:
        return ""
    tickers = meta["tickers"]
    keywords: list[str] = meta.get("keywords", [])
    per_ticker = max(MAX_CHARS_PER_QUESTION // len(tickers), MIN_CHARS_PER_TICKER)

    parts: list[str] = []
    for ticker in tickers:
        rows = df.filter(
            (pl.col("ticker") == ticker) & pl.col("item_code").is_in(meta["item_codes"])
        )
        if rows.is_empty():
            continue
        if keywords:
            # Most keyword hits first — document order would spend the budget on
            # whatever happens to come first in the section.
            # Word boundaries: a bare "AI" would otherwise match "obtain", "maintain"…
            pattern = r"(?i)\b(?:" + "|".join(keywords) + r")\b"
            rows = rows.with_columns(pl.col("text").str.count_matches(pattern).alias("hits"))
            hits = rows.filter(pl.col("hits") > 0).sort("hits", descending=True)
            rows = hits if not hits.is_empty() else rows
        else:
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
        "write a factual, concise ground-truth answer (2–5 sentences) to the question, "
        "including specific figures where the excerpts give them. Do not add information "
        "not present in the excerpts, and do not mention the excerpts themselves. If the "
        f"excerpts do not contain the answer, reply with exactly {INSUFFICIENT}.\n\n"
        f"EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
    msg = llm.invoke(prompt)
    # Direct Groq call — book it against the shared daily budget like the chain does.
    if msg.usage_metadata:
        get_quota_guard(get_settings().judge_model).record(
            msg.usage_metadata["input_tokens"], msg.usage_metadata["output_tokens"]
        )
    return str(msg.content).strip()


def main(ids: list[str] | None = None, regenerate_all: bool = False) -> None:
    settings = get_settings()
    # The judge model writes references: the generator under test must not
    # author its own answer key.
    llm = ChatGroq(
        model=settings.judge_model,
        api_key=SecretStr(settings.groq_api_key),
        temperature=0,
        max_tokens=1024,
        reasoning_effort="low",
    )

    df = _load_chunks()
    data = json.loads(GOLDEN_PATH.read_text())
    targets = [
        q
        for q in data["questions"]
        if regenerate_all or (q["id"] in ids if ids else q["ground_truth"] == "TODO")
    ]
    insufficient: list[str] = []

    for i, q in enumerate(targets):
        context = _get_context(df, q["id"])
        if not context:
            logger.warning("skip %s (no context found)", q["id"])
            continue

        get_quota_guard(settings.judge_model).check(len(context) // 4 + 1024)
        logger.info("generating %s: %s", q["id"], q["question"][:60])
        answer = _generate_answer(llm, q["question"], context)
        if answer == INSUFFICIENT:
            insufficient.append(q["id"])
            q["ground_truth"] = "TODO"
            logger.warning("%s: excerpts don't contain the answer — fix QUESTION_META", q["id"])
        else:
            q["ground_truth"] = answer
            logger.info("  → %s…", answer[:120])
        # Save as we go: a quota stop mid-run keeps everything generated so far.
        GOLDEN_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

        if i < len(targets) - 1:
            time.sleep(2)  # stay well under the requests-per-minute limit

    logger.info("Wrote %s", GOLDEN_PATH)
    if insufficient:
        logger.warning("Left as TODO (no answer in context): %s", insufficient)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ids", nargs="+", help="Regenerate only these question ids")
    group.add_argument("--all", action="store_true", help="Regenerate every question")
    args = parser.parse_args()
    main(ids=args.ids, regenerate_all=args.all)
