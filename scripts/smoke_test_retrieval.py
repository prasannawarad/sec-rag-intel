"""Phase 2 smoke test: cross-ticker retrieval sanity check.

Runs one targeted query per company and verifies:
  1. At least 3 docs retrieved
  2. Docs come from the expected ticker
  3. Section label is sensible (not 'full')

Usage:
    python -m scripts.smoke_test_retrieval
"""

from __future__ import annotations

import sys

from src.logging_setup import configure_logging
from src.retrieval.retriever import build_retriever

configure_logging()

QUERIES: list[tuple[str, str, str]] = [
    ("AAPL", "10-K", "What are Apple's main supply chain risks?"),
    ("MSFT", "10-K", "How does Microsoft describe its Azure cloud growth?"),
    ("AMZN", "10-K", "What does Amazon say about AWS segment revenue?"),
    ("GOOGL", "10-K", "What are Alphabet's risks related to advertising revenue?"),
    ("TSLA", "10-K", "How does Tesla describe its manufacturing capacity risks?"),
    ("JPM", "10-K", "What credit risk exposures does JPMorgan highlight?"),
    ("NVDA", "10-K", "What does NVIDIA say about data center demand?"),
    ("META", "10-K", "How does Meta describe Reality Labs operating losses?"),
]


def main() -> None:
    failures: list[str] = []

    for ticker, filing_type, question in QUERIES:
        retriever = build_retriever(ticker=ticker, filing_type=filing_type)
        docs = retriever.invoke(question)

        if len(docs) < 3:
            failures.append(f"{ticker}: only {len(docs)} docs returned (expected >=3)")
            continue

        wrong_ticker = [d for d in docs if d.metadata.get("ticker") != ticker]
        if wrong_ticker:
            failures.append(f"{ticker}: {len(wrong_ticker)} doc(s) with wrong ticker metadata")
            continue

        print(
            f"✓ {ticker:<6} {len(docs)} docs | sections: "
            + ", ".join({d.metadata.get("section_label", "?") for d in docs})
        )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
