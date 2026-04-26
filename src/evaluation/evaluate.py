"""RAGAS evaluation runner. Outputs scores to eval_results/scores.json."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.chain.rag_chain import build_rag_chain
from src.config import EVAL_DIR

logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
OUTPUT_PATH = EVAL_DIR / "scores.json"


def load_golden() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text())
    return data["questions"]


def run_predictions(questions: list[dict]) -> list[dict]:
    """Generate answers + retrieved contexts for each question."""
    results = []
    for q in questions:
        filters = q.get("filters", {}) or {}
        chain = build_rag_chain(
            ticker=filters.get("ticker"),
            year=filters.get("year"),
            filing_type=filters.get("filing_type"),
        )
        out = chain.invoke(q["question"])
        results.append(
            {
                "question": q["question"],
                "answer": out["answer"],
                "ground_truth": q["ground_truth"],
                "contexts": [str(s) for s in out["sources"]],
            }
        )
    return results


def evaluate() -> dict:
    """Run RAGAS metrics: faithfulness, answer_relevancy, context_recall."""
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_recall, faithfulness

    questions = load_golden()
    preds = run_predictions(questions)

    ds = Dataset.from_list(preds)
    result = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall])

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    scores = {k: float(v) for k, v in result.items()} if hasattr(result, "items") else dict(result)
    OUTPUT_PATH.write_text(json.dumps(scores, indent=2))
    logger.info("Saved RAGAS scores to %s", OUTPUT_PATH)
    return scores


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
