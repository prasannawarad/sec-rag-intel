"""RAGAS evaluation runner. Outputs scores to eval_results/scores.json.

Uses Groq Llama 3.3 70B as the LLM judge (instead of OpenAI default) and the
same local BGE embeddings as the retrieval pipeline. The LLM-as-judge being
the same family as the LLM-as-generator introduces a mild self-evaluation
bias — flag this in the README and spot-check faithfulness manually.

Groq constraints:
  - n=1 only (RAGAS default n=3 is rejected) → set via run_config
  - 100k tokens/day on free tier → use --subset N to evaluate fewer questions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.chain.rag_chain import build_rag_chain
from src.config import EVAL_DIR, get_settings

logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
OUTPUT_PATH = EVAL_DIR / "scores.json"


def load_golden(subset: int | None = None) -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text())
    qs = data["questions"]
    return qs[:subset] if subset else qs


def run_predictions(questions: list[dict]) -> list[dict]:
    """Generate answers + retrieved contexts for each question."""
    from src.retrieval.retriever import build_retriever

    results = []
    for q in questions:
        filters = q.get("filters", {}) or {}
        ticker = filters.get("ticker")
        year = filters.get("year")
        filing_type = filters.get("filing_type")

        retriever = build_retriever(ticker=ticker, year=year, filing_type=filing_type)
        docs = retriever.invoke(q["question"])
        contexts = [d.page_content for d in docs]

        chain = build_rag_chain(ticker=ticker, year=year, filing_type=filing_type)
        out = chain.invoke(q["question"])

        results.append(
            {
                "question": q["question"],
                "answer": out["answer"],
                "ground_truth": q["ground_truth"],
                "contexts": contexts,
            }
        )
    return results


def _build_judge():
    """Wrap Groq + local BGE for RAGAS."""
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    settings = get_settings()
    # max_tokens prevents runaway token usage on Groq free tier
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=512,
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    return judge_llm, judge_emb


def _extract_scores(result) -> dict:
    """Extract mean metric scores from a RAGAS EvaluationResult."""
    # EvaluationResult supports .scores (list of per-question dicts) in newer RAGAS
    if hasattr(result, "scores"):
        rows = result.scores
        if rows:
            keys = rows[0].keys()
            return {
                k: float(
                    sum(r[k] for r in rows if r[k] is not None)
                    / max(sum(1 for r in rows if r[k] is not None), 1)
                )
                for k in keys
            }
    # Older versions expose items() or act as a dict
    if hasattr(result, "items"):
        return {k: float(v) for k, v in result.items() if v is not None}
    # Fall back: convert to pandas and take column means
    try:
        df = result.to_pandas()
        numeric = df.select_dtypes("number")
        return {col: float(numeric[col].mean()) for col in numeric.columns}
    except Exception:
        return {}


def evaluate(subset: int | None = None) -> dict:
    """Run RAGAS metrics: faithfulness, answer_relevancy, context_recall.

    Args:
        subset: If given, evaluate only the first N questions (saves Groq tokens).
    """
    from datasets import Dataset
    from ragas import RunConfig
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics.collections import answer_relevancy, context_recall, faithfulness

    questions = load_golden(subset=subset)
    logger.info("Evaluating %d questions", len(questions))
    preds = run_predictions(questions)

    ds = Dataset.from_list(preds)
    judge_llm, judge_emb = _build_judge()

    # Groq free tier: n=1 only, timeout generous for rate-limit retries
    run_cfg = RunConfig(max_retries=5, max_wait=120, timeout=60)

    result = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_cfg,
    )

    scores = _extract_scores(result)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(scores, indent=2))
    logger.info("Saved RAGAS scores to %s: %s", OUTPUT_PATH, scores)
    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None, help="Evaluate first N questions only")
    args = parser.parse_args()
    print(json.dumps(evaluate(subset=args.subset), indent=2))
