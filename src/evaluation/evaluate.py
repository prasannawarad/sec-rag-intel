"""RAGAS evaluation runner. Outputs scores to eval_results/scores.json.

Uses Groq Llama 3.3 70B as the LLM judge (instead of OpenAI default) and the
same local BGE embeddings as the retrieval pipeline. The LLM-as-judge being
the same family as the LLM-as-generator introduces a mild self-evaluation
bias — flag this in the README and spot-check faithfulness manually.

Groq constraints:
  - n=1 only (RAGAS default n=3 is rejected) → set via run_config
  - 100k tokens/day on free tier → use --subset N to evaluate fewer questions
  - A pre-flight check against the shared quota guard (src/chain/quota.py)
    refuses to start a run that would blow the daily budget; --force overrides.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.chain.quota import get_quota_guard
from src.chain.rag_chain import build_rag_chain
from src.config import EVAL_DIR, get_settings

logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
OUTPUT_PATH = EVAL_DIR / "scores.json"

# Conservative per-question cost: ~3k tokens generation (5×512-token chunks
# + answer) + ~4k for the 3 RAGAS judge metrics. A full 20-question run is
# ~140k tokens — more than the whole free-tier day, hence the pre-flight check.
EST_GENERATION_TOKENS_PER_Q = 3_000
EST_JUDGE_TOKENS_PER_Q = 4_000
EST_TOKENS_PER_QUESTION = EST_GENERATION_TOKENS_PER_Q + EST_JUDGE_TOKENS_PER_Q


def _preflight_budget_check(n_questions: int, force: bool) -> None:
    """Refuse to start a run the daily Groq budget can't cover."""
    status = get_quota_guard().status()
    est = n_questions * EST_TOKENS_PER_QUESTION
    if est <= status["tokens_remaining"]:
        return
    affordable = status["tokens_remaining"] // EST_TOKENS_PER_QUESTION
    msg = (
        f"Evaluating {n_questions} questions needs ~{est:,} Groq tokens but only "
        f"{status['tokens_remaining']:,} remain in today's free-tier budget. "
        + (
            f"Try: make eval-subset N={affordable}"
            if affordable
            else "Try again after midnight UTC."
        )
    )
    if force:
        logger.warning("%s — continuing anyway (--force)", msg)
        return
    raise SystemExit(f"{msg}  (Override with --force.)")


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

        # use_cache=False: eval must measure fresh generations, not cached ones.
        # Unique session per question: sharing one session would grow the chat
        # history every question — wrong context AND ever-increasing token burn.
        chain = build_rag_chain(ticker=ticker, year=year, filing_type=filing_type, use_cache=False)
        out = chain.invoke({"question": q["question"], "session_id": f"eval-{q['id']}"})

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


def evaluate(subset: int | None = None, force: bool = False) -> dict:
    """Run RAGAS metrics: faithfulness, answer_relevancy, context_recall.

    Args:
        subset: If given, evaluate only the first N questions (saves Groq tokens).
        force: Skip the pre-flight budget check (run anyway, risking 429s).
    """
    from datasets import Dataset
    from ragas import RunConfig
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics.collections import answer_relevancy, context_recall, faithfulness

    questions = load_golden(subset=subset)
    _preflight_budget_check(len(questions), force=force)
    logger.info("Evaluating %d questions", len(questions))
    preds = run_predictions(questions)

    ds = Dataset.from_list(preds)
    judge_llm, judge_emb = _build_judge()

    # Groq free tier: n=1 only, timeout generous for rate-limit retries
    run_cfg = RunConfig(max_retries=5, max_wait=120, timeout=60)

    result = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall],  # type: ignore[list-item]
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_cfg,
    )

    scores = _extract_scores(result)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(scores, indent=2))
    logger.info("Saved RAGAS scores to %s: %s", OUTPUT_PATH, scores)

    # RAGAS calls the judge internally, so its usage bypasses the guard —
    # book an estimate so the shared daily budget stays honest.
    get_quota_guard().record(len(questions) * EST_JUDGE_TOKENS_PER_Q, 0)
    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None, help="Evaluate first N questions only")
    parser.add_argument(
        "--force", action="store_true", help="Skip the pre-flight Groq budget check"
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(subset=args.subset, force=args.force), indent=2))
