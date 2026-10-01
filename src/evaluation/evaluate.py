"""RAGAS evaluation runner.

Per-question scores accumulate in eval_results/per_question.json (keyed by
question id), and eval_results/scores.json holds the means over every question
scored so far. A full 20-question run exceeds one free-tier day, so it can be
split: `--subset 10` today, `--subset 10 --offset 10` tomorrow.

Uses Groq gpt-oss-120b as the LLM judge (settings.judge_model) and the same
local BGE embeddings as the retrieval pipeline. The judge is a different model
family from the generator (settings.llm_model), avoiding self-evaluation bias.

Groq constraints:
  - n=1 only — RAGAS detects ChatGroq lacks multi-completion and loops instead
  - ~30 RPM / low TPM → RAGAS runs with max_workers=2, not the default 16
  - 100k tokens/day on free tier → use --subset/--offset to split the run
  - A pre-flight check against the shared quota guard (src/chain/quota.py)
    refuses to start a run that would blow the daily budget; --force overrides.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

from src.chain.quota import get_quota_guard
from src.chain.rag_chain import build_rag_chain
from src.config import EVAL_DIR, get_settings

logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
OUTPUT_PATH = EVAL_DIR / "scores.json"
PER_QUESTION_PATH = EVAL_DIR / "per_question.json"
METRICS = ("faithfulness", "answer_relevancy", "context_recall")

# Per-question cost measured on the 480-token-chunk index (Sept 2026): ~3.5k
# generator tokens (5 chunks + answer) and ~11k judge tokens (faithfulness makes
# 2 calls with the full contexts, answer_relevancy 3, context_recall 1).
# Generator and judge are separate Groq models with separate daily budgets.
EST_GENERATION_TOKENS_PER_Q = 3_500
EST_JUDGE_TOKENS_PER_Q = 11_000


def _preflight_budget_check(n_questions: int, force: bool) -> None:
    """Refuse to start a run either model's daily Groq budget can't cover."""
    settings = get_settings()
    gen_left = get_quota_guard(settings.llm_model).status()["tokens_remaining"]
    judge_left = get_quota_guard(settings.judge_model).status()["tokens_remaining"]
    affordable = min(gen_left // EST_GENERATION_TOKENS_PER_Q, judge_left // EST_JUDGE_TOKENS_PER_Q)
    if n_questions <= affordable:
        return
    msg = (
        f"Evaluating {n_questions} questions needs ~"
        f"{n_questions * EST_GENERATION_TOKENS_PER_Q:,} generator + "
        f"~{n_questions * EST_JUDGE_TOKENS_PER_Q:,} judge tokens, but today's budgets "
        f"have {gen_left:,} and {judge_left:,} left. "
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


def load_golden(subset: int | None = None, offset: int = 0) -> list[dict]:
    qs = json.loads(GOLDEN_PATH.read_text())["questions"][offset:]
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
        if out["degraded"]:
            # A retrieval-only fallback is not an LLM answer — scoring it would
            # silently corrupt the results. Stop; the finished questions are lost
            # but no judge tokens are spent on them.
            raise SystemExit(
                f"Quota guard tripped at question {q['id']} — aborting before judging. "
                "Re-run later with a smaller --subset/--offset."
            )

        # RAGAS v2 column names (the v1 names still work but go through a converter)
        results.append(
            {
                "id": q["id"],
                "user_input": q["question"],
                "response": out["answer"],
                "reference": q["ground_truth"],
                "retrieved_contexts": contexts,
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
    # gpt-oss is a reasoning model: reasoning tokens count toward max_tokens, so
    # keep effort low and leave headroom for the JSON verdict RAGAS parses.
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model=settings.judge_model,
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=2048,  # 1024 truncated faithfulness statement extraction
            reasoning_effort="low",
        ),
        # ChatGroq has an `n` attribute, so RAGAS would set n=3 for answer_relevancy
        # and Groq rejects n>1 — bypass_n makes it issue 3 separate n=1 calls instead.
        bypass_n=True,
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    return judge_llm, judge_emb


def _clean(value: object) -> float | None:
    """RAGAS reports a failed judge call as NaN — store it as null, not a score."""
    if value is None:
        return None
    f = float(value)  # type: ignore[arg-type]
    return None if math.isnan(f) else round(f, 4)


def _aggregate(rows: dict[str, dict]) -> dict:
    """Mean of each metric over all scored questions, skipping failed (null) ones."""
    summary: dict = {"n_questions": len(rows)}
    for m in METRICS:
        vals = [r[m] for r in rows.values() if r.get(m) is not None]
        summary[m] = round(sum(vals) / len(vals), 4) if vals else None
        summary[f"{m}_n_scored"] = len(vals)
    return summary


def evaluate(subset: int | None = None, offset: int = 0, force: bool = False) -> dict:
    """Run RAGAS metrics: faithfulness, answer_relevancy, context_recall.

    Args:
        subset: If given, evaluate only N questions (saves Groq tokens).
        offset: Skip the first `offset` questions (to split a run across days).
        force: Skip the pre-flight budget check (run anyway, risking 429s).
    """
    from datasets import Dataset
    from ragas import RunConfig
    from ragas import evaluate as ragas_evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.dataset_schema import EvaluationResult

    # The legacy metric instances are what ragas.evaluate() accepts; the
    # ragas.metrics.collections names are modules/classes for a different API.
    from ragas.metrics import answer_relevancy, context_recall, faithfulness

    questions = load_golden(subset=subset, offset=offset)
    if not questions:
        raise SystemExit(f"No questions at offset {offset}.")
    _preflight_budget_check(len(questions), force=force)
    logger.info("Evaluating %d questions", len(questions))
    preds = run_predictions(questions)

    ds = Dataset.from_list([{k: v for k, v in p.items() if k != "id"} for p in preds])
    judge_llm, judge_emb = _build_judge()

    # Groq free tier: one judge call at a time — gpt-oss's tokens-per-minute limit
    # made parallel calls queue behind 429 retries until they hit the timeout
    # (null scores). Sequential + a long timeout trades wall-clock for complete data.
    run_cfg = RunConfig(max_workers=1, max_retries=6, max_wait=60, timeout=300)

    result = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_cfg,
        token_usage_parser=get_token_usage_for_openai,  # Groq returns OpenAI-shaped usage
    )
    assert isinstance(result, EvaluationResult)  # only an Executor if return_executor=True

    # RAGAS calls the judge directly, bypassing the guard — book its real usage
    # (fall back to the estimate if the parser saw nothing) so the budget stays honest.
    usage = result.total_tokens()
    in_tok, out_tok = getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0)
    if not in_tok + out_tok:
        in_tok = len(questions) * EST_JUDGE_TOKENS_PER_Q
    get_quota_guard(get_settings().judge_model).record(in_tok, out_tok)
    logger.info("Judge usage: %d input + %d output tokens", in_tok, out_tok)

    run_date = datetime.now(UTC).date().isoformat()
    settings = get_settings()
    df = result.to_pandas()
    rows = json.loads(PER_QUESTION_PATH.read_text()) if PER_QUESTION_PATH.exists() else {}
    for pred, (_, rec) in zip(preds, df.iterrows(), strict=True):
        rows[pred["id"]] = {
            **{m: _clean(rec.get(m)) for m in METRICS},
            "response": pred["response"],  # kept so a bad score can be inspected for free
            "model": settings.llm_model,
            "date": run_date,
        }

    summary = {
        **_aggregate(rows),
        "llm": settings.llm_model,
        "judge": settings.judge_model,
        "embeddings": settings.embedding_model_name,
        "vector_store": settings.vector_store_mode,
        "updated": run_date,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PER_QUESTION_PATH.write_text(json.dumps(dict(sorted(rows.items())), indent=2))
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    logger.info("Saved RAGAS scores to %s", OUTPUT_PATH)
    return summary


if __name__ == "__main__":
    import argparse

    from src.logging_setup import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None, help="Evaluate N questions only")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N questions")
    parser.add_argument(
        "--force", action="store_true", help="Skip the pre-flight Groq budget check"
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(subset=args.subset, offset=args.offset, force=args.force), indent=2))
