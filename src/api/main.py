"""FastAPI app: POST /query, GET /companies, GET /metrics, DELETE /session."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.chain.rag_chain import build_rag_chain, clear_session
from src.config import EVAL_DIR
from src.ingest.downloader import DEFAULT_TICKERS
from src.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="SEC RAG Intel", version="0.2.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    session_id: str = Field(default="default", max_length=64)
    ticker: str | None = None
    year: int | None = None
    filing_type: str | None = None


class Source(BaseModel):
    ticker: str
    year: int | str
    filing_type: str
    section_label: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    chain = build_rag_chain(ticker=req.ticker, year=req.year, filing_type=req.filing_type)
    result = chain.invoke({"question": req.question, "session_id": req.session_id})
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=req.session_id,
    )


@app.delete("/session/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    """Clear conversation history for a session (e.g. when user hits 'New chat')."""
    clear_session(session_id)


@app.get("/companies")
def companies() -> dict[str, list[str]]:
    return {"tickers": DEFAULT_TICKERS}


@app.get("/metrics")
def metrics() -> dict:
    scores_path: Path = EVAL_DIR / "scores.json"
    if not scores_path.exists():
        raise HTTPException(status_code=404, detail="Run evaluation first: make eval")
    return json.loads(scores_path.read_text())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
