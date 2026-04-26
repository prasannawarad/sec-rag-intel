"""FastAPI app: POST /query, GET /companies, GET /metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.chain.rag_chain import build_rag_chain
from src.config import EVAL_DIR
from src.ingest.downloader import DEFAULT_TICKERS
from src.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="SEC RAG Intel", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    ticker: str | None = None
    year: int | None = None
    filing_type: str | None = None


class Source(BaseModel):
    ticker: str
    year: int | str
    filing_type: str
    section: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    chain = build_rag_chain(ticker=req.ticker, year=req.year, filing_type=req.filing_type)
    result = chain.invoke(req.question)
    return QueryResponse(answer=result["answer"], sources=result["sources"])


@app.get("/companies")
def companies() -> dict[str, list[str]]:
    return {"tickers": DEFAULT_TICKERS}


@app.get("/metrics")
def metrics() -> dict:
    scores_path: Path = EVAL_DIR / "scores.json"
    if not scores_path.exists():
        raise HTTPException(status_code=404, detail="Run evaluation first")
    return json.loads(scores_path.read_text())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
