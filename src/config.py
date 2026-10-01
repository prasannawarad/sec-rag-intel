"""Centralised settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = ROOT_DIR / "eval_results"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    groq_api_key: str = ""
    # Groq retired llama-3.3-70b-versatile (Sept 2026). The generator and the RAGAS
    # judge are deliberately different model families — no self-evaluation bias.
    llm_model: str = "qwen/qwen3.8-27b"
    judge_model: str = "openai/gpt-oss-120b"
    # Cap answer length. Must stay under Groq's free-tier output-tokens-per-minute
    # limit for qwen (1,000): max_tokens counts as "requested", and 1024 got the
    # whole request rejected as too large.
    llm_max_tokens: int = 900

    # Groq free-tier guardrails. Budgets sit below the real per-model limits
    # (console.groq.com/settings/limits) so a miscount never hard-blocks the account.
    groq_rpm_limit: int = 25
    groq_daily_request_budget: int = 900
    groq_daily_token_budget: int = 90_000
    # gpt-oss-120b's real free-tier limit is 200k tokens/day (from Groq's 429 message);
    # the judge only runs offline (eval, ground truth), so it gets most of that.
    judge_daily_token_budget: int = 190_000
    quota_state_path: str = str(DATA_DIR / "quota_state.json")

    # Answer cache — repeat questions are served from disk, costing 0 tokens
    answer_cache_enabled: bool = True
    answer_cache_path: str = str(DATA_DIR / "cache" / "answers.json")
    answer_cache_max_entries: int = 500

    # Vector store
    vector_store_mode: Literal["local", "pinecone"] = "local"
    chroma_persist_dir: str = str(ROOT_DIR / "chroma_db")
    pinecone_api_key: str = ""
    pinecone_index_name: str = "sec-rag-intel"

    # Embeddings (local sentence-transformers, no API key)
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"  # "mps" on Apple Silicon for ~2-3x speedup

    # SEC EDGAR
    sec_user_agent: str = "sec-rag-intel example@example.com"

    # Chunking + retrieval
    # Measured in the embedding model's own tokens: bge-small embeds at most 512
    # (incl. [CLS]/[SEP]) and silently truncates the rest, so chunks stay under it.
    chunk_size: int = 480
    chunk_overlap: int = 50
    retriever_k: int = 5
    retriever_fetch_k: int = 20
    # Route unfiltered multi-company questions to one search per named company
    # (src/retrieval/retriever.py). False reproduces the pre-routing baseline.
    company_routing: bool = True

    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()
