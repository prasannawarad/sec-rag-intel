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

    openai_api_key: str = ""
    groq_api_key: str = ""

    vector_store_mode: Literal["local", "pinecone"] = "local"
    chroma_persist_dir: str = str(ROOT_DIR / "chroma_db")

    pinecone_api_key: str = ""
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "sec-rag-intel"

    sec_user_agent: str = "sec-rag-intel example@example.com"

    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "llama-3.3-70b-versatile"

    chunk_size: int = 512
    chunk_overlap: int = 50
    retriever_k: int = 5
    retriever_fetch_k: int = 20

    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()
