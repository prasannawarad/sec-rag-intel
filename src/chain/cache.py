"""Disk-backed answer cache: repeat questions cost zero Groq tokens.

The corpus is static between re-indexes and the LLM runs at temperature 0,
so identical (question, filters, model) triples produce identical answers —
caching them is safe and free. Only first-turn questions are cached
(follow-ups depend on chat history; see rag_chain.py).

Single JSON file, insertion-ordered, oldest entries evicted beyond
max_entries. The model name is part of the key, so swapping llm_model
invalidates the cache automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)


class AnswerCache:
    def __init__(self, path: Path, *, max_entries: int = 500) -> None:
        self._path = path
        self._max_entries = max_entries
        self._lock = threading.Lock()

    @staticmethod
    def _key(question: str, filters: dict[str, Any], model: str) -> str:
        material = json.dumps(
            {
                "q": " ".join(question.lower().split()),  # normalise whitespace + case
                "f": {k: v for k, v in sorted(filters.items()) if v is not None},
                "m": model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def _load(self) -> dict[str, dict]:
        try:
            return json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, entries: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(entries, f)
            os.replace(tmp, self._path)
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, question: str, filters: dict[str, Any], model: str) -> dict | None:
        with self._lock:
            entry = self._load().get(self._key(question, filters, model))
        if entry:
            logger.info("Answer cache hit — 0 Groq tokens spent")
        return entry

    def set(
        self,
        question: str,
        filters: dict[str, Any],
        model: str,
        *,
        answer: str,
        sources: list[dict],
    ) -> None:
        with self._lock:
            entries = self._load()
            entries[self._key(question, filters, model)] = {
                "answer": answer,
                "sources": sources,
            }
            while len(entries) > self._max_entries:
                entries.pop(next(iter(entries)))
            self._save(entries)


@lru_cache(maxsize=1)
def get_answer_cache() -> AnswerCache:
    settings = get_settings()
    return AnswerCache(
        Path(settings.answer_cache_path),
        max_entries=settings.answer_cache_max_entries,
    )
