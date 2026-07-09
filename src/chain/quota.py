"""Groq free-tier quota guard.

Tracks daily token + request usage in a small JSON file so the budget
survives process restarts, and throttles requests-per-minute in-process.
Every LLM call site goes through the same guard:

    guard = get_quota_guard()
    guard.check(est_tokens)   # raises QuotaExceededError if budget is gone
    guard.throttle()          # sleeps if the RPM window is full
    ... call Groq ...
    guard.record(input_tokens, output_tokens)

Accounting is exact where possible (usage_metadata from the Groq response)
and conservative where not (pre-flight estimates use ~4 chars/token).
State resets at midnight UTC, matching Groq's daily quota window.

Known limitation (fine for a single-instance demo): concurrent processes
share the state file with last-writer-wins semantics, so parallel API +
Streamlit usage can undercount slightly — the budget headroom in
Settings absorbs that.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from src.config import get_settings

logger = logging.getLogger(__name__)

_RPM_WINDOW_SECONDS = 60.0


class QuotaExceededError(RuntimeError):
    """Raised when today's Groq free-tier budget is exhausted."""


class QuotaStatus(TypedDict):
    date: str
    tokens_used: int
    tokens_budget: int
    tokens_remaining: int
    requests_used: int
    requests_budget: int
    requests_remaining: int


def estimate_tokens(text: str) -> int:
    """Cheap, conservative token estimate (~4 chars/token for English)."""
    return max(len(text) // 4, 1)


class QuotaGuard:
    """Daily token/request budget (persisted) + RPM throttle (persisted timestamps)."""

    def __init__(
        self,
        state_path: Path,
        *,
        daily_token_budget: int,
        daily_request_budget: int,
        rpm_limit: int,
        now_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._path = state_path
        self._token_budget = daily_token_budget
        self._request_budget = daily_request_budget
        self._rpm_limit = rpm_limit
        self._now = now_fn
        self._sleep = sleep_fn
        self._lock = threading.Lock()

    # -- state ---------------------------------------------------------

    def _today(self) -> str:
        return datetime.fromtimestamp(self._now(), tz=UTC).date().isoformat()

    def _fresh_state(self) -> dict:
        return {"date": self._today(), "tokens_used": 0, "requests_used": 0, "timestamps": []}

    def _load(self) -> dict:
        try:
            state = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return self._fresh_state()
        if state.get("date") != self._today():  # midnight UTC rollover
            return self._fresh_state()
        return state

    def _save(self, state: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-write can't corrupt the state file
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f)
            os.replace(tmp, self._path)
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- public API ----------------------------------------------------

    def check(self, est_tokens: int = 0) -> None:
        """Raise QuotaExceededError if est_tokens more would bust today's budget."""
        with self._lock:
            state = self._load()
            if state["requests_used"] + 1 > self._request_budget:
                raise QuotaExceededError(
                    f"Daily request budget spent ({state['requests_used']}"
                    f"/{self._request_budget}); resets at midnight UTC."
                )
            if state["tokens_used"] + est_tokens > self._token_budget:
                raise QuotaExceededError(
                    f"Daily token budget spent ({state['tokens_used']}"
                    f"/{self._token_budget}); resets at midnight UTC."
                )

    def throttle(self) -> None:
        """Sleep just long enough to stay under the requests-per-minute limit."""
        with self._lock:
            state = self._load()
            now = self._now()
            window = [t for t in state["timestamps"] if now - t < _RPM_WINDOW_SECONDS]
            if len(window) >= self._rpm_limit:
                wait = _RPM_WINDOW_SECONDS - (now - window[0]) + 0.1
                logger.info("RPM limit (%d/min) reached — sleeping %.1fs", self._rpm_limit, wait)
                self._sleep(wait)
                now = self._now()
                window = [t for t in window if now - t < _RPM_WINDOW_SECONDS]
            window.append(now)
            state["timestamps"] = window
            self._save(state)

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Record actual usage after a successful Groq call."""
        with self._lock:
            state = self._load()
            state["tokens_used"] += input_tokens + output_tokens
            state["requests_used"] += 1
            self._save(state)
            logger.debug(
                "Groq usage today: %d/%d tokens, %d/%d requests",
                state["tokens_used"],
                self._token_budget,
                state["requests_used"],
                self._request_budget,
            )

    def status(self) -> QuotaStatus:
        with self._lock:
            state = self._load()
        return QuotaStatus(
            date=state["date"],
            tokens_used=state["tokens_used"],
            tokens_budget=self._token_budget,
            tokens_remaining=max(self._token_budget - state["tokens_used"], 0),
            requests_used=state["requests_used"],
            requests_budget=self._request_budget,
            requests_remaining=max(self._request_budget - state["requests_used"], 0),
        )


@lru_cache(maxsize=1)
def get_quota_guard() -> QuotaGuard:
    settings = get_settings()
    return QuotaGuard(
        Path(settings.quota_state_path),
        daily_token_budget=settings.groq_daily_token_budget,
        daily_request_budget=settings.groq_daily_request_budget,
        rpm_limit=settings.groq_rpm_limit,
    )
