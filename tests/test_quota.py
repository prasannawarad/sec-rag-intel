"""Unit tests for the Groq free-tier quota guard (no network, no model load)."""

from __future__ import annotations

import pytest

from src.chain.quota import QuotaExceededError, QuotaGuard, estimate_tokens


class FakeClock:
    def __init__(self, start: float = 1_750_000_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_guard(tmp_path, clock: FakeClock, **overrides) -> QuotaGuard:
    kwargs = {
        "daily_token_budget": 1_000,
        "daily_request_budget": 10,
        "rpm_limit": 3,
        "now_fn": clock.time,
        "sleep_fn": clock.sleep,
    }
    kwargs.update(overrides)
    return QuotaGuard(tmp_path / "quota_state.json", **kwargs)


def test_fresh_guard_reports_zero_usage(tmp_path):
    guard = make_guard(tmp_path, FakeClock())
    status = guard.status()
    assert status["tokens_used"] == 0
    assert status["requests_used"] == 0
    assert status["tokens_remaining"] == 1_000
    assert status["requests_remaining"] == 10


def test_record_accumulates_and_persists_across_instances(tmp_path):
    clock = FakeClock()
    make_guard(tmp_path, clock).record(100, 50)
    status = make_guard(tmp_path, clock).status()  # new instance, same file
    assert status["tokens_used"] == 150
    assert status["requests_used"] == 1


def test_check_raises_when_token_budget_would_be_exceeded(tmp_path):
    clock = FakeClock()
    guard = make_guard(tmp_path, clock)
    guard.record(900, 0)
    guard.check(est_tokens=50)  # 950 < 1000 — fine
    with pytest.raises(QuotaExceededError, match="token budget"):
        guard.check(est_tokens=200)


def test_check_raises_when_request_budget_spent(tmp_path):
    clock = FakeClock()
    guard = make_guard(tmp_path, clock, daily_request_budget=2)
    guard.record(1, 1)
    guard.record(1, 1)
    with pytest.raises(QuotaExceededError, match="request budget"):
        guard.check()


def test_budget_resets_at_utc_midnight(tmp_path):
    clock = FakeClock()
    guard = make_guard(tmp_path, clock)
    guard.record(999, 0)
    clock.now += 86_400  # next UTC day
    assert guard.status()["tokens_used"] == 0
    guard.check(est_tokens=500)  # would have raised yesterday


def test_throttle_sleeps_when_rpm_window_full(tmp_path):
    clock = FakeClock()
    guard = make_guard(tmp_path, clock, rpm_limit=3)
    for _ in range(3):
        guard.throttle()
        clock.now += 1
    assert clock.sleeps == []
    guard.throttle()  # 4th call inside the same minute must wait
    assert len(clock.sleeps) == 1
    assert 55 < clock.sleeps[0] <= 61


def test_throttle_does_not_sleep_after_window_passes(tmp_path):
    clock = FakeClock()
    guard = make_guard(tmp_path, clock, rpm_limit=2)
    guard.throttle()
    guard.throttle()
    clock.now += 61
    guard.throttle()
    assert clock.sleeps == []


def test_corrupt_state_file_resets_cleanly(tmp_path):
    path = tmp_path / "quota_state.json"
    path.write_text("{not json")
    guard = make_guard(tmp_path, FakeClock())
    assert guard.status()["tokens_used"] == 0


def test_estimate_tokens_is_conservative_and_positive():
    assert estimate_tokens("") == 1
    assert estimate_tokens("word " * 100) == pytest.approx(125, abs=5)
