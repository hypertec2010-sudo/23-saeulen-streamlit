from __future__ import annotations

from datetime import datetime, timedelta

from modules.live_refresh_policy import (
    build_cache_key,
    build_schedule_key,
    evaluate_refresh,
    normalized_tickers,
    trigger_is_recent,
)


def test_ticker_and_schedule_keys_are_stable() -> None:
    tickers = normalized_tickers([" aapl ", "sap.de", ""])
    assert tickers == ("AAPL", "SAP.DE")
    assert build_schedule_key("Depot", tickers, "Charttechnik", "Swing", 900) == (
        "Depot|AAPL,SAP.DE|Charttechnik|Swing|900"
    )


def test_missing_or_mismatching_cache_is_due() -> None:
    now = datetime(2026, 7, 27, 10, 0, 0)
    expected = build_cache_key("Depot", ["AAPL"], "Charttechnik", "Swing")

    missing = evaluate_refresh(
        now=now,
        cache={},
        expected_cache_key=expected,
        interval_seconds=900,
    )
    assert missing.due is True
    assert missing.remaining_seconds == 0

    mismatch = evaluate_refresh(
        now=now,
        cache={"key": build_cache_key("Andere", ["AAPL"], "Charttechnik", "Swing"), "ts": now.isoformat()},
        expected_cache_key=expected,
        interval_seconds=900,
    )
    assert mismatch.due is True
    assert mismatch.cache_matches is False


def test_valid_cache_becomes_due_from_last_successful_scan() -> None:
    now = datetime(2026, 7, 27, 10, 15, 0)
    expected = build_cache_key("Depot", ["AAPL"], "Charttechnik", "Swing")

    waiting = evaluate_refresh(
        now=now,
        cache={"key": expected, "ts": (now - timedelta(minutes=5)).isoformat()},
        expected_cache_key=expected,
        interval_seconds=900,
    )
    assert waiting.due is False
    assert 599 <= waiting.remaining_seconds <= 600

    due = evaluate_refresh(
        now=now,
        cache={"key": expected, "ts": (now - timedelta(minutes=15)).isoformat()},
        expected_cache_key=expected,
        interval_seconds=900,
    )
    assert due.due is True
    assert due.remaining_seconds == 0


def test_duplicate_trigger_is_throttled() -> None:
    now = datetime(2026, 7, 27, 10, 0, 0)
    assert trigger_is_recent(now=now, last_trigger=(now - timedelta(seconds=30)).isoformat()) is True
    assert trigger_is_recent(now=now, last_trigger=(now - timedelta(seconds=121)).isoformat()) is False
    assert trigger_is_recent(now=now, last_trigger="invalid") is False
