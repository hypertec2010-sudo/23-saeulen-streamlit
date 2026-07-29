# -*- coding: utf-8 -*-
"""Pure scheduling helpers for the Live-Screener auto refresh.

The Streamlit fragment remains responsible for displaying status text and
triggering ``st.rerun``. This module contains only deterministic calculations,
so the refresh behaviour can be tested without a running browser session.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class RefreshDecision:
    """Result of one heartbeat evaluation."""

    due: bool
    remaining_seconds: int
    cache_matches: bool
    cache_timestamp: Optional[datetime]


def normalized_tickers(tickers: Iterable[Any]) -> tuple[str, ...]:
    """Return normalized, stable ticker symbols for cache comparisons."""
    return tuple(str(ticker).strip().upper() for ticker in (tickers or []) if str(ticker).strip())


def build_cache_key(
    watchlist: Any,
    tickers: Iterable[Any],
    style: Any,
    horizon: Any,
) -> dict[str, Any]:
    """Build the cache identity shared by the scanner and heartbeat."""
    return {
        "watchlist": str(watchlist or ""),
        "tickers": normalized_tickers(tickers),
        "style": str(style or ""),
        "horizon": str(horizon or ""),
    }


def build_schedule_key(
    watchlist: Any,
    tickers: Iterable[Any],
    style: Any,
    horizon: Any,
    interval_seconds: int,
) -> str:
    """Build a compact identity for scheduler initialization and resets."""
    cache_key = build_cache_key(watchlist, tickers, style, horizon)
    return "|".join(
        [
            cache_key["watchlist"],
            ",".join(cache_key["tickers"]),
            cache_key["style"],
            cache_key["horizon"],
            str(max(60, int(interval_seconds))),
        ]
    )


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def evaluate_refresh(
    *,
    now: datetime,
    cache: Mapping[str, Any] | None,
    expected_cache_key: Mapping[str, Any],
    interval_seconds: int,
    tolerance_seconds: int = 2,
) -> RefreshDecision:
    """Decide whether a scan is due for the current heartbeat.

    A missing, malformed or mismatching cache is immediately due. For a valid
    cache, the interval is measured from the last successful scan timestamp.
    """
    safe_interval = max(60, int(interval_seconds))
    safe_tolerance = max(0, min(int(tolerance_seconds), safe_interval))
    cache_mapping = cache if isinstance(cache, Mapping) else {}
    cache_matches = bool(cache_mapping) and cache_mapping.get("key") == dict(expected_cache_key)
    cache_timestamp = _parse_timestamp(cache_mapping.get("ts")) if cache_matches else None

    if cache_timestamp is None:
        return RefreshDecision(
            due=True,
            remaining_seconds=0,
            cache_matches=cache_matches,
            cache_timestamp=None,
        )

    elapsed = max(0.0, (now - cache_timestamp).total_seconds())
    remaining = max(0, int(safe_interval - elapsed))
    due = elapsed >= (safe_interval - safe_tolerance)
    return RefreshDecision(
        due=due,
        remaining_seconds=remaining,
        cache_matches=True,
        cache_timestamp=cache_timestamp,
    )


def trigger_is_recent(
    *,
    now: datetime,
    last_trigger: Any,
    cooldown_seconds: int = 120,
) -> bool:
    """Return whether a previous rerun trigger is still inside the cooldown."""
    parsed = _parse_timestamp(last_trigger)
    if parsed is None:
        return False
    return max(0.0, (now - parsed).total_seconds()) < max(1, int(cooldown_seconds))
