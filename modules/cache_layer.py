"""Shared cache helpers and deterministic market buckets."""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Any


def market_bucket(minutes: int = 15, now: datetime | None = None) -> int:
    try:
        mins = max(1, int(minutes))
    except Exception:
        mins = 15
    current = now or datetime.now()
    return int(current.timestamp() // (mins * 60))


def make_cached_analyzer(st_module, analyzer: Callable[..., Any], ttl_seconds: int = 900):
    """Create a Streamlit-cached facade without importing Streamlit in this module."""
    @st_module.cache_data(ttl=ttl_seconds, show_spinner=False)
    def cached(*, ticker, horizon, depot, risk_pct, override, buy_in_override,
               smart_money_default, strict_mode, market_bucket):
        return analyzer(
            ticker=ticker, horizon=horizon, depot=depot, risk_pct=risk_pct,
            override=override, buy_in_override=buy_in_override,
            smart_money_default=smart_money_default, strict_mode=strict_mode,
        )
    return cached
