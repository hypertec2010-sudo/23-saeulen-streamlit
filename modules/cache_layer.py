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
    """Bridge manual scan identity into provider history; retain normal caching."""
    from datetime import timezone
    from modules.provider_manager import history_refresh_scope

    @st_module.cache_data(ttl=ttl_seconds, show_spinner=False)
    def cached(*, ticker, horizon, depot, risk_pct, override, buy_in_override,
               smart_money_default, strict_mode, market_bucket, _execution_marker=None):
        if _execution_marker is not None:
            _execution_marker["executed"] = True
        token = str(market_bucket or "")
        manual = token.startswith(("manual-v287b-", "selective-v3021d-"))
        with history_refresh_scope(token if manual else ""):
            return analyzer(
                ticker=ticker, horizon=horizon, depot=depot, risk_pct=risk_pct,
                override=override, buy_in_override=buy_in_override,
                smart_money_default=smart_money_default, strict_mode=strict_mode,
            )

    def serve(**kwargs):
        marker = {}
        result = cached(**kwargs, _execution_marker=marker)
        if isinstance(result, dict):
            result = dict(result)
            meta = dict(result.get("history_metadata") or {})
            meta["analysis_cache_hit"] = not bool(marker.get("executed"))
            meta["analysis_served_at"] = datetime.now(timezone.utc).isoformat()
            result["history_metadata"] = meta
        return result
    if hasattr(cached, "clear"):
        serve.clear = cached.clear
    return serve
