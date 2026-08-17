"""Conservative ticker resolution for market-data providers (v28.4.5b1)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

ALIASES = {
    "SPX": "^GSPC", "S&P500": "^GSPC", "S&P 500": "^GSPC",
    "NDX": "^NDX", "VIX": "^VIX", "DJI": "^DJI", "DOW": "^DJI",
    "RUT": "^RUT",
}

# Symbols whose current security has only recently begun trading.  The start
# date prevents recycled ticker history from being mixed into the new listing.
# SPCX is especially important: an ETF previously used SPCX before changing
# to SPCK in April 2026; SpaceX began trading as SPCX on 2026-06-12.
LISTING_STARTS = {
    "SKHY": "2026-07-13",
    "SPCX": "2026-06-12",
}

@dataclass(frozen=True)
class ResolvedTicker:
    original: str
    provider_symbol: str
    changed: bool
    reason: str = ""
    history_start: Optional[str] = None

def resolve_ticker(symbol: str) -> ResolvedTicker:
    original = str(symbol or "").strip().upper()
    if not original:
        return ResolvedTicker("", "", False, "empty", None)
    if original in ALIASES:
        target = ALIASES[original]
        return ResolvedTicker(original, target, True, "index_alias", None)
    if "." in original and original.count(".") == 1:
        left, right = original.split(".")
        if right in {"A", "B"} and 1 <= len(left) <= 5:
            target = f"{left}-{right}"
            return ResolvedTicker(original, target, True, "share_class", None)
    start = LISTING_STARTS.get(original)
    if start:
        return ResolvedTicker(original, original, False, "new_listing", start)
    return ResolvedTicker(original, original, False, "direct", None)
