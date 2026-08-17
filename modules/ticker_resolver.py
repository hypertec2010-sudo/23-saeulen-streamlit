"""Conservative ticker resolution for market-data providers (v28.4.5b)."""
from __future__ import annotations
from dataclasses import dataclass

ALIASES = {
    "SPX": "^GSPC", "S&P500": "^GSPC", "S&P 500": "^GSPC",
    "NDX": "^NDX", "VIX": "^VIX", "DJI": "^DJI", "DOW": "^DJI",
    "RUT": "^RUT",
}

@dataclass(frozen=True)
class ResolvedTicker:
    original: str
    provider_symbol: str
    changed: bool
    reason: str = ""

def resolve_ticker(symbol: str) -> ResolvedTicker:
    original = str(symbol or "").strip().upper()
    if not original:
        return ResolvedTicker("", "", False, "empty")
    if original in ALIASES:
        target = ALIASES[original]
        return ResolvedTicker(original, target, True, "index_alias")
    # Yahoo uses a dash for share classes such as BRK.B / BF.B.
    if "." in original and original.count(".") == 1:
        left, right = original.split(".")
        if right in {"A", "B"} and 1 <= len(left) <= 5:
            target = f"{left}-{right}"
            return ResolvedTicker(original, target, True, "share_class")
    # Exchange-qualified symbols (.DE, .MI, .L, ...) and ordinary symbols,
    # including new listings such as SKHY/SPCX, remain untouched.
    return ResolvedTicker(original, original, False, "direct")
