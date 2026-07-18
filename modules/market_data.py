"""Small, dependency-light market-data utilities shared by UI modules."""
from __future__ import annotations
import math


def finite_positive(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else default
    except Exception:
        return default


def latest_close(frame, default=None):
    try:
        if frame is None or len(frame) == 0 or "Close" not in frame:
            return default
        return finite_positive(frame["Close"].dropna().iloc[-1], default)
    except Exception:
        return default


def atr_percent(atr, price, default=None):
    atr_v = finite_positive(atr)
    price_v = finite_positive(price)
    if atr_v is None or price_v is None:
        return default
    return 100.0 * atr_v / price_v
