"""Common score normalization helpers."""
from __future__ import annotations
import math


def clip_score(value, low=0.0, high=100.0, default=0.0):
    try:
        number = float(value)
        if not math.isfinite(number):
            return float(default)
        return max(float(low), min(float(high), number))
    except Exception:
        return float(default)


def weighted_score(items, default=0.0):
    total = 0.0
    weight_sum = 0.0
    for value, weight in items:
        try:
            v = float(value); w = float(weight)
        except Exception:
            continue
        if math.isfinite(v) and math.isfinite(w) and w > 0:
            total += v * w
            weight_sum += w
    return clip_score(total / weight_sum if weight_sum else default)
