"""Unified technical CRV package for CHSM.

v30.21aj separates two questions that were previously mixed together:

* ``CRV jetzt``: is buying at the current market price attractive versus the
  next credible technical obstacle and the productive risk stop?
* ``CRV Entry``: what would the same setup look like at the conservative upper
  edge of CHSM's planned entry zone?

Synthetic 1.8R/2R planning targets are never used as technical targets here.
The target is selected conservatively from real chart/setup structure.
"""
from __future__ import annotations

import math
import re
from typing import Any

try:
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover - runtime dependency in CHSM
    np = None
    pd = None


def _finite(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        num = _finite(value)
        if num is not None:
            return num
    return None


def parse_price_zone(value: Any) -> tuple[float | None, float | None]:
    """Parse CHSM entry-zone text without assuming a specific currency suffix."""
    if value is None:
        return None, None
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        a, b = _finite(value[0]), _finite(value[1])
        if a is not None and b is not None:
            return min(a, b), max(a, b)
    text = str(value).strip()
    if not text or text.lower() in {"-", "n/a", "none", "nan"}:
        return None, None
    # CHSM prices use decimal dots in the computed entry-zone strings. The
    # regex is intentionally conservative so currency symbols / labels cannot
    # become numbers.
    raw = re.findall(r"[-+]?\d+(?:[\.,]\d+)?", text)
    nums: list[float] = []
    for token in raw:
        try:
            nums.append(float(token.replace(",", ".")))
        except Exception:
            continue
    if len(nums) >= 2:
        return min(nums[0], nums[1]), max(nums[0], nums[1])
    if len(nums) == 1:
        return nums[0], nums[0]
    return None, None


def _pivot_zones_from_df(df: Any, *, tolerance_pct: float = 1.5, min_touches: int = 2) -> list[dict[str, Any]]:
    """Compute the same style of swing-pivot clusters CHSM uses for S/R.

    This helper lives outside the Streamlit UI so the core analysis, screener
    and single-stock analysis can all use the same chart-derived CRV target.
    """
    if pd is None or df is None or not hasattr(df, "empty") or df.empty:
        return []
    if "High" not in df.columns or "Low" not in df.columns:
        return []
    basis = df.tail(260).copy()
    if len(basis) < 12:
        return []
    highs = pd.to_numeric(basis["High"], errors="coerce").reset_index(drop=True)
    lows = pd.to_numeric(basis["Low"], errors="coerce").reset_index(drop=True)
    points: list[float] = []
    left = right = 3
    for i in range(left, len(basis) - right):
        hi = highs.iloc[i]
        lo = lows.iloc[i]
        if pd.notna(hi):
            l = highs.iloc[i-left:i]
            r = highs.iloc[i+1:i+1+right]
            if len(l.dropna()) == left and len(r.dropna()) == right and hi >= l.max() and hi >= r.max():
                points.append(float(hi))
        if pd.notna(lo):
            l = lows.iloc[i-left:i]
            r = lows.iloc[i+1:i+1+right]
            if len(l.dropna()) == left and len(r.dropna()) == right and lo <= l.min() and lo <= r.min():
                points.append(float(lo))
    if not points:
        return []
    prices = sorted(points)
    clusters: list[list[float]] = [[prices[0]]]
    for price in prices[1:]:
        mean = sum(clusters[-1]) / len(clusters[-1])
        tolerance = mean * tolerance_pct / 100.0
        if abs(price - mean) <= tolerance:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    zones: list[dict[str, Any]] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        low = min(cluster)
        high = max(cluster)
        if low <= 0:
            continue
        width_pct = (high / low - 1.0) * 100.0
        if width_pct > 8.0:
            continue
        zones.append({
            "low": float(low),
            "high": float(high),
            "mid": float(sum(cluster) / len(cluster)),
            "touches": int(len(cluster)),
            "width_pct": float(width_pct),
        })
    return zones


def _zones_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    structures = result.get("chart_structures_analysis")
    if isinstance(structures, dict):
        zones = structures.get("zones")
        if isinstance(zones, list) and zones:
            return [dict(z) for z in zones if isinstance(z, dict)]
    return _pivot_zones_from_df(result.get("df"))


def _wave_candidates(result: dict[str, Any]) -> list[tuple[float, str, str]]:
    wave = result.get("wave_structure_pkg") or {}
    if not isinstance(wave, dict):
        return []
    out: list[tuple[float, str, str]] = []
    for key in ("wave_extension_127", "target_127", "wave_target_127"):
        value = _finite(wave.get(key))
        if value is not None:
            out.append((value, "Wave-Ziel 1,27", "wave"))
    zone = wave.get("wave_target_zone") or wave.get("wave_readable_target")
    low, high = parse_price_zone(zone)
    if low is not None:
        out.append((low, "Wave-Zielzone", "wave"))
    elif high is not None:
        out.append((high, "Wave-Zielzone", "wave"))
    return out


def _technical_target_candidates(result: dict[str, Any], current_price: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    # 1) Existing CHSM chart S/R: first contact with an overhead zone is the
    # conservative target. If price is currently inside a zone, the upper edge
    # is the immediate technical obstacle.
    for zone in _zones_from_result(result):
        low = _finite(zone.get("low"))
        high = _finite(zone.get("high"))
        touches = int(_finite(zone.get("touches")) or 0)
        if low is None or high is None:
            continue
        if low <= current_price <= high and high > current_price * 1.001:
            candidates.append({
                "value": high,
                "source": f"Oberkante aktive CHSM-Chartzone ({max(touches, 2)} Berührungen)",
                "kind": "chart_active_zone",
                "priority": 0,
            })
        elif low > current_price * 1.001:
            candidates.append({
                "value": low,
                "source": f"CHSM-Widerstandszone ({max(touches, 2)} Berührungen)",
                "kind": "chart_resistance",
                "priority": 0,
            })

    # 2) Setup structure. tp2_base_target is the real target behind a synthetic
    # floor and is therefore allowed; tp2 itself is not unless explicitly real.
    setup_candidates = [
        (result.get("structural_target"), result.get("structural_target_source") or "Strukturelles Setup-Ziel"),
        (result.get("tp2_base_target"), result.get("tp2_base_source") or "Setup-Basisziel"),
        (result.get("technical_target_1"), "Technisches Setup-Ziel"),
    ]
    for value, source in setup_candidates:
        num = _finite(value)
        if num is not None and num > current_price * 1.001:
            candidates.append({"value": num, "source": str(source), "kind": "setup", "priority": 1})

    # Explicitly non-synthetic TP2 is also a real target for older result sets.
    if not bool(result.get("tp2_is_synthetic")):
        num = _finite(result.get("tp2"))
        if num is not None and num > current_price * 1.001:
            candidates.append({"value": num, "source": str(result.get("tp2_source") or "TP2"), "kind": "setup", "priority": 1})

    # 3) Wave structure.
    for value, source, kind in _wave_candidates(result):
        if value > current_price * 1.001:
            candidates.append({"value": value, "source": source, "kind": kind, "priority": 2})

    # 4) 52-week high as a last structural orientation, never analyst target.
    high52 = _first_number(result.get("high52"), result.get("52W_High"), result.get("high_52w"))
    if high52 is not None and high52 > current_price * 1.001:
        candidates.append({"value": high52, "source": "52W-Hoch", "kind": "high52", "priority": 3})

    # Deduplicate near-identical levels while preferring chart/setup provenance.
    ordered = sorted(candidates, key=lambda c: (float(c["value"]), int(c["priority"])))
    deduped: list[dict[str, Any]] = []
    for cand in ordered:
        value = float(cand["value"])
        if any(abs(value - float(x["value"])) / max(value, 1e-9) < 0.002 for x in deduped):
            continue
        deduped.append(cand)
    return deduped


def build_technical_crv_package(result: dict[str, Any] | None) -> dict[str, Any]:
    """Return CRV-now / CRV-entry using real chart and setup structure."""
    r = result or {}
    price = _first_number(r.get("price"), r.get("current_price"), r.get("live_price"))
    stop = _first_number(r.get("stop_used"), r.get("stop"), r.get("risk_stop"))
    stop_source = str(r.get("stop_source") or "Stop nicht belegt").strip() or "Stop nicht belegt"
    zone_low, zone_high = parse_price_zone(r.get("suggested_entry_zone") or r.get("entry_zone"))
    entry_ref = zone_high if zone_high is not None and zone_high > 0 else price
    entry_source = "Oberer Rand der CHSM-Entry-Zone" if zone_high is not None else "Aktueller Kurs (keine belastbare Entry-Zone)"

    base = {
        "technical_crv_now": math.nan,
        "technical_crv_entry": math.nan,
        "technical_crv_target": math.nan,
        "technical_crv_target_source": "kein belastbares technisches Ziel",
        "technical_crv_target_kind": "missing",
        "technical_crv_entry_reference": entry_ref if entry_ref is not None else math.nan,
        "technical_crv_entry_reference_source": entry_source,
        "technical_crv_stop_now": stop if stop is not None else math.nan,
        "technical_crv_stop_entry": math.nan,
        "technical_crv_stop_source": stop_source,
        "technical_crv_entry_stop_source": stop_source,
        "technical_crv_target_candidates": [],
        "technical_crv_status": "missing",
    }
    if price is None or price <= 0:
        return base

    candidates = _technical_target_candidates(r, price)
    base["technical_crv_target_candidates"] = [
        {"value": round(float(c["value"]), 4), "source": c["source"], "kind": c["kind"]}
        for c in candidates[:4]
    ]
    if not candidates:
        return base

    # The nearest credible overhead obstacle is intentionally used. This is
    # more conservative than choosing the farthest target and better reflects
    # what price has to clear first.
    target = float(candidates[0]["value"])
    base["technical_crv_target"] = target
    base["technical_crv_target_source"] = str(candidates[0]["source"])
    base["technical_crv_target_kind"] = str(candidates[0]["kind"])

    if stop is not None and 0 < stop < price and target > price:
        risk_now = price - stop
        base["technical_crv_now"] = (target - price) / risk_now if risk_now > 0 else math.nan

    if entry_ref is not None and entry_ref > 0 and target > entry_ref:
        # Reuse the productive stop, but never understate risk merely because the
        # current-price 3.5% floor would sit too close to a lower planned entry.
        # If available, a real chart invalidation below the planned entry is an
        # even better base than a stop that ended up above the entry zone.
        entry_stop = stop
        entry_stop_source = stop_source
        invalidation = _first_number(r.get("chart_invalidation_level"), r.get("structure_stop"))
        invalidation_source = str(r.get("chart_invalidation_source") or "Chart-Invalidierung").strip()
        if (entry_stop is None or entry_stop >= entry_ref) and invalidation is not None and 0 < invalidation < entry_ref:
            entry_stop = invalidation
            entry_stop_source = invalidation_source
        min_stop = entry_ref * 0.965
        if entry_stop is None or entry_stop <= 0 or entry_stop >= entry_ref or entry_stop > min_stop:
            entry_stop = min_stop
            entry_stop_source = f"{entry_stop_source} · 3,5%-Praxisabstand zur Entry-Zone"
        base["technical_crv_stop_entry"] = entry_stop
        base["technical_crv_entry_stop_source"] = entry_stop_source
        risk_entry = entry_ref - entry_stop
        if risk_entry > 0:
            base["technical_crv_entry"] = (target - entry_ref) / risk_entry

    now = _finite(base["technical_crv_now"])
    entry = _finite(base["technical_crv_entry"])
    if now is not None:
        base["technical_crv_status"] = "attractive" if now >= 1.5 else "selective" if now >= 1.2 else "tight"
    elif entry is not None:
        base["technical_crv_status"] = "entry_only"
    return base


def crv_soft_gate_state(crv_now: Any, crv_entry: Any = None) -> dict[str, Any]:
    """Decision helper: low CRV is a timing brake, not an automatic hard gate."""
    now = _finite(crv_now)
    entry = _finite(crv_entry)
    evaluation = entry if entry is not None else now
    if now is None and entry is None:
        return {"severity": "unknown", "hard_gate": False, "score": 45.0, "evaluation_crv": None}
    if evaluation is None:
        evaluation = now
    if evaluation is not None and evaluation >= 3.0:
        score = 94.0
    elif evaluation is not None and evaluation >= 2.0:
        score = 82.0
    elif evaluation is not None and evaluation >= 1.5:
        score = 66.0
    elif evaluation is not None and evaluation >= 1.2:
        score = 52.0
    else:
        score = 38.0
    severity = "good"
    if now is None:
        severity = "unknown"
    elif now < 1.2:
        severity = "tight"
    elif now < 1.5:
        severity = "selective"
    return {"severity": severity, "hard_gate": False, "score": score, "evaluation_crv": evaluation}
