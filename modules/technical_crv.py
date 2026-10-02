"""Setup-aware technical CRV for CHSM (v30.21am).

The module separates two different questions:

1. ``Freiraum``: distance from the current price to the next credible chart
   obstacle / trigger zone.
2. ``Trade-CRV``: reward/risk from the current price or planned entry to a
   setup-appropriate technical profit target.

This matters for breakout setups. The resistance that is about to be broken is
an entry/trigger obstacle, not automatically the trade's profit target. For
pullback / trend-following / rebound setups, by contrast, the next overhead
resistance remains a conservative and valid first trade target.

Synthetic 1.8R/2R planning floors and analyst targets are never technical CRV
targets here.
"""
from __future__ import annotations

import math
import re
from typing import Any

try:
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover
    np = None
    pd = None


BREAKOUT_SETUP_TOKENS = ("breakout", "range-breakout", "breakout-retest")


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
    if value is None:
        return None, None
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        a, b = _finite(value[0]), _finite(value[1])
        if a is not None and b is not None:
            return min(a, b), max(a, b)
    text = str(value).strip()
    if not text or text.lower() in {"-", "n/a", "none", "nan"}:
        return None, None
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
    """Headless equivalent of CHSM's pivot-zone clustering."""
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
            rr = highs.iloc[i+1:i+1+right]
            if len(l.dropna()) == left and len(rr.dropna()) == right and hi >= l.max() and hi >= rr.max():
                points.append(float(hi))
        if pd.notna(lo):
            l = lows.iloc[i-left:i]
            rr = lows.iloc[i+1:i+1+right]
            if len(l.dropna()) == left and len(rr.dropna()) == right and lo <= l.min() and lo <= rr.min():
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


def _is_breakout_setup(result: dict[str, Any]) -> bool:
    texts = [
        result.get("setup_type"),
        result.get("candidate_type"),
        result.get("preferred_entry"),
        result.get("entry_source"),
    ]
    combined = " ".join(str(x or "").strip().lower() for x in texts)
    return any(token in combined for token in BREAKOUT_SETUP_TOKENS) or "ausbruch" in combined


def _chart_obstacles(result: dict[str, Any], current_price: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for zone in _zones_from_result(result):
        low = _finite(zone.get("low"))
        high = _finite(zone.get("high"))
        touches = int(_finite(zone.get("touches")) or 0)
        if low is None or high is None or low <= 0 or high <= 0:
            continue
        if low <= current_price <= high and high > current_price * 1.001:
            out.append({
                "value": high,
                "source": f"Oberkante aktive CHSM-Chartzone ({max(touches, 2)} Berührungen)",
                "kind": "chart_active_zone",
                "zone_low": low,
                "zone_high": high,
                "touches": touches,
            })
        elif low > current_price * 1.001:
            out.append({
                "value": low,
                "source": f"CHSM-Widerstandszone ({max(touches, 2)} Berührungen)",
                "kind": "chart_resistance",
                "zone_low": low,
                "zone_high": high,
                "touches": touches,
            })
    return sorted(out, key=lambda c: float(c["value"]))


def _wave_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    wave = result.get("wave_structure_pkg") or {}
    if not isinstance(wave, dict):
        return []
    out: list[dict[str, Any]] = []
    for key in ("wave_extension_127", "target_127", "wave_target_127"):
        value = _finite(wave.get(key))
        if value is not None:
            out.append({"value": value, "source": "Wave-Ziel 1,27", "kind": "wave", "priority": 2})
    zone = wave.get("wave_target_zone") or wave.get("wave_readable_target")
    low, high = parse_price_zone(zone)
    if low is not None:
        out.append({"value": low, "source": "Wave-Zielzone", "kind": "wave", "priority": 2})
    elif high is not None:
        out.append({"value": high, "source": "Wave-Zielzone", "kind": "wave", "priority": 2})
    return out


def _breakout_measured_move(result: dict[str, Any], current_price: float, trigger_reference: float | None) -> dict[str, Any] | None:
    """Conservative measured-move fallback from the pre-breakout 20T range.

    Uses one full range height (classic measured move), but only for reasonably
    formed ranges (2-25% height). It is a fallback behind actual overhead chart,
    setup and wave targets, never a synthetic R multiple.
    """
    if pd is None:
        return None
    df = result.get("df")
    if df is None or not hasattr(df, "empty") or df.empty or "High" not in df.columns or "Low" not in df.columns:
        return None
    basis = df.tail(25).copy()
    if len(basis) < 12:
        return None
    # Exclude the newest bar so today's breakout spike does not inflate the base.
    prior = basis.iloc[:-1].tail(20)
    highs = pd.to_numeric(prior["High"], errors="coerce").dropna()
    lows = pd.to_numeric(prior["Low"], errors="coerce").dropna()
    if highs.empty or lows.empty:
        return None
    breakout_level = float(highs.max())
    base_low = float(lows.min())
    if trigger_reference is not None:
        breakout_level = max(breakout_level, float(trigger_reference))
    height = breakout_level - base_low
    if breakout_level <= 0 or height <= 0:
        return None
    height_pct = height / breakout_level * 100.0
    if height_pct < 2.0 or height_pct > 25.0:
        return None
    target = breakout_level + height
    if target <= current_price * 1.01:
        return None
    return {
        "value": target,
        "source": f"Breakout-Projektion aus 20T-Range ({height_pct:.1f}% Range-Höhe)",
        "kind": "breakout_measured_move",
        "priority": 3,
    }


def _setup_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw = [
        (result.get("structural_target"), result.get("structural_target_source") or "Strukturelles Setup-Ziel"),
        (result.get("tp2_base_target"), result.get("tp2_base_source") or "Setup-Basisziel"),
        (result.get("technical_target_1"), "Technisches Setup-Ziel"),
        (result.get("technical_target_2"), "Technisches Sekundärziel"),
    ]
    out: list[dict[str, Any]] = []
    for value, source in raw:
        num = _finite(value)
        if num is not None:
            out.append({"value": num, "source": str(source), "kind": "setup", "priority": 1})
    if not bool(result.get("tp2_is_synthetic")):
        num = _finite(result.get("tp2"))
        if num is not None:
            out.append({"value": num, "source": str(result.get("tp2_source") or "TP2"), "kind": "setup", "priority": 1})
    return out


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda c: (float(c["value"]), int(c.get("priority", 9))))
    out: list[dict[str, Any]] = []
    for cand in ordered:
        value = float(cand["value"])
        duplicate = False
        for existing in out:
            ev = float(existing["value"])
            if abs(value - ev) / max(value, 1e-9) < 0.002:
                duplicate = True
                # Keep the more direct / higher-priority provenance.
                if int(cand.get("priority", 9)) < int(existing.get("priority", 9)):
                    existing.update(cand)
                break
        if not duplicate:
            out.append(dict(cand))
    return sorted(out, key=lambda c: float(c["value"]))


def _trade_target_package(result: dict[str, Any], current_price: float, entry_ref: float | None) -> dict[str, Any]:
    breakout_mode = _is_breakout_setup(result)
    obstacles = _chart_obstacles(result, current_price)
    clearance = obstacles[0] if obstacles else None

    # Freiraum is always the nearest obstacle. It is informative even when it
    # is not the correct profit target for the current setup.
    clearance_target = _finite(clearance.get("value")) if clearance else None
    clearance_source = str(clearance.get("source")) if clearance else "kein nahes CHSM-Hindernis"

    chart_trade_candidates: list[dict[str, Any]] = []
    trigger_reference = None
    trigger_source = "-"

    if breakout_mode:
        # For breakout / retest setups the near resistance or active zone is the
        # trigger obstacle. The trade target must sit meaningfully beyond it.
        trigger_values = [v for v in (clearance_target, entry_ref, current_price) if v is not None and v > 0]
        trigger_reference = max(trigger_values) if trigger_values else current_price
        trigger_source = clearance_source if clearance_target is not None and clearance_target >= current_price else "Breakout-/Entry-Referenz"
        min_trade_target = trigger_reference * 1.01
        for obstacle in obstacles:
            value = float(obstacle["value"])
            if value > min_trade_target:
                chart_trade_candidates.append({
                    "value": value,
                    "source": str(obstacle["source"]),
                    "kind": "chart_resistance_above_breakout",
                    "priority": 0,
                })
    else:
        min_trade_target = current_price * 1.001
        for obstacle in obstacles:
            value = float(obstacle["value"])
            if value > min_trade_target:
                chart_trade_candidates.append({
                    "value": value,
                    "source": str(obstacle["source"]),
                    "kind": str(obstacle.get("kind") or "chart_resistance"),
                    "priority": 0,
                })

    candidates = list(chart_trade_candidates)
    for cand in _setup_candidates(result) + _wave_candidates(result):
        if float(cand["value"]) > min_trade_target:
            candidates.append(cand)

    high52 = _first_number(result.get("high52"), result.get("52W_High"), result.get("high_52w"))
    if high52 is not None and high52 > min_trade_target:
        candidates.append({"value": high52, "source": "52W-Hoch", "kind": "high52", "priority": 4})

    # Only when real structure above the trigger is sparse do we add the
    # technical measured-move projection. It competes by distance with 52W etc.
    if breakout_mode:
        measured = _breakout_measured_move(result, current_price, trigger_reference)
        if measured is not None and float(measured["value"]) > min_trade_target:
            candidates.append(measured)

    candidates = _dedupe_candidates(candidates)
    chosen = candidates[0] if candidates else None
    return {
        "breakout_mode": breakout_mode,
        "clearance_target": clearance_target,
        "clearance_source": clearance_source,
        "trigger_reference": trigger_reference,
        "trigger_source": trigger_source,
        "target": _finite(chosen.get("value")) if chosen else None,
        "target_source": str(chosen.get("source")) if chosen else "kein belastbares setupgerechtes Trade-Ziel",
        "target_kind": str(chosen.get("kind")) if chosen else "missing",
        "candidates": candidates,
    }


def build_technical_crv_package(result: dict[str, Any] | None) -> dict[str, Any]:
    """Return setup-aware Trade-CRV plus chart clearance diagnostics."""
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
        "technical_crv_target_source": "kein belastbares setupgerechtes Trade-Ziel",
        "technical_crv_target_kind": "missing",
        "technical_crv_entry_reference": entry_ref if entry_ref is not None else math.nan,
        "technical_crv_entry_reference_source": entry_source,
        "technical_crv_stop_now": stop if stop is not None else math.nan,
        "technical_crv_stop_entry": math.nan,
        "technical_crv_stop_source": stop_source,
        "technical_crv_entry_stop_source": stop_source,
        "technical_crv_target_candidates": [],
        "technical_crv_status": "missing",
        "technical_crv_setup_mode": "breakout" if _is_breakout_setup(r) else "standard",
        "technical_clearance_target": math.nan,
        "technical_clearance_source": "kein nahes CHSM-Hindernis",
        "technical_clearance_pct": math.nan,
        "technical_clearance_r": math.nan,
        "technical_crv_trigger_reference": math.nan,
        "technical_crv_trigger_source": "-",
    }
    if price is None or price <= 0:
        return base

    target_pkg = _trade_target_package(r, price, entry_ref)
    clearance_target = target_pkg["clearance_target"]
    base["technical_clearance_target"] = clearance_target if clearance_target is not None else math.nan
    base["technical_clearance_source"] = target_pkg["clearance_source"]
    if clearance_target is not None and clearance_target > price:
        base["technical_clearance_pct"] = (clearance_target - price) / price * 100.0
        if stop is not None and 0 < stop < price:
            risk_now = price - stop
            if risk_now > 0:
                base["technical_clearance_r"] = (clearance_target - price) / risk_now

    trigger_reference = target_pkg["trigger_reference"]
    base["technical_crv_trigger_reference"] = trigger_reference if trigger_reference is not None else math.nan
    base["technical_crv_trigger_source"] = target_pkg["trigger_source"]

    target = target_pkg["target"]
    base["technical_crv_target"] = target if target is not None else math.nan
    base["technical_crv_target_source"] = target_pkg["target_source"]
    base["technical_crv_target_kind"] = target_pkg["target_kind"]
    base["technical_crv_target_candidates"] = [
        {"value": round(float(c["value"]), 4), "source": c["source"], "kind": c["kind"]}
        for c in target_pkg["candidates"][:5]
    ]

    if target is None:
        return base

    if stop is not None and 0 < stop < price and target > price:
        risk_now = price - stop
        base["technical_crv_now"] = (target - price) / risk_now if risk_now > 0 else math.nan

    if entry_ref is not None and entry_ref > 0 and target > entry_ref:
        # Planned-entry CRV uses the same productive stop unless that stop sits
        # above / too close to the lower planned entry. Then use a genuine chart
        # invalidation if available, otherwise enforce the 3.5% practice floor.
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
    """CRV remains a timing / entry-quality brake, not an automatic hard gate."""
    now = _finite(crv_now)
    entry = _finite(crv_entry)
    evaluation = entry if entry is not None else now
    if now is None and entry is None:
        return {"severity": "unknown", "hard_gate": False, "score": 45.0, "evaluation_crv": None}
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
