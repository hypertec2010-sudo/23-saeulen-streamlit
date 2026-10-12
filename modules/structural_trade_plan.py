"""Provider-free structural stop evidence and pullback plan continuity (v30.21be).

The swing-stop and plan-history functions are deliberately independent of the
live-score decision. No signal is made executable merely because a pullback is
expected. A previous plan expires if its invalidation is broken or a hard gate
is active; cached plan references never override fresh market analysis.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def confirmed_support_stop(df: Any, *, current_price: Any, entry_reference: Any,
                           atr: Any = None, lookback: int = 180) -> dict[str, Any]:
    """Look for *two separate, completed* pivot lows at a clustered support.

    The stop is below the entire cluster with a volatility buffer and a practical
    distance floor. The result is a reference, not an automatic purchase signal.
    """
    missing = {"quality": "unverified", "stop": None, "support": None,
               "source": "Kein mehrfach bestätigtes Swing-Tief", "touches": 0}
    price = finite(current_price)
    entry = finite(entry_reference)
    if price is None or entry is None or min(price, entry) <= 0 or df is None:
        return dict(missing)
    try:
        import pandas as pd
        if not hasattr(df, "columns") or "Low" not in df.columns:
            return dict(missing)
        lows = pd.to_numeric(df["Low"], errors="coerce").tail(lookback).reset_index(drop=True)
    except Exception:
        return dict(missing)
    if len(lows) < 22:
        return dict(missing)
    pivots = []
    for i in range(3, len(lows) - 3):
        v = finite(lows.iloc[i])
        if v is None or v <= 0:
            continue
        window = lows.iloc[i - 3:i + 4]
        if window.notna().all() and v == min(window) and v < min(lows.iloc[i-3:i]) and v < min(lows.iloc[i+1:i+4]):
            pivots.append((i, v))
    if not pivots:
        return dict(missing)
    # Never use a support already above the planned entry as a stop.
    max_support = min(price, entry) * 0.994
    pivot_candidates = [(i, val) for i, val in pivots if val < max_support]
    clusters = []
    for i, val in reversed(pivot_candidates):
        cluster = [(j, other) for j, other in pivot_candidates
                   if abs(other - val) / val <= 0.015 and abs(j - i) >= 6]
        if cluster:
            group = [(i, val)] + cluster
            support = min(p for _, p in group)
            latest = max(j for j, _ in group)
            clusters.append((support, latest, len(group)))
    if not clusters:
        return dict(missing)
    # Closest qualified support beneath the entry; recency breaks ties.
    support, last_idx, touches = sorted(clusters, key=lambda x: (x[0], x[1]), reverse=True)[0]
    atr_val = finite(atr)
    buffer = max(support * 0.006, (atr_val * 0.25 if atr_val and atr_val > 0 else 0.0))
    stop = support - buffer
    # v30.21bf: chart support plus buffer only; no practice-distance substitute.
    if stop <= 0 or stop >= min(price, entry) or (entry - stop) / entry > 0.22:
        return dict(missing)
    return {
        "quality": "confirmed_swing",
        "stop": round(stop, 6), "support": round(support, 6),
        "support_buffer": round(support - stop, 6),
        "source": f"Bestätigtes Swing-Support-Tief ({touches} Pivot-Berührungen, Puffer)",
        "touches": int(touches), "last_pivot_bar": int(last_idx),
    }


def stop_evidence(stop: Any, stop_source: str, structural: dict[str, Any]) -> dict[str, Any]:
    """Classify the *actual used* stop; a nearby reference alone is not proof."""
    actual = finite(stop)
    source = str(stop_source or "-")
    lower = source.lower()
    if actual is None:
        quality, explanation = "missing", "Kein verwendbarer Stop"
    elif any(term in lower for term in ("atr", "praxis", "fallback", "mindestabstand", "nähe", "naeherung")):
        quality, explanation = "proxy", "Ersatz-/Mindestabstands-Stop; nicht chartbestätigt"
    elif (structural.get("quality") == "confirmed_swing"
          and finite(structural.get("stop")) is not None
          and abs(actual - float(structural["stop"])) / max(actual, 1e-9) < 0.002):
        quality, explanation = "confirmed", "Stop an bestätigter Pivot-Unterstützung"
    elif any(term in lower for term in ("atr", "praxis", "fallback", "mindestabstand")):
        quality, explanation = "proxy", "Volatilitäts-/Mindestabstands-Stop; nicht chartbestätigt"
    else:
        quality, explanation = "estimated", "Setup-/MA-/Level-basierte technische Näherung"
    return {
        "quality": quality, "label": explanation, "used_stop": actual,
        "used_stop_source": source,
        "confirmed_support": finite(structural.get("support")),
        "confirmed_structure_stop": finite(structural.get("stop")),
        "reference_source": str(structural.get("source") or "-"),
    }


def plan_stop_reference(result: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """Keep stop value, provenance and quality aligned for history anchors."""
    structure = finite(decision.get("crv_structure_stop"))
    if structure is not None:
        return {"stop": structure, "source": decision.get("crv_structure_source") or "Swing-Support-Referenz",
                "quality": "confirmed"}
    invalidation = finite(result.get("chart_invalidation_level"))
    if invalidation is not None:
        return {"stop": invalidation, "source": result.get("chart_invalidation_source") or "Chart-Invalidierung",
                "quality": "estimated"}
    return {"stop": finite(result.get("stop_used")), "source": result.get("stop_source") or "Stop nicht belegt",
            "quality": decision.get("crv_stop_quality") or "missing"}


def evaluate_pullback_continuity(previous: dict[str, Any], current: dict[str, Any],
                                 *, now_at: datetime | None = None) -> str:
    """Interpret an anchored pullback plan without overriding the current signal."""
    old = previous or {}
    now = current or {}
    p0 = finite(old.get("price"))
    p1 = finite(now.get("price"))
    low = finite(old.get("entry_low"))
    high = finite(old.get("entry_high"))
    invalidation = finite(old.get("invalidation"))
    if any(v is None for v in (p0, low, high, invalidation)):
        return "—"
    if not (0 < invalidation < low <= high):
        return "—"
    if bool(now.get("hard_gate")) or bool(now.get("invalidated")):
        return "⛔ Vorheriger Plan durch aktuelle Sperre ungültig"
    if p1 is not None and p1 <= invalidation:
        return "⛔ Vorherige Struktur invalidiert"
    # Avoid perpetuating an obsolete plan across unrelated market regimes.
    opened = str(old.get("created_at") or "").strip()
    if opened:
        try:
            opened_at = datetime.strptime(opened, "%d.%m.%Y %H:%M:%S")
            if (now_at or datetime.now()) - opened_at > timedelta(days=30):
                return "⏳ Plan nach 30 Tagen abgelaufen – neu prüfen"
        except ValueError:
            return "⏳ Plan-Datum unklar – neu prüfen"
    else:
        return "⏳ Plan-Datum unklar – neu prüfen"
    if p1 is None or p1 <= 0:
        return "⚠️ Frische Kursbasis fehlt – Plan nicht aktuell prüfbar"
    if p1 >= p0 * 0.999:
        status = "Plan intakt – kein Rücksetzer"
    elif low <= p1 <= high:
        status = "🎯 Pullback in Plan-Zone – Trigger neu prüfen"
    elif p1 > high:
        status = "↘ Rücksetzer Richtung Plan-Zone – nicht hinterherlaufen"
    else:
        status = "⚠️ Unter Plan-Zone, über Invalidierung – neu bestätigen"
    # A target-method switch can explain a sudden CRV/status change even when
    # price hardly moved. No later target is auto-promoted into operative CRV.
    prev_target = finite(old.get("target"))
    new_target = finite(now.get("target"))
    if (prev_target is not None and new_target is not None
            and abs(prev_target - new_target) / max(prev_target, 1e-9) > 0.02):
        status += " · Zielbasis gewechselt – CRV prüfen"
    return status


def update_pullback_plan(previous: dict[str, Any], current: dict[str, Any],
                         candidate: dict[str, Any], *, now_at: datetime) -> dict[str, Any]:
    """Retain a valid anchor even on a new buy scan; create only from fresh readiness.

    This returns history metadata only, never a decision-state or buy permission.
    Invalid, expired or undated anchors cannot perpetuate themselves.
    """
    status = evaluate_pullback_continuity(previous, current, now_at=now_at)
    terminal = status == "—" or any(token in status for token in ("⛔", "abgelaufen", "Datum unklar"))
    if previous and not terminal:
        return dict(previous)
    if current.get("decision_state") not in {"plan", "wait_entry", "buy"}:
        return {}
    if current.get("hard_gate") or current.get("invalidated"):
        return {}
    price, low, high, stop, target = (
        finite(candidate.get(key)) for key in ("price", "entry_low", "entry_high", "invalidation", "target")
    )
    if None in (price, low, high, stop, target) or not (0 < stop < low <= high and stop < price < target):
        return {}
    plan = dict(candidate)
    plan["created_at"] = now_at.strftime("%d.%m.%Y %H:%M:%S")
    return plan
