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
                           atr: Any = None, lookback: int = 180,
                           max_pivot_age: int | None = None, max_distance: float = 0.22) -> dict[str, Any]:
    """Look for *two separate, completed* pivot lows at a clustered support.

    The stop is below the entire cluster with a volatility buffer, without a
    practice-distance floor. The result is a reference, not a purchase signal.
    """
    missing = {"quality": "unverified", "stop": None, "support": None,
               "source": "Kein mehrfach bestätigtes Swing-Tief", "touches": 0}
    def rejected(code, reason, **details):
        return {**missing, "reason_code": code, "reason": reason, **details}
    price = finite(current_price)
    entry = finite(entry_reference)
    if price is None or entry is None or min(price, entry) <= 0 or df is None:
        return rejected("inputs_missing", "Kurs, Entry-Referenz oder Kurshistorie fehlt/ist ungültig.")
    try:
        import pandas as pd
        if not hasattr(df, "columns") or "Low" not in df.columns:
            return rejected("lows_missing", "Tiefkurse (Low) fehlen in der Kurshistorie.")
        lows = pd.to_numeric(df["Low"], errors="coerce").tail(lookback).reset_index(drop=True)
    except Exception:
        return rejected("lows_unreadable", "Tiefkurse lassen sich nicht auswerten.")
    if len(lows) < 22:
        return rejected("history_short", f"Nur {len(lows)} Kerzen; mindestens 22 für diese Swing-Prüfung erforderlich.")
    pivots = []
    for i in range(3, len(lows) - 3):
        v = finite(lows.iloc[i])
        if v is None or v <= 0:
            continue
        window = lows.iloc[i - 3:i + 4]
        if window.notna().all() and v == min(window) and v < min(lows.iloc[i-3:i]) and v < min(lows.iloc[i+1:i+4]):
            pivots.append((i, v))
    if not pivots:
        return rejected("no_pivots", "Kein striktes Swing-Tief mit je drei Bestätigungskerzen davor und danach erkannt.")
    # Never use a support already above the planned entry as a stop.
    max_support = min(price, entry) * 0.994
    pivot_candidates = [(i, val) for i, val in pivots if val < max_support]
    if not pivot_candidates:
        return rejected("no_downside_pivot", "Erkannte Swing-Tiefs liegen nicht mindestens 0,6 % unter Kurs und Entry-Referenz.",
                        pivot_count=len(pivots))
    clusters = []
    for i, val in reversed(pivot_candidates):
        cluster = [(j, other) for j, other in pivot_candidates
                   if abs(other - val) / val <= 0.015 and abs(j - i) >= 6]
        if cluster:
            group = [(i, val)] + cluster
            support = min(p for _, p in group)
            latest = max(j for j, _ in group)
            clusters.append((support, latest, len(group)))
    if max_pivot_age is not None and clusters:
        clusters = [c for c in clusters if len(lows) - 1 - c[1] <= max_pivot_age]
        if not clusters:
            return rejected("old_retest", "Letzte Bestätigung liegt außerhalb der jüngeren Setup-Struktur.")
    if not clusters:
        return rejected("no_retest", "Kein zweites Swing-Tief innerhalb 1,5 % Preisabstand und mit mindestens sechs Kerzen Abstand erkannt.",
                        candidate_support=max(val for _, val in pivot_candidates),
                        candidate_quality="single_pivot", pivot_count=len(pivots))
    # Closest qualified support beneath the entry; recency breaks ties.
    support, last_idx, touches = sorted(clusters, key=lambda x: (x[0], x[1]), reverse=True)[0]
    atr_val = finite(atr)
    buffer = max(support * 0.006, (atr_val * 0.25 if atr_val and atr_val > 0 else 0.0))
    stop = support - buffer
    # v30.21bf: chart support plus buffer only; no practice-distance substitute.
    if stop <= 0 or stop >= min(price, entry) or (entry - stop) / entry > max_distance:
        reason = ("Stop nach Puffer ist nicht positiv." if stop <= 0 else
                  "Stop nach Puffer liegt nicht unter Kurs und Entry-Referenz." if stop >= min(price, entry) else
                  f"Stop nach Puffer liegt {(entry-stop)/entry*100:.1f} % unter Entry; Grenze ist {max_distance*100:g} %.")
        return rejected("stop_distance", reason, candidate_support=support, candidate_stop=stop,
                        candidate_quality="retested_but_rejected", candidate_touches=int(touches))
    return {
        "quality": "confirmed_swing",
        "stop": round(stop, 6), "support": round(support, 6),
        "support_buffer": round(support - stop, 6),
        "source": f"Bestätigtes Swing-Support-Tief ({touches} Pivot-Berührungen, Puffer)",
        "touches": int(touches), "last_pivot_bar": int(last_idx),
        "reason_code": "confirmed", "reason": "Mehrfach bestätigtes Swing-Tief; Stop mit Puffer erfüllt die Abstandsregeln.",
    }


# Initial trade-relevance heuristics, based on the existing 1–4 week horizon and
# the existing 12% shadow distance guide; not an optimized/backtested forecast.
TRADE_STOP_POLICY = "trade_structure_v30.21bv"


def trade_support_stop(df, *, current_price, entry_reference, atr=None):
    package = confirmed_support_stop(df, current_price=current_price,
        entry_reference=entry_reference, atr=atr, lookback=60,
        max_pivot_age=20, max_distance=0.12)
    package.update(scope="current_setup", lookback=60, max_pivot_age=20,
                   max_distance_pct=12.0, policy=TRADE_STOP_POLICY)
    if package.get("quality") == "confirmed_swing":
        package["source"] += " · jüngere Setup-Struktur"
    return package


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
    structure_reason = structural.get("reason") or "Bestätigte Swing-Support-Referenz fehlt; frisch analysieren."
    if quality == "confirmed":
        reason = "Verwendeter Stop passt zur bestätigten Swing-Support-Referenz."
    elif quality == "missing":
        reason = "Verwendeter Stop fehlt. " + structure_reason
    elif structural.get("quality") != "confirmed_swing":
        reason = explanation + ". " + structure_reason
    elif quality == "proxy":
        reason = "Bestätigte Support-Referenz vorhanden; verwendet wird ein Ersatzstop (" + source + ")."
    else:
        reason = "Bestätigte Support-Referenz vorhanden; verwendeter Stop passt nicht innerhalb der 0,2-%-Toleranz zu deren Strukturstop."
    return {
        "quality": quality, "label": explanation, "used_stop": actual,
        "used_stop_source": source,
        "confirmed_support": finite(structural.get("support")),
        "confirmed_structure_stop": finite(structural.get("stop")),
        "reference_source": str(structural.get("source") or "-"),
        "reason": reason,
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
