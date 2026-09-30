"""Classical daily Pivot Points as an informational chart context.

The package is deliberately score-neutral. It derives PP/R1/R2/S1/S2 from the
previous completed daily OHLC bar and can describe proximity/confluence with
existing CHSM support/resistance zones. It never changes trading decisions.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LEVEL_ORDER = ("S2", "S1", "PP", "R1", "R2")


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _index_date(value: Any) -> Optional[date]:
    try:
        ts = pd.Timestamp(value)
        return ts.date()
    except Exception:
        return None


def _today_utc(now: Any = None) -> date:
    if now is None:
        return datetime.now(timezone.utc).date()
    try:
        ts = pd.Timestamp(now)
        if ts.tzinfo is None:
            return ts.date()
        return ts.tz_convert("UTC").date()
    except Exception:
        return datetime.now(timezone.utc).date()


def select_previous_completed_daily_bar(df: pd.DataFrame, now: Any = None) -> Optional[pd.Series]:
    """Return the daily bar used for the next/current session's classic pivots.

    If the latest row is dated today, it can be an in-progress daily candle and
    the second-latest row is used. Before the market opens, when the latest row
    is still yesterday's completed candle, the latest row is used.
    """
    if df is None or df.empty:
        return None
    required = {"High", "Low", "Close"}
    if not required.issubset(set(df.columns)):
        return None

    work = df.loc[:, ["High", "Low", "Close"]].copy()
    for col in ("High", "Low", "Close"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["High", "Low", "Close"])
    if work.empty:
        return None

    last_pos = len(work) - 1
    last_date = _index_date(work.index[last_pos])
    if last_date is not None and last_date >= _today_utc(now) and len(work) >= 2:
        last_pos -= 1
    return work.iloc[last_pos]


def _bar_source_date(df: pd.DataFrame, bar: pd.Series, now: Any = None) -> Optional[str]:
    if df is None or df.empty or bar is None:
        return None
    work = df.loc[:, ["High", "Low", "Close"]].copy()
    for col in ("High", "Low", "Close"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["High", "Low", "Close"])
    if work.empty:
        return None
    pos = len(work) - 1
    last_date = _index_date(work.index[pos])
    if last_date is not None and last_date >= _today_utc(now) and len(work) >= 2:
        pos -= 1
    src = _index_date(work.index[pos])
    return src.isoformat() if src else str(work.index[pos])


def calculate_classic_pivot_levels(high: float, low: float, close: float) -> Dict[str, float]:
    h = float(high)
    l = float(low)
    c = float(close)
    pp = (h + l + c) / 3.0
    return {
        "PP": pp,
        "R1": 2.0 * pp - l,
        "R2": pp + (h - l),
        "S1": 2.0 * pp - h,
        "S2": pp - (h - l),
    }


def _distance_pct(price: Optional[float], level: Optional[float]) -> Optional[float]:
    if price is None or level is None or price == 0:
        return None
    return ((level - price) / price) * 100.0


def _zone_iter(structures: Optional[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if not isinstance(structures, dict):
        return []
    groups = (
        ("aktive CHSM-Zone", structures.get("active_zones", []) or []),
        ("CHSM-Support", structures.get("supports", []) or []),
        ("CHSM-Widerstand", structures.get("resistances", []) or []),
    )
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for label, zones in groups:
        for zone in zones:
            if isinstance(zone, dict):
                rows.append((label, zone))
    return rows


def find_pivot_confluences(
    levels: Dict[str, float],
    structures: Optional[Dict[str, Any]],
    near_pct: float = 0.60,
) -> List[Dict[str, Any]]:
    """Find descriptive proximity between classic pivots and CHSM S/R zones."""
    hits: List[Dict[str, Any]] = []
    for level_name in LEVEL_ORDER:
        level = _finite_float(levels.get(level_name))
        if level is None or level <= 0:
            continue
        for zone_label, zone in _zone_iter(structures):
            low = _finite_float(zone.get("low"))
            high = _finite_float(zone.get("high"))
            mid = _finite_float(zone.get("mid"))
            if low is None or high is None:
                continue
            if low > high:
                low, high = high, low
            inside = low <= level <= high
            if mid is None:
                mid = (low + high) / 2.0
            dist_pct = abs((level - mid) / level) * 100.0 if level else None
            if inside or (dist_pct is not None and dist_pct <= near_pct):
                hits.append({
                    "level": level_name,
                    "price": level,
                    "zone_label": zone_label,
                    "zone_low": low,
                    "zone_high": high,
                    "distance_pct": 0.0 if inside else dist_pct,
                    "inside": inside,
                    "text": (
                        f"{level_name} {level:.2f} liegt in {zone_label} {low:.2f}-{high:.2f}."
                        if inside
                        else f"{level_name} {level:.2f} liegt nahe {zone_label} {low:.2f}-{high:.2f}."
                    ),
                })
    hits.sort(key=lambda item: (float(item.get("distance_pct") or 0.0), LEVEL_ORDER.index(item["level"])))
    return hits


def _level_distance_text(price: float, level: float) -> str:
    pct = ((level - price) / price) * 100.0 if price else 0.0
    return f"{pct:+.2f}%"


def _level_value(levels: Dict[str, float], name: Optional[str]) -> Optional[float]:
    if not name:
        return None
    return _finite_float(levels.get(name))


def _neighbor_name(name: Optional[str], step: int) -> Optional[str]:
    if not name or name not in LEVEL_ORDER:
        return None
    idx = LEVEL_ORDER.index(name) + step
    return LEVEL_ORDER[idx] if 0 <= idx < len(LEVEL_ORDER) else None


def _display_level_row(levels: Dict[str, float], price: float, name: str) -> Optional[Dict[str, Any]]:
    value = _level_value(levels, name)
    if value is None:
        return None
    return {
        "name": name,
        "price": round(value, 6),
        "distance_pct": ((value - price) / price) * 100.0 if price else None,
        "text": f"{name} {value:.2f} ({_level_distance_text(price, value)})",
    }


def build_pivot_reading(levels: Dict[str, float], current_price: Optional[float]) -> Dict[str, Any]:
    """Build a score-neutral, action-oriented reading of classic daily pivots.

    The wording deliberately follows a simple pattern:
    current situation -> positive if -> negative if -> what to do now.
    This is descriptive chart guidance only and never changes CHSM decisions.
    """
    price = _finite_float(current_price)
    if price is None or price <= 0:
        return {
            "location": "Aktueller Kurs nicht belastbar verfügbar.",
            "recommendation": "Pivot-Level nur als Orientierung anzeigen; keine technische Handlungshilfe ohne aktuellen Kurs.",
            "nearest": "n/a",
            "guidance": {
                "headline": "Pivot-Lage aktuell nicht belastbar",
                "current": "Ohne belastbaren aktuellen Kurs ist keine Wenn-dann-Einordnung möglich.",
                "positive": "n/a",
                "negative": "n/a",
                "action": "CHSM-Hauptsignal bleibt maßgeblich; Pivot-Kontext heute nicht verwenden.",
                "decision_level": None,
                "near_decision_level": False,
                "relevant_level_names": [],
                "display_levels": [],
            },
        }

    ordered = sorted(((name, float(value)) for name, value in levels.items()), key=lambda x: x[1])
    nearest_name, nearest_price = min(ordered, key=lambda item: abs(item[1] - price))
    nearest_abs_pct = abs((nearest_price - price) / price) * 100.0

    below = [(name, value) for name, value in ordered if value <= price]
    above = [(name, value) for name, value in ordered if value > price]
    lower = max(below, key=lambda item: item[1]) if below else None
    upper = min(above, key=lambda item: item[1]) if above else None

    if lower and upper:
        location = f"Kurs {price:.2f} liegt zwischen {lower[0]} {lower[1]:.2f} und {upper[0]} {upper[1]:.2f}."
    elif lower:
        location = f"Kurs {price:.2f} liegt oberhalb {lower[0]} {lower[1]:.2f}; kein höheres Standard-Pivot-Level mehr im Paket."
    elif upper:
        location = f"Kurs {price:.2f} liegt unterhalb {upper[0]} {upper[1]:.2f}; kein tieferes Standard-Pivot-Level mehr im Paket."
    else:
        location = f"Kurs {price:.2f}; Pivot-Lage nicht eindeutig."

    near = nearest_abs_pct <= 0.35
    relevant_names: List[str] = []

    def _add(name: Optional[str]) -> None:
        if name and name in levels and name not in relevant_names:
            relevant_names.append(name)

    if near:
        _add(nearest_name)
        _add(_neighbor_name(nearest_name, +1))
        _add(_neighbor_name(nearest_name, -1))
    else:
        if lower:
            _add(lower[0])
        if upper:
            _add(upper[0])

    display_levels = []
    for name in relevant_names:
        row = _display_level_row(levels, price, name)
        if row:
            display_levels.append(row)

    if near:
        side = "über" if price >= nearest_price else "unter"
        headline = f"Entscheidungsmarke {nearest_name} direkt am Kurs"
        current = (
            f"Kurs {price:.2f} liegt {nearest_abs_pct:.2f}% {side} {nearest_name} {nearest_price:.2f}. "
            f"{nearest_name} ist damit aktuell die relevante Pivot-Marke."
        )
        up_name = _neighbor_name(nearest_name, +1)
        down_name = _neighbor_name(nearest_name, -1)
        up_value = _level_value(levels, up_name)
        down_value = _level_value(levels, down_name)

        if nearest_name == "PP":
            positive = (
                f"Bleibt bzw. stabilisiert sich der Kurs über PP {nearest_price:.2f}, bestätigt das kurzfristige Stärke. "
                + (f"Nächste obere Orientierung: {up_name} {up_value:.2f}." if up_name and up_value is not None else "")
            ).strip()
            negative = (
                f"Fällt der Kurs klar unter PP {nearest_price:.2f}, kippt die kurzfristige Pivot-Lage nach unten. "
                + (f"Nächste untere Orientierung: {down_name} {down_value:.2f}." if down_name and down_value is not None else "")
            ).strip()
            action = (
                "Aktuell nichts allein wegen des Pivots auslösen. CHSM-Ampel und Setup bleiben führend; "
                "oberhalb PP die obere Route, unterhalb PP die untere Route als Zusatzorientierung nutzen."
            )
        elif nearest_name.startswith("R"):
            positive = (
                f"Stabilisiert sich der Kurs oberhalb {nearest_name} {nearest_price:.2f}, gilt der Pivot-Widerstand als überwunden. "
                + (f"Nächste obere Orientierung: {up_name} {up_value:.2f}." if up_name and up_value is not None else "Oberhalb liegt kein weiteres Standard-Pivot-Level.")
            )
            negative = (
                f"Wird der Kurs an {nearest_name} {nearest_price:.2f} abgewiesen und fällt wieder darunter, bleibt der Widerstand intakt. "
                + (f"Nächste untere Orientierung: {down_name} {down_value:.2f}." if down_name and down_value is not None else "")
            ).strip()
            action = (
                f"Nicht direkt in {nearest_name} hinein nachlaufen. Für Pivot-Kontext erst eine Stabilisierung oberhalb als positive Zusatzbestätigung nutzen; "
                "bei Abweisung die darunterliegende Pivot-Marke beobachten."
            )
        else:
            positive = (
                f"Hält {nearest_name} {nearest_price:.2f} bzw. wird die Marke zurückerobert, stabilisiert sich die kurzfristige Pivot-Lage. "
                + (f"Nächste obere Orientierung: {up_name} {up_value:.2f}." if up_name and up_value is not None else "")
            ).strip()
            negative = (
                f"Fällt der Kurs klar unter {nearest_name} {nearest_price:.2f}, ist diese Pivot-Unterstützung gebrochen. "
                + (f"Nächste untere Orientierung: {down_name} {down_value:.2f}." if down_name and down_value is not None else "Unterhalb liegt kein weiteres Standard-Pivot-Level.")
            )
            action = (
                f"Nicht auf die bloße Berührung von {nearest_name} reagieren. Halt/Rückeroberung als positive Zusatzbestätigung abwarten; "
                "bei Bruch die Marke nicht mehr als Unterstützung anrechnen."
            )
    elif lower and upper:
        headline = f"Kurs zwischen {lower[0]} und {upper[0]}"
        current = (
            f"Kurs {price:.2f} liegt nicht direkt an einer Pivot-Marke: unten {lower[0]} {lower[1]:.2f}, "
            f"oben {upper[0]} {upper[1]:.2f}."
        )
        next_up_name = _neighbor_name(upper[0], +1)
        next_up_value = _level_value(levels, next_up_name)
        next_down_name = _neighbor_name(lower[0], -1)
        next_down_value = _level_value(levels, next_down_name)
        positive = (
            f"Erst ein Anstieg über {upper[0]} {upper[1]:.2f} verbessert die Pivot-Lage nach oben. "
            + (f"Danach wäre {next_up_name} {next_up_value:.2f} die nächste Orientierung." if next_up_name and next_up_value is not None else "")
        ).strip()
        negative = (
            f"Erst ein Fall unter {lower[0]} {lower[1]:.2f} verschlechtert die Pivot-Lage nach unten. "
            + (f"Danach wäre {next_down_name} {next_down_value:.2f} die nächste Orientierung." if next_down_name and next_down_value is not None else "")
        ).strip()
        action = (
            "Aktuell keine Pivot-spezifische Aktion. CHSM-Ampel und Setup bleiben maßgeblich; "
            "erst beim Annähern an eine der beiden Grenzen wird der Pivot-Kontext handlungsrelevant."
        )
    elif lower:
        headline = "Kurs oberhalb der Standard-Pivot-Spanne"
        current = f"Kurs {price:.2f} liegt oberhalb des höchsten Standard-Pivot-Levels {lower[0]} {lower[1]:.2f}."
        positive = f"Solange {lower[0]} {lower[1]:.2f} bei einem Rücklauf hält, bleibt die Pivot-Lage kurzfristig konstruktiv."
        negative = f"Ein Rückfall unter {lower[0]} {lower[1]:.2f} wäre das erste Pivot-Warnsignal; darunter die nächste Marke beobachten."
        action = "Kein zusätzliches Hochjagen wegen der Pivot-Lage; bei Rücklauf den Re-Test der höchsten Pivot-Marke beobachten."
    elif upper:
        headline = "Kurs unterhalb der Standard-Pivot-Spanne"
        current = f"Kurs {price:.2f} liegt unterhalb des tiefsten Standard-Pivot-Levels {upper[0]} {upper[1]:.2f}."
        positive = f"Eine Rückeroberung von {upper[0]} {upper[1]:.2f} wäre das erste positive Pivot-Signal."
        negative = f"Bleibt der Kurs unter {upper[0]} {upper[1]:.2f}, liefert das Pivot-Paket keine technische Unterstützung darunter."
        action = "Nicht allein wegen tiefer Pivot-Lage auf eine Gegenbewegung setzen; zuerst Rückeroberung der tiefsten Marke abwarten."
    else:
        headline = "Pivot-Lage nicht eindeutig"
        current = location
        positive = "n/a"
        negative = "n/a"
        action = "Pivot-Level nur ergänzend zur bestehenden CHSM-Chartstruktur verwenden."

    recommendation = action
    return {
        "location": location,
        "recommendation": recommendation,
        "nearest": f"{nearest_name} {nearest_price:.2f} ({nearest_abs_pct:.2f}% Abstand)",
        "guidance": {
            "headline": headline,
            "current": current,
            "positive": positive,
            "negative": negative,
            "action": action,
            "decision_level": nearest_name if near else None,
            "near_decision_level": bool(near),
            "relevant_level_names": relevant_names,
            "display_levels": display_levels,
        },
    }

def build_classic_daily_pivot_package(
    daily_df: pd.DataFrame,
    current_price: Any = None,
    structures: Optional[Dict[str, Any]] = None,
    now: Any = None,
) -> Dict[str, Any]:
    base = {
        "available": False,
        "score_neutral": True,
        "timeframe": "Daily",
        "levels": {},
        "rows": [],
        "confluences": [],
        "location": "n/a",
        "recommendation": "n/a",
        "nearest": "n/a",
        "guidance": {},
        "source_date": None,
    }
    bar = select_previous_completed_daily_bar(daily_df, now=now)
    if bar is None:
        base["reason"] = "Keine belastbare abgeschlossene Tageskerze verfügbar."
        return base

    high = _finite_float(bar.get("High"))
    low = _finite_float(bar.get("Low"))
    close = _finite_float(bar.get("Close"))
    if high is None or low is None or close is None or high <= 0 or low <= 0 or close <= 0 or high < low:
        base["reason"] = "OHLC-Basis für Pivot Points ist unvollständig oder ungültig."
        return base

    levels = calculate_classic_pivot_levels(high, low, close)
    price = _finite_float(current_price)
    reading = build_pivot_reading(levels, price)
    confluences = find_pivot_confluences(levels, structures)

    rows = []
    for name in ("PP", "R1", "R2", "S1", "S2"):
        value = levels[name]
        dist = _distance_pct(price, value)
        if name == "PP":
            meaning = "zentraler Tages-Pivot"
        elif name.startswith("R"):
            meaning = "Pivot-Widerstand"
        else:
            meaning = "Pivot-Unterstützung"
        rows.append({
            "Level": name,
            "Kurs": round(value, 2),
            "Abstand": "n/a" if dist is None else f"{dist:+.2f}%",
            "Bedeutung": meaning,
        })

    base.update({
        "available": True,
        "source_date": _bar_source_date(daily_df, bar, now=now),
        "source_ohlc": {"high": high, "low": low, "close": close},
        "current_price": price,
        "levels": {key: round(value, 6) for key, value in levels.items()},
        "rows": rows,
        "confluences": confluences[:4],
        "location": reading["location"],
        "recommendation": reading["recommendation"],
        "nearest": reading["nearest"],
        "guidance": reading.get("guidance", {}),
        "reason": "Aus High/Low/Close der vorherigen abgeschlossenen Tageskerze berechnet.",
    })
    return base
