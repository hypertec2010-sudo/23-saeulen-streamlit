"""MA-Cross event analysis for CHSM.

v30.21aq adds *fresh crossover events* as a soft confirmation layer. CHSM
already scores the MA structure (price/MA20/MA50/MA200 and MA ordering), so
this module deliberately does not reward the same information twice. Only a
recent cross can contribute to trigger confluence; older crosses stay
informational.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


PAIR_CONFIG = {
    "ma20_50": {
        "fast": 20,
        "slow": 50,
        "label": "MA20/50",
        "fresh_days": 10,
        "weight": 0.65,
    },
    "ma50_200": {
        "fast": 50,
        "slow": 200,
        "label": "MA50/200",
        "fresh_days": 25,
        "weight": 0.45,
    },
}


def _finite(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _close_series(df: Any) -> pd.Series | None:
    if df is None or not hasattr(df, "columns") or len(df) == 0:
        return None
    close = None
    for key in ("Close", "close", "Adj Close", "adj_close"):
        try:
            if key in df.columns:
                close = df[key]
                break
        except Exception:
            continue
    if close is None:
        return None
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            return None
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        return None
    return close


def _slope_pct(series: pd.Series, lookback: int = 5) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < lookback + 1:
        return None
    now = _finite(s.iloc[-1])
    old = _finite(s.iloc[-1 - lookback])
    if now is None or old is None or old == 0:
        return None
    return (now / old - 1.0) * 100.0


def _date_label(index_value: Any) -> str:
    try:
        ts = pd.Timestamp(index_value)
        return ts.strftime("%d.%m.%Y")
    except Exception:
        return "-"


def _pair_package(close: pd.Series, *, fast: int, slow: int, label: str, fresh_days: int, weight: float) -> dict[str, Any]:
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    valid = pd.DataFrame({"fast": fast_ma, "slow": slow_ma}).dropna()
    if len(valid) < 2:
        return {
            "available": False,
            "label": label,
            "status": "nicht genug Historie",
            "direction": "neutral",
            "fresh": False,
            "days_since": None,
            "event_date": None,
            "weight": weight,
        }

    diff = valid["fast"] - valid["slow"]
    prev = diff.shift(1)
    bullish = (diff > 0) & (prev <= 0)
    bearish = (diff < 0) & (prev >= 0)

    event_positions: list[tuple[int, str]] = []
    for pos in range(1, len(valid)):
        if bool(bullish.iloc[pos]):
            event_positions.append((pos, "bullish"))
        elif bool(bearish.iloc[pos]):
            event_positions.append((pos, "bearish"))

    current_fast = _finite(valid["fast"].iloc[-1])
    current_slow = _finite(valid["slow"].iloc[-1])
    if current_fast is None or current_slow is None:
        structure = "neutral"
    elif current_fast > current_slow:
        structure = "bullish"
    elif current_fast < current_slow:
        structure = "bearish"
    else:
        structure = "neutral"

    fast_slope = _slope_pct(valid["fast"], 5)
    slow_slope = _slope_pct(valid["slow"], 5)

    last_direction = None
    days_since = None
    event_date = None
    if event_positions:
        last_pos, last_direction = event_positions[-1]
        days_since = int(len(valid) - 1 - last_pos)
        event_date = _date_label(valid.index[last_pos])

    fresh = bool(days_since is not None and days_since <= fresh_days)
    confirmed = False
    if fresh and last_direction == "bullish":
        confirmed = bool(
            fast_slope is not None and fast_slope > 0
            and (slow_slope is None or slow_slope >= 0)
        )
    elif fresh and last_direction == "bearish":
        confirmed = bool(
            fast_slope is not None and fast_slope < 0
            and (slow_slope is None or slow_slope <= 0)
        )

    if fresh and last_direction == "bullish":
        status = "frischer bullischer Cross"
        direction = "bullish"
    elif fresh and last_direction == "bearish":
        status = "frischer bärischer Cross"
        direction = "bearish"
    elif structure == "bullish":
        status = "bullische MA-Struktur"
        direction = "neutral"
    elif structure == "bearish":
        status = "bärische MA-Struktur"
        direction = "neutral"
    else:
        status = "MA-Struktur neutral"
        direction = "neutral"

    if days_since is None:
        age_text = "kein Cross im verfügbaren Zeitraum"
    elif days_since == 0:
        age_text = "heute"
    elif days_since == 1:
        age_text = "vor 1 Handelstag"
    else:
        age_text = f"vor {days_since} Handelstagen"

    slope_bits = []
    if fast_slope is not None:
        slope_bits.append(f"MA{fast} {fast_slope:+.2f}%/5T")
    if slow_slope is not None:
        slope_bits.append(f"MA{slow} {slow_slope:+.2f}%/5T")
    slopes_text = " · ".join(slope_bits) if slope_bits else "Steigung n/a"

    if fresh and last_direction:
        event_word = "bullisch" if last_direction == "bullish" else "bärisch"
        summary = f"{label} {event_word} gekreuzt {age_text} ({event_date}); {slopes_text}."
        if confirmed:
            summary += " Cross wird von den MA-Steigungen bestätigt."
        else:
            summary += " Steigungen bestätigen den Cross noch nicht vollständig."
    else:
        relation = "über" if structure == "bullish" else "unter" if structure == "bearish" else "nahe"
        summary = f"MA{fast} liegt {relation} MA{slow}; {age_text}; {slopes_text}."

    return {
        "available": True,
        "label": label,
        "fast_period": fast,
        "slow_period": slow,
        "fast_value": current_fast,
        "slow_value": current_slow,
        "structure": structure,
        "status": status,
        "direction": direction,
        "fresh": fresh,
        "fresh_days": fresh_days,
        "days_since": days_since,
        "event_date": event_date,
        "last_cross_direction": last_direction,
        "confirmed_by_slopes": confirmed,
        "fast_slope_5d_pct": fast_slope,
        "slow_slope_5d_pct": slow_slope,
        "summary": summary,
        "weight": float(weight),
    }


def build_ma_cross_package(df: Any) -> dict[str, Any]:
    """Build MA20/50 and MA50/200 cross context from daily price history."""
    close = _close_series(df)
    if close is None or len(close) < 55:
        return {
            "available": False,
            "score_neutral_when_stale": True,
            "summary": "Nicht genug Tageshistorie für MA-Cross-Auswertung.",
            "pairs": {},
            "confluence_components": [],
        }

    pairs: dict[str, dict[str, Any]] = {}
    for key, cfg in PAIR_CONFIG.items():
        pairs[key] = _pair_package(close, **cfg)

    components = []
    for key in ("ma20_50", "ma50_200"):
        pair = pairs.get(key) or {}
        # Important: only fresh cross EVENTS affect confluence. The standing
        # MA order is already part of CHSM trend quality and is not double-counted.
        if pair.get("available") and pair.get("fresh") and pair.get("direction") in {"bullish", "bearish"}:
            weight = float(pair.get("weight") or 0.0)
            if not pair.get("confirmed_by_slopes"):
                weight *= 0.75
            components.append({
                "name": f"{pair.get('label')} Cross",
                "direction": pair.get("direction"),
                "text": pair.get("summary") or "-",
                "weight": round(weight, 3),
            })

    fresh_labels = [
        str(pair.get("status"))
        for pair in pairs.values()
        if pair.get("available") and pair.get("fresh")
    ]
    if fresh_labels:
        summary = " · ".join(fresh_labels)
    else:
        summary = "Kein frischer MA20/50- oder MA50/200-Cross; MA-Struktur bleibt separat im Trend-Score berücksichtigt."

    return {
        "available": any(bool(p.get("available")) for p in pairs.values()),
        "score_neutral_when_stale": True,
        "summary": summary,
        "pairs": pairs,
        "confluence_components": components,
        "source": "daily-close-v30.21aq",
    }


def pair_display_row(pair: dict[str, Any] | None) -> dict[str, str]:
    """Compact UI row used by the technical chart-analysis section."""
    p = pair or {}
    if not p.get("available"):
        return {
            "Cross": str(p.get("label") or "-"),
            "Status": "n/a",
            "Alter": "-",
            "MA-Steigung (5T)": "-",
            "Wirkung": "keine",
        }
    days = p.get("days_since")
    if days is None:
        age = "kein Cross im Zeitraum"
    elif int(days) == 0:
        age = "heute"
    elif int(days) == 1:
        age = "1 Handelstag"
    else:
        age = f"{int(days)} Handelstage"
    fs = _finite(p.get("fast_slope_5d_pct"))
    ss = _finite(p.get("slow_slope_5d_pct"))
    slope = " / ".join([
        f"MA{p.get('fast_period')} {fs:+.2f}%" if fs is not None else f"MA{p.get('fast_period')} n/a",
        f"MA{p.get('slow_period')} {ss:+.2f}%" if ss is not None else f"MA{p.get('slow_period')} n/a",
    ])
    if p.get("fresh") and p.get("direction") == "bullish":
        effect = "positive Konfluenz"
    elif p.get("fresh") and p.get("direction") == "bearish":
        effect = "bremsende Konfluenz"
    else:
        effect = "nur Struktur · bereits im Trend berücksichtigt"
    return {
        "Cross": str(p.get("label") or "-"),
        "Status": str(p.get("status") or "-"),
        "Alter": age,
        "MA-Steigung (5T)": slope,
        "Wirkung": effect,
    }
