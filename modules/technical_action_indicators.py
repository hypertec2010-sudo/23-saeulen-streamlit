"""Short-term / swing technical action indicators for CHSM.

v30.21ar adds four complementary technical context blocks that deliberately
avoid duplicating CHSM's existing RSI/MACD/MA/SR logic:

* DMI (+DI/-DI) direction around the existing ADX trend-strength concept.
* Anchored VWAP from a recent swing-low / 60T low anchor.
* Bollinger-inside-Keltner squeeze (TTM-style) and recent release context.
* True price-gap hold/fill context.

Only a *fresh DMI cross* may contribute a small soft confluence component.
AVWAP, Squeeze and Gap remain informational / shadow in this release. None of
these indicators creates a hard gate by itself.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _finite(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _series(df: Any, *names: str) -> pd.Series | None:
    if df is None or not hasattr(df, "columns") or len(df) == 0:
        return None
    for name in names:
        if name in df.columns:
            s = df[name]
            if isinstance(s, pd.DataFrame):
                if s.shape[1] == 0:
                    continue
                s = s.iloc[:, 0]
            s = pd.to_numeric(s, errors="coerce")
            return s
    return None


def _ohlcv(df: Any):
    h = _series(df, "High", "high")
    l = _series(df, "Low", "low")
    c = _series(df, "Close", "close", "Adj Close", "adj_close")
    o = _series(df, "Open", "open")
    v = _series(df, "Volume", "volume")
    if h is None or l is None or c is None:
        return None
    data = pd.DataFrame({"High": h, "Low": l, "Close": c})
    if o is not None:
        data["Open"] = o
    if v is not None:
        data["Volume"] = v
    return data.dropna(subset=["High", "Low", "Close"])


def _true_range(data: pd.DataFrame) -> pd.Series:
    h, l, c = data["High"], data["Low"], data["Close"]
    return pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def _fmt_date(x: Any) -> str:
    try:
        return pd.Timestamp(x).strftime("%d.%m.%Y")
    except Exception:
        return "-"


def _pct_change(now: Any, old: Any) -> float | None:
    n, o = _finite(now), _finite(old)
    if n is None or o is None or o == 0:
        return None
    return (n / o - 1.0) * 100.0


def build_dmi_package(df: Any) -> dict[str, Any]:
    data = _ohlcv(df)
    if data is None or len(data) < 35:
        return {"available": False, "signal": "n/a", "meaning": "Nicht genug Historie.", "action": "Keine DMI-Handlung ableiten.", "confluence_components": []}

    tr = _true_range(data)
    atr = tr.rolling(14).mean()
    up = data["High"].diff()
    dn = -data["Low"].diff()
    pdm = up.where((up > dn) & (up > 0), 0.0).rolling(14).mean()
    ndm = dn.where((dn > up) & (dn > 0), 0.0).rolling(14).mean()
    pdi = 100 * pdm / atr.replace(0, np.nan)
    ndi = 100 * ndm / atr.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.rolling(14).mean()
    valid = pd.DataFrame({"pdi": pdi, "ndi": ndi, "adx": adx}).dropna()
    if len(valid) < 3:
        return {"available": False, "signal": "n/a", "meaning": "DMI noch nicht stabil berechenbar.", "action": "Keine DMI-Handlung ableiten.", "confluence_components": []}

    diff = valid["pdi"] - valid["ndi"]
    prev = diff.shift(1)
    bull = (diff > 0) & (prev <= 0)
    bear = (diff < 0) & (prev >= 0)
    events: list[tuple[int, str]] = []
    for i in range(1, len(valid)):
        if bool(bull.iloc[i]):
            events.append((i, "bullish"))
        elif bool(bear.iloc[i]):
            events.append((i, "bearish"))

    pdi_now = float(valid["pdi"].iloc[-1])
    ndi_now = float(valid["ndi"].iloc[-1])
    adx_now = float(valid["adx"].iloc[-1])
    adx_5 = _pct_change(valid["adx"].iloc[-1], valid["adx"].iloc[-6]) if len(valid) >= 6 else None
    last_dir = None
    days_since = None
    event_date = None
    if events:
        pos, last_dir = events[-1]
        days_since = int(len(valid) - 1 - pos)
        event_date = _fmt_date(valid.index[pos])
    fresh = bool(days_since is not None and days_since <= 10)
    strong = adx_now >= 25.0
    rising = adx_5 is not None and adx_5 > 0

    if pdi_now > ndi_now:
        direction = "bullish"
        if strong:
            signal = "🟢 Bullischer DMI-Trend"
            meaning = f"+DI {pdi_now:.1f} liegt über -DI {ndi_now:.1f}; ADX {adx_now:.1f} bestätigt Trendstärke."
            action = "Long-Setup technisch unterstützt. Entry/CRV und Trigger prüfen; bei deutlicher Überdehnung nicht nachjagen."
        else:
            signal = "🟡 Bullische Richtung, Trend noch schwach"
            meaning = f"+DI {pdi_now:.1f} > -DI {ndi_now:.1f}, aber ADX {adx_now:.1f} zeigt noch keinen starken Trend."
            action = "Eher vorbereiten/beobachten; für Breakout zusätzliche Volumen- oder Struktur-Bestätigung abwarten."
    elif ndi_now > pdi_now:
        direction = "bearish"
        if strong:
            signal = "🔴 Bearischer DMI-Trend"
            meaning = f"-DI {ndi_now:.1f} liegt über +DI {pdi_now:.1f}; ADX {adx_now:.1f} bestätigt negativen Trenddruck."
            action = "Neue Long-Einstiege zurückstellen; Reclaim/Trendwende oder klaren Gegen-Trigger abwarten."
        else:
            signal = "🟡 Bearische Richtung, Trend noch schwach"
            meaning = f"-DI {ndi_now:.1f} > +DI {pdi_now:.1f}, ADX {adx_now:.1f} bleibt jedoch niedrig."
            action = "Kein Long-Vorteil aus DMI; nur bei starker anderer Konfluenz weiter prüfen."
    else:
        direction = "neutral"
        signal = "⚪ DMI neutral"
        meaning = f"+DI und -DI liegen nahe beieinander; ADX {adx_now:.1f}."
        action = "Keine DMI-Richtung erzwingen; Range-/Strukturtrigger abwarten."

    if fresh and last_dir == "bullish":
        signal = "🟢 Frischer +DI/-DI Bull-Cross"
        meaning = f"+DI kreuzte -DI vor {days_since} Handelstagen ({event_date}); ADX {adx_now:.1f}{' steigt' if rising else ''}."
        action = "Frischen Richtungswechsel als Bestätigung nutzen, aber Entry erst mit passender Zone/CRV und Preis-/Volumentrigger freigeben."
    elif fresh and last_dir == "bearish":
        signal = "🔴 Frischer +DI/-DI Bear-Cross"
        meaning = f"-DI kreuzte +DI vor {days_since} Handelstagen ({event_date}); ADX {adx_now:.1f}{' steigt' if rising else ''}."
        action = "Longs defensiver behandeln; Reclaim bzw. neuen bullischen Richtungswechsel abwarten."

    components = []
    if fresh and last_dir in {"bullish", "bearish"}:
        weight = 0.55 if (strong or rising) else 0.40
        components.append({
            "name": "DMI +DI/-DI Cross",
            "direction": last_dir,
            "text": meaning,
            "weight": weight,
        })

    return {
        "available": True,
        "plus_di": pdi_now,
        "minus_di": ndi_now,
        "adx": adx_now,
        "adx_change_5d_pct": adx_5,
        "direction": direction,
        "fresh_cross": fresh,
        "last_cross_direction": last_dir,
        "days_since_cross": days_since,
        "signal": signal,
        "meaning": meaning,
        "action": action,
        "effect": "weiche Konfluenz bei frischem Cross" if fresh else "informativ; Trendstärke bereits via ADX berücksichtigt",
        "confluence_components": components,
    }


def _find_anchor(data: pd.DataFrame) -> tuple[Any, str]:
    # Recent confirmed swing low (3 bars left/right), searched over ~90T.
    low = data["Low"]
    start = max(3, len(data) - 90)
    end = len(data) - 3
    pivots = []
    for i in range(start, max(start, end)):
        v = low.iloc[i]
        if pd.isna(v):
            continue
        if v <= low.iloc[i-3:i].min() and v <= low.iloc[i+1:i+4].min():
            pivots.append(i)
    if pivots:
        i = pivots[-1]
        return data.index[i], "letztes bestätigtes Swing-Low"
    tail = low.tail(60)
    idx = tail.idxmin()
    return idx, "60T-Tief"


def build_avwap_package(df: Any) -> dict[str, Any]:
    data = _ohlcv(df)
    if data is None or "Volume" not in data.columns or len(data) < 25:
        return {"available": False, "signal": "n/a", "meaning": "AVWAP benötigt belastbare Kurs- und Volumendaten.", "action": "Keine AVWAP-Handlung ableiten.", "effect": "informativ / Shadow"}
    volume = pd.to_numeric(data["Volume"], errors="coerce").fillna(0.0)
    if float(volume.tail(60).sum()) <= 0:
        return {"available": False, "signal": "n/a", "meaning": "Volumendaten fehlen für AVWAP.", "action": "Keine AVWAP-Handlung ableiten.", "effect": "informativ / Shadow"}
    anchor_idx, anchor_kind = _find_anchor(data)
    anchored = data.loc[anchor_idx:].copy()
    vol = pd.to_numeric(anchored["Volume"], errors="coerce").fillna(0.0)
    typical = (anchored["High"] + anchored["Low"] + anchored["Close"]) / 3.0
    cumv = vol.cumsum().replace(0, np.nan)
    avwap = (typical * vol).cumsum() / cumv
    avwap_now = _finite(avwap.iloc[-1])
    price = _finite(anchored["Close"].iloc[-1])
    if avwap_now is None or price is None or avwap_now <= 0:
        return {"available": False, "signal": "n/a", "meaning": "AVWAP nicht stabil berechenbar.", "action": "Keine AVWAP-Handlung ableiten.", "effect": "informativ / Shadow"}
    dist = (price / avwap_now - 1.0) * 100.0
    slope5 = _pct_change(avwap.iloc[-1], avwap.iloc[-6]) if len(avwap.dropna()) >= 6 else None
    if dist >= 1.0:
        signal = "🟢 Kurs über Anchored VWAP"
        meaning = f"Kurs liegt {dist:+.1f}% über AVWAP {avwap_now:.2f}; Anker: {anchor_kind} vom {_fmt_date(anchor_idx)}."
        action = "AVWAP als dynamische Pullback-/Support-Referenz nutzen; bei großer Distanz nicht hinterherlaufen, sondern Rücklauf/Halten prüfen."
    elif dist <= -1.0:
        signal = "🔴 Kurs unter Anchored VWAP"
        meaning = f"Kurs liegt {dist:+.1f}% unter AVWAP {avwap_now:.2f}; Anker: {anchor_kind} vom {_fmt_date(anchor_idx)}."
        action = "Neue Longs zurückstellen, bis AVWAP zurückerobert und anschließend gehalten wird."
    else:
        signal = "🟡 AVWAP-Entscheidungszone"
        meaning = f"Kurs liegt nur {dist:+.1f}% vom AVWAP {avwap_now:.2f} entfernt; Anker: {anchor_kind}."
        action = "Reaktion an AVWAP beobachten: Hold/Reclaim unterstützt Entry, klarer Bruch spricht für weiteres Abwarten."
    return {
        "available": True,
        "value": avwap_now,
        "distance_pct": dist,
        "slope_5d_pct": slope5,
        "anchor_date": _fmt_date(anchor_idx),
        "anchor_kind": anchor_kind,
        "signal": signal,
        "meaning": meaning,
        "action": action,
        "effect": "informativ / Shadow · keine Score-Wirkung",
    }


def build_squeeze_package(df: Any) -> dict[str, Any]:
    data = _ohlcv(df)
    if data is None or len(data) < 35:
        return {"available": False, "signal": "n/a", "meaning": "Nicht genug Historie für Squeeze.", "action": "Keine Squeeze-Handlung ableiten.", "effect": "informativ / Shadow"}
    close = data["Close"]
    sma20 = close.rolling(20).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    std20 = close.rolling(20).std()
    # TTM-style squeeze: classic Bollinger (20 SMA, 2 sigma) inside a
    # Keltner envelope around EMA20 using 1.5 x ATR20.
    bb_u, bb_l = sma20 + 2.0 * std20, sma20 - 2.0 * std20
    atr20 = _true_range(data).rolling(20).mean()
    kc_u, kc_l = ema20 + 1.5 * atr20, ema20 - 1.5 * atr20
    squeeze = (bb_u < kc_u) & (bb_l > kc_l)
    valid = pd.DataFrame({"close": close, "ema": ema20, "sq": squeeze}).dropna()
    if valid.empty:
        return {"available": False, "signal": "n/a", "meaning": "Squeeze nicht stabil berechenbar.", "action": "Keine Squeeze-Handlung ableiten.", "effect": "informativ / Shadow"}
    on = bool(valid["sq"].iloc[-1])
    release_days = None
    for back in range(0, min(10, len(valid)-1)):
        i = len(valid)-1-back
        if i > 0 and (not bool(valid["sq"].iloc[i])) and bool(valid["sq"].iloc[i-1]):
            release_days = back
            break
    ema_slope = _pct_change(valid["ema"].iloc[-1], valid["ema"].iloc[-6]) if len(valid) >= 6 else None
    price = float(valid["close"].iloc[-1])
    ema = float(valid["ema"].iloc[-1])
    bullish = price > ema and (ema_slope is None or ema_slope >= 0)
    if on:
        signal = "🔵 TTM-Squeeze aktiv"
        meaning = "Bollinger-Bänder liegen innerhalb der Keltner Channels: Volatilität ist komprimiert."
        action = "Ausbruch nicht vorwegnehmen. 20T-Range-/Strukturbruch plus Volumen und Richtung bestätigen lassen."
    elif release_days is not None and release_days <= 5 and bullish:
        signal = "🟢 Bullischer Squeeze-Release"
        meaning = f"Squeeze löste sich vor {release_days} Handelstagen; Kurs liegt über EMA20 und die kurzfristige Struktur ist positiv."
        action = "Breakout-Fenster aktiv: Trigger, Volumen und CRV prüfen; Rückfall unter Ausbruchs-/EMA20-Zone wäre Warnsignal."
    elif release_days is not None and release_days <= 5:
        signal = "🔴 Squeeze-Release ohne bullische Bestätigung"
        meaning = f"Squeeze löste sich vor {release_days} Handelstagen, aber Kurs/EMA20 bestätigen keinen bullischen Release."
        action = "Long-Entry nicht allein wegen des Squeeze handeln; bullischen Reclaim oder neue Struktur abwarten."
    else:
        signal = "⚪ Kein aktiver Squeeze"
        meaning = "Aktuell keine besondere Bollinger/Keltner-Kompression oder frische Release-Situation."
        action = "Keine Sonderhandlung aus Squeeze; normale CHSM-Trigger, Trend und CRV verwenden."
    return {
        "available": True,
        "squeeze_on": on,
        "release_days": release_days,
        "signal": signal,
        "meaning": meaning,
        "action": action,
        "effect": "informativ / Shadow · keine Score-Wirkung",
    }


def build_gap_package(df: Any) -> dict[str, Any]:
    data = _ohlcv(df)
    if data is None or "Open" not in data.columns or len(data) < 25:
        return {"available": False, "signal": "n/a", "meaning": "Open/High/Low-Historie reicht für Gap-Analyse nicht aus.", "action": "Keine Gap-Handlung ableiten.", "effect": "informativ / Shadow"}
    atr = _true_range(data).rolling(14).mean()
    candidates: list[dict[str, Any]] = []
    start = max(1, len(data)-25)
    for i in range(start, len(data)):
        prev_hi = _finite(data["High"].iloc[i-1])
        prev_lo = _finite(data["Low"].iloc[i-1])
        op = _finite(data["Open"].iloc[i])
        atr_i = _finite(atr.iloc[i-1])
        if prev_hi is None or prev_lo is None or op is None:
            continue
        if op > prev_hi:
            gap = op - prev_hi
            direction = "up"
            boundary = prev_hi
        elif op < prev_lo:
            gap = prev_lo - op
            direction = "down"
            boundary = prev_lo
        else:
            continue
        gap_pct = gap / max(boundary, 1e-9) * 100.0
        gap_atr = gap / atr_i if atr_i and atr_i > 0 else None
        if gap_pct < 0.6 and (gap_atr is None or gap_atr < 0.25):
            continue
        candidates.append({"i": i, "direction": direction, "gap_pct": gap_pct, "gap_atr": gap_atr, "open": op, "boundary": boundary})
    if not candidates:
        return {"available": True, "signal": "⚪ Kein relevantes offenes Gap", "meaning": "In den letzten 25 Handelstagen kein ausreichend großes True Gap gefunden.", "action": "Keine Sonderhandlung aus Gap-Struktur; normale CHSM-Level verwenden.", "effect": "informativ / Shadow · keine Score-Wirkung"}
    g = candidates[-1]
    i = g["i"]
    after = data.iloc[i:]
    direction = g["direction"]
    if direction == "up":
        filled = bool((after["Low"] <= g["boundary"]).any())
        hold = _finite(data["Close"].iloc[-1]) is not None and float(data["Close"].iloc[-1]) > g["boundary"]
    else:
        filled = bool((after["High"] >= g["boundary"]).any())
        hold = _finite(data["Close"].iloc[-1]) is not None and float(data["Close"].iloc[-1]) < g["boundary"]
    days = int(len(data)-1-i)
    atr_txt = "" if g["gap_atr"] is None else f" / {g['gap_atr']:.2f} ATR"
    if direction == "up" and not filled and hold:
        signal = "🟢 Gap-up hält"
        meaning = f"True Gap +{g['gap_pct']:.1f}%{atr_txt} vor {days} Handelstagen ist bislang nicht geschlossen."
        action = "Gap-Unterkante als Support/Invalidierungsreferenz beobachten; Long nur solange Gap-Hold und übrige Trigger intakt bleiben."
    elif direction == "up" and filled:
        signal = "🟡 Gap-up gefüllt"
        meaning = f"Gap-up +{g['gap_pct']:.1f}%{atr_txt} wurde inzwischen geschlossen."
        action = "Gap liefert keinen Stärkevorteil mehr; neuen Reclaim/Support oder frischen Trigger abwarten."
    elif direction == "down" and not filled and hold:
        signal = "🔴 Gap-down bleibt offen"
        meaning = f"True Gap-down -{g['gap_pct']:.1f}%{atr_txt} vor {days} Handelstagen bleibt technisch offen."
        action = "Overhead-Angebot respektieren; Longs erst nach überzeugendem Gap-Reclaim/Strukturbruch aufwerten."
    else:
        signal = "🟡 Gap-Struktur neutralisiert"
        meaning = f"Das jüngste relevante Gap ({g['gap_pct']:.1f}%{atr_txt}) ist technisch weitgehend neutralisiert."
        action = "Gap nicht mehr als Primärsignal verwenden; aktuelle S/R- und Triggerstruktur priorisieren."
    return {
        "available": True,
        "direction": direction,
        "days_since": days,
        "gap_pct": g["gap_pct"],
        "gap_atr": g["gap_atr"],
        "filled": filled,
        "signal": signal,
        "meaning": meaning,
        "action": action,
        "effect": "informativ / Shadow · keine Score-Wirkung",
    }


def build_technical_action_package(df: Any, result: dict[str, Any] | None = None) -> dict[str, Any]:
    dmi = build_dmi_package(df)
    avwap = build_avwap_package(df)
    squeeze = build_squeeze_package(df)
    gap = build_gap_package(df)
    components = list(dmi.get("confluence_components") or []) if isinstance(dmi, dict) else []
    return {
        "available": any(bool(x.get("available")) for x in (dmi, avwap, squeeze, gap) if isinstance(x, dict)),
        "dmi": dmi,
        "avwap": avwap,
        "squeeze": squeeze,
        "gap": gap,
        "confluence_components": components,
        "source": "daily-ohlcv-v30.21ar",
    }


def action_rows(pkg: dict[str, Any] | None) -> list[dict[str, str]]:
    p = pkg or {}
    specs = [
        ("DMI (+DI/-DI)", p.get("dmi") or {}),
        ("Anchored VWAP", p.get("avwap") or {}),
        ("TTM-Squeeze", p.get("squeeze") or {}),
        ("Gap Hold / Fill", p.get("gap") or {}),
    ]
    rows: list[dict[str, str]] = []
    for label, item in specs:
        rows.append({
            "Indikator": label,
            "Signal": str(item.get("signal") or "n/a"),
            "Bedeutung": str(item.get("meaning") or "Keine belastbare Lesart."),
            "Konkrete Handlung": str(item.get("action") or "Keine Sonderhandlung ableiten."),
            "Wirkung": str(item.get("effect") or "informativ"),
        })
    return rows
