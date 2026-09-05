"""v30.4b - Short-Term Trader & Profit Harvest Engine calibration.

Advisory/provider-free tactical layer. v30.4b adds scan-breadth and
RS-deterioration calibration while it complements the existing classic
TP/Trend path with a dynamic short-term profit-harvest perspective. The module
never places orders, changes stops, or alters productive Live/Shadow scores.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import re
import numpy as np
import pandas as pd

_time_provider = None


def configure_context(*, time_provider=None) -> None:
    global _time_provider
    if time_provider is not None:
        _time_provider = time_provider


def _now() -> datetime:
    if _time_provider is not None:
        try:
            value = _time_provider()
            if isinstance(value, datetime):
                return value
        except Exception:
            pass
    return datetime.now()


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"nan", "none", "n/a", "na", "-"}:
                return default
            if "/" in text:
                text = text.split("/", 1)[0]
            text = text.replace("%", "").replace("R", "").replace(",", ".").strip()
            value = text
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _text(value: Any, default: str = "-") -> str:
    try:
        out = str(value or "").strip()
    except Exception:
        out = ""
    if not out or out.lower() in {"nan", "none", "n/a", "na"}:
        return default
    return out


def _clamp(value: Any, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        return float(max(lo, min(hi, float(value))))
    except Exception:
        return float(lo)


def _first_num(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        val = _num(row.get(key), None)
        if val is not None:
            return val
    return None


def _first_text_number(value: Any) -> float | None:
    try:
        text = str(value or "").replace(",", ".")
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None
    except Exception:
        return None


def _rs_delta_pp(live: dict[str, Any]) -> float | None:
    for key in ("RS-Dynamik Delta PP", "RS-Dynamik Delta", "rs_dynamics_delta_pp", "rs_delta_pp"):
        val = _num(live.get(key), None)
        if val is not None:
            return val
    text = _text(live.get("RS-Dynamik"), "")
    if text:
        match = re.search(r"\(([-+]?\d+(?:[\.,]\d+)?)\s*PP\)", text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", "."))
            except Exception:
                pass
    return None


def _volatility_pct(live: dict[str, Any]) -> float | None:
    direct = _first_num(live, "ATR-%", "ATR %", "atr_pct", "Volatilitaet %", "Volatilitaets-%")
    if direct is not None:
        return direct
    return _first_text_number(live.get("Volatilit\u00e4t"))


def _rs_deterioration_pressure(live: dict[str, Any]) -> float:
    label = _text(live.get("RS-Dynamik"), "").lower()
    delta = _rs_delta_pp(live)
    if "verschlechter" in label:
        if delta is None:
            return 52.0
        return _clamp(35.0 + abs(min(delta, 0.0)) * 2.6, 35.0, 100.0)
    if delta is not None and delta <= -5.0:
        return _clamp(abs(delta) * 2.2, 10.0, 75.0)
    if "verbessert" in label or (delta is not None and delta >= 5.0):
        return 0.0
    if delta is not None and delta < 0:
        return _clamp(abs(delta) * 1.5, 0.0, 18.0)
    return 0.0


def _scan_chop_context(df: pd.DataFrame | None) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or len(df) < 8:
        return {"score": None, "sample": int(len(df)) if isinstance(df, pd.DataFrame) else 0, "reasons": []}

    records = [row.to_dict() for _, row in df.iterrows()]
    n = max(1, len(records))

    def share(predicate) -> float | None:
        hits = 0
        known = 0
        for row in records:
            value = predicate(row)
            if value is None:
                continue
            known += 1
            hits += int(bool(value))
        return (hits / known) if known else None

    active_share = share(lambda r: str(r.get("Ampel") or "").strip() in {"\U0001f7e2", "\U0001f7e1"} if str(r.get("Ampel") or "").strip() else None)
    deteriorating_share = share(
        lambda r: (
            "verschlechter" in _text(r.get("RS-Dynamik"), "").lower()
            or ((_rs_delta_pp(r) is not None) and float(_rs_delta_pp(r)) <= -5.0)
        ) if _text(r.get("RS-Dynamik"), "") else None
    )
    weak_rs_share = share(
        lambda r: any(x in _text(r.get("Relative St\u00e4rke"), "").lower() for x in ("sehr schwach", "schwach", "neutral"))
        if _text(r.get("Relative St\u00e4rke"), "") else None
    )
    elevated_vol_share = share(
        lambda r: any(x in _text(r.get("Volatilit\u00e4tsregime"), "").lower() for x in ("erh\u00f6ht", "erhoeht", "hoch"))
        if _text(r.get("Volatilit\u00e4tsregime"), "") else ((_volatility_pct(r) or 0.0) >= 4.5 if _volatility_pct(r) is not None else None)
    )
    unstable_share = share(
        lambda r: any(x in _text(r.get("Signal-Stabilit\u00e4t"), "").lower() for x in ("wackelig", "fragil", "instabil", "unruh", "defensiv"))
        if _text(r.get("Signal-Stabilit\u00e4t"), "") else None
    )
    positive_market_share = share(
        lambda r: any(x in _text(r.get("Marktregime"), "").lower() for x in ("positiv", "bull", "risk-on"))
        if _text(r.get("Marktregime"), "") else None
    )

    weighted: list[tuple[float, float]] = []
    if active_share is not None:
        weighted.append(((1.0 - active_share) * 100.0, 0.44))
    if deteriorating_share is not None:
        weighted.append((deteriorating_share * 100.0, 0.24))
    if weak_rs_share is not None:
        weighted.append((weak_rs_share * 100.0, 0.14))
    if elevated_vol_share is not None:
        weighted.append((elevated_vol_share * 100.0, 0.10))
    if unstable_share is not None:
        weighted.append((unstable_share * 100.0, 0.08))
    if not weighted:
        return {"score": None, "sample": n, "reasons": []}
    weight_sum = sum(w for _, w in weighted)
    score = sum(v * w for v, w in weighted) / max(weight_sum, 1e-9)

    reasons: list[str] = []
    if active_share is not None and active_share < 0.50:
        reasons.append(f"nur {active_share * 100.0:.0f}% des Scans gruen/gelb")
    if deteriorating_share is not None and deteriorating_share >= 0.22:
        reasons.append(f"RS verschlechtert sich bei {deteriorating_share * 100.0:.0f}%")
    if weak_rs_share is not None and weak_rs_share >= 0.45:
        reasons.append(f"schwache/neutrale RS bei {weak_rs_share * 100.0:.0f}%")
    if elevated_vol_share is not None and elevated_vol_share >= 0.30:
        reasons.append(f"erhoehte Volatilitaet bei {elevated_vol_share * 100.0:.0f}%")

    # A positive index/regime with weak watchlist breadth is a classic mixed/choppy
    # condition: the headline market looks constructive while many individual names
    # do not participate cleanly.
    if positive_market_share is not None and active_share is not None and positive_market_share >= 0.60 and active_share < 0.50:
        score += 10.0
        reasons.append("positives Marktregime, aber schwache Scan-Breite")
    if active_share is not None and active_share < 0.32:
        score += 6.0
    if deteriorating_share is not None and deteriorating_share >= 0.38:
        score += 6.0

    score = _clamp(score)

    def pct(value: float | None) -> float | None:
        return None if value is None else round(value * 100.0, 1)

    return {
        "score": round(float(score), 1),
        "sample": n,
        "active_share_pct": pct(active_share),
        "rs_deteriorating_share_pct": pct(deteriorating_share),
        "weak_rs_share_pct": pct(weak_rs_share),
        "elevated_vol_share_pct": pct(elevated_vol_share),
        "unstable_share_pct": pct(unstable_share),
        "positive_market_share_pct": pct(positive_market_share),
        "reasons": reasons[:4],
    }

def attach_scan_context(df: pd.DataFrame | None) -> pd.DataFrame:
    """Attach provider-free cross-sectional chop context to each Atomic row.

    Hidden columns are additive only. They do not modify productive Live/Shadow
    inputs and can safely be carried into the tactical position layer.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    ctx = _scan_chop_context(out)
    score = ctx.get("score")
    out["__v304b_scan_chop"] = np.nan if score is None else float(score)
    out["__v304b_scan_chop_reason"] = " \u00b7 ".join(ctx.get("reasons") or [])
    out["__v304b_scan_active_share_pct"] = ctx.get("active_share_pct")
    out["__v304b_scan_rs_deteriorating_share_pct"] = ctx.get("rs_deteriorating_share_pct")
    return out


def _holding_days(pos: dict[str, Any], now: datetime | None = None) -> int | None:
    raw_iso = pos.get("opened_at_iso")
    start: date | None = None
    if raw_iso:
        try:
            ts = pd.to_datetime(raw_iso, errors="coerce")
            if pd.notna(ts):
                start = ts.date()
        except Exception:
            start = None
    if start is None and pos.get("created_at"):
        try:
            ts = pd.to_datetime(pos.get("created_at"), dayfirst=True, errors="coerce")
            if pd.notna(ts):
                start = ts.date()
        except Exception:
            start = None
    if start is None:
        return None
    end = (now or _now()).date()
    if end < start:
        return 1
    try:
        return max(1, int(np.busday_count(start.isoformat(), end.isoformat())) + 1)
    except Exception:
        return max(1, (end - start).days + 1)


def _market_context(live: dict[str, Any]) -> tuple[str, str, str, str]:
    market = _text(live.get("Marktregime"), "n/a")
    volatility = _text(live.get("Volatilit\u00e4tsregime"), "n/a")
    if volatility == "n/a":
        atr_pct = _first_num(live, "ATR-%", "ATR %", "atr_pct")
        if atr_pct is not None:
            if atr_pct >= 6.0:
                volatility = "hoch"
            elif atr_pct >= 3.5:
                volatility = "erhoeht"
            else:
                volatility = "normal"
    rs = _text(live.get("RS-Dynamik"), "n/a")
    stability = _text(live.get("Signal-Stabilit\u00e4t"), "n/a")
    return market, volatility, rs, stability


def _risk_inputs(live: dict[str, Any]) -> dict[str, float | None]:
    return {
        "exit": _first_num(live, "Exit-Score", "Exit Score"),
        "tactical": _first_num(live, "Tactical-Exit-Risk", "Tactical Exit Risk"),
        "trend_break": _first_num(live, "Trendbruch-Score", "Trendbruch Score"),
        "momentum": _first_num(live, "Momentum-Collapse-Score", "Momentum Collapse Score"),
        "distribution": _first_num(live, "Distribution-Score", "Distribution Score"),
        "relweak": _first_num(live, "Relative-Schw\u00e4che-Score", "Relative Schwaeche-Score"),
        "accumulation": _first_num(live, "Akkumulation-Score", "Accumulation-Score"),
    }


def _live_score(live: dict[str, Any]) -> float | None:
    return _first_num(live, "Live-Score", "Guarded Engine-Score", "Engine-Score")


def _chop_score(live: dict[str, Any]) -> tuple[float, list[str]]:
    market, volatility, rs, stability = _market_context(live)
    ml, vl, rl, sl = market.lower(), volatility.lower(), rs.lower(), stability.lower()
    risks = _risk_inputs(live)
    score = 36.0
    reasons: list[str] = []

    if any(x in ml for x in ("negativ", "bear", "risk-off")):
        score += 15.0
        reasons.append("schwieriges/negatives Marktregime")
    elif any(x in ml for x in ("neutral", "gemischt", "mixed", "seitw", "unklar")):
        score += 9.0
        reasons.append("gemischtes Marktregime")
    elif any(x in ml for x in ("positiv", "bull", "risk-on")):
        score -= 3.0

    if "hoch" in vl:
        score += 18.0
        reasons.append("hohe Volatilitaet")
    elif any(x in vl for x in ("erh\u00f6ht", "erhoeht", "mittel")):
        score += 10.0
        reasons.append("erhoehte Volatilitaet")
    elif any(x in vl for x in ("niedrig", "ruhig")):
        score -= 2.0

    vol_pct = _volatility_pct(live)
    if vol_pct is not None:
        if vol_pct >= 6.0:
            score += 5.0
        elif vol_pct >= 4.5:
            score += 3.0
        elif vol_pct <= 2.0:
            score -= 1.0

    vals = [v for k, v in risks.items() if k != "accumulation" and v is not None]
    risk_mean = float(np.mean(vals)) if vals else None
    if risk_mean is not None:
        score += _clamp((risk_mean - 42.0) * 0.36, -4.0, 20.0)
        if risk_mean >= 58:
            reasons.append(f"technischer Gegen-/Exit-Druck {risk_mean:.0f}/100")

    rs_delta = _rs_delta_pp(live)
    if "verschlechter" in rl:
        if rs_delta is not None and rs_delta <= -20.0:
            score += 24.0
        elif rs_delta is not None and rs_delta <= -12.0:
            score += 18.0
        elif rs_delta is not None and rs_delta <= -6.0:
            score += 12.0
        else:
            score += 9.0
        if rs_delta is None:
            reasons.append("RS-Dynamik verschlechtert sich")
        else:
            reasons.append(f"RS-Dynamik {rs_delta:+.1f} PP")
    elif "verbessert" in rl:
        score -= 5.0
    elif rs_delta is not None and rs_delta <= -6.0:
        score += 7.0
        reasons.append(f"RS-Dynamik {rs_delta:+.1f} PP")

    rel = _text(live.get("Relative St\u00e4rke"), "").lower()
    if "sehr schwach" in rel:
        score += 8.0
        reasons.append("Relative Staerke sehr schwach")
    elif "schwach" in rel:
        score += 5.0
    elif "neutral" in rel:
        score += 3.0
    elif "stark" in rel:
        score -= 2.0

    if any(x in sl for x in ("wechsel", "instabil", "fragil", "unruh", "wackelig")):
        score += 9.0
        reasons.append("Signalbild wenig stabil")
    elif "defensiv" in sl:
        score += 5.0
    elif any(x in sl for x in ("stabil", "best\u00e4tigt", "bestaetigt")):
        score -= 2.0

    ls = _live_score(live)
    setup = _text(live.get("Setup-Alert"), "-").lower()
    rs_pressure = _rs_deterioration_pressure(live)
    if ls is not None:
        clean_trend_bonus = (
            ls >= 75
            and rs_pressure < 25
            and (risk_mean is None or risk_mean < 50)
            and not any(x in sl for x in ("wackelig", "fragil", "instabil", "unruh"))
            and ("verbessert" in rl or any(x in setup for x in ("breakout", "trend", "trigger aktiv")))
        )
        if clean_trend_bonus:
            score -= 4.0
        elif ls < 50:
            score += 4.0

    scan_chop = _num(live.get("__v304b_scan_chop"), None)
    if scan_chop is not None:
        scan_uplift = _clamp((scan_chop - 42.0) * 0.40, 0.0, 14.0)
        score += scan_uplift
        if scan_uplift >= 3.0:
            scan_reason = _text(live.get("__v304b_scan_chop_reason"), "")
            reasons.append(f"Scan-Chop {scan_chop:.0f}/100" + (f": {scan_reason}" if scan_reason else ""))

    score = _clamp(score)
    if not reasons:
        reasons.append("kein dominanter Chop-Treiber; Markt-/Technikkontext eher geordnet")
    return round(score, 1), reasons[:6]

def _trend_quality(live: dict[str, Any]) -> float:
    market, _, rs, stability = _market_context(live)
    ml, rl, sl = market.lower(), rs.lower(), stability.lower()
    risks = _risk_inputs(live)
    ls = _live_score(live)
    score = 50.0 if ls is None else ls
    if "verbessert" in rl:
        score += 10.0
    elif "verschlechter" in rl:
        score -= 12.0
    if any(x in ml for x in ("positiv", "bull")):
        score += 7.0
    elif any(x in ml for x in ("negativ", "bear")):
        score -= 9.0
    if any(x in sl for x in ("stabil", "best\u00e4tigt", "bestaetigt")):
        score += 5.0
    for key in ("exit", "trend_break", "momentum", "distribution", "relweak"):
        val = risks.get(key)
        if val is not None and val >= 65:
            score -= 5.0
    return _clamp(score)


def _nearest_resistance_pct(live: dict[str, Any], anchor: float) -> float | None:
    if anchor <= 0:
        return None
    candidates = []
    for key in (
        "TP1", "tp1", "Kursziel 1", "Target 1", "target1", "technical_target_1",
        "Technisches Ziel", "Prim\u00e4rziel", "Primaerziel",
    ):
        val = _num(live.get(key), None)
        if val is not None and val > anchor:
            candidates.append(val)
    if not candidates:
        return None
    nearest = min(candidates)
    return (nearest / anchor - 1.0) * 100.0


def _target_structure(live: dict[str, Any], anchor: float, chop: float, trend: float) -> dict[str, Any]:
    atr_pct = _volatility_pct(live)
    if anchor <= 0 or atr_pct is None or atr_pct <= 0:
        return {
            "atr_pct": atr_pct,
            "target_pct": None,
            "target_price": None,
            "secure_pct": None,
            "secure_price": None,
            "horizon": "n/a",
            "resistance_pct": None,
        }

    multiplier = 1.12 - (chop - 35.0) * 0.0045
    if trend >= 72:
        multiplier += 0.10
    elif trend <= 42:
        multiplier -= 0.08
    multiplier = _clamp(multiplier, 0.72, 1.25)
    target_pct = atr_pct * multiplier
    cap = 5.5 if trend >= 72 and chop < 50 else 4.5
    target_pct = _clamp(target_pct, 1.2, cap)

    resistance_pct = _nearest_resistance_pct(live, anchor)
    if resistance_pct is not None and 0.8 <= resistance_pct < target_pct:
        target_pct = max(0.8, resistance_pct * 0.92)

    secure_pct = max(0.8, target_pct * (0.80 if chop >= 65 else 0.84))
    target_price = anchor * (1.0 + target_pct / 100.0)
    secure_price = anchor * (1.0 + secure_pct / 100.0)
    if chop >= 70:
        horizon = "1-3 Handelstage"
    elif chop >= 50:
        horizon = "2-5 Handelstage"
    else:
        horizon = "3-8 Handelstage"
    return {
        "atr_pct": round(float(atr_pct), 2),
        "target_pct": round(float(target_pct), 2),
        "target_price": round(float(target_price), 4),
        "secure_pct": round(float(secure_pct), 2),
        "secure_price": round(float(secure_price), 4),
        "horizon": horizon,
        "resistance_pct": None if resistance_pct is None else round(float(resistance_pct), 2),
    }


def _mode(score: float) -> str:
    if score >= 75:
        return "Profit-Harvest priorisieren"
    if score >= 60:
        return "Kurzfrist-Ziel relevant"
    if score >= 45:
        return "Hybrid beobachten"
    return "Trendpfad priorisieren"


def build_screener_plan(live_row: dict[str, Any] | None) -> dict[str, Any]:
    """Provider-free tactical plan for a current screener row.

    The anchor is the current screener price. This is an additional tactical
    perspective only; classic TP1/TP2/TP3 remain untouched.
    """
    live = dict(live_row or {})
    price = _first_num(live, "Kurs", "price", "Price")
    chop, chop_reasons = _chop_score(live)
    trend = _trend_quality(live)
    target = _target_structure(live, float(price or 0.0), chop, trend)
    risks = _risk_inputs(live)
    risk_vals = [v for k, v in risks.items() if k != "accumulation" and v is not None]
    risk_pressure = float(np.mean(risk_vals)) if risk_vals else 40.0
    rs_pressure = _rs_deterioration_pressure(live)
    scan_chop = _num(live.get("__v304b_scan_chop"), None)
    harvest = (
        chop * 0.62
        + risk_pressure * 0.14
        + (100.0 - trend) * 0.08
        + rs_pressure * 0.16
    )
    if scan_chop is not None:
        harvest += _clamp((scan_chop - 50.0) * 0.18, 0.0, 7.0)
    if trend >= 78 and chop < 45 and rs_pressure < 25:
        harvest -= 6.0
    harvest = _clamp(harvest)
    mode = _mode(harvest)

    if price is None:
        level, ampel, action = "neutral", "\u26aa", "Aktuellen Kurs abwarten"
    elif target.get("target_pct") is None:
        level, ampel, action = "neutral", "\u26aa", "ATR-Datenbasis fehlt"
    elif harvest >= 75:
        level, ampel, action = "orange", "\U0001F7E0", "Kurzfristigen Gewinnpfad priorisieren"
    elif harvest >= 60:
        level, ampel, action = "yellow", "\U0001F7E1", "Kurzfrist-Ziel aktiv einplanen"
    elif harvest >= 45:
        level, ampel, action = "green", "\U0001F7E2", "Hybrid: Trader-Ziel plus Trendpfad"
    else:
        level, ampel, action = "green", "\U0001F7E2", "Klassische Trendziele priorisieren"

    reasons = list(chop_reasons)
    if trend >= 72:
        reasons.append("starke Trendqualit\u00e4t spricht fuer Restposition")
    if target.get("resistance_pct") is not None:
        reasons.append("naher technischer Ziel-/Widerstandsbereich beruecksichtigt")

    confidence_points = sum([
        price is not None,
        target.get("atr_pct") is not None,
        _text(live.get("Marktregime"), "n/a") != "n/a",
        _text(live.get("Volatilit\u00e4tsregime"), "n/a") != "n/a",
        any(v is not None for v in risk_vals),
    ])
    confidence = "Hoch" if confidence_points >= 5 else "Mittel" if confidence_points >= 3 else "Reduziert"

    return {
        "ampel": ampel,
        "level": level,
        "action": action,
        "mode": mode,
        "harvest_score": round(float(harvest), 1),
        "chop_score": round(float(chop), 1),
        "trend_quality": round(float(trend), 1),
        "scan_chop_score": None if scan_chop is None else round(float(scan_chop), 1),
        "rs_deterioration_pressure": round(float(rs_pressure), 1),
        "anchor_price": price,
        **target,
        "partial_pct": 50 if harvest >= 78 else 40 if harvest >= 68 else 33 if harvest >= 60 else 25 if harvest >= 50 else 0,
        "confidence": confidence,
        "why": reasons[:6],
        "why_text": " \u00b7 ".join(reasons[:6]),
        "classic_targets_unchanged": True,
        "provider_calls": 0,
    }


def assess_position(
    pos: dict[str, Any] | None,
    live_row: dict[str, Any] | None,
    *,
    early_profit: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assess an open position for tactical profit harvesting.

    Uses only persisted position data, the already completed Atomic row, and the
    already computed Early-Profit package. It does not execute any management.
    """
    pos = dict(pos or {})
    live = dict(live_row or {})
    early = dict(early_profit or {})
    entry = _num(pos.get("entry"), None)
    current_live = _first_num(live, "Kurs", "price", "Price")
    has_current_live = current_live is not None
    current = current_live if current_live is not None else _num(pos.get("last_price"), None)
    days = early.get("holding_days") if early.get("holding_days") is not None else _holding_days(pos, now=now)
    pnl_pct = _num(early.get("pnl_pct"), None)
    if pnl_pct is None and entry is not None and entry > 0 and current is not None:
        pnl_pct = (current / entry - 1.0) * 100.0

    base = build_screener_plan(live if has_current_live else {})
    anchor = float(entry or 0.0)
    chop = float(base.get("chop_score") or 0.0)
    trend = float(base.get("trend_quality") or 0.0)
    target = _target_structure(live, anchor, chop, trend) if anchor > 0 and has_current_live else {
        "atr_pct": base.get("atr_pct"), "target_pct": None, "target_price": None,
        "secure_pct": None, "secure_price": None, "horizon": "n/a", "resistance_pct": None,
    }

    velocity = _num(early.get("profit_velocity"), None)
    exhaustion = _num(early.get("exhaustion_risk"), None)
    giveback = _num(early.get("giveback_risk"), None)
    hist_n = int(_num(early.get("history_sample"), 0) or 0)
    if hist_n < 5:
        giveback_for_score = None
    else:
        giveback_for_score = giveback

    progress = 0.0
    if pnl_pct is not None and target.get("target_pct") not in (None, 0):
        progress = _clamp((pnl_pct / float(target["target_pct"])) * 100.0)

    components: list[tuple[float, float]] = [(chop, 0.38), (float(base.get("harvest_score") or 0.0), 0.18)]
    if velocity is not None:
        components.append((velocity, 0.17))
    if exhaustion is not None:
        components.append((exhaustion, 0.13))
    if giveback_for_score is not None:
        components.append((giveback_for_score, 0.08))
    components.append((progress, 0.06))
    wsum = sum(w for _, w in components)
    harvest = sum(v * w for v, w in components) / max(wsum, 1e-9)

    if pnl_pct is None or pnl_pct <= 0:
        harvest = min(harvest, 42.0)
    else:
        if days is not None and int(days) <= 3 and pnl_pct >= 1.5:
            harvest += 5.0
        if target.get("secure_pct") is not None and pnl_pct >= float(target["secure_pct"]):
            harvest += 7.0
        if target.get("target_pct") is not None and pnl_pct >= float(target["target_pct"]):
            harvest += 9.0
    if trend >= 75 and (exhaustion is None or exhaustion < 42) and (giveback_for_score is None or giveback_for_score < 55):
        harvest -= 10.0
    harvest = _clamp(harvest)

    if not has_current_live:
        level, ampel, action = "neutral", "\u26aa", "Aktuellen Atomic-Kurs pruefen"
        recommendation = "Keine Kurzfrist-Entscheidung auf Basis eines gespeicherten Alt-Kurses."
    elif entry is None or entry <= 0:
        level, ampel, action = "neutral", "\u26aa", "Entry-Daten ergaenzen"
        recommendation = "Kurzfrist-Ziel braucht einen gueltigen Entry als Renditeanker."
    elif target.get("target_pct") is None:
        level, ampel, action = "neutral", "\u26aa", "ATR-Datenbasis pruefen"
        recommendation = "Ohne aktuelle ATR-Basis wird kein kuenstlich praezises Kurzfrist-Ziel erzeugt."
    elif pnl_pct is None or pnl_pct <= 0:
        level, ampel, action = "green", "\U0001F7E2", "Noch kein Profit-Harvest"
        recommendation = "Kurzfrist-Trader-Pfad beobachten; aktuell besteht noch kein positiver Gewinnpuffer zum Ernten."
    elif harvest >= 78 or (pnl_pct >= float(target.get("target_pct") or 999) and harvest >= 65):
        level, ampel, action = "orange", "\U0001F7E0", "Teilgewinn 40-50% pruefen"
        recommendation = "Taktisches Kurzfrist-Ziel ist erreicht/nahe und Giveback-/Chop-Kontext ist hoch. Gewinn teilweise realisieren; Restposition nur bei weiter intaktem Trend laufen lassen."
    elif harvest >= 65 or (pnl_pct >= float(target.get("secure_pct") or 999) and harvest >= 50):
        level, ampel, action = "yellow", "\U0001F7E1", "Teilgewinn 25-40% pruefen"
        recommendation = "In diesem schwankenden Kontext wird der positive Puffer wertvoll. Teilgewinn wird pruefbar, ohne die klassische Restposition aufzugeben."
    elif harvest >= 55:
        level, ampel, action = "yellow", "\U0001F7E1", "Kurzfrist-Ziel eng beobachten"
        recommendation = "Taktischer Gewinnpfad ist relevant. Bei Zielnaehe Teilgewinn vorbereiten; Trend-Restposition separat weiterfuehren."
    else:
        level, ampel, action = "green", "\U0001F7E2", "Klassischen Trendpfad priorisieren"
        recommendation = "Aktuell spricht die Datenlage eher dafuer, den bestehenden TP-/Trendpfad nicht durch zu fruehes Harvesting abzuschneiden."

    if harvest >= 78:
        partial = 50
    elif harvest >= 68:
        partial = 40
    elif harvest >= 60:
        partial = 33
    elif harvest >= 52:
        partial = 25
    else:
        partial = 0

    risks = _risk_inputs(live)
    exit_pressure_vals = [v for k, v in risks.items() if k in {"exit", "tactical", "trend_break", "momentum", "distribution", "relweak"} and v is not None]
    exit_pressure = max(exit_pressure_vals) if exit_pressure_vals else None
    rs = _text(live.get("RS-Dynamik"), "n/a")
    remainder_ok = bool(
        (exit_pressure is None or exit_pressure < 45)
        and (giveback_for_score is None or giveback_for_score < 60)
        and "verschlechter" not in rs.lower()
    )
    if remainder_ok:
        remainder_rule = "Rest laufen lassen, solange Exit-Druck <45, Giveback <60 und RS nicht verschlechtert."
    else:
        remainder_rule = "Restposition enger fuehren; Exit-Druck/Giveback/RS bestaetigen den aggressiveren Gewinnschutz."

    reasons = list(base.get("why") or [])
    if pnl_pct is not None:
        reasons.insert(0, f"Positionsgewinn {pnl_pct:+.1f}%")
    if velocity is not None:
        reasons.append(f"Profit Velocity {velocity:.0f}/100")
    if exhaustion is not None:
        reasons.append(f"Exhaustion {exhaustion:.0f}/100")
    if giveback_for_score is not None:
        reasons.append(f"historischer Giveback {giveback_for_score:.0f}% (n={hist_n})")

    return {
        "ticker": str(pos.get("ticker") or live.get("Ticker") or "").strip().upper(),
        "ampel": ampel,
        "level": level,
        "action": action,
        "recommendation": recommendation,
        "mode": _mode(harvest),
        "harvest_score": round(float(harvest), 1),
        "chop_score": round(float(chop), 1),
        "trend_quality": round(float(trend), 1),
        "scan_chop_score": base.get("scan_chop_score"),
        "rs_deterioration_pressure": base.get("rs_deterioration_pressure"),
        "holding_days": days,
        "pnl_pct": None if pnl_pct is None else round(float(pnl_pct), 2),
        "current": current,
        "entry": entry,
        "has_current_live": has_current_live,
        "profit_velocity": velocity,
        "exhaustion_risk": exhaustion,
        "giveback_risk": giveback_for_score,
        "history_sample": hist_n,
        **target,
        "partial_pct": partial,
        "remainder_rule": remainder_rule,
        "remainder_ok": remainder_ok,
        "exit_pressure_max": None if exit_pressure is None else round(float(exit_pressure), 1),
        "confidence": base.get("confidence", "Reduziert"),
        "why": reasons[:7],
        "why_text": " \u00b7 ".join(reasons[:7]),
        "event_active": bool(has_current_live and pnl_pct is not None and pnl_pct > 0 and harvest >= 55),
        "classic_targets_unchanged": True,
        "provider_calls": 0,
    }


def enrich_live_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    """Add read-only short-term trader columns to an existing Atomic frame."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = attach_scan_context(df)
    trader_target = []
    secure = []
    harvest = []
    chop = []
    scan_chop = []
    mode = []
    horizon = []
    for _, row in out.iterrows():
        pkg = build_screener_plan(row.to_dict())
        tp = pkg.get("target_pct")
        price = pkg.get("target_price")
        sp = pkg.get("secure_pct")
        trader_target.append("n/a" if tp is None or price is None else f"+{float(tp):.1f}% @ {float(price):.2f}")
        secure.append("n/a" if sp is None else f"+{float(sp):.1f}%")
        harvest.append(f"{float(pkg.get('harvest_score') or 0):.0f}/100")
        chop.append(f"{float(pkg.get('chop_score') or 0):.0f}/100")
        sc = pkg.get("scan_chop_score")
        scan_chop.append("n/a" if sc is None else f"{float(sc):.0f}/100")
        mode.append(f"{pkg.get('ampel','\u26aa')} {pkg.get('mode','-')}")
        horizon.append(str(pkg.get("horizon") or "n/a"))
    out["Trader-Ziel"] = trader_target
    out["Trader-Sicherung ab"] = secure
    out["Harvest-Score"] = harvest
    out["Chop-Risk"] = chop
    out["Scan-Chop"] = scan_chop
    out["Trader-Modus"] = mode
    out["Trader-Horizont"] = horizon
    return out
