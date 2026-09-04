"""v30.2 - Early Profit Protection & Giveback Engine.

Observational/advisory helper for open long positions.  The module does not
place orders, change stops, or alter the productive Live/Shadow score.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Iterable

import numpy as np
import pandas as pd

_storage = None
_time_provider = None
_NAMESPACE = "profit_protection_v302"


def configure_context(*, storage=None, time_provider=None) -> None:
    global _storage, _time_provider
    if storage is not None:
        _storage = storage
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
            text = text.replace("%", "").replace(",", ".").strip()
            value = text
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _text(value: Any, default: str = "-") -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        text = ""
    if not text or text.lower() in {"nan", "none", "n/a", "na"}:
        return default
    return text


def _clamp(value: Any, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        return float(max(lo, min(hi, float(value))))
    except Exception:
        return float(lo)


def _parse_position_date(pos: dict[str, Any]) -> date | None:
    # opened_at_iso is ISO-8601 and must not be parsed with dayfirst=True.
    raw_iso = pos.get("opened_at_iso")
    if raw_iso:
        try:
            parsed = pd.to_datetime(raw_iso, errors="coerce")
            if pd.notna(parsed):
                return parsed.date()
        except Exception:
            pass
    raw_created = pos.get("created_at")
    if raw_created:
        try:
            parsed = pd.to_datetime(raw_created, dayfirst=True, errors="coerce")
            if pd.notna(parsed):
                return parsed.date()
        except Exception:
            pass
    return None


def holding_business_days(pos: dict[str, Any], now: datetime | None = None) -> int | None:
    start = _parse_position_date(pos or {})
    if start is None:
        return None
    end = (now or _now()).date()
    if end < start:
        return 1
    try:
        # Include the opening day.  Weekend/holiday noise is preferable to using
        # raw calendar days for the early-move detector.
        days = int(np.busday_count(start.isoformat(), end.isoformat())) + 1
        return max(1, days)
    except Exception:
        return max(1, (end - start).days + 1)


def _initial_stop(pos: dict[str, Any], entry: float | None) -> float | None:
    if entry is None:
        return None
    candidates = [pos.get("initial_stop")]
    try:
        for item in list(pos.get("stop_history") or []):
            if isinstance(item, dict):
                candidates.extend([item.get("old_stop"), item.get("new_stop")])
    except Exception:
        pass
    candidates.append(pos.get("stop"))
    for raw in candidates:
        val = _num(raw, None)
        if val is not None and val > 0 and val < entry:
            return val
    return None


def sample_label(n: int) -> str:
    n = int(n or 0)
    if n < 5:
        return "Zu klein"
    if n < 10:
        return "Frühphase"
    if n < 20:
        return "Mittel"
    if n < 40:
        return "Gut"
    return "Breiter"


def _profile_store() -> dict[str, Any]:
    if _storage is None:
        return {}
    try:
        data = _storage.load_namespace(_NAMESPACE, default={})
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_profile(ticker: str) -> dict[str, Any]:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return {}
    store = _profile_store()
    profiles = store.get("profiles") if isinstance(store.get("profiles"), dict) else {}
    profile = profiles.get(symbol)
    return dict(profile) if isinstance(profile, dict) else {}


def save_profile(ticker: str, profile: dict[str, Any]) -> bool:
    symbol = str(ticker or "").strip().upper()
    if not symbol or not isinstance(profile, dict) or _storage is None:
        return False
    store = _profile_store()
    profiles = store.get("profiles") if isinstance(store.get("profiles"), dict) else {}
    profiles = dict(profiles)
    profiles[symbol] = dict(profile)
    store = dict(store)
    store["profiles"] = profiles
    store["updated_at"] = _now().isoformat()
    try:
        return bool(_storage.save_namespace(_NAMESPACE, store))
    except Exception:
        return False


def assess_position(
    pos: dict[str, Any],
    live_row: dict[str, Any] | None = None,
    *,
    history_profile: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assess whether an open winner has moved unusually fast, unusually far.

    No provider calls are performed here.  Only persisted position data and the
    already completed Atomic live row are used.  Historical giveback evidence is
    optional and must be supplied/persisted by an explicit on-demand analysis.
    """
    pos = dict(pos or {})
    live = dict(live_row or {})
    profile = dict(history_profile or {})

    ticker = str(pos.get("ticker") or live.get("Ticker") or "").strip().upper()
    entry = _num(pos.get("entry"), None)
    current_live = _num(live.get("Kurs"), None)
    current = current_live if current_live is not None else _num(pos.get("last_price"), None)
    has_current_live = current_live is not None
    days = holding_business_days(pos, now=now)

    pnl_pct = None
    if entry is not None and entry > 0 and current is not None:
        pnl_pct = (current / entry - 1.0) * 100.0

    ref_stop = _initial_stop(pos, entry)
    r_mult = None
    if entry is not None and ref_stop is not None and current is not None and entry > ref_stop:
        r_mult = (current - entry) / (entry - ref_stop)

    atr_pct = _num(live.get("ATR-%"), None)
    ma10_dist = _num(live.get("MA10-Abstand %"), None)
    if ma10_dist is None:
        ma10_dist = _num(live.get("MA10_Abstand_%"), None)
    exit_score = _num(live.get("Exit-Score"), None)
    tactical = _num(live.get("Tactical-Exit-Risk"), None)
    trend_break = _num(live.get("Trendbruch-Score"), None)
    momentum = _num(live.get("Momentum-Collapse-Score"), None)
    distribution = _num(live.get("Distribution-Score"), None)
    relweak = _num(live.get("Relative-Schwäche-Score"), None)
    accumulation = _num(live.get("Akkumulation-Score"), None)
    market = _text(live.get("Marktregime"), "n/a")
    volatility = _text(live.get("Volatilitätsregime"), "n/a")
    rs_dyn = _text(live.get("RS-Dynamik"), "n/a")
    setup_alert = _text(live.get("Setup-Alert"), "-")

    # Historical profile: risk is the empirically observed share of similar
    # fast-move events that gave back at least half of the impulse in 5 sessions.
    hist_risk = _num(profile.get("combined_giveback_rate"), None)
    hist_sample = int(_num(profile.get("combined_n"), 0) or 0)
    hist_follow = _num(profile.get("combined_followthrough_rate"), None)
    hist_revisit = _num(profile.get("combined_revisit_start_rate"), None)
    hist_drawdown = _num(profile.get("combined_median_drawdown_pct"), None)
    history_compatible = bool(profile)
    profile_days = int(_num(profile.get("signal_holding_days"), 0) or 0)
    profile_pnl = _num(profile.get("signal_pnl_pct"), None)
    if history_compatible and days is not None and profile_days:
        if abs(min(int(days), 3) - min(profile_days, 3)) > 1:
            history_compatible = False
    if history_compatible and pnl_pct is not None and pnl_pct > 0 and profile_pnl is not None and profile_pnl > 0:
        ratio = pnl_pct / profile_pnl
        if ratio < 0.65 or ratio > 1.55:
            history_compatible = False
    # An old profile remains visible for audit/transparency, but it must not
    # influence today's recommendation when the fast-move shape changed a lot.
    hist_risk_for_action = hist_risk if history_compatible else None

    if not has_current_live:
        return {
            "ticker": ticker,
            "ampel": "⚪",
            "level": "neutral",
            "active": False,
            "action": "Aktuellen Scanwert prüfen",
            "profit_velocity": None,
            "exhaustion_risk": None,
            "giveback_risk": hist_risk,
            "holding_days": days,
            "pnl_pct": pnl_pct,
            "r_multiple": r_mult,
            "atr_units": None,
            "daily_gain_pct": None,
            "history_sample": hist_sample,
            "history_label": sample_label(hist_sample),
            "history_compatible": history_compatible,
            "history_profile": profile,
            "why": ["Kein aktueller Kurs im letzten Atomic-Vollscan"],
            "why_text": "Kein aktueller Kurs im letzten Atomic-Vollscan",
            "recommendation": "Keine Early-Profit-Entscheidung auf Basis eines gespeicherten Alt-Kurses.",
        }

    if pnl_pct is None or days is None or days <= 0:
        return {
            "ticker": ticker,
            "ampel": "⚪",
            "level": "neutral",
            "active": False,
            "action": "Datenbasis ergänzen",
            "profit_velocity": None,
            "exhaustion_risk": None,
            "giveback_risk": hist_risk,
            "holding_days": days,
            "pnl_pct": pnl_pct,
            "r_multiple": r_mult,
            "atr_units": None,
            "daily_gain_pct": None,
            "history_sample": hist_sample,
            "history_label": sample_label(hist_sample),
            "history_compatible": history_compatible,
            "history_profile": profile,
            "why": ["Entry oder Eröffnungszeitpunkt fehlt"],
            "why_text": "Entry oder Eröffnungszeitpunkt fehlt",
            "recommendation": "Early-Profit-Schutz benötigt Entry, aktuellen Kurs und Eröffnungsdatum.",
        }

    daily_gain = pnl_pct / max(days, 1)
    atr_units = (pnl_pct / atr_pct) if (atr_pct is not None and atr_pct > 0 and pnl_pct > 0) else None

    if pnl_pct <= 0:
        velocity = 0.0
    else:
        speed_score = _clamp((daily_gain - 0.75) * 24.0)
        atr_score = _clamp(((atr_units or 0.0) - 0.75) * 42.0) if atr_units is not None else 0.0
        r_score = _clamp(((r_mult or 0.0) - 0.35) * 48.0) if r_mult is not None else 0.0
        pnl_score = _clamp((pnl_pct - 2.0) * 7.5)
        velocity = speed_score * 0.38 + atr_score * 0.30 + r_score * 0.18 + pnl_score * 0.14
        if days <= 3:
            time_factor = 1.0
        elif days <= 5:
            time_factor = 0.90
        elif days <= 8:
            time_factor = 0.68
        elif days <= 12:
            time_factor = 0.45
        else:
            time_factor = 0.25
        velocity = _clamp(velocity * time_factor)

    # Extension / exhaustion is deliberately separate from velocity.  A fast
    # move can remain healthy if the chart, RS and market still confirm it.
    extension = 0.0
    if atr_units is not None:
        extension = max(extension, _clamp((atr_units - 1.25) * 35.0))
    if ma10_dist is not None:
        extension = max(extension, _clamp((ma10_dist - 2.0) * 12.5))

    tech_values = [v for v in (momentum, distribution, tactical, exit_score, trend_break, relweak) if v is not None]
    tech_max = max(tech_values) if tech_values else 0.0
    components: list[tuple[float, float]] = [(velocity, 0.18), (extension, 0.24)]
    for value, weight in (
        (momentum, 0.15),
        (distribution, 0.14),
        (tactical, 0.10),
        (exit_score, 0.07),
        (trend_break, 0.06),
        (relweak, 0.06),
    ):
        if value is not None:
            components.append((value, weight))
    weight_sum = sum(w for _, w in components)
    exhaustion = sum(v * w for v, w in components) / max(weight_sum, 1e-9)

    market_l = market.lower()
    vol_l = volatility.lower()
    rs_l = rs_dyn.lower()
    setup_l = setup_alert.lower()
    if "verschlechter" in rs_l:
        exhaustion += 10.0
    elif "verbessert" in rs_l:
        exhaustion -= 7.0
    if "negativ" in market_l or "bear" in market_l:
        exhaustion += 6.0
    elif "positiv" in market_l or "bull" in market_l:
        exhaustion -= 3.0
    if "hoch" in vol_l:
        exhaustion += 7.0
    if any(token in setup_l for token in ("climax", "exhaust", "überdehnt", "ueberdehnt", "wide & loose", "fomo")):
        exhaustion += 10.0
    if accumulation is not None and distribution is not None and accumulation >= distribution + 15:
        exhaustion -= 5.0
    if hist_risk_for_action is not None and hist_sample >= 5:
        exhaustion += (hist_risk_for_action - 50.0) * 0.18
    exhaustion = _clamp(exhaustion)

    reasons: list[str] = []
    if pnl_pct > 0:
        reasons.append(f"{pnl_pct:+.1f}% in {days} Handelstag(en)")
    if atr_units is not None and atr_units >= 1.5:
        reasons.append(f"Impuls ca. {atr_units:.1f} ATR")
    if r_mult is not None and r_mult >= 1.0:
        reasons.append(f"Positionspuffer {r_mult:.2f}R")
    if ma10_dist is not None and ma10_dist >= 4.0:
        reasons.append(f"{ma10_dist:.1f}% über MA10")
    if distribution is not None and distribution >= 60:
        reasons.append(f"Distribution {distribution:.0f}/100")
    if momentum is not None and momentum >= 60:
        reasons.append(f"Momentum-Abbau {momentum:.0f}/100")
    if "verschlechter" in rs_l:
        reasons.append("RS-Dynamik verschlechtert sich")
    if hist_risk_for_action is not None and hist_sample >= 5:
        reasons.append(f"historischer Giveback {hist_risk_for_action:.0f}% bei n={hist_sample}")
    elif profile and not history_compatible:
        reasons.append("historisches Fast-Move-Profil passt nicht mehr zum aktuellen Move")

    active = bool(pnl_pct > 0 and days <= 12 and velocity >= 45)
    if not active:
        level, ampel, action = "green", "🟢", "Normaler Gewinnpfad"
        recommendation = "Kein ungewöhnlich schneller Frühgewinn. Führung primär über Exit Engine 2.0 und Stop-/Trendstruktur."
    elif velocity >= 78 and exhaustion < 42 and (hist_risk_for_action is None or hist_risk_for_action < 60):
        level, ampel, action = "green", "🟢", "Healthy Acceleration · laufen lassen"
        recommendation = "Der Anstieg ist schnell, aber aktuell noch konstruktiv bestätigt. Nicht allein wegen der Geschwindigkeit verkaufen."
    elif velocity >= 86 and exhaustion >= 82 and (hist_risk_for_action is not None and hist_risk_for_action >= 70) and tech_max >= 70:
        level, ampel, action = "red", "🔴", "Teilgewinn / Exit prüfen"
        recommendation = "Sehr schneller Frühgewinn plus klare Ermüdung und historisch hohes Giveback-Risiko. Deutlichen Gewinnschutz bzw. Teil-/Gesamtreduktion prüfen."
    elif velocity >= 75 and (exhaustion >= 62 or (hist_risk_for_action is not None and hist_risk_for_action >= 65)):
        level, ampel, action = "orange", "🟠", "Teilgewinn 25–50% prüfen"
        recommendation = "Schneller Frühgewinn mit erhöhtem Giveback-/Ermüdungsrisiko. Teilgewinn und technischen Trail der Restposition prüfen."
    elif velocity >= 62 and (exhaustion >= 45 or (hist_risk_for_action is not None and hist_risk_for_action >= 55)):
        level, ampel, action = "yellow", "🟡", "Gewinnschutz prüfen"
        recommendation = "Der Gewinn ist ungewöhnlich schnell entstanden. Kein Vollausstiegssignal, aber Einstand/Strukturstop und ggf. kleiner Teilgewinn werden prüfbar."
    else:
        level, ampel, action = "green", "🟢", "Schnell, aber noch konstruktiv"
        recommendation = "Tempo beobachten, aber ohne zusätzliche Ermüdungssignale nicht vorschnell verkaufen."

    if not reasons:
        reasons = ["kein dominanter Early-Profit-/Giveback-Treiber"]

    return {
        "ticker": ticker,
        "ampel": ampel,
        "level": level,
        "active": active,
        "action": action,
        "profit_velocity": round(float(velocity), 1),
        "exhaustion_risk": round(float(exhaustion), 1),
        "giveback_risk": None if hist_risk_for_action is None else round(float(hist_risk_for_action), 1),
        "history_compatible": history_compatible,
        "holding_days": days,
        "pnl_pct": round(float(pnl_pct), 2),
        "r_multiple": None if r_mult is None else round(float(r_mult), 2),
        "atr_pct": atr_pct,
        "atr_units": None if atr_units is None else round(float(atr_units), 2),
        "daily_gain_pct": round(float(daily_gain), 2),
        "history_sample": hist_sample,
        "history_label": sample_label(hist_sample),
        "history_followthrough_rate": hist_follow,
        "history_revisit_start_rate": hist_revisit,
        "history_median_drawdown_pct": hist_drawdown,
        "history_profile": profile,
        "why": reasons[:6],
        "why_text": " · ".join(reasons[:6]),
        "recommendation": recommendation,
        "technical_max": round(float(tech_max), 1),
        "market": market,
        "volatility": volatility,
        "rs_dynamics": rs_dyn,
    }


def _normalize_history(df: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename = {str(c).lower(): c for c in out.columns}
    required = {}
    for canonical in ("Open", "High", "Low", "Close", "Volume"):
        found = rename.get(canonical.lower())
        if found is not None:
            required[found] = canonical
    out = out.rename(columns=required)
    if "Close" not in out.columns:
        return pd.DataFrame()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in out.columns:
            if col in ("Open", "High", "Low"):
                out[col] = out["Close"]
            elif col == "Volume":
                out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close"]).sort_index()
    return out


def _atr_pct_series(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=10).mean()
    return (atr / close.replace(0, np.nan)) * 100.0


def _event_rows(
    df: pd.DataFrame,
    *,
    symbol: str,
    window: int,
    horizon: int,
    threshold_move_pct: float,
    threshold_atr: float,
    max_events: int = 80,
) -> list[dict[str, Any]]:
    data = _normalize_history(df)
    if data.empty or len(data) < 60:
        return []
    data = data.copy()
    data["ATR_Pct"] = _atr_pct_series(data)
    rows: list[dict[str, Any]] = []
    last_event_i = -999
    start_i = max(20, window)
    stop_i = len(data) - horizon - 1
    for i in range(start_i, max(start_i, stop_i)):
        if i - last_event_i <= max(window, 2) + 2:
            continue
        start_close = _num(data["Close"].iloc[i - window], None)
        event_close = _num(data["Close"].iloc[i], None)
        atr_pct = _num(data["ATR_Pct"].iloc[i], None)
        if start_close is None or event_close is None or start_close <= 0 or event_close <= 0 or atr_pct is None or atr_pct <= 0:
            continue
        move_pct = (event_close / start_close - 1.0) * 100.0
        move_atr = move_pct / atr_pct
        if move_pct < threshold_move_pct or move_atr < threshold_atr:
            continue
        future = data.iloc[i + 1:i + 1 + horizon]
        if len(future) < horizon:
            continue
        future_low = _num(future["Low"].min(), None)
        future_high = _num(future["High"].max(), None)
        end_close = _num(future["Close"].iloc[-1], None)
        if future_low is None or future_high is None or end_close is None:
            continue
        drawdown_pct = max(0.0, (event_close - future_low) / event_close * 100.0) if event_close > 0 else np.nan
        follow_pct = max(0.0, (future_high / event_close - 1.0) * 100.0)
        end_return_pct = (end_close / event_close - 1.0) * 100.0
        giveback_fraction = drawdown_pct / max(move_pct, 1e-9)
        half_giveback = bool(giveback_fraction >= 0.50)
        revisit_start = bool(future_low <= start_close * 1.01)
        direct_follow = bool(end_return_pct > 0 and follow_pct >= max(1.0, move_pct * 0.25) and not half_giveback)
        idx_val = data.index[i]
        try:
            event_date = pd.to_datetime(idx_val).date().isoformat()
        except Exception:
            event_date = str(idx_val)
        rows.append({
            "Ticker": symbol,
            "Datum": event_date,
            "Move_%": round(move_pct, 2),
            "Move_ATR": round(move_atr, 2),
            "Giveback_%": round(drawdown_pct, 2),
            "Giveback_Anteil_%": round(giveback_fraction * 100.0, 1),
            "Halber_Giveback": half_giveback,
            "Zurueck_zum_Start": revisit_start,
            "Followthrough_%": round(follow_pct, 2),
            "Ende_5T_%": round(end_return_pct, 2),
            "Direkter_Followthrough": direct_follow,
        })
        last_event_i = i
        if len(rows) >= max_events:
            break
    return rows


def _summarize_events(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not rows:
        return {
            "n": 0,
            "giveback_rate": None,
            "revisit_start_rate": None,
            "followthrough_rate": None,
            "median_drawdown_pct": None,
            "median_followthrough_pct": None,
        }
    frame = pd.DataFrame(rows)
    return {
        "n": n,
        "giveback_rate": round(float(frame["Halber_Giveback"].astype(float).mean() * 100.0), 1),
        "revisit_start_rate": round(float(frame["Zurueck_zum_Start"].astype(float).mean() * 100.0), 1),
        "followthrough_rate": round(float(frame["Direkter_Followthrough"].astype(float).mean() * 100.0), 1),
        "median_drawdown_pct": round(float(pd.to_numeric(frame["Giveback_%"], errors="coerce").median()), 2),
        "median_followthrough_pct": round(float(pd.to_numeric(frame["Followthrough_%"], errors="coerce").median()), 2),
    }


def _weighted_metric(same: dict[str, Any], peer: dict[str, Any], key: str) -> float | None:
    a = _num(same.get(key), None)
    b = _num(peer.get(key), None)
    na = int(same.get("n") or 0)
    nb = int(peer.get("n") or 0)
    if a is None and b is None:
        return None
    if a is not None and (nb < 5 or b is None):
        return a
    if b is not None and (na < 3 or a is None):
        return b
    if na >= 8:
        wa, wb = 0.70, 0.30
    elif na >= 4:
        wa, wb = 0.45, 0.55
    else:
        wa, wb = 0.25, 0.75
    return round(float(a * wa + b * wb), 1)


def analyze_fast_move_history(
    histories: dict[str, pd.DataFrame],
    target_ticker: str,
    current_signal: dict[str, Any],
    *,
    horizon_days: int = 5,
) -> dict[str, Any]:
    """Backtest similar 1-3 day fast moves for the stock and compact peers."""
    target = str(target_ticker or "").strip().upper()
    window = int(max(1, min(3, int(current_signal.get("holding_days") or 2))))
    current_pnl = max(0.0, float(_num(current_signal.get("pnl_pct"), 0.0) or 0.0))
    current_atr_units = max(0.0, float(_num(current_signal.get("atr_units"), 0.0) or 0.0))
    threshold_move = max(3.0, min(12.0, current_pnl * 0.65 if current_pnl > 0 else 4.0))
    threshold_atr = max(1.25, min(3.0, current_atr_units * 0.60 if current_atr_units > 0 else 1.5))

    same_rows: list[dict[str, Any]] = []
    peer_rows: list[dict[str, Any]] = []
    symbols_used: list[str] = []
    for symbol, frame in (histories or {}).items():
        sym = str(symbol or "").strip().upper()
        if not sym:
            continue
        rows = _event_rows(
            frame,
            symbol=sym,
            window=window,
            horizon=int(horizon_days),
            threshold_move_pct=threshold_move,
            threshold_atr=threshold_atr,
        )
        if rows:
            symbols_used.append(sym)
        if sym == target:
            same_rows.extend(rows)
        else:
            peer_rows.extend(rows)

    same = _summarize_events(same_rows)
    peer = _summarize_events(peer_rows)
    combined_n = int(same.get("n") or 0) + int(peer.get("n") or 0)
    combined = {
        "combined_giveback_rate": _weighted_metric(same, peer, "giveback_rate"),
        "combined_revisit_start_rate": _weighted_metric(same, peer, "revisit_start_rate"),
        "combined_followthrough_rate": _weighted_metric(same, peer, "followthrough_rate"),
        "combined_median_drawdown_pct": _weighted_metric(same, peer, "median_drawdown_pct"),
        "combined_median_followthrough_pct": _weighted_metric(same, peer, "median_followthrough_pct"),
    }
    return {
        "ticker": target,
        "updated_at": _now().isoformat(),
        "horizon_days": int(horizon_days),
        "event_window_days": window,
        "signal_holding_days": int(current_signal.get("holding_days") or window),
        "signal_pnl_pct": round(float(current_pnl), 2),
        "signal_atr_units": round(float(current_atr_units), 2),
        "threshold_move_pct": round(float(threshold_move), 2),
        "threshold_atr": round(float(threshold_atr), 2),
        "same_stock_n": int(same.get("n") or 0),
        "same_stock_giveback_rate": same.get("giveback_rate"),
        "same_stock_revisit_start_rate": same.get("revisit_start_rate"),
        "same_stock_followthrough_rate": same.get("followthrough_rate"),
        "same_stock_median_drawdown_pct": same.get("median_drawdown_pct"),
        "same_stock_median_followthrough_pct": same.get("median_followthrough_pct"),
        "peer_n": int(peer.get("n") or 0),
        "peer_giveback_rate": peer.get("giveback_rate"),
        "peer_revisit_start_rate": peer.get("revisit_start_rate"),
        "peer_followthrough_rate": peer.get("followthrough_rate"),
        "peer_median_drawdown_pct": peer.get("median_drawdown_pct"),
        "peer_median_followthrough_pct": peer.get("median_followthrough_pct"),
        "combined_n": combined_n,
        "sample_label": sample_label(combined_n),
        "symbols_used": symbols_used,
        "events": (same_rows[-20:] + peer_rows[-20:])[-30:],
        **combined,
    }
