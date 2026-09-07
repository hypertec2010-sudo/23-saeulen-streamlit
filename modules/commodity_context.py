"""v30.5 - Commodity / WTI context helpers.

Provider-neutral calculations for WTI-specific context. Network access is kept
outside this module so the normal Watchlist/Atomic path never creates hidden
provider requests.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _close_series(frame: pd.DataFrame | pd.Series | None) -> pd.Series:
    if frame is None:
        return pd.Series(dtype=float)
    if isinstance(frame, pd.Series):
        out = pd.to_numeric(frame, errors="coerce").dropna()
        return out.sort_index()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(dtype=float)
    try:
        if "Close" in frame.columns:
            out = frame["Close"]
        elif "Adj Close" in frame.columns:
            out = frame["Adj Close"]
        else:
            return pd.Series(dtype=float)
        if isinstance(out, pd.DataFrame):
            if out.shape[1] != 1:
                return pd.Series(dtype=float)
            out = out.iloc[:, 0]
        out = pd.to_numeric(out, errors="coerce").dropna()
        return out.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def _return_pct(series: pd.Series, periods: int) -> float | None:
    if not isinstance(series, pd.Series) or len(series.dropna()) <= periods:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    try:
        a = float(s.iloc[-periods - 1])
        b = float(s.iloc[-1])
        if not np.isfinite(a) or not np.isfinite(b) or a == 0:
            return None
        return (b / a - 1.0) * 100.0
    except Exception:
        return None


def _true_range_atr_pct(frame: pd.DataFrame | None, period: int = 14) -> float | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if not all(c in frame.columns for c in ("High", "Low", "Close")):
        return None
    try:
        high = pd.to_numeric(frame["High"], errors="coerce")
        low = pd.to_numeric(frame["Low"], errors="coerce")
        close = pd.to_numeric(frame["Close"], errors="coerce")
        prev = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=max(5, period // 2)).mean()
        last_close = _num(close.dropna().iloc[-1] if not close.dropna().empty else None)
        last_atr = _num(atr.dropna().iloc[-1] if not atr.dropna().empty else None)
        if last_close is None or last_close <= 0 or last_atr is None:
            return None
        return last_atr / last_close * 100.0
    except Exception:
        return None


def volatility_regime(atr_pct: float | None) -> str:
    if atr_pct is None:
        return "n/a"
    x = float(atr_pct)
    if x >= 6.0:
        return "hoch"
    if x >= 4.0:
        return "erhöht"
    if x >= 2.0:
        return "normal"
    return "niedrig"


def _trend_label(r5: float | None, r21: float | None, r63: float | None) -> str:
    vals = [v for v in (r5, r21, r63) if v is not None]
    if not vals:
        return "n/a"
    positives = sum(v > 0 for v in vals)
    negatives = sum(v < 0 for v in vals)
    if positives == len(vals):
        return "Breit aufwärts"
    if negatives == len(vals):
        return "Breit abwärts"
    if r5 is not None and r21 is not None and r5 > 0 and r21 < 0:
        return "Kurzfristige Erholung"
    if r5 is not None and r21 is not None and r5 < 0 and r21 > 0:
        return "Kurzfristiger Rücksetzer"
    return "Gemischt / wechselhaft"


def build_local_wti_context(history_df: pd.DataFrame | None, *, atr_pct: float | None = None) -> dict[str, Any]:
    close = _close_series(history_df)
    r5 = _return_pct(close, 5)
    r21 = _return_pct(close, 21)
    r63 = _return_pct(close, 63)
    atr = atr_pct if atr_pct is not None else _true_range_atr_pct(history_df)
    return {
        "wti_price": _num(close.iloc[-1] if not close.empty else None),
        "ret5": None if r5 is None else round(r5, 2),
        "ret21": None if r21 is None else round(r21, 2),
        "ret63": None if r63 is None else round(r63, 2),
        "trend_label": _trend_label(r5, r21, r63),
        "atr_pct": None if atr is None else round(float(atr), 2),
        "volatility_regime": volatility_regime(atr),
    }


def extract_close_frame(raw: pd.DataFrame | None, symbols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    symbols = [str(x).upper().strip() for x in symbols if str(x).strip()]
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = [str(x).upper() for x in raw.columns.get_level_values(0)]
            lvl1 = [str(x).upper() for x in raw.columns.get_level_values(1)]
            if "CLOSE" in lvl0:
                close = raw.xs(raw.columns.get_level_values(0)[lvl0.index("CLOSE")], axis=1, level=0)
            elif "CLOSE" in lvl1:
                close = raw.xs(raw.columns.get_level_values(1)[lvl1.index("CLOSE")], axis=1, level=1)
            else:
                return pd.DataFrame()
        else:
            if "Close" not in raw.columns or len(symbols) != 1:
                return pd.DataFrame()
            close = pd.DataFrame({symbols[0]: raw["Close"]})
        if isinstance(close, pd.Series):
            close = close.to_frame(name=symbols[0] if symbols else "Close")
        close.columns = [str(c).upper().strip() for c in close.columns]
        close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all").sort_index()
        return close
    except Exception:
        return pd.DataFrame()


def _aligned_returns(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    df = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1, join="inner").dropna()
    if df.shape[1] == 2:
        df.columns = ["a", "b"]
    return df


def build_external_wti_context(wti_history: pd.DataFrame | None, peer_close: pd.DataFrame | None) -> dict[str, Any]:
    """Combine WTI history with optional Brent/XLE/DXY close series."""
    wti = _close_series(wti_history)
    peer_close = peer_close.copy() if isinstance(peer_close, pd.DataFrame) else pd.DataFrame()
    if not peer_close.empty:
        peer_close.columns = [str(c).upper().strip() for c in peer_close.columns]

    out: dict[str, Any] = {
        "brent_price": None,
        "spread_usd": None,
        "spread_pct": None,
        "spread_change_21d": None,
        "xle_ret21": None,
        "xle_ret63": None,
        "wti_vs_xle_21": None,
        "wti_vs_xle_63": None,
        "energy_leadership": "n/a",
        "dxy_ret21": None,
        "wti_dxy_corr63": None,
        "dollar_effect": "n/a",
        "reference_date": None,
    }
    dates = []
    if not wti.empty:
        dates.append(wti.index[-1])

    # Brent-WTI spread (front-month Yahoo futures, therefore roll-sensitive).
    if "BZ=F" in peer_close.columns and not wti.empty:
        brent = pd.to_numeric(peer_close["BZ=F"], errors="coerce").dropna()
        aligned = _aligned_returns(wti, brent)
        if not aligned.empty:
            latest_wti = float(aligned["a"].iloc[-1])
            latest_brent = float(aligned["b"].iloc[-1])
            spread = latest_brent - latest_wti
            out["brent_price"] = round(latest_brent, 4)
            out["spread_usd"] = round(spread, 4)
            out["spread_pct"] = round((spread / latest_wti) * 100.0, 2) if latest_wti else None
            if len(aligned) > 21:
                old_spread = float(aligned["b"].iloc[-22] - aligned["a"].iloc[-22])
                out["spread_change_21d"] = round(spread - old_spread, 4)
            dates.append(aligned.index[-1])

    # XLE as liquid US energy-equity proxy, not as a substitute for crude itself.
    if "XLE" in peer_close.columns and not wti.empty:
        xle = pd.to_numeric(peer_close["XLE"], errors="coerce").dropna()
        aligned = _aligned_returns(wti, xle)
        if not aligned.empty:
            wti21 = _return_pct(aligned["a"], 21)
            wti63 = _return_pct(aligned["a"], 63)
            xle21 = _return_pct(aligned["b"], 21)
            xle63 = _return_pct(aligned["b"], 63)
            out["xle_ret21"] = None if xle21 is None else round(xle21, 2)
            out["xle_ret63"] = None if xle63 is None else round(xle63, 2)
            out["wti_vs_xle_21"] = None if wti21 is None or xle21 is None else round(wti21 - xle21, 2)
            out["wti_vs_xle_63"] = None if wti63 is None or xle63 is None else round(wti63 - xle63, 2)
            rel = out["wti_vs_xle_21"]
            if rel is not None and rel >= 4:
                out["energy_leadership"] = "WTI führt XLE deutlich"
            elif rel is not None and rel <= -4:
                out["energy_leadership"] = "XLE führt WTI deutlich"
            elif rel is not None:
                out["energy_leadership"] = "WTI und XLE ähnlich"
            dates.append(aligned.index[-1])

    # Dollar effect: context, not causal claim. DXY 21d direction plus 63d return correlation.
    if "DX-Y.NYB" in peer_close.columns and not wti.empty:
        dxy = pd.to_numeric(peer_close["DX-Y.NYB"], errors="coerce").dropna()
        aligned = _aligned_returns(wti, dxy)
        if not aligned.empty:
            dxy21 = _return_pct(aligned["b"], 21)
            out["dxy_ret21"] = None if dxy21 is None else round(dxy21, 2)
            ret = aligned.pct_change().dropna()
            if len(ret) >= 20:
                corr_window = ret.tail(min(63, len(ret)))
                corr = corr_window["a"].corr(corr_window["b"])
                out["wti_dxy_corr63"] = None if pd.isna(corr) else round(float(corr), 2)
            corr = out["wti_dxy_corr63"]
            if dxy21 is not None and corr is not None and corr <= -0.15:
                if dxy21 >= 1.0:
                    out["dollar_effect"] = "Dollar-Gegenwind für WTI"
                elif dxy21 <= -1.0:
                    out["dollar_effect"] = "Dollar-Rückenwind für WTI"
                else:
                    out["dollar_effect"] = "Dollar-Effekt aktuell neutral"
            elif dxy21 is not None:
                out["dollar_effect"] = "Dollar-Zusammenhang aktuell schwach/uneindeutig"
            dates.append(aligned.index[-1])

    if dates:
        try:
            out["reference_date"] = str(pd.Timestamp(max(dates)).date())
        except Exception:
            pass
    return out
