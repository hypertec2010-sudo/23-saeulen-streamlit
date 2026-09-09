"""v30.6 Harvest Outcome & Learning Validation.

Observational, provider-free learning layer for the Short-Term Trader / Profit
Harvest engine. It stores one latest Atomic full-scan snapshot per watchlist and
Berlin trade date, then evaluates 1/3/5 business-day outcomes only when the
corresponding future full-scan date is actually available.

No score, threshold, stop, order or productive decision is modified here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any

import pandas as pd

_NAMESPACE = "harvest_outcome_learning_v306"
_SCHEMA = "harvest-outcome-v30.6"
_MAX_DAYS_PER_WATCHLIST = 140
_HORIZONS = (1, 3, 5)

_storage = None
_time_provider = lambda: datetime.now(timezone.utc)


def configure_context(*, storage=None, time_provider=None):
    global _storage, _time_provider
    if storage is not None:
        _storage = storage
    if callable(time_provider):
        _time_provider = time_provider


def _now_iso() -> str:
    try:
        return _time_provider().isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _empty_store() -> dict[str, Any]:
    return {"schema": _SCHEMA, "watchlists": {}, "updated_at": _now_iso()}


def _load_store() -> dict[str, Any]:
    if _storage is None:
        return _empty_store()
    try:
        payload = _storage.load_namespace(_NAMESPACE, default={}) or {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    out = _empty_store()
    if isinstance(payload.get("watchlists"), dict):
        out["watchlists"] = payload.get("watchlists") or {}
    if payload.get("updated_at"):
        out["updated_at"] = payload.get("updated_at")
    return out


def _save_store(payload: dict[str, Any]) -> bool:
    if _storage is None:
        return False
    try:
        clean = dict(payload or {})
        clean["schema"] = _SCHEMA
        clean["updated_at"] = _now_iso()
        return bool(_storage.save_namespace(_NAMESPACE, clean))
    except Exception:
        return False


def _num(value: Any, default=None):
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        try:
            f = float(value)
            return f if math.isfinite(f) else default
        except Exception:
            return default
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "n/a", "na", "-"}:
        return default
    text = text.replace("\u2212", "-")
    # German decimal comma is accepted when no decimal point is present.
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return default
    try:
        f = float(match.group(0))
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _text(value: Any, default="-") -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        text = ""
    return text if text and text.lower() not in {"none", "nan"} else default


def _ticker(value: Any) -> str:
    return _text(value, "").upper()


def _target_pct(value: Any):
    text = _text(value, "")
    if not text:
        return None
    # Main UI representation: "+4.5% @ 222.69".
    match = re.search(r"([+-]?\d+(?:[\.,]\d+)?)\s*%", text)
    if match:
        try:
            return abs(float(match.group(1).replace(",", ".")))
        except Exception:
            return None
    return None


def _target_price(value: Any):
    text = _text(value, "")
    match = re.search(r"@\s*([+-]?\d+(?:[\.,]\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except Exception:
        return None


def _parse_time(value: Any, *, naive_is_utc: bool = True):
    if value in (None, ""):
        return None
    # Visible Berlin timestamp format from v30.5b. Remove timezone abbreviation;
    # date itself is already Berlin-local and is sufficient for daily learning.
    text = str(value).strip()
    text = re.sub(r"\s+(MESZ|MEZ)$", "", text, flags=re.IGNORECASE)
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except Exception:
            pass
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None and naive_is_utc:
        try:
            ts = ts.tz_localize("UTC").tz_convert("Europe/Berlin")
        except Exception:
            pass
    elif ts.tzinfo is not None:
        try:
            ts = ts.tz_convert("Europe/Berlin")
        except Exception:
            pass
    return ts


def _scan_date(scan_time: Any, live_df: pd.DataFrame | None = None) -> tuple[str, str]:
    # Prefer the v30.5b visible Berlin scan timestamp if present.
    if isinstance(live_df, pd.DataFrame) and not live_df.empty and "Scan-Zeit" in live_df.columns:
        for raw in live_df["Scan-Zeit"].tolist():
            ts = _parse_time(raw, naive_is_utc=False)
            if ts is not None:
                return ts.date().isoformat(), ts.isoformat()
    ts = _parse_time(scan_time, naive_is_utc=True)
    if ts is None:
        ts = _parse_time(_now_iso(), naive_is_utc=False)
    if ts is None:
        ts = pd.Timestamp.now(tz="Europe/Berlin")
    return ts.date().isoformat(), ts.isoformat()


def _row_record(row: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(row.get("Ticker"))
    price = _num(row.get("Kurs"), None)
    if not ticker or price is None or price <= 0:
        return None
    target_text = _text(row.get("Trader-Ziel"), "n/a")
    return {
        "ticker": ticker,
        "name": _text(row.get("Name"), ticker),
        "price": round(float(price), 8),
        "harvest": _num(row.get("Harvest-Score"), None),
        "chop": _num(row.get("Chop-Risk"), None),
        "scan_chop": _num(row.get("Scan-Chop"), None),
        "trader_mode": _text(row.get("Trader-Modus"), "-"),
        "trader_target": target_text,
        "target_pct": _target_pct(target_text),
        "target_price": _target_price(target_text),
        "live_score": _num(row.get("Live-Score"), None),
        "ampel": _text(row.get("Ampel"), "-"),
        "rs_dynamics": _text(row.get("RS-Dynamik"), "-"),
        "relative_strength": _text(row.get("Relative Stärke"), "-"),
        "volatility": _text(row.get("Volatilitätsregime"), _text(row.get("Volatilität"), "-")),
        "market_regime": _text(row.get("Marktregime"), "-"),
    }


def capture_scan(
    watchlist_name: str,
    live_df: pd.DataFrame | None,
    *,
    scan_id: str | None = None,
    scan_time: Any = None,
    atomic_complete: bool = True,
) -> dict[str, Any]:
    """Persist one latest complete scan per Berlin date.

    Re-renders of the same run are ignored. A later full scan on the same Berlin
    date replaces that day's snapshot, so calibration does not overweight days
    with many Streamlit reruns or manual refreshes.
    """
    if not atomic_complete or not isinstance(live_df, pd.DataFrame) or live_df.empty:
        return {"saved": False, "reason": "no-complete-frame", "rows": 0}

    rows = []
    for _, series in live_df.iterrows():
        rec = _row_record(series.to_dict())
        if rec is not None:
            rows.append(rec)
    if not rows:
        return {"saved": False, "reason": "no-priced-rows", "rows": 0}

    wl = _text(watchlist_name, "default")
    trade_date, observed_at = _scan_date(scan_time, live_df)
    sid = _text(scan_id, "")
    if not sid:
        sid = f"{wl}|{trade_date}|{observed_at}|{len(rows)}"

    store = _load_store()
    watchlists = dict(store.get("watchlists") or {})
    bucket = dict(watchlists.get(wl) or {})
    days = list(bucket.get("days") or [])

    for existing in days:
        if str(existing.get("scan_id") or "") == sid:
            return {"saved": False, "reason": "same-scan", "rows": len(rows), "trade_date": trade_date}

    snapshot = {
        "trade_date": trade_date,
        "observed_at": observed_at,
        "scan_id": sid,
        "rows": rows,
    }
    replaced = False
    for i, existing in enumerate(days):
        if str(existing.get("trade_date") or "") == trade_date:
            days[i] = snapshot
            replaced = True
            break
    if not replaced:
        days.append(snapshot)

    days = sorted(days, key=lambda x: str(x.get("trade_date") or ""))[-_MAX_DAYS_PER_WATCHLIST:]
    bucket["days"] = days
    bucket["last_scan_id"] = sid
    bucket["last_trade_date"] = trade_date
    bucket["last_observed_at"] = observed_at
    watchlists[wl] = bucket
    store["watchlists"] = watchlists
    ok = _save_store(store)
    return {
        "saved": bool(ok),
        "reason": "replaced-day" if replaced else "new-day",
        "rows": len(rows),
        "trade_date": trade_date,
        "days": len(days),
    }


def _business_target(date_text: str, horizon: int) -> str:
    try:
        ts = pd.Timestamp(date_text).normalize() + pd.offsets.BDay(int(horizon))
        return ts.date().isoformat()
    except Exception:
        return ""


def _sample_label(n: int) -> str:
    n = int(n or 0)
    if n < 5:
        return "Zu klein"
    if n < 15:
        return "Früh"
    if n < 30:
        return "Beobachtbar"
    return "Reifer"


def _harvest_band(value: Any) -> str:
    n = _num(value, None)
    if n is None:
        return "n/a"
    if n < 45:
        return "0-44 · Trendpfad"
    if n < 60:
        return "45-59 · Hybrid/Vorwarnung"
    if n < 75:
        return "60-74 · Gelb"
    return "75-100 · Orange"


def _chop_band(value: Any) -> str:
    n = _num(value, None)
    if n is None:
        return "n/a"
    if n < 40:
        return "0-39 · ruhig"
    if n < 60:
        return "40-59 · gemischt"
    if n < 75:
        return "60-74 · hoch"
    return "75-100 · sehr hoch"


def _classification(final_return: float, peak_return: float, giveback_peak: float, target_reached: bool) -> str:
    # Validation heuristic only. It never changes the production score.
    if target_reached and giveback_peak >= 1.0:
        return "Teilgewinn am Trader-Ziel bestätigt"
    if giveback_peak >= 1.5 or final_return <= -1.0:
        return "Teilgewinn eher sinnvoll"
    if final_return >= 2.0 and giveback_peak < 1.0:
        return "Laufenlassen eher besser"
    return "Gemischt / kein klarer Vorteil"


def _daily_maps(days: list[dict[str, Any]]):
    by_date: dict[str, dict[str, dict[str, Any]]] = {}
    observed_at: dict[str, str] = {}
    for day in days:
        date = str(day.get("trade_date") or "")
        if not date:
            continue
        rows = {}
        for row in list(day.get("rows") or []):
            tk = _ticker(row.get("ticker"))
            if tk:
                rows[tk] = dict(row)
        by_date[date] = rows
        observed_at[date] = str(day.get("observed_at") or "")
    return by_date, observed_at


def _path_prices(by_date: dict[str, dict[str, dict[str, Any]]], ticker: str, start_date: str, end_date: str):
    out = []
    for date in sorted(by_date):
        if date <= start_date or date > end_date:
            continue
        row = by_date.get(date, {}).get(ticker)
        price = _num((row or {}).get("price"), None)
        if price is not None and price > 0:
            out.append((date, float(price)))
    return out


def _eval_one(
    *,
    ticker: str,
    name: str,
    base_date: str,
    base_time: str,
    base_price: float,
    harvest: Any,
    chop: Any,
    scan_chop: Any,
    target_pct: Any,
    target_text: str,
    trader_mode: str,
    horizon: int,
    by_date: dict[str, dict[str, dict[str, Any]]],
    source: str,
    extra: dict[str, Any] | None = None,
):
    target_date = _business_target(base_date, horizon)
    future = by_date.get(target_date, {}).get(ticker)
    final_price = _num((future or {}).get("price"), None)
    if not target_date or final_price is None or final_price <= 0 or base_price <= 0:
        return None

    path = _path_prices(by_date, ticker, base_date, target_date)
    path_values = [p for _, p in path]
    if not path_values:
        path_values = [float(final_price)]
    peak_price = max([float(base_price)] + path_values)
    trough_price = min([float(base_price)] + path_values)
    final_return = (float(final_price) / float(base_price) - 1.0) * 100.0
    peak_return = (peak_price / float(base_price) - 1.0) * 100.0
    trough_return = (trough_price / float(base_price) - 1.0) * 100.0
    giveback_peak = max(0.0, (peak_price - float(final_price)) / peak_price * 100.0) if peak_price > 0 else 0.0
    t_pct = _num(target_pct, None)
    target_reached = bool(t_pct is not None and peak_return >= float(t_pct))
    verdict = _classification(final_return, peak_return, giveback_peak, target_reached)

    out = {
        "Quelle": source,
        "Datum": base_date,
        "Zeit": base_time,
        "Ticker": ticker,
        "Name": name,
        "Horizont": f"{int(horizon)}T",
        "Zieltag": target_date,
        "Startkurs": round(float(base_price), 4),
        "Zielkurs": round(float(final_price), 4),
        "Return %": round(final_return, 2),
        "Scan-Max %": round(peak_return, 2),
        "Scan-Min %": round(trough_return, 2),
        "Giveback vom Scan-Peak %": round(giveback_peak, 2),
        "Harvest Score": None if _num(harvest, None) is None else round(float(_num(harvest)), 1),
        "Harvest-Band": _harvest_band(harvest),
        "Chop Risk": None if _num(chop, None) is None else round(float(_num(chop)), 1),
        "Chop-Band": _chop_band(chop),
        "Scan-Chop": None if _num(scan_chop, None) is None else round(float(_num(scan_chop)), 1),
        "Trader-Ziel": target_text,
        "Trader-Ziel %": None if t_pct is None else round(float(t_pct), 2),
        "Trader-Ziel erreicht": "Ja" if target_reached else "Nein",
        "Trader-Modus": trader_mode,
        "Bewertung": verdict,
        "Pfadpunkte": int(len(path_values)),
    }
    if isinstance(extra, dict):
        out.update(extra)
    return out


def _build_scan_detail(days: list[dict[str, Any]]) -> pd.DataFrame:
    by_date, observed = _daily_maps(days)
    rows = []
    for base_date in sorted(by_date):
        for ticker, base in by_date.get(base_date, {}).items():
            price = _num(base.get("price"), None)
            harvest = _num(base.get("harvest"), None)
            if price is None or price <= 0 or harvest is None:
                continue
            for horizon in _HORIZONS:
                result = _eval_one(
                    ticker=ticker,
                    name=_text(base.get("name"), ticker),
                    base_date=base_date,
                    base_time=observed.get(base_date, ""),
                    base_price=float(price),
                    harvest=harvest,
                    chop=base.get("chop"),
                    scan_chop=base.get("scan_chop"),
                    target_pct=base.get("target_pct"),
                    target_text=_text(base.get("trader_target"), "n/a"),
                    trader_mode=_text(base.get("trader_mode"), "-"),
                    horizon=horizon,
                    by_date=by_date,
                    source="Atomic Vollscan",
                    extra={
                        "Live Score": base.get("live_score"),
                        "Live-Ampel": base.get("ampel"),
                        "RS-Dynamik": base.get("rs_dynamics"),
                        "Marktregime": base.get("market_regime"),
                        "Volatilitätsregime": base.get("volatility"),
                    },
                )
                if result is not None:
                    rows.append(result)
    return pd.DataFrame(rows)


def _event_time(row: dict[str, Any]):
    raw = row.get("Zeit") or row.get("time") or row.get("Timestamp")
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except Exception:
            pass
    try:
        return pd.Timestamp(raw)
    except Exception:
        return None


def _event_target_pct(row: dict[str, Any], base_price: float):
    # v30.4 position events may contain either a percent or a target price,
    # depending on runtime generation. Prefer explicit percent/text first.
    for key in ("Trader-Ziel %", "Target %", "target_pct"):
        val = _num(row.get(key), None)
        if val is not None and 0 < val < 30:
            return float(val)
    text = row.get("Trader-Ziel") or row.get("Trader Target") or row.get("target")
    pct = _target_pct(text)
    if pct is not None:
        return pct
    val = _num(text, None)
    if val is not None and base_price > 0:
        if 0 < val < 30:
            return float(val)
        if val > base_price:
            return (float(val) / base_price - 1.0) * 100.0
    return None


def _build_position_event_detail(event_df: pd.DataFrame | None, days: list[dict[str, Any]]) -> pd.DataFrame:
    if not isinstance(event_df, pd.DataFrame) or event_df.empty:
        return pd.DataFrame()
    if "Ereignis" not in event_df.columns:
        return pd.DataFrame()
    work = event_df[event_df["Ereignis"].astype(str).str.strip() == "Short-Term Profit Harvest"].copy()
    if work.empty:
        return pd.DataFrame()
    by_date, _ = _daily_maps(days)
    rows = []
    # Event log is usually newest first. Evaluation order is irrelevant.
    for _, series in work.iterrows():
        event = series.to_dict()
        ts = _event_time(event)
        if ts is None:
            continue
        base_date = ts.date().isoformat()
        ticker = _ticker(event.get("Ticker"))
        base_price = _num(event.get("Kurs"), None)
        if not ticker or base_price is None or base_price <= 0:
            continue
        target_pct = _event_target_pct(event, float(base_price))
        target_text = _text(event.get("Trader-Ziel"), "n/a")
        harvest = event.get("Harvest Score")
        chop = event.get("Chop Risk")
        for horizon in _HORIZONS:
            result = _eval_one(
                ticker=ticker,
                name=_text(event.get("Name"), ticker),
                base_date=base_date,
                base_time=str(event.get("Zeit") or ""),
                base_price=float(base_price),
                harvest=harvest,
                chop=chop,
                scan_chop=event.get("Scan-Chop"),
                target_pct=target_pct,
                target_text=target_text,
                trader_mode=_text(event.get("Status"), _text(event.get("Trader-Modus"), "-")),
                horizon=horizon,
                by_date=by_date,
                source="Positions-Harvest-Event",
                extra={
                    "P/L % beim Hinweis": _num(event.get("P/L %"), None),
                    "Profit Velocity": _num(event.get("Profit Velocity"), None),
                    "Exhaustion Risk": _num(event.get("Exhaustion Risk"), None),
                    "Historical Giveback Risk": _num(event.get("Giveback Risk"), None),
                    "Haltedauer": _num(event.get("Haltedauer"), None),
                    "Teilgewinn-Idee %": _num(event.get("Teilgewinn"), _num(event.get("Partial %"), None)),
                },
            )
            if result is not None:
                rows.append(result)
    return pd.DataFrame(rows)


def _pct_true(series: pd.Series, value: str) -> float | None:
    if series is None or len(series) == 0:
        return None
    return float((series.astype(str) == value).mean() * 100.0)


def _pct_contains(series: pd.Series, needle: str) -> float | None:
    if series is None or len(series) == 0:
        return None
    return float(series.astype(str).str.contains(needle, case=False, na=False).mean() * 100.0)


def _aggregate_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for group, part in df.groupby(group_col, dropna=False):
        n = int(len(part))
        rows.append({
            group_col: str(group),
            "Fälle": n,
            "Median Return %": round(float(pd.to_numeric(part["Return %"], errors="coerce").median()), 2),
            "Median Scan-Max %": round(float(pd.to_numeric(part["Scan-Max %"], errors="coerce").median()), 2),
            "Median Giveback %": round(float(pd.to_numeric(part["Giveback vom Scan-Peak %"], errors="coerce").median()), 2),
            "Trader-Ziel erreicht %": round(float(_pct_true(part["Trader-Ziel erreicht"], "Ja") or 0.0), 1),
            "Teilgewinn eher bestätigt %": round(float(_pct_contains(part["Bewertung"], "Teilgewinn") or 0.0), 1),
            "Laufenlassen eher besser %": round(float(_pct_contains(part["Bewertung"], "Laufenlassen") or 0.0), 1),
            "Stichprobe": _sample_label(n),
        })
    return pd.DataFrame(rows)


def _horizon_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return pd.DataFrame()
    rows = []
    for horizon in ("1T", "3T", "5T"):
        part = detail[detail["Horizont"].astype(str) == horizon]
        if part.empty:
            continue
        alerts = part[pd.to_numeric(part["Harvest Score"], errors="coerce") >= 60]
        rows.append({
            "Horizont": horizon,
            "Alle Fälle": int(len(part)),
            "Harvest ≥60": int(len(alerts)),
            "Median Return ≥60 %": None if alerts.empty else round(float(pd.to_numeric(alerts["Return %"], errors="coerce").median()), 2),
            "Median Giveback ≥60 %": None if alerts.empty else round(float(pd.to_numeric(alerts["Giveback vom Scan-Peak %"], errors="coerce").median()), 2),
            "Trader-Ziel erreicht ≥60 %": None if alerts.empty else round(float(_pct_true(alerts["Trader-Ziel erreicht"], "Ja") or 0.0), 1),
            "Teilgewinn bestätigt ≥60 %": None if alerts.empty else round(float(_pct_contains(alerts["Bewertung"], "Teilgewinn") or 0.0), 1),
            "Laufenlassen besser ≥60 %": None if alerts.empty else round(float(_pct_contains(alerts["Bewertung"], "Laufenlassen") or 0.0), 1),
            "Stichprobe": _sample_label(len(alerts)),
        })
    return pd.DataFrame(rows)


def _insights(detail_3t: pd.DataFrame) -> list[str]:
    if not isinstance(detail_3t, pd.DataFrame) or detail_3t.empty:
        return []
    out = []
    scores = pd.to_numeric(detail_3t.get("Harvest Score"), errors="coerce")
    high = detail_3t[scores >= 60]
    low = detail_3t[scores < 60]
    if len(high) >= 5:
        gb = float(pd.to_numeric(high["Giveback vom Scan-Peak %"], errors="coerce").median())
        partial = float(_pct_contains(high["Bewertung"], "Teilgewinn") or 0.0)
        run = float(_pct_contains(high["Bewertung"], "Laufenlassen") or 0.0)
        out.append(
            f"Harvest ≥60: n={len(high)}, Median-Giveback nach 3T {gb:.2f}%, "
            f"Teilgewinn eher bestätigt {partial:.0f}%, Laufenlassen eher besser {run:.0f}%."
        )
    if len(high) >= 5 and len(low) >= 5:
        high_gb = float(pd.to_numeric(high["Giveback vom Scan-Peak %"], errors="coerce").median())
        low_gb = float(pd.to_numeric(low["Giveback vom Scan-Peak %"], errors="coerce").median())
        diff = high_gb - low_gb
        if diff >= 0.75:
            out.append(f"Die aktuelle Harvest-Trennung zeigt nach 3T mehr Giveback im ≥60-Band (+{diff:.2f} PP Median gegenüber <60).")
        elif diff <= -0.75:
            out.append(f"Das ≥60-Band zeigt bisher nicht mehr, sondern {abs(diff):.2f} PP weniger Median-Giveback als <60; weiter Daten sammeln, noch keine Schwelle ändern.")
        else:
            out.append("Zwischen Harvest <60 und ≥60 ist der 3T-Giveback bisher noch nicht klar getrennt; Stichprobe weiter aufbauen.")
    if not out:
        out.append("Noch zu wenige 3T-Fälle für einen belastbaren Kalibrierungshinweis. Die Engine sammelt weiter rein beobachtend.")
    return out


def build_learning_package(watchlist_name: str, event_df: pd.DataFrame | None = None) -> dict[str, Any]:
    wl = _text(watchlist_name, "default")
    store = _load_store()
    bucket = dict((store.get("watchlists") or {}).get(wl) or {})
    days = list(bucket.get("days") or [])
    scan_detail = _build_scan_detail(days)
    event_detail = _build_position_event_detail(event_df, days)

    detail_3t = scan_detail[scan_detail["Horizont"].astype(str) == "3T"].copy() if not scan_detail.empty else pd.DataFrame()
    alerts_3t = detail_3t[pd.to_numeric(detail_3t.get("Harvest Score"), errors="coerce") >= 60].copy() if not detail_3t.empty else pd.DataFrame()

    summary = {
        "scan_days": int(len(days)),
        "first_day": str(days[0].get("trade_date") or "") if days else "",
        "last_day": str(days[-1].get("trade_date") or "") if days else "",
        "evaluable_1t": int(len(scan_detail[scan_detail["Horizont"] == "1T"])) if not scan_detail.empty else 0,
        "evaluable_3t": int(len(detail_3t)),
        "evaluable_5t": int(len(scan_detail[scan_detail["Horizont"] == "5T"])) if not scan_detail.empty else 0,
        "alerts_3t": int(len(alerts_3t)),
        "target_hit_3t_pct": None if alerts_3t.empty else round(float(_pct_true(alerts_3t["Trader-Ziel erreicht"], "Ja") or 0.0), 1),
        "partial_confirmed_3t_pct": None if alerts_3t.empty else round(float(_pct_contains(alerts_3t["Bewertung"], "Teilgewinn") or 0.0), 1),
        "run_better_3t_pct": None if alerts_3t.empty else round(float(_pct_contains(alerts_3t["Bewertung"], "Laufenlassen") or 0.0), 1),
        "position_events_evaluable_3t": int(len(event_detail[event_detail["Horizont"] == "3T"])) if not event_detail.empty else 0,
        "sample_label": _sample_label(len(alerts_3t)),
    }

    return {
        "summary": summary,
        "horizon_summary": _horizon_summary(scan_detail),
        "harvest_band_summary": _aggregate_group(detail_3t, "Harvest-Band"),
        "chop_band_summary": _aggregate_group(detail_3t, "Chop-Band"),
        "detail": scan_detail,
        "position_event_detail": event_detail,
        "position_event_summary": _aggregate_group(
            event_detail[event_detail["Horizont"].astype(str) == "3T"] if not event_detail.empty else pd.DataFrame(),
            "Harvest-Band",
        ),
        "insights": _insights(detail_3t),
        "storage": {
            "namespace": _NAMESPACE,
            "schema": _SCHEMA,
            "max_days": _MAX_DAYS_PER_WATCHLIST,
        },
    }
