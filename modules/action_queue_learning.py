"""v30.10 - Decision Action Queue Outcome Validation.

Provider-free, observational validation for the v30.9 Decision Action Queue.
The module stores one latest complete Atomic queue snapshot per watchlist and
Berlin trade date, then evaluates exact 1/3/5 business-day outcomes when a
matching future full-scan snapshot exists.

It never changes queue categories, scores, gates, stops, targets or orders.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any

import pandas as pd

_NAMESPACE = "decision_action_queue_learning_v3010"
_SCHEMA = "decision-action-queue-outcome-v30.10"
_MAX_DAYS_PER_WATCHLIST = 180
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
        raw = _storage.load_namespace(_NAMESPACE, default={}) or {}
    except Exception:
        raw = {}
    out = _empty_store()
    if isinstance(raw, dict) and isinstance(raw.get("watchlists"), dict):
        out["watchlists"] = raw.get("watchlists") or {}
        out["updated_at"] = raw.get("updated_at") or out["updated_at"]
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


def _text(value: Any, default="-") -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        text = ""
    if not text or text.lower() in {"none", "nan", "n/a", "na"}:
        return default
    return text


def _num(value: Any, default=None):
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        try:
            val = float(value)
            return val if math.isfinite(val) else default
        except Exception:
            return default
    text = str(value).strip().replace("\u2212", "-")
    if not text or text.lower() in {"none", "nan", "n/a", "na", "-"}:
        return default
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return default
    try:
        val = float(m.group(0))
        return val if math.isfinite(val) else default
    except Exception:
        return default


def _parse_time(value: Any, *, naive_is_utc=True):
    if value in (None, ""):
        return None
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
    try:
        if ts.tzinfo is None and naive_is_utc:
            ts = ts.tz_localize("UTC").tz_convert("Europe/Berlin")
        elif ts.tzinfo is not None:
            ts = ts.tz_convert("Europe/Berlin")
    except Exception:
        pass
    return ts


def _trade_date(scan_time: Any) -> tuple[str, str]:
    ts = _parse_time(scan_time, naive_is_utc=True)
    if ts is None:
        ts = _parse_time(_now_iso(), naive_is_utc=False)
    if ts is None:
        ts = pd.Timestamp.now(tz="Europe/Berlin")
    return ts.date().isoformat(), ts.isoformat()


def _clean_category(value: Any) -> str:
    text = _text(value, "")
    if "jetzt" in text.lower() and "prüf" in text.lower():
        return "🎯 Jetzt prüfen"
    if "block" in text.lower():
        return "⛔ Blockiert"
    if "beob" in text.lower():
        return "👀 Beobachten"
    return text or "👀 Beobachten"


def _clean_confidence(value: Any) -> str:
    text = _text(value, "Nicht bewertet")
    low = text.lower()
    if "hoch" in low:
        return "Hoch"
    if "mittel" in low:
        return "Mittel"
    if "niedrig" in low:
        return "Niedrig"
    return "Nicht bewertet"


def _row_record(row: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _text(row.get("Ticker"), "").upper()
    price = _num(row.get("Kurs"), None)
    if not ticker or price is None or price <= 0:
        return None
    return {
        "ticker": ticker,
        "name": _text(row.get("Name"), ticker),
        "price": round(float(price), 8),
        "category": _clean_category(row.get("Priorität")),
        "confidence": _clean_confidence(row.get("Decision-Confidence")),
        "live_score": _num(row.get("Live-Score"), None),
        "harvest": _num(row.get("Harvest"), None),
        "ampel": _text(row.get("Ampel"), "-"),
        "status": _text(row.get("Status"), "-"),
        "trade_state": _text(row.get("Trade-State"), "-"),
        "crv": _text(row.get("CRV"), "-"),
        "entry_distance": _text(row.get("Entry-Abstand"), "-"),
        "changed": _text(row.get("Änderung"), "-"),
        "reason": _text(row.get("Fokus-Grund"), "-"),
        "action": _text(row.get("Nächste Handlung"), "-"),
        "evidence": _text(row.get("Evidenz"), "-"),
        "limits": _text(row.get("Grenzen"), "-"),
        "freshness": _text(row.get("Aktualität"), "-"),
    }


def capture_queue_snapshot(
    watchlist_name: str,
    queue_df: pd.DataFrame | None,
    *,
    scan_id: str | None = None,
    scan_time: Any = None,
    atomic_complete: bool = True,
) -> dict[str, Any]:
    """Store one latest complete queue snapshot per Berlin date."""
    if not atomic_complete or not isinstance(queue_df, pd.DataFrame) or queue_df.empty:
        return {"saved": False, "reason": "no-complete-queue", "rows": 0}

    rows = []
    for _, series in queue_df.iterrows():
        rec = _row_record(series.to_dict())
        if rec is not None:
            rows.append(rec)
    if not rows:
        return {"saved": False, "reason": "no-priced-rows", "rows": 0}

    wl = _text(watchlist_name, "default")
    date_text, observed_at = _trade_date(scan_time)
    sid = _text(scan_id, "")
    if not sid:
        sid = f"{wl}|{date_text}|{observed_at}|{len(rows)}"

    store = _load_store()
    watchlists = dict(store.get("watchlists") or {})
    bucket = dict(watchlists.get(wl) or {})
    days = list(bucket.get("days") or [])

    for existing in days:
        if str(existing.get("scan_id") or "") == sid:
            return {"saved": False, "reason": "same-scan", "rows": len(rows), "trade_date": date_text}

    snapshot = {
        "trade_date": date_text,
        "observed_at": observed_at,
        "scan_id": sid,
        "rows": rows,
    }
    replaced = False
    for i, existing in enumerate(days):
        if str(existing.get("trade_date") or "") == date_text:
            days[i] = snapshot
            replaced = True
            break
    if not replaced:
        days.append(snapshot)
    days = sorted(days, key=lambda x: str(x.get("trade_date") or ""))[-_MAX_DAYS_PER_WATCHLIST:]

    bucket["days"] = days
    bucket["last_scan_id"] = sid
    bucket["last_trade_date"] = date_text
    bucket["last_observed_at"] = observed_at
    watchlists[wl] = bucket
    store["watchlists"] = watchlists
    ok = _save_store(store)
    return {
        "saved": bool(ok),
        "reason": "replaced-day" if replaced else "new-day",
        "rows": len(rows),
        "trade_date": date_text,
        "days": len(days),
    }


def _business_target(date_text: str, horizon: int) -> str:
    try:
        ts = pd.Timestamp(date_text).normalize() + pd.offsets.BDay(int(horizon))
        return ts.date().isoformat()
    except Exception:
        return ""


def _maps(days: list[dict[str, Any]]):
    by_date: dict[str, dict[str, dict[str, Any]]] = {}
    observed: dict[str, str] = {}
    for day in days:
        d = str(day.get("trade_date") or "")
        if not d:
            continue
        rows = {}
        for raw in list(day.get("rows") or []):
            ticker = _text(raw.get("ticker"), "").upper()
            if ticker:
                rows[ticker] = dict(raw)
        by_date[d] = rows
        observed[d] = str(day.get("observed_at") or "")
    return by_date, observed


def _path_prices(by_date: dict[str, dict[str, dict[str, Any]]], ticker: str, start_date: str, end_date: str):
    out = []
    for d in sorted(by_date):
        if d <= start_date or d > end_date:
            continue
        row = by_date.get(d, {}).get(ticker)
        price = _num((row or {}).get("price"), None)
        if price is not None and price > 0:
            out.append((d, float(price)))
    return out


def _sample_label(n: int) -> str:
    n = int(n or 0)
    if n < 5:
        return "Zu klein"
    if n < 15:
        return "Früh"
    if n < 30:
        return "Beobachtbar"
    return "Reifer"


def _eval_detail(days: list[dict[str, Any]]) -> pd.DataFrame:
    by_date, observed = _maps(days)
    rows = []
    for base_date in sorted(by_date):
        for ticker, base in by_date.get(base_date, {}).items():
            start_price = _num(base.get("price"), None)
            if start_price is None or start_price <= 0:
                continue
            for horizon in _HORIZONS:
                target_date = _business_target(base_date, horizon)
                future = by_date.get(target_date, {}).get(ticker)
                end_price = _num((future or {}).get("price"), None)
                if not target_date or end_price is None or end_price <= 0:
                    continue
                path = _path_prices(by_date, ticker, base_date, target_date)
                path_values = [p for _, p in path] or [float(end_price)]
                peak = max([float(start_price)] + path_values)
                trough = min([float(start_price)] + path_values)
                ret = (float(end_price) / float(start_price) - 1.0) * 100.0
                max_ret = (peak / float(start_price) - 1.0) * 100.0
                min_ret = (trough / float(start_price) - 1.0) * 100.0
                target_category = _clean_category((future or {}).get("category"))
                rows.append({
                    "Datum": base_date,
                    "Zeit": observed.get(base_date, ""),
                    "Ticker": ticker,
                    "Name": _text(base.get("name"), ticker),
                    "Horizont": f"{horizon}T",
                    "Zieltag": target_date,
                    "Priorität": _clean_category(base.get("category")),
                    "Decision-Confidence": _clean_confidence(base.get("confidence")),
                    "Live-Score": _num(base.get("live_score"), None),
                    "Startkurs": round(float(start_price), 4),
                    "Zielkurs": round(float(end_price), 4),
                    "Return %": round(ret, 2),
                    "Scan-Max %": round(max_ret, 2),
                    "Scan-Min %": round(min_ret, 2),
                    "Positiv": "Ja" if ret > 0 else "Nein",
                    "+2% im Pfad": "Ja" if max_ret >= 2.0 else "Nein",
                    "-2% im Pfad": "Ja" if min_ret <= -2.0 else "Nein",
                    "Folge-Priorität": target_category,
                    "Kategorie-Wechsel": f"{_clean_category(base.get('category'))} → {target_category}",
                    "Fokus-Grund": _text(base.get("reason"), "-"),
                    "Änderung": _text(base.get("changed"), "-"),
                    "Pfadpunkte": len(path_values),
                })
    return pd.DataFrame(rows)


def _pct(series: pd.Series, wanted: str) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float((series.astype(str) == wanted).mean() * 100.0)


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if any(c not in df.columns for c in group_cols):
        return pd.DataFrame()
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for key, part in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: str(val) for col, val in zip(group_cols, key)}
        n = len(part)
        returns = pd.to_numeric(part["Return %"], errors="coerce")
        maxes = pd.to_numeric(part["Scan-Max %"], errors="coerce")
        mins = pd.to_numeric(part["Scan-Min %"], errors="coerce")
        row.update({
            "Fälle": int(n),
            "Median Return %": round(float(returns.median()), 2),
            "Positiv %": round(_pct(part["Positiv"], "Ja"), 1),
            "+2% im Pfad %": round(_pct(part["+2% im Pfad"], "Ja"), 1),
            "-2% im Pfad %": round(_pct(part["-2% im Pfad"], "Ja"), 1),
            "Median Scan-Max %": round(float(maxes.median()), 2),
            "Median Scan-Min %": round(float(mins.median()), 2),
            "Stichprobe": _sample_label(n),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _category_order(value: str) -> int:
    return {"🎯 Jetzt prüfen": 0, "👀 Beobachten": 1, "⛔ Blockiert": 2}.get(str(value), 9)


def _confidence_order(value: str) -> int:
    return {"Hoch": 0, "Mittel": 1, "Niedrig": 2, "Nicht bewertet": 3}.get(str(value), 9)


def _insights(detail_3t: pd.DataFrame) -> list[str]:
    if not isinstance(detail_3t, pd.DataFrame) or detail_3t.empty:
        return ["Noch keine 3T-Fälle aus exakt passenden Folge-Vollscans verfügbar."]
    out = []
    groups = {}
    for cat in ("🎯 Jetzt prüfen", "👀 Beobachten", "⛔ Blockiert"):
        part = detail_3t[detail_3t["Priorität"].astype(str) == cat]
        if len(part) >= 5:
            groups[cat] = {
                "n": len(part),
                "median": float(pd.to_numeric(part["Return %"], errors="coerce").median()),
                "positive": _pct(part["Positiv"], "Ja"),
                "down2": _pct(part["-2% im Pfad"], "Ja"),
            }
    ready = groups.get("🎯 Jetzt prüfen")
    watch = groups.get("👀 Beobachten")
    blocked = groups.get("⛔ Blockiert")
    if ready and watch:
        diff = ready["median"] - watch["median"]
        if diff >= 0.5:
            out.append(f"'Jetzt prüfen' liegt nach 3T beim Median-Return aktuell {diff:+.2f} PP vor 'Beobachten' (n={ready['n']} vs. {watch['n']}).")
        elif diff <= -0.5:
            out.append(f"'Jetzt prüfen' liegt nach 3T aktuell {abs(diff):.2f} PP hinter 'Beobachten'. Das spricht noch nicht für eine saubere Priorisierung; weiter beobachten, keine automatische Änderung.")
        else:
            out.append("'Jetzt prüfen' und 'Beobachten' sind beim 3T-Median bisher kaum getrennt; die Queue braucht weitere Outcomes.")
    if ready and blocked:
        downside_diff = blocked["down2"] - ready["down2"]
        if downside_diff >= 15:
            out.append(f"Blockierte Werte zeigen bisher häufiger mindestens -2% im beobachteten 3T-Pfad ({downside_diff:+.0f} PP gegenüber 'Jetzt prüfen').")
        elif downside_diff <= -15:
            out.append("Blockierte Werte zeigen bisher nicht mehr 3T-Downside als 'Jetzt prüfen'; Gate-/Blockierungswirkung weiter validieren.")
    high = detail_3t[detail_3t["Decision-Confidence"].astype(str) == "Hoch"]
    lower = detail_3t[detail_3t["Decision-Confidence"].astype(str).isin(["Mittel", "Niedrig"])]
    if len(high) >= 5 and len(lower) >= 5:
        hp = _pct(high["Positiv"], "Ja")
        lp = _pct(lower["Positiv"], "Ja")
        out.append(f"Decision-Confidence 'Hoch' hat nach 3T aktuell {hp:.0f}% positive Fälle gegenüber {lp:.0f}% bei Mittel/Niedrig (rein beobachtend).")
    if not out:
        out.append("Noch zu wenige vergleichbare 3T-Fälle pro Queue-Kategorie für eine belastbare Trennschärfe-Aussage.")
    return out


def build_learning_package(watchlist_name: str) -> dict[str, Any]:
    wl = _text(watchlist_name, "default")
    store = _load_store()
    bucket = dict((store.get("watchlists") or {}).get(wl) or {})
    days = list(bucket.get("days") or [])
    detail = _eval_detail(days)
    detail_3t = detail[detail["Horizont"].astype(str) == "3T"].copy() if not detail.empty else pd.DataFrame()

    category_summary = _aggregate(detail, ["Horizont", "Priorität"])
    if not category_summary.empty:
        category_summary["__h"] = category_summary["Horizont"].map({"1T": 1, "3T": 3, "5T": 5}).fillna(99)
        category_summary["__c"] = category_summary["Priorität"].map(_category_order)
        category_summary = category_summary.sort_values(["__h", "__c"]).drop(columns=["__h", "__c"])

    confidence_summary = _aggregate(detail_3t, ["Decision-Confidence"])
    if not confidence_summary.empty:
        confidence_summary["__c"] = confidence_summary["Decision-Confidence"].map(_confidence_order)
        confidence_summary = confidence_summary.sort_values("__c").drop(columns=["__c"])

    transition_summary = pd.DataFrame()
    if not detail_3t.empty:
        transition_summary = (
            detail_3t.groupby(["Priorität", "Folge-Priorität"], dropna=False)
            .size().reset_index(name="Fälle")
        )
        transition_summary["__c1"] = transition_summary["Priorität"].map(_category_order)
        transition_summary["__c2"] = transition_summary["Folge-Priorität"].map(_category_order)
        transition_summary = transition_summary.sort_values(["__c1", "__c2"]).drop(columns=["__c1", "__c2"])

    summary = {
        "scan_days": len(days),
        "first_day": str(days[0].get("trade_date") or "") if days else "",
        "last_day": str(days[-1].get("trade_date") or "") if days else "",
        "evaluable_1t": int(len(detail[detail["Horizont"] == "1T"])) if not detail.empty else 0,
        "evaluable_3t": int(len(detail_3t)),
        "evaluable_5t": int(len(detail[detail["Horizont"] == "5T"])) if not detail.empty else 0,
        "ready_3t": int((detail_3t["Priorität"].astype(str) == "🎯 Jetzt prüfen").sum()) if not detail_3t.empty else 0,
        "watch_3t": int((detail_3t["Priorität"].astype(str) == "👀 Beobachten").sum()) if not detail_3t.empty else 0,
        "blocked_3t": int((detail_3t["Priorität"].astype(str) == "⛔ Blockiert").sum()) if not detail_3t.empty else 0,
        "sample_label": _sample_label(len(detail_3t)),
    }
    return {
        "summary": summary,
        "category_summary": category_summary,
        "confidence_summary": confidence_summary,
        "transition_summary": transition_summary,
        "detail": detail,
        "insights": _insights(detail_3t),
        "storage": {"namespace": _NAMESPACE, "schema": _SCHEMA, "max_days": _MAX_DAYS_PER_WATCHLIST},
    }
