"""v30.12 - Calibration Stability & Experiment Tracker.

Provider-free history layer for the v30.11 Calibration Advisor. It stores one
latest advisor snapshot per watchlist and Berlin date and measures whether the
same shadow recommendation persists across multiple independent daily outcome
states.

The tracker never changes productive thresholds, queue categories, confidence
rules, gates, stops, targets, orders or the advisor itself.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

import pandas as pd

_NAMESPACE = "calibration_stability_v3012"
_SCHEMA = "calibration-stability-v30.12"
_MAX_DAYS_PER_WATCHLIST = 180
_STORAGE = None
_TIME_PROVIDER = lambda: datetime.now(timezone.utc)


def configure_context(*, storage=None, time_provider=None):
    global _STORAGE, _TIME_PROVIDER
    if storage is not None:
        _STORAGE = storage
    if callable(time_provider):
        _TIME_PROVIDER = time_provider


def _now():
    try:
        value = _TIME_PROVIDER()
        if isinstance(value, datetime):
            return value
    except Exception:
        pass
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _trade_date() -> str:
    try:
        return _now().date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def _empty_store() -> dict[str, Any]:
    return {"schema": _SCHEMA, "watchlists": {}, "updated_at": _now_iso()}


def _load_store() -> dict[str, Any]:
    if _STORAGE is None:
        return _empty_store()
    try:
        raw = _STORAGE.load_namespace(_NAMESPACE, default={}) or {}
    except Exception:
        raw = {}
    out = _empty_store()
    if isinstance(raw, dict) and isinstance(raw.get("watchlists"), dict):
        out["watchlists"] = raw.get("watchlists") or {}
        out["updated_at"] = raw.get("updated_at") or out["updated_at"]
    return out


def _save_store(payload: dict[str, Any]) -> bool:
    if _STORAGE is None:
        return False
    try:
        clean = dict(payload or {})
        clean["schema"] = _SCHEMA
        clean["updated_at"] = _now_iso()
        return bool(_STORAGE.save_namespace(_NAMESPACE, clean))
    except Exception:
        return False


def _text(value: Any, default="-") -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        text = ""
    return text if text and text.lower() not in {"none", "nan", "n/a", "na"} else default


def _stance(status: Any) -> str:
    text = _text(status, "").lower()
    if "stichprobe" in text or "aufbauen" in text:
        return "Stichprobe aufbauen"
    if "audit" in text or "prüfen" in text or "pruefen" in text:
        return "Shadow prüfen"
    if "✅" in str(status) or "halten" in text or "plausibel" in text or "bestätigt" in text or "bestaetigt" in text or "gestützt" in text or "gestuetzt" in text:
        return "Halten / bestätigt"
    if "weiter beobachten" in text or "beobachten" in text:
        return "Weiter beobachten"
    return _text(status, "Unklassifiziert")


def _sample_numbers(sample: Any) -> list[int]:
    text = _text(sample, "")
    return [int(x) for x in re.findall(r"(?<![\d.])\d+(?![\d.])", text)]


def _min_sample(sample: Any):
    nums = _sample_numbers(sample)
    return min(nums) if nums else None


def _row_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "area": _text(row.get("area"), ""),
        "status": _text(row.get("status"), ""),
        "sample": _text(row.get("sample"), ""),
        "statement": _text(row.get("statement"), ""),
        "advice": _text(row.get("advice"), ""),
        "evidence": _text(row.get("evidence"), ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _evidence_fingerprint(rows: list[dict[str, Any]], summary: dict[str, Any] | None = None) -> str:
    # Only evidence-bearing fields are hashed. Merely reopening the same advisor
    # on another calendar day must not count as an independent stability state.
    payload = {
        "rows": [
            {
                "area": _text(r.get("area"), ""),
                "status": _text(r.get("status"), ""),
                "sample": _text(r.get("sample"), ""),
                "statement": _text(r.get("statement"), ""),
                "advice": _text(r.get("advice"), ""),
                "evidence": _text(r.get("evidence"), ""),
            }
            for r in rows
        ],
        "summary": {
            k: (summary or {}).get(k)
            for k in ("queue_3t", "harvest_3t", "actionable_shadow_checks", "overall_maturity")
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def capture_advisor_snapshot(
    watchlist_name: str,
    overview: pd.DataFrame | None,
    *,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store one latest advisor state per Berlin date and watchlist."""
    if not isinstance(overview, pd.DataFrame) or overview.empty:
        return {"saved": False, "reason": "no-advisor-rows", "rows": 0}

    rows = []
    for _, series in overview.iterrows():
        row = series.to_dict()
        area = _text(row.get("Bereich"), "")
        if not area:
            continue
        status = _text(row.get("Status"), "-")
        sample = _text(row.get("Stichprobe"), "-")
        rec = {
            "area": area,
            "status": status,
            "stance": _stance(status),
            "sample": sample,
            "min_sample": _min_sample(sample),
            "statement": _text(row.get("Aussage"), "-"),
            "advice": _text(row.get("Shadow-Empfehlung"), "-"),
            "evidence": _text(row.get("Evidenz"), "-"),
        }
        rec["row_fingerprint"] = _row_fingerprint(rec)
        rows.append(rec)
    if not rows:
        return {"saved": False, "reason": "no-usable-rows", "rows": 0}

    wl = _text(watchlist_name, "default")
    day = _trade_date()
    fingerprint = _evidence_fingerprint(rows, summary)
    snapshot = {
        "trade_date": day,
        "observed_at": _now_iso(),
        "evidence_fingerprint": fingerprint,
        "rows": rows,
        "summary": dict(summary or {}),
    }

    store = _load_store()
    watchlists = dict(store.get("watchlists") or {})
    bucket = dict(watchlists.get(wl) or {})
    days = list(bucket.get("days") or [])

    # Independent-state guard: unchanged evidence on a later day is not a new
    # stability observation. Same-day rerenders may still replace the day's
    # state if the evidence actually changed.
    if days and str(days[-1].get("evidence_fingerprint") or "") == fingerprint:
        return {
            "saved": False, "reason": "same-evidence", "rows": len(rows),
            "days": len(days), "trade_date": day,
        }

    replaced = False
    for idx, existing in enumerate(days):
        if str(existing.get("trade_date") or "") == day:
            days[idx] = snapshot
            replaced = True
            break
    if not replaced:
        days.append(snapshot)
    days = sorted(days, key=lambda x: str(x.get("trade_date") or ""))[-_MAX_DAYS_PER_WATCHLIST:]
    bucket["days"] = days
    bucket["last_trade_date"] = day
    watchlists[wl] = bucket
    store["watchlists"] = watchlists
    ok = _save_store(store)
    return {
        "saved": bool(ok),
        "reason": "replaced-day" if replaced else "new-day",
        "rows": len(rows),
        "days": len(days),
        "trade_date": day,
    }


def _area_history(days: list[dict[str, Any]], area: str) -> list[dict[str, Any]]:
    out = []
    last_fp = None
    for day in sorted(days, key=lambda x: str(x.get("trade_date") or "")):
        for row in list(day.get("rows") or []):
            if _text(row.get("area"), "") == area:
                item = dict(row)
                fp = _text(item.get("row_fingerprint"), "") or _row_fingerprint(item)
                # Per-area guard: a change in another advisor area must not count
                # as a fresh independent state for this unchanged area.
                if fp == last_fp:
                    break
                item["trade_date"] = str(day.get("trade_date") or "")
                item["observed_at"] = str(day.get("observed_at") or "")
                out.append(item)
                last_fp = fp
                break
    return out


def _streak(history: list[dict[str, Any]], current_stance: str) -> int:
    count = 0
    for item in reversed(history):
        if _text(item.get("stance"), "") == current_stance:
            count += 1
        else:
            break
    return count


def _transition_count(history: list[dict[str, Any]], window: int = 8) -> int:
    part = history[-int(window):]
    stances = [_text(x.get("stance"), "") for x in part]
    return sum(1 for a, b in zip(stances, stances[1:]) if a != b)


def _stability_status(history: list[dict[str, Any]], strong_group_n: int):
    if not history:
        return "⏳ Keine Historie", False
    current = history[-1]
    stance = _text(current.get("stance"), "Unklassifiziert")
    n = len(history)
    streak = _streak(history, stance)
    window = history[-min(8, n):]
    ratio = sum(1 for x in window if _text(x.get("stance"), "") == stance) / max(1, len(window))
    transitions = _transition_count(history, window=8)
    min_sample = current.get("min_sample")
    try:
        sample_ok = min_sample is not None and int(min_sample) >= int(strong_group_n)
    except Exception:
        sample_ok = False

    if stance == "Stichprobe aufbauen":
        return "⏳ Daten sammeln", False
    if n < 3:
        return "⏳ Noch zu kurz", False
    if transitions >= 3 and n >= 5:
        return "⚪ Wechselhaft", False
    if streak >= 5 and ratio >= 0.80:
        candidate = bool(sample_ok and stance == "Shadow prüfen")
        return ("🟢 Stabil · manuell prüfbar" if candidate else "🟢 Stabil"), candidate
    if streak >= 3 and ratio >= 0.67:
        return "🟡 Vorläufig stabil", False
    if ratio < 0.60:
        return "⚪ Wechselhaft", False
    return "⚪ Beobachten", False


def build_stability_package(watchlist_name: str, *, strong_group_n: int = 30) -> dict[str, Any]:
    wl = _text(watchlist_name, "default")
    store = _load_store()
    bucket = dict((store.get("watchlists") or {}).get(wl) or {})
    days = list(bucket.get("days") or [])

    areas = []
    seen = set()
    for day in days:
        for row in list(day.get("rows") or []):
            area = _text(row.get("area"), "")
            if area and area not in seen:
                seen.add(area)
                areas.append(area)

    rows = []
    history_rows = []
    candidate_count = 0
    stable_count = 0
    volatile_count = 0
    for area in areas:
        hist = _area_history(days, area)
        if not hist:
            continue
        current = hist[-1]
        current_stance = _text(current.get("stance"), "-")
        streak = _streak(hist, current_stance)
        window = hist[-min(8, len(hist)):]
        ratio = sum(1 for x in window if _text(x.get("stance"), "") == current_stance) / max(1, len(window))
        transitions = _transition_count(hist, window=8)
        stability, candidate = _stability_status(hist, strong_group_n)
        if candidate:
            candidate_count += 1
        if stability.startswith("🟢"):
            stable_count += 1
        if "Wechselhaft" in stability:
            volatile_count += 1

        rows.append({
            "Bereich": area,
            "Aktuelle Empfehlung": _text(current.get("status"), "-"),
            "Stance": current_stance,
            "Stabilität": stability,
            "Gleiche Empfehlung in Folge": streak,
            "Trefferquote letzte 8": round(ratio * 100.0, 0),
            "Wechsel letzte 8": transitions,
            "Aktuelle Mindest-Stichprobe": current.get("min_sample"),
            "Erster Historientag": hist[0].get("trade_date"),
            "Letzter Historientag": hist[-1].get("trade_date"),
            "Manuell prüfbarer Kandidat": "Ja" if candidate else "Nein",
            "Shadow-Empfehlung": _text(current.get("advice"), "-"),
        })
        for item in hist:
            history_rows.append({
                "Datum": item.get("trade_date"),
                "Bereich": area,
                "Status": item.get("status"),
                "Stance": item.get("stance"),
                "Stichprobe": item.get("sample"),
                "Mindest-Stichprobe": item.get("min_sample"),
                "Shadow-Empfehlung": item.get("advice"),
                "Evidenz": item.get("evidence"),
            })

    table = pd.DataFrame(rows)
    history_df = pd.DataFrame(history_rows)
    return {
        "summary": {
            "history_days": len(days),
            "areas": len(rows),
            "stable": stable_count,
            "manual_candidates": candidate_count,
            "volatile": volatile_count,
            "strong_group_n": int(strong_group_n),
        },
        "stability": table,
        "history": history_df,
        "policy": {
            "mode": "Shadow only",
            "auto_apply": False,
            "manual_candidate_requires": f">=5 unabhängige Evidenzstände mit gleicher Empfehlung und Mindest-Stichprobe >= {int(strong_group_n)}",
        },
        "storage": {"namespace": _NAMESPACE, "schema": _SCHEMA, "max_days": _MAX_DAYS_PER_WATCHLIST},
    }
