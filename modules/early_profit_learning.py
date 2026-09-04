"""v30.3 - Early Profit Learning & Calibration.

Read-only learning layer for v30.2 Early Profit Protection events.
It matches defensive Early-Profit events to safely timestamped closed trades and
checks whether the trade subsequently gave back R or continued to improve.

Important: this is observational. It does not change orders, stops, live/shadow
scores, thresholds, or v30.2 recommendations.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


_EVENT_NAMES = {"early profit protection", "early-profit protection", "early profit"}


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


def _ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        # ISO strings are unambiguous and should be parsed without dayfirst.
        raw = str(value).strip()
        if not raw or raw.lower() in {"nan", "none", "n/a", "na", "-"}:
            return None
        if "T" in raw or (len(raw) >= 10 and raw[4:5] == "-"):
            out = pd.to_datetime(raw, errors="coerce")
        else:
            out = pd.to_datetime(raw, dayfirst=True, errors="coerce")
        if pd.isna(out):
            return None
        if getattr(out, "tzinfo", None) is not None:
            # Event-log timestamps and journal timestamps are compared in local wall
            # time.  Drop timezone information without shifting the clock first.
            try:
                out = out.tz_localize(None)
            except Exception:
                try:
                    out = out.tz_convert(None)
                except Exception:
                    pass
        return pd.Timestamp(out)
    except Exception:
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


def _first(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row:
            val = row.get(key)
            if val is None:
                continue
            try:
                if isinstance(val, float) and np.isnan(val):
                    continue
            except Exception:
                pass
            if isinstance(val, str) and val.strip().lower() in {"", "nan", "none", "n/a", "na", "-"}:
                continue
            return val
    return default


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    direct = _first(row, ["Payload", "payload", "Daten", "data"], None)
    if isinstance(direct, dict):
        return dict(direct)
    if isinstance(direct, str):
        text = direct.strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return {}


def _event_value(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    val = _first(row, keys, None)
    if val is not None:
        return val
    payload = _payload(row)
    return _first(payload, keys, default)


def _normalize_events(events_df: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, src in events_df.iterrows():
        raw = src.to_dict()
        event_type = _text(_event_value(raw, ["Ereignis", "Event", "event_type", "Typ", "Type"], ""), "")
        event_l = event_type.lower().strip()
        if event_l not in _EVENT_NAMES and "early profit" not in event_l:
            continue
        ticker = _text(_event_value(raw, ["Ticker", "ticker", "Symbol"], ""), "").upper()
        when = _ts(_event_value(raw, ["Zeit", "Timestamp", "timestamp", "created_at", "Datum", "Date"], None))
        if not ticker or when is None:
            continue
        rows.append({
            "Ticker": ticker,
            "Event-Zeit": when,
            "Aktion": _text(_event_value(raw, ["Status", "Aktion", "Action", "action"], "-")),
            "Event-Kurs": _num(_event_value(raw, ["Kurs", "Price", "price"], None), None),
            "Event-R": _num(_event_value(raw, ["R", "R-Multiple", "R Multiple", "r_multiple"], None), None),
            "Event-P/L %": _num(_event_value(raw, ["P/L %", "P/L", "pnl_pct", "PnL %"], None), None),
            "Profit Velocity": _num(_event_value(raw, ["Profit Velocity", "profit_velocity"], None), None),
            "Exhaustion Risk": _num(_event_value(raw, ["Exhaustion Risk", "exhaustion_risk"], None), None),
            "Giveback Risk": _num(_event_value(raw, ["Giveback Risk", "giveback_risk"], None), None),
            "Details": _text(_event_value(raw, ["Details", "details", "Warum", "why_text"], "-")),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Ticker", "Event-Zeit"]).reset_index(drop=True)
    return out


def _normalize_trades(trades_df: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(trades_df, pd.DataFrame) or trades_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, src in trades_df.iterrows():
        raw = src.to_dict()
        ticker = _text(_first(raw, ["Ticker", "ticker", "Symbol"], ""), "").upper()
        entry_time = _ts(_first(raw, ["Entry-Zeit", "Entry Zeit", "Entry_Time", "opened_at_iso", "Eröffnet"], None))
        exit_time = _ts(_first(raw, ["Exit-Zeit", "Exit Zeit", "Exit_Time", "closed_at_iso", "Geschlossen"], None))
        if not ticker or entry_time is None or exit_time is None or exit_time < entry_time:
            continue
        rows.append({
            "Ticker": ticker,
            "Name": _text(_first(raw, ["Name", "Unternehmen"], ticker), ticker),
            "Entry-Zeit": entry_time,
            "Exit-Zeit": exit_time,
            "Final R": _num(_first(raw, ["Gesamt R", "Gesamt_R", "R", "Total R"], None), None),
            "Final P/L": _num(_first(raw, ["Gesamt P/L", "Gesamt_P_L", "P/L", "Total P/L"], None), None),
            "Kapitalrendite %": _num(_first(raw, ["Kapitalrendite %", "Return %", "Kapitalrendite"], None), None),
            "Haltedauer Tage": _num(_first(raw, ["Haltedauer Tage", "Haltedauer", "Hold Days"], None), None),
            "Outcome": _text(_first(raw, ["Outcome", "Ergebnis"], "-")),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Ticker", "Entry-Zeit", "Exit-Zeit"]).reset_index(drop=True)
    return out


def _severity(action: str) -> int:
    text = str(action or "").lower()
    if any(x in text for x in ["exit", "gesamtreduk", "deutlich", "50%", "50 %"]):
        return 3
    if any(x in text for x in ["teilgewinn", "25-50", "25–50", "reduz"]):
        return 2
    if any(x in text for x in ["gewinnschutz", "trail", "absicher"]):
        return 1
    return 0


def _band(value: float | None, *, kind: str) -> str:
    if value is None:
        return "n/a"
    v = float(value)
    if kind == "velocity":
        if v >= 85:
            return "85-100"
        if v >= 70:
            return "70-84"
        if v >= 55:
            return "55-69"
        return "<55"
    if kind == "risk":
        if v >= 75:
            return "75-100"
        if v >= 60:
            return "60-74"
        if v >= 45:
            return "45-59"
        return "<45"
    return "n/a"


def _outcome(delta_r: float | None) -> tuple[str, str]:
    if delta_r is None:
        return "⚪", "Nicht auswertbar"
    if delta_r <= -0.75:
        return "🟢", "Gewinnschutz stark bestätigt"
    if delta_r <= -0.25:
        return "🟢", "Gewinnschutz bestätigt"
    if delta_r >= 0.75:
        return "🔴", "Laufenlassen klar besser"
    if delta_r >= 0.25:
        return "🟠", "Laufenlassen besser"
    return "🟡", "Nahezu neutral"


def _summary_table(detail: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if detail.empty or group_col not in detail.columns:
        return pd.DataFrame()
    work = detail[detail["Auswertbar"] == True].copy()  # noqa: E712
    if work.empty:
        return pd.DataFrame()
    rows = []
    for key, grp in work.groupby(group_col, dropna=False):
        delta = pd.to_numeric(grp["ΔR nach Warnung"], errors="coerce").dropna()
        if delta.empty:
            continue
        protect = float((delta <= -0.25).mean() * 100.0)
        hold = float((delta >= 0.25).mean() * 100.0)
        rows.append({
            group_col: key,
            "n": int(len(delta)),
            "Stichprobe": sample_label(len(delta)),
            "Gewinnschutz bestätigt %": round(protect, 1),
            "Laufenlassen besser %": round(hold, 1),
            "Median ΔR": round(float(delta.median()), 2),
            "Ø ΔR": round(float(delta.mean()), 2),
        })
    return pd.DataFrame(rows).sort_values(["n", "Gewinnschutz bestätigt %"], ascending=[False, False]).reset_index(drop=True) if rows else pd.DataFrame()


def _risk_calibration(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    work = detail[(detail["Auswertbar"] == True) & pd.to_numeric(detail["Giveback Risk"], errors="coerce").notna()].copy()  # noqa: E712
    if work.empty:
        return pd.DataFrame()
    rows = []
    for band, grp in work.groupby("Giveback-Risk-Band", dropna=False):
        predicted = pd.to_numeric(grp["Giveback Risk"], errors="coerce").dropna()
        delta = pd.to_numeric(grp["ΔR nach Warnung"], errors="coerce").dropna()
        if predicted.empty or delta.empty:
            continue
        actual = (delta <= -0.25).astype(float) * 100.0
        obs = float(actual.mean())
        pred = float(predicted.mean())
        rows.append({
            "Giveback-Risk-Band": band,
            "n": int(len(grp)),
            "Stichprobe": sample_label(len(grp)),
            "Ø prognostiziert %": round(pred, 1),
            "Real bestätigt %": round(obs, 1),
            "Kalibrierungs-Lücke PP": round(obs - pred, 1),
        })
    if not rows:
        return pd.DataFrame()
    order = {"<45": 0, "45-59": 1, "60-74": 2, "75-100": 3, "n/a": 9}
    out = pd.DataFrame(rows)
    out["__order"] = out["Giveback-Risk-Band"].map(order).fillna(9)
    return out.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)


def build_learning_package(trades_df: pd.DataFrame | None, events_df: pd.DataFrame | None) -> dict[str, Any]:
    """Match first Early-Profit warning in each closed trade and evaluate later R.

    The comparison is intentionally *not* a hypothetical exit backtest.  It only
    asks whether the final realized trade R ended materially below, near, or above
    the R that was present when the first Early-Profit warning was logged.
    """
    trades = _normalize_trades(trades_df)
    events = _normalize_events(events_df)
    empty_summary = {
        "matched_trades": 0,
        "evaluable": 0,
        "coverage_pct": 0.0,
        "sample_label": "Zu klein",
        "protect_confirmed_pct": None,
        "hold_better_pct": None,
        "neutral_pct": None,
        "median_delta_r": None,
        "median_giveback_r": None,
        "status": "Daten sammeln",
    }
    if trades.empty or events.empty:
        return {
            "summary": empty_summary,
            "detail": pd.DataFrame(),
            "action_summary": pd.DataFrame(),
            "velocity_summary": pd.DataFrame(),
            "exhaustion_summary": pd.DataFrame(),
            "risk_calibration": pd.DataFrame(),
            "insights": [],
        }

    rows = []
    for _, tr in trades.iterrows():
        candidates = events[
            (events["Ticker"] == tr["Ticker"])
            & (events["Event-Zeit"] >= tr["Entry-Zeit"])
            & (events["Event-Zeit"] <= tr["Exit-Zeit"])
        ].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values("Event-Zeit")
        first = candidates.iloc[0]
        strongest_idx = candidates["Aktion"].astype(str).map(_severity).idxmax()
        strongest = candidates.loc[strongest_idx]
        event_r = _num(first.get("Event-R"), None)
        final_r = _num(tr.get("Final R"), None)
        delta_r = None if event_r is None or final_r is None else final_r - event_r
        giveback_r = None if delta_r is None else max(0.0, -delta_r)
        amp, outcome = _outcome(delta_r)
        velocity = _num(first.get("Profit Velocity"), None)
        exhaust = _num(first.get("Exhaustion Risk"), None)
        gb_risk = _num(first.get("Giveback Risk"), None)
        rows.append({
            "Ticker": tr["Ticker"],
            "Name": tr["Name"],
            "Entry-Zeit": tr["Entry-Zeit"],
            "Erste Warnung": first["Event-Zeit"],
            "Exit-Zeit": tr["Exit-Zeit"],
            "Warn-Aktion": first["Aktion"],
            "Stärkste Aktion": strongest["Aktion"],
            "Warn-R": event_r,
            "Final R": final_r,
            "ΔR nach Warnung": None if delta_r is None else round(float(delta_r), 3),
            "Giveback R": None if giveback_r is None else round(float(giveback_r), 3),
            "Ampel": amp,
            "Bewertung": outcome,
            "Auswertbar": bool(delta_r is not None),
            "Profit Velocity": velocity,
            "Velocity-Band": _band(velocity, kind="velocity"),
            "Exhaustion Risk": exhaust,
            "Exhaustion-Band": _band(exhaust, kind="risk"),
            "Giveback Risk": gb_risk,
            "Giveback-Risk-Band": _band(gb_risk, kind="risk"),
            "Warn-P/L %": _num(first.get("Event-P/L %"), None),
            "Final Kapitalrendite %": _num(tr.get("Kapitalrendite %"), None),
            "Haltedauer Tage": _num(tr.get("Haltedauer Tage"), None),
            "Trade Outcome": tr.get("Outcome"),
            "Events im Trade": int(len(candidates)),
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {
            "summary": empty_summary,
            "detail": detail,
            "action_summary": pd.DataFrame(),
            "velocity_summary": pd.DataFrame(),
            "exhaustion_summary": pd.DataFrame(),
            "risk_calibration": pd.DataFrame(),
            "insights": [],
        }

    evaluable = detail[detail["Auswertbar"] == True].copy()  # noqa: E712
    n_eval = int(len(evaluable))
    n_match = int(len(detail))
    coverage = 100.0 * n_eval / max(1, n_match)
    if n_eval:
        delta = pd.to_numeric(evaluable["ΔR nach Warnung"], errors="coerce").dropna()
        protect_pct = float((delta <= -0.25).mean() * 100.0) if len(delta) else None
        hold_pct = float((delta >= 0.25).mean() * 100.0) if len(delta) else None
        neutral_pct = float(((delta > -0.25) & (delta < 0.25)).mean() * 100.0) if len(delta) else None
        median_delta = float(delta.median()) if len(delta) else None
        median_giveback = float(pd.to_numeric(evaluable["Giveback R"], errors="coerce").dropna().median()) if evaluable["Giveback R"].notna().any() else None
    else:
        protect_pct = hold_pct = neutral_pct = median_delta = median_giveback = None

    if n_eval < 10:
        status = "Daten sammeln"
    elif n_eval < 20:
        status = "Frühe Kalibrierung"
    else:
        status = "Beobachtbar kalibriert"

    summary = {
        "matched_trades": n_match,
        "evaluable": n_eval,
        "coverage_pct": round(coverage, 1),
        "sample_label": sample_label(n_eval),
        "protect_confirmed_pct": None if protect_pct is None else round(protect_pct, 1),
        "hold_better_pct": None if hold_pct is None else round(hold_pct, 1),
        "neutral_pct": None if neutral_pct is None else round(neutral_pct, 1),
        "median_delta_r": None if median_delta is None else round(median_delta, 2),
        "median_giveback_r": None if median_giveback is None else round(median_giveback, 2),
        "status": status,
    }

    action_summary = _summary_table(detail, "Warn-Aktion")
    velocity_summary = _summary_table(detail, "Velocity-Band")
    exhaustion_summary = _summary_table(detail, "Exhaustion-Band")
    risk_calibration = _risk_calibration(detail)

    insights: list[str] = []
    if n_eval < 5:
        insights.append("Noch zu wenige sicher zuordenbare Early-Profit-Fälle; keine belastbare persönliche Regel ableiten.")
    else:
        if protect_pct is not None and protect_pct >= 60:
            insights.append(f"Bei {protect_pct:.0f}% der auswertbaren Warnungen lag das final realisierte R mindestens 0,25R unter dem Warnzeitpunkt; früher Gewinnschutz war häufig sinnvoll.")
        if hold_pct is not None and hold_pct >= 60:
            insights.append(f"Bei {hold_pct:.0f}% der auswertbaren Warnungen stieg das final realisierte R noch mindestens 0,25R weiter; die Warnungen waren in dieser Stichprobe häufig früh.")
        if protect_pct is not None and hold_pct is not None and max(protect_pct, hold_pct) < 60:
            insights.append("Die bisherigen Fälle sind gemischt; Teilgewinn/Trail bleibt sinnvoller als eine starre Alles-oder-Nichts-Regel.")

    if not exhaustion_summary.empty:
        high = exhaustion_summary[exhaustion_summary["Exhaustion-Band"].isin(["60-74", "75-100"])]
        if not high.empty and int(high["n"].sum()) >= 5:
            weighted = np.average(high["Gewinnschutz bestätigt %"], weights=high["n"])
            insights.append(f"Bei Exhaustion Risk ab 60 lag die Gewinnschutz-Bestätigung bisher bei rund {weighted:.0f}% (zusammen n={int(high['n'].sum())}).")

    if not risk_calibration.empty:
        mature = risk_calibration[risk_calibration["n"] >= 5]
        if not mature.empty:
            gap = float(np.average(mature["Kalibrierungs-Lücke PP"], weights=mature["n"]))
            if abs(gap) >= 15:
                direction = "unterschätzt" if gap > 0 else "überschätzt"
                insights.append(f"Das historische Giveback-Risiko wirkt in den reiferen Bändern derzeit eher {direction} (gewichtete Kalibrierungslücke {gap:+.0f} PP).")

    return {
        "summary": summary,
        "detail": detail.sort_values("Erste Warnung", ascending=False).reset_index(drop=True),
        "action_summary": action_summary,
        "velocity_summary": velocity_summary,
        "exhaustion_summary": exhaustion_summary,
        "risk_calibration": risk_calibration,
        "insights": insights[:6],
    }
