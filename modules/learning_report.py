from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

REPORT_SCHEMA = "chsm_learning_report_v1"

REVIEW_CRITERIA = {
    "Action Queue": "mind. 15 auswertbare 3T-Fälle in 'Jetzt prüfen' und 15 in 'Beobachten'",
    "Harvest / Chop": "mind. 15 auswertbare 3T-Harvest-Alarme",
    "Guarded / Shadow": "mind. 40 auswertbare 5T-Episoden; harte Release-Gates bleiben zusätzlich erforderlich",
    "Real Trade Learning": "mind. 20 geschlossene Trades, ØR > 0 und Entry-Kontextabdeckung >= 60%",
    "Early Profit / Exit": "mind. 20 auswertbare, sicher gematchte Warn-/Trade-Fälle",
    "Hybrid Stop": "mind. 50 entscheidbare Paarfälle plus >=2 Setup- und >=2 Volatilitätsgruppen mit jeweils n>=5",
}


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _records(frame: Any, limit: int = 100) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    safe = frame.head(max(0, int(limit))).copy()
    return _json_safe(safe)


def _selected(source: dict[str, Any] | None, keys: list[str]) -> dict[str, Any]:
    source = source or {}
    return {key: _json_safe(source.get(key)) for key in keys if key in source}


def build_review_status(
    *,
    queue_pkg: dict[str, Any] | None = None,
    harvest_pkg: dict[str, Any] | None = None,
    shadow_cal: dict[str, Any] | None = None,
    trade_pkg: dict[str, Any] | None = None,
    early_pkg: dict[str, Any] | None = None,
    stop_pkg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    queue = (queue_pkg or {}).get("summary") or {}
    harvest = (harvest_pkg or {}).get("summary") or {}
    shadow = (shadow_cal or {}).get("overview") or {}
    trade = (trade_pkg or {}).get("summary") or {}
    early = (early_pkg or {}).get("summary") or {}
    stop = (stop_pkg or {}).get("summary") or {}

    ready_n = _int(queue.get("ready_3t"))
    watch_n = _int(queue.get("watch_3t"))
    queue_due = ready_n >= 15 and watch_n >= 15

    harvest_n = _int(harvest.get("alerts_3t"))
    harvest_due = harvest_n >= 15

    shadow_n = _int(shadow.get("events_evaluable"))
    shadow_due = shadow_n >= 40

    trades_n = _int(trade.get("closed_trades"))
    avg_r = _num(trade.get("avg_r"))
    context = _num(trade.get("context_coverage"), 0.0) or 0.0
    trade_due = trades_n >= 20 and avg_r is not None and avg_r > 0 and context >= 60.0

    early_n = _int(early.get("evaluable"))
    early_due = early_n >= 20

    stop_n = _int(stop.get("pair_evaluable"))
    setup_groups = _int(stop.get("setup_groups_n5"))
    vola_groups = _int(stop.get("volatility_regimes_n5"))
    stop_due = stop_n >= 50 and setup_groups >= 2 and vola_groups >= 2

    rows = [
        {
            "Engine": "Action Queue",
            "Prüfung fällig": queue_due,
            "Evidenz": f"Jetzt prüfen {ready_n}/15 · Beobachten {watch_n}/15",
            "Kriterium": REVIEW_CRITERIA["Action Queue"],
        },
        {
            "Engine": "Harvest / Chop",
            "Prüfung fällig": harvest_due,
            "Evidenz": f"{harvest_n}/15 Harvest-Alarme · 3T",
            "Kriterium": REVIEW_CRITERIA["Harvest / Chop"],
        },
        {
            "Engine": "Guarded / Shadow",
            "Prüfung fällig": shadow_due,
            "Evidenz": f"{shadow_n}/40 · 5T",
            "Kriterium": REVIEW_CRITERIA["Guarded / Shadow"],
        },
        {
            "Engine": "Real Trade Learning",
            "Prüfung fällig": trade_due,
            "Evidenz": f"{trades_n}/20 Trades · ØR {('n/a' if avg_r is None else f'{avg_r:+.2f}R')} · Kontext {context:.0f}%",
            "Kriterium": REVIEW_CRITERIA["Real Trade Learning"],
        },
        {
            "Engine": "Early Profit / Exit",
            "Prüfung fällig": early_due,
            "Evidenz": f"{early_n}/20 auswertbar",
            "Kriterium": REVIEW_CRITERIA["Early Profit / Exit"],
        },
        {
            "Engine": "Hybrid Stop",
            "Prüfung fällig": stop_due,
            "Evidenz": f"{stop_n}/50 Paarfälle · Setup-Gruppen {setup_groups}/2 · Vola-Gruppen {vola_groups}/2",
            "Kriterium": REVIEW_CRITERIA["Hybrid Stop"],
        },
    ]
    for row in rows:
        row["Status"] = "🟢 Prüfung fällig" if row["Prüfung fällig"] else "⏳ Weiter sammeln"
    return rows


def _trade_summary_for_export(summary: dict[str, Any] | None) -> dict[str, Any]:
    return _selected(
        summary or {},
        [
            "closed_trades",
            "wins",
            "losses",
            "win_rate",
            "avg_r",
            "median_r",
            "profit_factor",
            "avg_return_pct",
            "avg_hold_days",
            "context_coverage",
            "sample_label",
        ],
    )


def _segment_export(segments: Any) -> dict[str, Any]:
    if not isinstance(segments, dict):
        return {}
    return {str(name): _records(frame, limit=50) for name, frame in segments.items() if isinstance(frame, pd.DataFrame)}


def _stop_breakdowns(stop_pkg: dict[str, Any] | None) -> dict[str, Any]:
    detail = (stop_pkg or {}).get("detail")
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return {}
    pair = detail[detail.get("Vergleich auswertbar", pd.Series(False, index=detail.index)).fillna(False).astype(bool)].copy()
    if pair.empty:
        return {}

    out: dict[str, Any] = {}
    for column, label in [("Setup", "setup"), ("Marktregime", "market_regime"), ("Vola-Regime", "volatility_regime")]:
        if column not in pair.columns:
            continue
        rows = []
        for key, grp in pair.groupby(column, dropna=False):
            if len(grp) < 2:
                continue
            prod = grp.get("Produktiv Ergebnis", pd.Series(index=grp.index, dtype=object)).astype(str)
            hyb = grp.get("Hybrid Ergebnis", pd.Series(index=grp.index, dtype=object)).astype(str)
            rows.append(
                {
                    "segment": str(key),
                    "n": int(len(grp)),
                    "productive_target_first_pct": round(float((prod == "Ziel zuerst").mean() * 100.0), 1),
                    "hybrid_target_first_pct": round(float((hyb == "Ziel zuerst").mean() * 100.0), 1),
                    "hybrid_whipsaw_pct": round(float(grp.get("Hybrid Whipsaw", pd.Series(False, index=grp.index)).fillna(False).astype(bool).mean() * 100.0), 1),
                    "median_mae_pct": _num(pd.to_numeric(grp.get("MAE %"), errors="coerce").median()),
                    "median_mfe_pct": _num(pd.to_numeric(grp.get("MFE %"), errors="coerce").median()),
                }
            )
        if rows:
            out[label] = rows
    return _json_safe(out)


def build_learning_report(
    *,
    app_version: str,
    watchlist_name: str,
    queue_pkg: dict[str, Any] | None = None,
    harvest_pkg: dict[str, Any] | None = None,
    shadow_cal: dict[str, Any] | None = None,
    trade_pkg: dict[str, Any] | None = None,
    early_pkg: dict[str, Any] | None = None,
    stop_pkg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    queue_pkg = queue_pkg or {}
    harvest_pkg = harvest_pkg or {}
    shadow_cal = shadow_cal or {}
    trade_pkg = trade_pkg or {}
    early_pkg = early_pkg or {}
    stop_pkg = stop_pkg or {}

    review = build_review_status(
        queue_pkg=queue_pkg,
        harvest_pkg=harvest_pkg,
        shadow_cal=shadow_cal,
        trade_pkg=trade_pkg,
        early_pkg=early_pkg,
        stop_pkg=stop_pkg,
    )

    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "app_version": str(app_version or "unknown"),
        "watchlist": str(watchlist_name or "default"),
        "privacy": {
            "aggregated_only": True,
            "contains_secrets": False,
            "contains_account_ids": False,
            "contains_broker_pnl_totals": False,
            "contains_ticker_level_trade_rows": False,
        },
        "governance": {
            "automatic_rule_changes": False,
            "process": [
                "Sammeln",
                "Auswerten",
                "Shadow-Empfehlung",
                "Evidenz-Gate",
                "A/B-Phase",
                "manueller Cutover",
            ],
            "review_criteria": REVIEW_CRITERIA,
        },
        "review_status": review,
        "review_due_engines": [row["Engine"] for row in review if row["Prüfung fällig"]],
        "engines": {
            "action_queue": {
                "summary": _json_safe(queue_pkg.get("summary") or {}),
                "category_summary": _records(queue_pkg.get("category_summary")),
                "confidence_summary": _records(queue_pkg.get("confidence_summary")),
            },
            "harvest_chop": {
                "summary": _json_safe(harvest_pkg.get("summary") or {}),
                "horizon_summary": _records(harvest_pkg.get("horizon_summary")),
                "harvest_band_summary": _records(harvest_pkg.get("harvest_band_summary")),
                "chop_band_summary": _records(harvest_pkg.get("chop_band_summary")),
            },
            "guarded_shadow": {
                "overview": _json_safe(shadow_cal.get("overview") or {}),
                "horizons": _records(shadow_cal.get("horizons")),
                "coverage": _records(shadow_cal.get("coverage")),
                "guardrail_backtest": _records(shadow_cal.get("guardrail_backtest")),
                "market_segments": _records(shadow_cal.get("market")),
                "volatility_segments": _records(shadow_cal.get("volatility")),
            },
            "real_trade_learning": {
                "summary": _trade_summary_for_export(trade_pkg.get("summary") or {}),
                "segments": _segment_export(trade_pkg.get("segments")),
                "exit_summary": _records(trade_pkg.get("exit_summary")),
            },
            "early_profit_exit": {
                "summary": _json_safe(early_pkg.get("summary") or {}),
                "action_summary": _records(early_pkg.get("action_summary")),
                "velocity_summary": _records(early_pkg.get("velocity_summary")),
                "exhaustion_summary": _records(early_pkg.get("exhaustion_summary")),
                "risk_calibration": _records(early_pkg.get("risk_calibration")),
            },
            "hybrid_stop": {
                "summary": _json_safe(stop_pkg.get("summary") or {}),
                "model_summary": _records(stop_pkg.get("model_summary")),
                "segment_breakdowns": _stop_breakdowns(stop_pkg),
            },
        },
    }
    return _json_safe(report)


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False)
