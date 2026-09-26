"""v30.21u - provider-free outcome validation for the Hybrid Stop shadow model.

The module is strictly observational. It stores stop variants frozen at trade
planning/execution time and evaluates them against OHLC bars that are already
present in completed Atomic live scans. No market-data request is performed and
no productive stop, CRV, sizing, gate or order is changed automatically.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import re
from typing import Any

import pandas as pd

_NAMESPACE = "stop_outcome_learning_v3021u"
_SCHEMA = "stop-outcome-learning-v30.21u"
_MAX_EPISODES = 240
_MAX_BARS_PER_TICKER = 260
_MAX_EVAL_BARS = 20

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


def _text(value: Any, default="") -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        text = ""
    if not text or text.lower() in {"none", "nan", "n/a", "na", "-", "<na>"}:
        return default
    return text


def _num(value: Any, default=None):
    if value is None or isinstance(value, bool):
        return default
    try:
        if isinstance(value, str):
            text = value.strip().replace("\u2212", "-").replace("%", "").replace(",", ".")
            match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
            if not match:
                return default
            value = match.group(0)
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _ts(value: Any):
    if value in (None, ""):
        return None
    try:
        out = pd.Timestamp(value)
    except Exception:
        return None
    try:
        if out.tzinfo is None:
            out = out.tz_localize("Europe/Berlin")
        else:
            out = out.tz_convert("Europe/Berlin")
    except Exception:
        pass
    return out


def _date_text(value: Any) -> str:
    ts = _ts(value)
    if ts is None:
        return ""
    try:
        return ts.date().isoformat()
    except Exception:
        return ""


def _episode_id(watchlist: str, ticker: str, opened_at: str, entry: float) -> str:
    raw = f"{watchlist}|{ticker}|{opened_at}|{entry:.8f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _watchlist_bucket(store: dict[str, Any], watchlist_name: str) -> tuple[dict[str, Any], str]:
    wl = _text(watchlist_name, "default")
    watchlists = dict(store.get("watchlists") or {})
    bucket = dict(watchlists.get(wl) or {})
    bucket.setdefault("episodes", [])
    bucket.setdefault("bars", {})
    return bucket, wl


def sync_position_episodes(watchlist_name: str, positions: dict | None) -> dict[str, Any]:
    """Register/update stop episodes from persisted Screener positions.

    Planned positions are stored but are not evaluable until execution is visible
    (shares > 0 or execution_status is no longer planned). Historical positions
    without a frozen ``entry_context.stop_shadow`` are deliberately not backfilled
    with today's structure/ATR values.
    """
    if not isinstance(positions, dict) or not positions:
        return {"saved": False, "registered": 0, "activated": 0, "reason": "no-positions"}

    store = _load_store()
    bucket, wl = _watchlist_bucket(store, watchlist_name)
    episodes = [dict(x) for x in list(bucket.get("episodes") or []) if isinstance(x, dict)]
    by_id = {str(x.get("episode_id") or ""): i for i, x in enumerate(episodes) if str(x.get("episode_id") or "")}
    registered = 0
    activated = 0
    changed = False

    for ticker_key, raw_pos in positions.items():
        if not isinstance(raw_pos, dict):
            continue
        if _text(raw_pos.get("strategy_origin"), "").lower() not in {"", "screener"}:
            continue
        ticker = _text(raw_pos.get("ticker") or ticker_key, "").upper()
        context = raw_pos.get("entry_context") if isinstance(raw_pos.get("entry_context"), dict) else {}
        frozen = context.get("stop_shadow") if isinstance(context.get("stop_shadow"), dict) else {}
        if not ticker or not frozen:
            continue
        # Provisional ATR-only shadows without a real structural invalidation are
        # intentionally excluded from Hybrid-vs-productive evidence.
        if frozen.get("full_trade_plan") is False:
            continue

        frozen_entry = _num(frozen.get("entry"), _num(raw_pos.get("entry"), None))
        entry = frozen_entry
        productive = _num(frozen.get("productive_stop"), _num(raw_pos.get("initial_stop"), _num(raw_pos.get("stop"), None)))
        hybrid = _num(frozen.get("hybrid_stop"), None)
        structure = _num(frozen.get("structure_stop"), None)
        target = _num(frozen.get("target"), _num(raw_pos.get("target"), None))

        shares = _num(raw_pos.get("shares"), 0.0) or 0.0
        exec_status = _text(raw_pos.get("execution_status"), "").lower()
        is_active = bool(shares > 0 or exec_status not in {"planned", "geplant", "vorgemerkt"})
        # Once a broker fill exists, evaluate the frozen absolute stops against
        # the actual position entry rather than the earlier planning limit.
        if is_active:
            actual_entry = _num(raw_pos.get("entry"), None)
            if actual_entry is not None and actual_entry > 0:
                entry = actual_entry

        if entry is None or entry <= 0 or productive is None or productive <= 0 or productive >= entry:
            continue
        if hybrid is None or hybrid <= 0 or hybrid >= entry:
            continue

        opened_at = _text(frozen.get("captured_at") or raw_pos.get("opened_at_iso") or raw_pos.get("created_at"), _now_iso())
        # Identity stays tied to the frozen planning geometry so a later broker
        # fill at a slightly different entry activates the same episode.
        _id_entry = frozen_entry if frozen_entry is not None and frozen_entry > 0 else entry
        eid = _text(frozen.get("episode_id"), "") or _episode_id(wl, ticker, opened_at, _id_entry)
        activation_at = _text(
            raw_pos.get("broker_opened_at")
            or (raw_pos.get("updated_at") if is_active and exec_status == "open" else None)
            or raw_pos.get("opened_at_iso"),
            opened_at,
        ) if is_active else ""

        rec = {
            "episode_id": eid,
            "ticker": ticker,
            "name": _text(raw_pos.get("name"), ticker),
            "watchlist": wl,
            "captured_at": opened_at,
            "activation_at": activation_at,
            "execution_status": "active" if is_active else "planned",
            "entry": float(entry),
            "target": target,
            "productive_stop": float(productive),
            "structure_stop": structure,
            "hybrid_stop": float(hybrid),
            "atr_pct": _num(frozen.get("atr_pct"), None),
            "setup_type": _text(frozen.get("setup_type"), "n/a"),
            "market_regime": _text(frozen.get("market_regime"), "n/a"),
            "volatility_regime": _text(frozen.get("volatility_regime"), "n/a"),
            "tradeability": _text(frozen.get("tradeability"), "n/a"),
            "source": _text(frozen.get("source"), "Hybrid Stop Shadow"),
            "productive_distance_pct": round((entry - productive) / entry * 100.0, 6),
            "hybrid_distance_pct": round((entry - hybrid) / entry * 100.0, 6),
        }

        if eid in by_id:
            idx = by_id[eid]
            old = episodes[idx]
            if is_active and not _text(old.get("activation_at"), ""):
                old["activation_at"] = activation_at or _now_iso()
                old["execution_status"] = "active"
                activated += 1
                changed = True
            # Keep frozen stop geometry immutable; only execution status may advance.
            if is_active and _text(old.get("execution_status"), "") != "active":
                old["execution_status"] = "active"
                changed = True
            episodes[idx] = old
        else:
            episodes.append(rec)
            by_id[eid] = len(episodes) - 1
            registered += 1
            if is_active:
                activated += 1
            changed = True

    if not changed:
        return {"saved": False, "registered": 0, "activated": 0, "episodes": len(episodes), "reason": "no-change"}

    episodes = sorted(episodes, key=lambda x: str(x.get("captured_at") or ""))[-_MAX_EPISODES:]
    bucket["episodes"] = episodes
    bucket["updated_at"] = _now_iso()
    watchlists = dict(store.get("watchlists") or {})
    watchlists[wl] = bucket
    store["watchlists"] = watchlists
    ok = _save_store(store)
    return {"saved": bool(ok), "registered": registered, "activated": activated, "episodes": len(episodes), "reason": "updated"}


def capture_market_snapshot(
    watchlist_name: str,
    live_df: pd.DataFrame | None,
    *,
    scan_id: str | None = None,
    scan_time: Any = None,
    atomic_complete: bool = True,
) -> dict[str, Any]:
    """Store daily OHLC already present in a complete Atomic scan.

    No provider call is performed. Partial rescans are intentionally ignored so
    the learning history cannot be polluted by mixed-age rows.
    """
    if not atomic_complete or not isinstance(live_df, pd.DataFrame) or live_df.empty:
        return {"saved": False, "rows": 0, "reason": "no-complete-scan"}
    if "Ticker" not in live_df.columns:
        return {"saved": False, "rows": 0, "reason": "no-ticker"}

    store = _load_store()
    bucket, wl = _watchlist_bucket(store, watchlist_name)
    sid = _text(scan_id, "")
    observed_at = _text(scan_time, _now_iso())
    if sid and _text(bucket.get("last_scan_id"), "") == sid:
        return {"saved": False, "rows": 0, "reason": "same-scan"}
    if not sid and observed_at and _text(bucket.get("last_scan_at"), "") == observed_at:
        return {"saved": False, "rows": 0, "reason": "same-scan-time"}
    bars = {str(k): list(v or []) for k, v in dict(bucket.get("bars") or {}).items()}
    written = 0

    for _, series in live_df.iterrows():
        row = series.to_dict()
        ticker = _text(row.get("Ticker"), "").upper()
        if not ticker:
            continue
        high = _num(row.get("__diag_stop_day_high"), None)
        low = _num(row.get("__diag_stop_day_low"), None)
        close = _num(row.get("__diag_stop_day_close"), _num(row.get("Kurs"), None))
        data_date = _text(row.get("__diag_stop_data_date") or row.get("__diag_setup_data_date"), "")
        if data_date:
            try:
                data_date = pd.Timestamp(data_date).date().isoformat()
            except Exception:
                data_date = data_date[:10]
        if not data_date:
            data_date = _date_text(observed_at)
        if not data_date or close is None or close <= 0:
            continue
        if high is None or high <= 0:
            high = close
        if low is None or low <= 0:
            low = close
        if high < low:
            high, low = low, high
        bar = {
            "date": data_date,
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "observed_at": observed_at,
            "scan_id": _text(scan_id, ""),
        }
        seq = [dict(x) for x in bars.get(ticker, []) if isinstance(x, dict)]
        replaced = False
        for i, old in enumerate(seq):
            if str(old.get("date") or "") == data_date:
                seq[i] = bar
                replaced = True
                break
        if not replaced:
            seq.append(bar)
        seq = sorted(seq, key=lambda x: str(x.get("date") or ""))[-_MAX_BARS_PER_TICKER:]
        bars[ticker] = seq
        written += 1

    if written <= 0:
        return {"saved": False, "rows": 0, "reason": "no-ohlc"}
    bucket["bars"] = bars
    bucket["last_scan_id"] = sid
    bucket["last_scan_at"] = observed_at
    watchlists = dict(store.get("watchlists") or {})
    watchlists[wl] = bucket
    store["watchlists"] = watchlists
    ok = _save_store(store)
    return {"saved": bool(ok), "rows": written, "reason": "stored"}


def _model_outcome(episode: dict[str, Any], bars: list[dict[str, Any]], stop_key: str) -> dict[str, Any]:
    entry = _num(episode.get("entry"), None)
    stop = _num(episode.get(stop_key), None)
    target = _num(episode.get("target"), None)
    if entry is None or stop is None or stop <= 0 or stop >= entry:
        return {"status": "nicht auswertbar", "decisive": False}
    activation_date = _date_text(episode.get("activation_at"))
    future = [b for b in bars if str(b.get("date") or "") > activation_date]
    future = future[:_MAX_EVAL_BARS]
    if not future:
        return {"status": "offen", "decisive": False, "bars": 0}

    min_low = min((_num(b.get("low"), entry) for b in future), default=entry)
    max_high = max((_num(b.get("high"), entry) for b in future), default=entry)
    mae_pct = max(0.0, (entry - min_low) / entry * 100.0)
    mfe_pct = max(0.0, (max_high - entry) / entry * 100.0)

    first_stop_idx = None
    first_target_idx = None
    same_day_ambiguous = False
    terminal_date = ""
    for idx, bar in enumerate(future):
        low = _num(bar.get("low"), None)
        high = _num(bar.get("high"), None)
        stop_hit = bool(low is not None and low <= stop)
        target_hit = bool(target is not None and target > entry and high is not None and high >= target)
        if first_stop_idx is None and stop_hit:
            first_stop_idx = idx
        if first_target_idx is None and target_hit:
            first_target_idx = idx
        if stop_hit or target_hit:
            terminal_date = str(bar.get("date") or "")
            if stop_hit and target_hit:
                same_day_ambiguous = True
            break

    if same_day_ambiguous:
        status = "gleichentags unklar"
        decisive = False
    elif first_stop_idx is not None and (first_target_idx is None or first_stop_idx < first_target_idx):
        status = "Stop zuerst"
        decisive = True
    elif first_target_idx is not None and (first_stop_idx is None or first_target_idx < first_stop_idx):
        status = "Ziel zuerst"
        decisive = True
    else:
        status = "20T offen" if len(future) >= _MAX_EVAL_BARS else "offen"
        decisive = False

    whipsaw = False
    if status == "Stop zuerst" and target is not None and target > entry and first_stop_idx is not None:
        for later in future[first_stop_idx + 1 :]:
            high = _num(later.get("high"), None)
            if high is not None and high >= target:
                whipsaw = True
                break

    return {
        "status": status,
        "decisive": decisive,
        "bars": len(future),
        "terminal_date": terminal_date,
        "whipsaw": whipsaw,
        "mae_pct": mae_pct,
        "mfe_pct": mfe_pct,
    }


def _maturity(n: int) -> tuple[str, str]:
    n = int(n or 0)
    if n < 10:
        return "🔴 Sammeln", "mindestens 10 aktive/auswertbare Fälle aufbauen"
    if n < 30:
        return "🟠 Früh", "bis mindestens 30 Vergleichsfälle weiter sammeln"
    if n < 50:
        return "🟡 Beobachtbar", "Regime-/Setup-Stabilität prüfen; noch kein Cutover"
    return "🟢 Reif für Cutover-Prüfung", "kontrollierten A/B-Cutover prüfen; niemals automatisch umstellen"


def build_learning_package(watchlist_name: str) -> dict[str, Any]:
    store = _load_store()
    wl = _text(watchlist_name, "default")
    bucket = dict((store.get("watchlists") or {}).get(wl) or {})
    episodes = [dict(x) for x in list(bucket.get("episodes") or []) if isinstance(x, dict)]
    bars_by_ticker = {str(k).upper(): list(v or []) for k, v in dict(bucket.get("bars") or {}).items()}

    rows: list[dict[str, Any]] = []
    for ep in episodes:
        active = _text(ep.get("execution_status"), "") == "active" and bool(_text(ep.get("activation_at"), ""))
        prod = _model_outcome(ep, bars_by_ticker.get(_text(ep.get("ticker"), "").upper(), []), "productive_stop") if active else {"status": "geplant", "decisive": False}
        hybrid = _model_outcome(ep, bars_by_ticker.get(_text(ep.get("ticker"), "").upper(), []), "hybrid_stop") if active else {"status": "geplant", "decisive": False}
        pair_decisive = bool(prod.get("decisive") and hybrid.get("decisive"))
        rows.append({
            "Episode": _text(ep.get("episode_id"), ""),
            "Ticker": _text(ep.get("ticker"), ""),
            "Status": "Aktiv" if active else "Geplant",
            "Entry": _num(ep.get("entry"), None),
            "Ziel": _num(ep.get("target"), None),
            "Produktiv Stop": _num(ep.get("productive_stop"), None),
            "Hybrid Stop": _num(ep.get("hybrid_stop"), None),
            "Struktur": _num(ep.get("structure_stop"), None),
            "Produktiv Abstand %": _num(ep.get("productive_distance_pct"), None),
            "Hybrid Abstand %": _num(ep.get("hybrid_distance_pct"), None),
            "Setup": _text(ep.get("setup_type"), "n/a"),
            "Marktregime": _text(ep.get("market_regime"), "n/a"),
            "Vola-Regime": _text(ep.get("volatility_regime"), "n/a"),
            "Produktiv Ergebnis": prod.get("status"),
            "Hybrid Ergebnis": hybrid.get("status"),
            "Produktiv Whipsaw": bool(prod.get("whipsaw", False)),
            "Hybrid Whipsaw": bool(hybrid.get("whipsaw", False)),
            "MAE %": _num(hybrid.get("mae_pct"), None),
            "MFE %": _num(hybrid.get("mfe_pct"), None),
            "Vergleich auswertbar": pair_decisive,
            "Aktiv seit": _text(ep.get("activation_at"), ""),
        })

    detail = pd.DataFrame(rows)
    active_n = int((detail["Status"] == "Aktiv").sum()) if not detail.empty else 0
    planned_n = int((detail["Status"] == "Geplant").sum()) if not detail.empty else 0
    pair = detail[detail["Vergleich auswertbar"] == True].copy() if not detail.empty else pd.DataFrame()  # noqa: E712
    n_pair = int(len(pair))
    maturity, next_step = _maturity(n_pair)

    def _pct(series: pd.Series, value: Any) -> float | None:
        if series.empty:
            return None
        return round(float((series == value).mean() * 100.0), 1)

    model_rows = []
    for label, result_col, whipsaw_col, distance_col in [
        ("Produktiv", "Produktiv Ergebnis", "Produktiv Whipsaw", "Produktiv Abstand %"),
        ("Hybrid Shadow", "Hybrid Ergebnis", "Hybrid Whipsaw", "Hybrid Abstand %"),
    ]:
        active_detail = detail[detail["Status"] == "Aktiv"].copy() if not detail.empty else pd.DataFrame()
        decisive = active_detail[active_detail[result_col].isin(["Stop zuerst", "Ziel zuerst"])].copy() if not active_detail.empty else pd.DataFrame()
        model_rows.append({
            "Modell": label,
            "Auswertbar": int(len(decisive)),
            "Ziel zuerst %": _pct(decisive[result_col], "Ziel zuerst") if not decisive.empty else None,
            "Stop zuerst %": _pct(decisive[result_col], "Stop zuerst") if not decisive.empty else None,
            "Whipsaw %": round(float(pd.to_numeric(decisive[whipsaw_col], errors="coerce").fillna(False).astype(bool).mean() * 100.0), 1) if not decisive.empty else None,
            "Median Stop-Abstand %": round(float(pd.to_numeric(active_detail[distance_col], errors="coerce").dropna().median()), 2) if not active_detail.empty and pd.to_numeric(active_detail[distance_col], errors="coerce").notna().any() else None,
        })
    model_summary = pd.DataFrame(model_rows)

    width_reduction = None
    if not detail.empty:
        prod_dist = pd.to_numeric(detail["Produktiv Abstand %"], errors="coerce")
        hyb_dist = pd.to_numeric(detail["Hybrid Abstand %"], errors="coerce")
        diffs = (prod_dist - hyb_dist).dropna()
        if not diffs.empty:
            width_reduction = round(float(diffs.median()), 2)

    hybrid_target = _pct(pair["Hybrid Ergebnis"], "Ziel zuerst") if not pair.empty else None
    prod_target = _pct(pair["Produktiv Ergebnis"], "Ziel zuerst") if not pair.empty else None
    target_delta = None if hybrid_target is None or prod_target is None else round(hybrid_target - prod_target, 1)

    setup_coverage = 0
    regime_coverage = 0
    if not pair.empty:
        setup_counts = pair["Setup"].astype(str).replace("n/a", pd.NA).dropna().value_counts()
        regime_counts = pair["Vola-Regime"].astype(str).replace("n/a", pd.NA).dropna().value_counts()
        setup_coverage = int((setup_counts >= 5).sum())
        regime_coverage = int((regime_counts >= 5).sum())

    insights = []
    if n_pair < 10:
        insights.append("Noch zu wenige auswertbare Stop-Vergleiche; keinerlei produktive Stop-Regel ableiten.")
    else:
        if width_reduction is not None:
            insights.append(f"Der Hybrid reduziert den medianen Stop-Abstand aktuell um {width_reduction:+.2f} Prozentpunkte gegenüber produktiv.")
        if target_delta is not None:
            insights.append(f"Ziel-vor-Stop: Hybrid liegt aktuell {target_delta:+.1f} Prozentpunkte gegenüber produktiv (nur entscheidbare Paarfälle).")
        if n_pair < 30:
            insights.append("Frühphase: mindestens 30 entscheidbare Paarfälle abwarten, bevor ein A/B-Cutover überhaupt diskutiert wird.")
        elif setup_coverage < 2 or regime_coverage < 2:
            insights.append("Stichprobe ist größer, aber noch zu wenig über Setup-/Volatilitätsregime verteilt; weiter sammeln.")
        else:
            insights.append("Datenbasis ist für eine kontrollierte Cutover-Prüfung näher gekommen; automatische Umstellung bleibt ausgeschlossen.")

    summary = {
        "episodes_total": len(episodes),
        "active": active_n,
        "planned": planned_n,
        "pair_evaluable": n_pair,
        "maturity": maturity,
        "next_step": next_step,
        "median_width_reduction_pp": width_reduction,
        "hybrid_target_first_pct": hybrid_target,
        "productive_target_first_pct": prod_target,
        "target_first_delta_pp": target_delta,
        "setup_groups_n5": setup_coverage,
        "volatility_regimes_n5": regime_coverage,
        "max_eval_bars": _MAX_EVAL_BARS,
        "automatic_cutover": False,
    }
    return {
        "summary": summary,
        "model_summary": model_summary,
        "detail": detail.sort_values("Aktiv seit", ascending=False).reset_index(drop=True) if not detail.empty else detail,
        "insights": insights,
        "storage": {"namespace": _NAMESPACE, "schema": _SCHEMA},
    }
