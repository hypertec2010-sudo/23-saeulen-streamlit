"""Trading Journal & Learning Engine introduced in v29.0.

This module is intentionally observational. It builds learning datasets from
persisted journal/event data and never changes live scores, shadow thresholds,
guardrails, positions or orders.
"""
from __future__ import annotations

from typing import Any
import math
import re

import pandas as pd


AMPel_RANK = {"🔴": 0, "⚪": 1, "🟡": 2, "🟢": 3}


def _blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "n/a", "na", "-"}


def _num(value: Any, default=None):
    if _blank(value):
        return default
    try:
        if isinstance(value, str):
            value = value.replace("%", "").replace("R", "").replace("/100", "").replace(",", ".").strip()
        out = float(value)
        return default if math.isnan(out) else out
    except Exception:
        return default


def _dt(value: Any):
    if _blank(value):
        return pd.NaT
    try:
        text = str(value).strip()
        iso_like = bool(re.match(r"^\d{4}-\d{2}-\d{2}", text))
        out = pd.to_datetime(value, errors="coerce", dayfirst=not iso_like)
        if isinstance(out, pd.Timestamp) and out.tzinfo is not None:
            out = out.tz_localize(None)
        return out
    except Exception:
        return pd.NaT


def _text(value: Any, default="") -> str:
    return default if _blank(value) else str(value).strip()


def _score_band(value: Any) -> str:
    score = _num(value, None)
    if score is None:
        return "n/a"
    if score < 28:
        return "0-27 · Rot"
    if score < 55:
        return "28-54 · Weiß"
    if score < 72:
        return "55-71 · Gelb"
    return "72-100 · Grün"


def _sample_label(n: int) -> str:
    n = int(n or 0)
    if n < 5:
        return "Zu klein"
    if n < 15:
        return "Frühphase"
    if n < 30:
        return "Mittel"
    if n < 60:
        return "Gut"
    return "Breiter"


def _shadow_relation(live: Any, shadow: Any) -> str:
    l = _text(live, "")
    s = _text(shadow, "")
    if l not in AMPel_RANK or s not in AMPel_RANK:
        return "n/a"
    if AMPel_RANK[s] > AMPel_RANK[l]:
        return "Shadow stärker"
    if AMPel_RANK[s] < AMPel_RANK[l]:
        return "Shadow schwächer"
    return "Gleich"


def capture_entry_context(live_row: dict | None) -> dict:
    """Capture the already-computed atomic screener context for a new trade.

    No market-data request is performed here. Missing fields remain missing.
    """
    row = dict(live_row or {})
    return {
        "captured_at": _text(row.get("Letztes Update"), ""),
        "status": _text(row.get("Status"), ""),
        "live_ampel": _text(row.get("Ampel"), ""),
        "shadow_ampel": _text(row.get("Shadow-Ampel"), ""),
        "live_score": _num(row.get("Live-Score"), None),
        "engine_score": _num(row.get("Engine-Score"), None),
        "guarded_score": _num(row.get("Guarded Engine-Score"), None),
        "engine_recommendation": _text(row.get("Engine-Empfehlung"), ""),
        "guardrail": _text(row.get("Engine-Guardrail"), ""),
        "context_adjustment": _num(row.get("Kontext-Anpassung"), None),
        "context_confidence": _text(row.get("Kontext-Verlässlichkeit"), ""),
        "market_regime": _text(row.get("Marktregime"), ""),
        "volatility_regime": _text(row.get("Volatilitätsregime"), ""),
        "rs_dynamics": _text(row.get("RS-Dynamik"), ""),
        "relative_strength": _text(row.get("Relative Stärke"), ""),
        "radar_bucket": _text(row.get("Radar-Bucket"), ""),
        "grade": _text(row.get("Grade"), ""),
        "crv": _num(row.get("CRV"), None),
        "entry_distance": _text(row.get("Entry-Abstand"), ""),
        "setup_alert": _text(row.get("Setup-Alert"), ""),
        "active_gates": _text(row.get("Aktive Einstiegsgates"), ""),
        "benchmark": _text(row.get("Primärbenchmark") or row.get("Benchmark"), ""),
        "live_horizon": _text(row.get("Live-Horizont"), ""),
        "atr_pct": _num(row.get("ATR-%"), None),
    }


def _context_from_close(row: dict, snapshot: dict) -> dict:
    ctx = {}
    if isinstance(snapshot, dict):
        raw = snapshot.get("entry_context")
        if isinstance(raw, dict):
            ctx.update(raw)

    # v29.0 journal rows carry flattened context too. Prefer explicit row values
    # so CSV exports remain self-contained, but do not invent legacy values.
    mapping = {
        "status": "Entry Status",
        "live_ampel": "Entry Live-Ampel",
        "shadow_ampel": "Entry Shadow-Ampel",
        "live_score": "Entry Live-Score",
        "engine_score": "Entry Engine-Score",
        "guarded_score": "Entry Guarded Score",
        "engine_recommendation": "Entry Engine-Empfehlung",
        "guardrail": "Entry Guardrail",
        "context_adjustment": "Entry Kontext-Anpassung",
        "context_confidence": "Entry Kontext-Verlässlichkeit",
        "market_regime": "Entry Marktregime",
        "volatility_regime": "Entry Volatilitätsregime",
        "rs_dynamics": "Entry RS-Dynamik",
        "relative_strength": "Entry Relative Stärke",
        "radar_bucket": "Entry Radar-Bucket",
        "grade": "Entry Grade",
        "crv": "Entry CRV",
        "entry_distance": "Entry Abstand",
        "setup_alert": "Entry Setup-Alert",
        "active_gates": "Entry Gates",
        "benchmark": "Entry Benchmark",
        "live_horizon": "Entry Horizont",
        "captured_at": "Entry Kontext-Zeit",
    }
    for key, col in mapping.items():
        val = row.get(col)
        if not _blank(val):
            ctx[key] = val
    return ctx


def build_trade_dataset(journal_df: pd.DataFrame | None) -> pd.DataFrame:
    """Build one row per valid completed trade cycle.

    Undo rows are audit-only and are never counted as completed trades. Partial
    exits are folded into the subsequent valid full close for initial-share and
    capital-return reconstruction where possible.
    """
    if not isinstance(journal_df, pd.DataFrame) or journal_df.empty:
        return pd.DataFrame()

    work = journal_df.copy().reset_index(drop=True)
    if "Typ" not in work.columns or "Ticker" not in work.columns:
        return pd.DataFrame()

    work["__dt"] = work.get("Zeit", pd.Series(index=work.index, dtype=object)).map(_dt)
    if "Datum" in work.columns:
        missing = work["__dt"].isna()
        work.loc[missing, "__dt"] = work.loc[missing, "Datum"].map(_dt)
    work["__seq"] = range(len(work))
    # Journal dataframe is normally newest-first. Time sort restores the cycle.
    work = work.sort_values(["__dt", "__seq"], ascending=[True, False], na_position="last").reset_index(drop=True)

    cycle_state: dict[tuple[str, str], dict] = {}
    rows = []

    for _, series in work.iterrows():
        rec = series.to_dict()
        ticker = _text(rec.get("Ticker"), "").upper()
        watchlist = _text(rec.get("Watchlist"), "Standard")
        typ = _text(rec.get("Typ"), "")
        if not ticker:
            continue
        key = (watchlist, ticker)
        state = cycle_state.setdefault(key, {"partial_shares": 0, "partial_pnl": 0.0, "partial_count": 0})

        if typ == "Teilverkauf":
            state["partial_shares"] += max(0, int(_num(rec.get("Stück"), 0) or 0))
            state["partial_pnl"] += _num(rec.get("Realisiert P/L"), 0.0) or 0.0
            state["partial_count"] += 1
            continue

        if typ == "Schließung rückgängig":
            # Audit row only. Existing partial exits remain part of the still-open cycle.
            continue

        if typ != "Position geschlossen":
            continue

        snapshot = rec.get("Position vorher") if isinstance(rec.get("Position vorher"), dict) else {}
        context = _context_from_close(rec, snapshot)

        close_shares = max(0, int(_num(rec.get("Stück"), 0) or 0))
        initial_shares = int(_num(snapshot.get("initial_shares"), 0) or 0) if snapshot else 0
        if initial_shares <= 0:
            initial_shares = close_shares + int(state.get("partial_shares", 0) or 0)

        entry = _num(rec.get("Entry"), _num(snapshot.get("entry"), None) if snapshot else None)
        total_pnl = _num(rec.get("Gesamt P/L"), None)
        if total_pnl is None:
            total_pnl = float(state.get("partial_pnl", 0.0) or 0.0) + (_num(rec.get("Realisiert P/L"), 0.0) or 0.0)
        total_r = _num(rec.get("Gesamt R"), None)

        capital_return = None
        if entry is not None and entry > 0 and initial_shares > 0 and total_pnl is not None:
            capital_return = (float(total_pnl) / (float(entry) * initial_shares)) * 100.0

        close_dt = rec.get("__dt")
        entry_dt = pd.NaT
        if snapshot:
            entry_dt = _dt(snapshot.get("opened_at_iso"))
            if pd.isna(entry_dt):
                entry_dt = _dt(snapshot.get("created_at"))
        if pd.isna(entry_dt):
            entry_dt = _dt(context.get("captured_at"))
        hold_days = None
        if not pd.isna(entry_dt) and not pd.isna(close_dt):
            hold_days = max(0.0, (close_dt - entry_dt).total_seconds() / 86400.0)

        live_ampel = _text(context.get("live_ampel"), "n/a")
        shadow_ampel = _text(context.get("shadow_ampel"), "n/a")
        live_score = _num(context.get("live_score"), None)
        guarded_score = _num(context.get("guarded_score"), None)
        pnl_value = _num(total_pnl, None)
        outcome = "n/a"
        if pnl_value is not None:
            outcome = "Gewinn" if pnl_value > 0 else ("Verlust" if pnl_value < 0 else "Flat")

        rows.append({
            "Watchlist": watchlist,
            "Ticker": ticker,
            "Name": _text(rec.get("Name"), ticker),
            "Entry-Zeit": entry_dt,
            "Exit-Zeit": close_dt,
            "Entry": entry,
            "Exit": _num(rec.get("Kurs"), None),
            "Initial-Stück": initial_shares if initial_shares > 0 else None,
            "Teilverkäufe": int(state.get("partial_count", 0) or 0),
            "Gesamt P/L": pnl_value,
            "Gesamt R": total_r,
            "Kapitalrendite %": capital_return,
            "Haltedauer Tage": hold_days,
            "Outcome": outcome,
            "Exit-Grund": _text(rec.get("Details"), ""),
            "Erkenntnis": _text(rec.get("Erkenntnis"), ""),
            "Entry Status": _text(context.get("status"), "n/a"),
            "Entry Live-Ampel": live_ampel,
            "Entry Shadow-Ampel": shadow_ampel,
            "Shadow vs Live": _shadow_relation(live_ampel, shadow_ampel),
            "Entry Live-Score": live_score,
            "Live-Score-Band": _score_band(live_score),
            "Entry Engine-Score": _num(context.get("engine_score"), None),
            "Entry Guarded Score": guarded_score,
            "Guarded-Score-Band": _score_band(guarded_score),
            "Entry Engine-Empfehlung": _text(context.get("engine_recommendation"), "n/a"),
            "Entry Guardrail": _text(context.get("guardrail"), "n/a"),
            "Entry Kontext-Anpassung": _num(context.get("context_adjustment"), None),
            "Entry Kontext-Verlässlichkeit": _text(context.get("context_confidence"), "n/a"),
            "Entry Marktregime": _text(context.get("market_regime"), "n/a"),
            "Entry Volatilitätsregime": _text(context.get("volatility_regime"), "n/a"),
            "Entry RS-Dynamik": _text(context.get("rs_dynamics"), "n/a"),
            "Entry Relative Stärke": _text(context.get("relative_strength"), "n/a"),
            "Entry Radar-Bucket": _text(context.get("radar_bucket"), "n/a"),
            "Entry Grade": _text(context.get("grade"), "n/a"),
            "Entry CRV": _num(context.get("crv"), None),
            "Entry Abstand": _text(context.get("entry_distance"), "n/a"),
            "Entry Setup-Alert": _text(context.get("setup_alert"), "n/a"),
            "Entry Gates": _text(context.get("active_gates"), "n/a"),
            "Entry Benchmark": _text(context.get("benchmark"), "n/a"),
            "Entry Horizont": _text(context.get("live_horizon"), "n/a"),
            "Kontext vorhanden": bool(context),
        })
        cycle_state[key] = {"partial_shares": 0, "partial_pnl": 0.0, "partial_count": 0}

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values("Exit-Zeit", ascending=False, na_position="last").reset_index(drop=True)
    return out


def _profit_factor(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return None
    gross_profit = vals[vals > 0].sum()
    gross_loss = abs(vals[vals < 0].sum())
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else None
    return float(gross_profit / gross_loss)


def summarize_trades(trades: pd.DataFrame | None) -> dict:
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        return {
            "closed_trades": 0, "wins": 0, "losses": 0, "win_rate": None,
            "total_pnl": 0.0, "avg_r": None, "median_r": None,
            "profit_factor": None, "avg_return_pct": None, "avg_hold_days": None,
            "context_coverage": 0.0, "sample_label": "Zu klein",
        }
    pnl = pd.to_numeric(trades.get("Gesamt P/L"), errors="coerce")
    rvals = pd.to_numeric(trades.get("Gesamt R"), errors="coerce")
    returns = pd.to_numeric(trades.get("Kapitalrendite %"), errors="coerce")
    holds = pd.to_numeric(trades.get("Haltedauer Tage"), errors="coerce")
    valid_pnl = pnl.dropna()
    wins = int((valid_pnl > 0).sum())
    losses = int((valid_pnl < 0).sum())
    context = trades.get("Kontext vorhanden", pd.Series(False, index=trades.index)).fillna(False).astype(bool)
    return {
        "closed_trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate": float((valid_pnl > 0).mean() * 100.0) if len(valid_pnl) else None,
        "total_pnl": float(valid_pnl.sum()) if len(valid_pnl) else 0.0,
        "avg_r": float(rvals.dropna().mean()) if len(rvals.dropna()) else None,
        "median_r": float(rvals.dropna().median()) if len(rvals.dropna()) else None,
        "profit_factor": _profit_factor(valid_pnl),
        "avg_return_pct": float(returns.dropna().mean()) if len(returns.dropna()) else None,
        "avg_hold_days": float(holds.dropna().mean()) if len(holds.dropna()) else None,
        "context_coverage": float(context.mean() * 100.0) if len(context) else 0.0,
        "sample_label": _sample_label(len(trades)),
    }


def segment_summary(trades: pd.DataFrame | None, column: str, label: str | None = None) -> pd.DataFrame:
    if not isinstance(trades, pd.DataFrame) or trades.empty or column not in trades.columns:
        return pd.DataFrame()
    work = trades.copy()
    work[column] = work[column].map(lambda x: "n/a" if _blank(x) else str(x).strip())
    work = work[work[column] != "n/a"]
    if work.empty:
        return pd.DataFrame()

    rows = []
    for value, group in work.groupby(column, dropna=False):
        pnl = pd.to_numeric(group.get("Gesamt P/L"), errors="coerce").dropna()
        rvals = pd.to_numeric(group.get("Gesamt R"), errors="coerce").dropna()
        returns = pd.to_numeric(group.get("Kapitalrendite %"), errors="coerce").dropna()
        rows.append({
            label or column: value,
            "Trades": int(len(group)),
            "Trefferquote %": round(float((pnl > 0).mean() * 100.0), 1) if len(pnl) else None,
            "Ø R": round(float(rvals.mean()), 3) if len(rvals) else None,
            "Median R": round(float(rvals.median()), 3) if len(rvals) else None,
            "Ø Kapitalrendite %": round(float(returns.mean()), 2) if len(returns) else None,
            "Profit Factor": None if _profit_factor(pnl) is None else round(float(_profit_factor(pnl)), 2) if math.isfinite(_profit_factor(pnl)) else float("inf"),
            "Stichprobe": _sample_label(len(group)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Trades", "Ø R"], ascending=[False, False], na_position="last").reset_index(drop=True)


def _event_time_frame(event_df: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(event_df, pd.DataFrame) or event_df.empty:
        return pd.DataFrame()
    events = event_df.copy()
    if "Ereignis" not in events.columns or "Ticker" not in events.columns:
        return pd.DataFrame()
    events = events[events["Ereignis"].astype(str) == "Exit Engine 2.0"].copy()
    if events.empty:
        return events
    events["__dt"] = events.get("Zeit", pd.Series(index=events.index, dtype=object)).map(_dt)
    events["Ticker"] = events["Ticker"].astype(str).str.upper().str.strip()
    return events.sort_values("__dt", ascending=True, na_position="last")


def build_exit_learning(trades: pd.DataFrame | None, event_df: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match v28.9 Exit Engine warnings to completed trades.

    Matching is intentionally conservative: an entry timestamp and a close
    timestamp must both be available, otherwise the trade is not guessed into
    an event window.
    """
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    events = _event_time_frame(event_df)
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    matched = []
    for _, trade in trades.iterrows():
        ticker = _text(trade.get("Ticker"), "").upper()
        watchlist = _text(trade.get("Watchlist"), "")
        entry_dt = trade.get("Entry-Zeit")
        close_dt = trade.get("Exit-Zeit")
        if not ticker or pd.isna(entry_dt) or pd.isna(close_dt):
            continue
        mask = (events["Ticker"] == ticker) & events["__dt"].notna() & (events["__dt"] >= entry_dt) & (events["__dt"] <= close_dt)
        if watchlist and "Watchlist" in events.columns:
            mask = mask & (events["Watchlist"].astype(str) == watchlist)
        ev = events[mask].copy()
        if ev.empty:
            continue
        first = ev.iloc[0]
        pressure_series = pd.to_numeric(ev.get("Exit-Druck 2.0"), errors="coerce") if "Exit-Druck 2.0" in ev.columns else pd.Series(dtype=float)
        max_pressure = float(pressure_series.max()) if len(pressure_series.dropna()) else None
        warning_r = _num(first.get("R"), None)
        final_r = _num(trade.get("Gesamt R"), None)
        r_change = final_r - warning_r if final_r is not None and warning_r is not None else None
        lead_days = max(0.0, (close_dt - first.get("__dt")).total_seconds() / 86400.0) if not pd.isna(first.get("__dt")) else None
        matched.append({
            "Ticker": ticker,
            "Erstwarnung": first.get("__dt"),
            "Exit": close_dt,
            "Erstaktion": _text(first.get("Aktion 2.0") or first.get("Status"), "n/a"),
            "Exit-Druck max": max_pressure,
            "R bei Warnung": warning_r,
            "P/L % bei Warnung": _num(first.get("P/L %"), None),
            "Schluss-R": final_r,
            "R-Veränderung danach": r_change,
            "Verschlechtert danach": bool(r_change < 0) if r_change is not None else None,
            "Vorlauf Tage": lead_days,
        })

    detail = pd.DataFrame(matched)
    if detail.empty:
        return detail, pd.DataFrame()

    summary_rows = []
    for action, group in detail.groupby("Erstaktion", dropna=False):
        wr = pd.to_numeric(group.get("R bei Warnung"), errors="coerce").dropna()
        fr = pd.to_numeric(group.get("Schluss-R"), errors="coerce").dropna()
        delta = pd.to_numeric(group.get("R-Veränderung danach"), errors="coerce").dropna()
        lead = pd.to_numeric(group.get("Vorlauf Tage"), errors="coerce").dropna()
        worsened = group.get("Verschlechtert danach", pd.Series(dtype=object)).dropna()
        summary_rows.append({
            "Erstaktion": action,
            "Trades": int(len(group)),
            "Ø R bei Warnung": round(float(wr.mean()), 3) if len(wr) else None,
            "Ø Schluss-R": round(float(fr.mean()), 3) if len(fr) else None,
            "Ø R-Veränderung danach": round(float(delta.mean()), 3) if len(delta) else None,
            "Verschlechtert danach %": round(float(pd.Series(worsened).astype(bool).mean() * 100.0), 1) if len(worsened) else None,
            "Ø Vorlauf Tage": round(float(lead.mean()), 1) if len(lead) else None,
            "Stichprobe": _sample_label(len(group)),
        })
    summary = pd.DataFrame(summary_rows).sort_values("Trades", ascending=False).reset_index(drop=True)
    detail = detail.sort_values("Exit", ascending=False).reset_index(drop=True)
    return detail, summary


def manual_learning_tags(journal_df: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(journal_df, pd.DataFrame) or journal_df.empty or "Erkenntnis" not in journal_df.columns:
        return pd.DataFrame()
    texts = [str(x).strip() for x in journal_df["Erkenntnis"].dropna().tolist() if str(x).strip()]
    if not texts:
        return pd.DataFrame()
    tags = {
        "FOMO / zu spät": ["fomo", "zu spät", "zu spaet", "hinterher"],
        "Entry / Timing": ["entry", "einstieg", "timing", "trigger"],
        "Stop / Risiko": ["stop", "risiko", "verlust", "invalid"],
        "Exit / zu früh": ["exit", "zu früh", "zu frueh", "verkauf"],
        "Positionsgröße": ["positionsgröße", "positionsgroesse", "stück", "stueck", "größe", "groesse"],
        "Marktumfeld": ["markt", "regime", "risk-off", "risk off"],
        "Earnings / Event": ["earnings", "zahlen", "event"],
        "Disziplin / Plan": ["disziplin", "plan", "regel", "geduld"],
    }
    counts = {key: 0 for key in tags}
    for text in texts:
        low = text.lower()
        for key, terms in tags.items():
            if any(term in low for term in terms):
                counts[key] += 1
    rows = [{"Lern-Thema": key, "Nennungen": val} for key, val in counts.items() if val > 0]
    return pd.DataFrame(rows).sort_values("Nennungen", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def _best_worst_observation(segment_tables: dict[str, pd.DataFrame]) -> tuple[dict | None, dict | None]:
    candidates = []
    for name, table in (segment_tables or {}).items():
        if not isinstance(table, pd.DataFrame) or table.empty or "Trades" not in table.columns or "Ø R" not in table.columns:
            continue
        first_col = table.columns[0]
        for _, row in table.iterrows():
            n = int(_num(row.get("Trades"), 0) or 0)
            avg_r = _num(row.get("Ø R"), None)
            if n < 5 or avg_r is None:
                continue
            candidates.append({"Bereich": name, "Segment": row.get(first_col), "Trades": n, "Ø R": avg_r})
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x["Ø R"])
    return candidates[-1], candidates[0]


def build_learning_package(journal_df: pd.DataFrame | None, event_df: pd.DataFrame | None = None) -> dict:
    trades = build_trade_dataset(journal_df)
    summary = summarize_trades(trades)
    segment_specs = [
        ("Setup / Radar-Bucket", "Entry Radar-Bucket"),
        ("Marktregime", "Entry Marktregime"),
        ("Volatilitätsregime", "Entry Volatilitätsregime"),
        ("RS-Dynamik", "Entry RS-Dynamik"),
        ("Live-Ampel", "Entry Live-Ampel"),
        ("Shadow vs Live", "Shadow vs Live"),
        ("Live-Score-Band", "Live-Score-Band"),
        ("Guarded-Score-Band", "Guarded-Score-Band"),
        ("Guardrail", "Entry Guardrail"),
        ("Grade", "Entry Grade"),
    ]
    segments = {label: segment_summary(trades, col, label) for label, col in segment_specs}
    segments = {k: v for k, v in segments.items() if isinstance(v, pd.DataFrame) and not v.empty}
    exit_detail, exit_summary = build_exit_learning(trades, event_df)
    manual_tags = manual_learning_tags(journal_df)

    insights = []
    n = int(summary.get("closed_trades") or 0)
    coverage = float(summary.get("context_coverage") or 0.0)
    if n == 0:
        insights.append("Noch keine gültig geschlossenen Trades: Die Learning Engine sammelt erst Daten und verändert keine Regeln.")
    elif n < 5:
        insights.append(f"Nur {n} geschlossene Trades: Ergebnisse sind Einzelfallbeobachtungen, noch keine Kalibrierungsbasis.")
    elif n < 15:
        insights.append(f"{n} geschlossene Trades: erste Hypothesen sind sichtbar, Schwellen/Weights sollten daraus noch nicht automatisch geändert werden.")
    else:
        insights.append(f"{n} geschlossene Trades: die Datenbasis ist {_sample_label(n).lower()} und erlaubt erste strukturierte Kalibrierungsvergleiche.")

    if n and coverage < 50:
        insights.append(f"Entry-Kontext ist erst bei {coverage:.0f}% der geschlossenen Trades vorhanden. Historische Trades bleiben nutzbar für P/L/R, aber nicht für saubere Setup-/Regime-Vergleiche.")
    elif n:
        insights.append(f"Entry-Kontextabdeckung {coverage:.0f}%: Setup-/Regime-Vergleiche werden zunehmend belastbar.")

    best, worst = _best_worst_observation(segments)
    if best is not None:
        insights.append(f"Stärkstes Segment mit mindestens 5 Trades: {best['Bereich']} · {best['Segment']} mit Ø {best['Ø R']:+.2f}R bei n={best['Trades']}.")
    if worst is not None and (best is None or worst != best):
        insights.append(f"Schwächstes Segment mit mindestens 5 Trades: {worst['Bereich']} · {worst['Segment']} mit Ø {worst['Ø R']:+.2f}R bei n={worst['Trades']}.")

    if isinstance(exit_detail, pd.DataFrame) and not exit_detail.empty:
        delta = pd.to_numeric(exit_detail.get("R-Veränderung danach"), errors="coerce").dropna()
        if len(delta) >= 5:
            worsened = float((delta < 0).mean() * 100.0)
            insights.append(f"Exit Engine 2.0 konnte bei {len(exit_detail)} abgeschlossenen Trades zeitlich zugeordnet werden; nach der Erstwarnung verschlechterte sich R in {worsened:.0f}% der auswertbaren Fälle.")
    else:
        insights.append("Exit-Engine-Lerncheck noch ohne sichere Trade-Zuordnung. Für Matching wird bewusst ein belastbarer Entry-Zeitpunkt verlangt.")

    return {
        "trades": trades,
        "summary": summary,
        "segments": segments,
        "exit_detail": exit_detail,
        "exit_summary": exit_summary,
        "manual_tags": manual_tags,
        "insights": insights,
    }
