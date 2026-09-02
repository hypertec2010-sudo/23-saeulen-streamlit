from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

STORE = Path("shadow_performance_v287.json")
NAMESPACE = "shadow_performance_v288"
HORIZONS = (1, 3, 5, 10, 20)
_STORAGE = None


def configure_storage(storage=None):
    """Bind the central app storage when available.

    v28.8 keeps a local JSON fallback for compatibility, but prefers the
    existing StorageManager so calibration data survives restarts/deploys.
    """
    global _STORAGE
    _STORAGE = storage


def _first(row, names, default=None):
    for name in names:
        try:
            value = row.get(name) if hasattr(row, "get") else None
        except Exception:
            value = None
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        if str(value).strip() not in ("", "-", "nan", "None"):
            return value
    return default


def _num(value):
    try:
        s = str(value).replace("%", "").replace(",", ".").replace("/100", "").strip()
        return float(s)
    except Exception:
        return None


def _blank(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() in ("", "-", "nan", "None", "n/a", "N/A")


def _local_load():
    try:
        payload = json.loads(STORE.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("events", [])
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _load():
    if _STORAGE is not None:
        try:
            payload = _STORAGE.load_namespace(NAMESPACE, default=None)
            if isinstance(payload, dict):
                rows = payload.get("events", [])
                if isinstance(rows, list):
                    return rows
            if isinstance(payload, list):
                return payload
        except Exception:
            pass

    rows = _local_load()
    if rows and _STORAGE is not None:
        try:
            _STORAGE.save_namespace(
                NAMESPACE,
                {
                    "schema": "v28.8",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "events": rows,
                },
            )
        except Exception:
            pass
    return rows


def _save(rows):
    rows = list(rows or [])[-1500:]
    saved = False
    if _STORAGE is not None:
        try:
            _STORAGE.save_namespace(
                NAMESPACE,
                {
                    "schema": "v28.8",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "events": rows,
                },
            )
            saved = True
        except Exception:
            saved = False
    if not saved:
        try:
            STORE.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass


def _direction(live, shadow):
    rank = {
        "\U0001f534": 0,
        "\u26aa": 1,
        "\U0001f7e1": 2,
        "\U0001f7e2": 3,
    }
    live_rank = rank.get(str(live), 1)
    shadow_rank = rank.get(str(shadow), 1)
    if shadow_rank > live_rank:
        return "Aufwertung"
    if shadow_rank < live_rank:
        return "Abwertung"
    return "Unver\u00e4ndert"


def _metadata_from_shadow_row(data):
    """Extract calibration metadata without inventing missing legacy values."""
    return {
        "base_score": _num(_first(data, ["Basis-Score", "Live-Score", "base_score"])),
        "engine_score": _num(_first(data, ["Engine-Score", "engine_score"])),
        "guarded_score": _num(_first(data, ["Guarded Engine", "Guarded Engine-Score", "guarded_score"])),
        "engine_recommendation": _first(data, ["Engine-Empfehlung", "engine_recommendation"], ""),
        "context": _first(data, ["Kontext", "Kontext-Beitr\u00e4ge", "context"], ""),
        "context_adjustment": _num(_first(data, ["Kontext-Anpassung", "context_adjustment"])),
        "context_confidence": _first(data, ["Kontext-Verl\u00e4sslichkeit", "context_confidence"], ""),
        "guardrail": _first(data, ["Engine-Guardrail", "guardrail"], ""),
        "rs_dynamics": _first(data, ["RS-Dynamik", "rs_dynamics"], ""),
        "market_regime": _first(data, ["Marktregime", "market_regime"], ""),
        "volatility_regime": _first(data, ["Volatilit\u00e4tsregime", "volatility_regime"], ""),
        "live_horizon": _first(data, ["Live-Horizont", "live_horizon"], ""),
        "benchmark": _first(data, ["Prim\u00e4rbenchmark", "Benchmark", "benchmark"], ""),
        "active_gates": _first(data, ["Aktive Einstiegsgates", "active_gates"], ""),
        "gate_details": _first(data, ["Gate-Details", "gate_details"], ""),
        "crv": _num(_first(data, ["CRV", "crv"])),
        "trigger_component": _num(_first(data, ["Trigger-Komponente", "__trigger_component", "trigger_component"])),
        "chart_component": _num(_first(data, ["Chart-Komponente", "__chart_component", "chart_component"])),
        "trend_component": _num(_first(data, ["Trend-Komponente", "__trend_component", "trend_component"])),
        "timing_component": _num(_first(data, ["Timing-Komponente", "__timing_component", "timing_component"])),
        "crv_component": _num(_first(data, ["CRV-Komponente", "__crv_component", "crv_component"])),
    }


def sync_events(shadow_df):
    rows = _load()
    by_id = {str(r.get("id")): r for r in rows if isinstance(r, dict) and r.get("id")}
    changed = False

    if not isinstance(shadow_df, pd.DataFrame) or shadow_df.empty:
        return pd.DataFrame(rows)

    for _, row in shadow_df.iterrows():
        data = row.to_dict()
        ticker = str(_first(data, ["Ticker", "ticker", "Symbol"], "")).upper().strip()
        ts = str(_first(data, ["Zeit", "Zeitpunkt", "Timestamp", "timestamp", "Datum", "date", "ts"], ""))
        live = str(_first(data, ["Live-Ampel", "Live Ampel", "Ampel", "live_ampel"], "-"))
        shadow = str(_first(data, ["Shadow-Ampel", "Shadow Ampel", "Engine-Ampel", "shadow_ampel"], "-"))
        price = _num(_first(data, ["Kurs", "Preis", "Price", "price", "Event-Kurs", "event_price"]))
        if not ticker or live == "-" or shadow == "-":
            continue
        if not ts:
            ts = datetime.now().isoformat(timespec="seconds")

        event_id = f"{ticker}|{ts}|{live}|{shadow}"
        metadata = _metadata_from_shadow_row(data)
        if event_id in by_id:
            target = by_id[event_id]
            if _blank(target.get("event_price")) and price is not None:
                target["event_price"] = price
                changed = True
            for key, value in metadata.items():
                if _blank(target.get(key)) and not _blank(value):
                    target[key] = value
                    changed = True
            continue

        rec = {
            "id": event_id,
            "ticker": ticker,
            "event_ts": ts,
            "live": live,
            "shadow": shadow,
            "richtung": _direction(live, shadow),
            "event_price": price,
            **metadata,
        }
        for h in HORIZONS:
            rec[f"r{h}"] = None
            rec[f"mfe{h}"] = None
            rec[f"mae{h}"] = None
        rows.append(rec)
        by_id[event_id] = rec
        changed = True

    if changed:
        _save(rows)
    return pd.DataFrame(rows)


def _event_date(value):
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if isinstance(dt, pd.Timestamp) and dt.tzinfo is not None:
            dt = dt.tz_localize(None)
        return dt
    except Exception:
        return pd.NaT


def _history_period_for_items(items):
    oldest = None
    now = pd.Timestamp.now().tz_localize(None)
    for item in items:
        dt = _event_date(item.get("event_ts"))
        if pd.isna(dt):
            continue
        oldest = dt if oldest is None or dt < oldest else oldest
    if oldest is None:
        return "6mo"
    age_days = max(0, int((now - oldest).days))
    if age_days > 650:
        return "5y"
    if age_days > 300:
        return "2y"
    if age_days > 150:
        return "1y"
    return "6mo"


def _direction_sign(direction):
    return 1.0 if str(direction) == "Aufwertung" else (-1.0 if str(direction) == "Abwertung" else 0.0)


def refresh_forward_returns(events, provider):
    """Refresh only on explicit user action.

    v28.8 fixes non-trading-day indexing and adds directional MFE/MAE. No
    provider calls are made by dashboard rendering itself.
    """
    rows = events.to_dict("records") if isinstance(events, pd.DataFrame) else list(events or [])
    by_ticker = {}
    for rec in rows:
        by_ticker.setdefault(str(rec.get("ticker", "")), []).append(rec)

    for ticker, items in by_ticker.items():
        if not ticker:
            continue
        try:
            period = _history_period_for_items(items)
            hist = provider.get_history(ticker, period=period, auto_adjust=True)
            if hist is None or len(hist) < 2:
                continue
            frame = hist.copy()
            frame.index = pd.to_datetime(frame.index)
            try:
                if getattr(frame.index, "tz", None) is not None:
                    frame.index = frame.index.tz_localize(None)
            except Exception:
                pass
            close = pd.to_numeric(frame.get("Close"), errors="coerce").dropna()
            if close.empty:
                continue
            high = pd.to_numeric(frame.get("High", frame.get("Close")), errors="coerce")
            low = pd.to_numeric(frame.get("Low", frame.get("Close")), errors="coerce")

            for rec in items:
                dt = _event_date(rec.get("event_ts"))
                if pd.isna(dt):
                    continue
                day = dt.normalize()
                pos = int(close.index.searchsorted(day, side="left"))
                if pos >= len(close):
                    continue
                same_day = bool(close.index[pos].normalize() == day)
                first_future = pos + 1 if same_day else pos
                if first_future >= len(close):
                    continue

                base = _num(rec.get("event_price"))
                if base is None or base <= 0:
                    base_pos = pos if same_day else max(0, pos - 1)
                    base = float(close.iloc[base_pos])
                    rec["event_price"] = round(base, 6)

                sign = _direction_sign(rec.get("richtung"))
                # Equal-state records are persisted only as episode boundaries.
                # They do not need forward-return calculations.
                if sign == 0:
                    rec["last_refresh"] = datetime.now().isoformat(timespec="seconds")
                    continue
                for horizon in HORIZONS:
                    target_pos = first_future + horizon - 1
                    if target_pos >= len(close):
                        continue
                    target_price = float(close.iloc[target_pos])
                    raw_return = (target_price / float(base) - 1.0) * 100.0
                    rec[f"r{horizon}"] = round(raw_return, 3)

                    if sign == 0:
                        continue
                    future_index = close.index[first_future : target_pos + 1]
                    if len(future_index) == 0:
                        continue
                    try:
                        hi = pd.to_numeric(high.reindex(future_index), errors="coerce").dropna()
                        lo = pd.to_numeric(low.reindex(future_index), errors="coerce").dropna()
                        if hi.empty or lo.empty:
                            continue
                        high_ret = (float(hi.max()) / float(base) - 1.0) * 100.0
                        low_ret = (float(lo.min()) / float(base) - 1.0) * 100.0
                        if sign > 0:
                            mfe = high_ret
                            mae = low_ret
                        else:
                            mfe = -low_ret
                            mae = -high_ret
                        rec[f"mfe{horizon}"] = round(mfe, 3)
                        rec[f"mae{horizon}"] = round(mae, 3)
                    except Exception:
                        continue
                rec["last_refresh"] = datetime.now().isoformat(timespec="seconds")
        except Exception:
            continue

    _save(rows)
    return pd.DataFrame(rows)


def _confidence_label(n):
    n = int(n or 0)
    if n < 5:
        return "Zu klein"
    if n < 15:
        return "Fr\u00fchphase"
    if n < 30:
        return "Mittel"
    if n < 60:
        return "Gut"
    return "Breiter"


def _fmt_pct(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):+.{digits}f}%"
    except Exception:
        return "n/a"


def _fmt_rate(value):
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value) * 100.0:.0f}%"
    except Exception:
        return "n/a"


def build_dashboard(events):
    if not isinstance(events, pd.DataFrame) or events.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = events.copy()
    if "richtung" in df.columns:
        df = df[df["richtung"].isin(["Aufwertung", "Abwertung"])].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    out = []
    for direction, group in df.groupby("richtung"):
        row = {"Shadow-Richtung": direction, "Events": len(group)}
        sign = _direction_sign(direction)
        for horizon in HORIZONS:
            vals = pd.to_numeric(group.get(f"r{horizon}"), errors="coerce").dropna()
            row[f"{horizon}T \u00d8"] = "n/a" if vals.empty else f"{vals.mean():+.2f}%"
            if vals.empty or sign == 0:
                row[f"{horizon}T Treffer"] = "n/a"
            else:
                edge = vals * sign
                row[f"{horizon}T Treffer"] = f"{((edge > 0).mean() * 100):.0f}%"
        out.append(row)

    rename = {
        "ticker": "Ticker",
        "event_ts": "Event",
        "live": "Live",
        "shadow": "Shadow",
        "richtung": "Richtung",
        "event_price": "Event-Kurs",
        "guarded_score": "Guarded Score",
        "market_regime": "Marktregime",
        "rs_dynamics": "RS-Dynamik",
    }
    rename.update({f"r{h}": f"{h}T %" for h in HORIZONS})
    detail = df.rename(columns=rename)
    cols = [
        "Ticker", "Event", "Live", "Shadow", "Richtung", "Event-Kurs", "Guarded Score",
        "Marktregime", "RS-Dynamik",
    ] + [f"{h}T %" for h in HORIZONS]
    return pd.DataFrame(out), detail[[c for c in cols if c in detail.columns]]


def _score_band(value):
    score = _num(value)
    if score is None:
        return "Keine Score-Daten"
    if score < 28:
        return "0-27 (Rot)"
    if score < 55:
        return "28-54 (Weiss)"
    if score < 72:
        return "55-71 (Gelb)"
    return "72-100 (Gruen)"


def _rs_bucket(value):
    text = str(value or "").lower()
    if "verbess" in text:
        return "Verbessert"
    if "verschlechter" in text:
        return "Verschlechtert"
    if "stabil" in text:
        return "Stabil"
    return "Keine Daten"


def _guardrail_bucket(value, recommendation=""):
    text = str(value or "").lower()
    rec = str(recommendation or "").lower()
    if not text or text in ("-", "n/a"):
        if "guardrail" in rec or "blockiert" in rec or "begrenzt" in rec:
            return "Guardrail aktiv (Legacy-Details fehlen)"
        return "Keine Daten"
    if "keine guardrail" in text:
        return "Keine Guardrail-Bremse"
    hard = "hart" in text or "invalid" in text
    trigger = "trigger" in text
    chart = "chart" in text
    crv = "crv" in text
    active = sum([hard, trigger, chart, crv])
    if active > 1:
        return "Mehrere Guardrails"
    if hard:
        return "Hartes Gate / Invalidierung"
    if trigger:
        return "Trigger fehlt"
    if chart:
        return "Chart-Bremse"
    if crv:
        return "CRV-Bremse"
    return "Sonstige Guardrail"


def _episode_entries(events):
    """Collapse correlated state changes into divergence episodes.

    Multiple guarded-score changes while the same ticker remains an upgrade or
    downgrade are not independent samples. Return-to-equal records reset the
    episode; an opposite divergence starts a new one immediately.
    """
    if not isinstance(events, pd.DataFrame) or events.empty:
        return pd.DataFrame()
    df = events.copy()
    if "event_ts" not in df.columns or "ticker" not in df.columns or "richtung" not in df.columns:
        return df[df.get("richtung", pd.Series(index=df.index, dtype=object)).isin(["Aufwertung", "Abwertung"])].copy()
    df["__dt"] = pd.to_datetime(df["event_ts"], errors="coerce", utc=True)
    df = df.sort_values(["ticker", "__dt", "event_ts"], kind="stable")
    keep = []
    state = {}
    for idx, row in df.iterrows():
        ticker = str(row.get("ticker") or "").strip().upper()
        direction = str(row.get("richtung") or "")
        if not ticker:
            continue
        active = state.get(ticker)
        if direction not in ("Aufwertung", "Abwertung"):
            state[ticker] = None
            continue
        if active != direction:
            keep.append(idx)
            state[ticker] = direction
    if not keep:
        return pd.DataFrame(columns=[c for c in df.columns if c != "__dt"])
    return df.loc[keep].drop(columns=["__dt"], errors="ignore").reset_index(drop=True)


def _prepare_calibration_frame(events, horizon):
    if not isinstance(events, pd.DataFrame) or events.empty:
        return pd.DataFrame()
    horizon = int(horizon)
    df = events.copy()
    df["raw_return"] = pd.to_numeric(df.get(f"r{horizon}"), errors="coerce")
    df["direction_sign"] = df.get("richtung", pd.Series(index=df.index, dtype=object)).map(
        {"Aufwertung": 1.0, "Abwertung": -1.0}
    )
    df["shadow_edge"] = df["raw_return"] * df["direction_sign"]
    df["hit"] = df["shadow_edge"] > 0
    df["mfe"] = pd.to_numeric(df.get(f"mfe{horizon}"), errors="coerce")
    df["mae"] = pd.to_numeric(df.get(f"mae{horizon}"), errors="coerce")
    df["guarded_num"] = pd.to_numeric(df.get("guarded_score"), errors="coerce")
    df["engine_num"] = pd.to_numeric(df.get("engine_score"), errors="coerce")
    df["base_num"] = pd.to_numeric(df.get("base_score"), errors="coerce")
    df["score_band"] = df.get("guarded_score", pd.Series(index=df.index, dtype=object)).map(_score_band)
    df["rs_bucket"] = df.get("rs_dynamics", pd.Series(index=df.index, dtype=object)).map(_rs_bucket)
    df["guardrail_bucket"] = [
        _guardrail_bucket(v, r)
        for v, r in zip(
            df.get("guardrail", pd.Series([""] * len(df), index=df.index)),
            df.get("engine_recommendation", pd.Series([""] * len(df), index=df.index)),
        )
    ]
    return df


def _segment_summary(df, column, label):
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    work = df[df["raw_return"].notna() & df["direction_sign"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    rows = []
    for value, group in work.groupby(column, dropna=False):
        value_text = str(value) if not _blank(value) else "Keine Daten"
        n = len(group)
        up = group[group.get("richtung") == "Aufwertung"]
        down = group[group.get("richtung") == "Abwertung"]
        rows.append(
            {
                label: value_text,
                "Events": n,
                "Aufwertungen": len(up),
                "Abwertungen": len(down),
                "\u00d8 Shadow-Edge": _fmt_pct(group["shadow_edge"].mean()),
                "Median Edge": _fmt_pct(group["shadow_edge"].median()),
                "Trefferquote": _fmt_rate(group["hit"].mean()),
                "\u00d8 Return Aufwertung": _fmt_pct(up["raw_return"].mean()) if len(up) else "n/a",
                "\u00d8 Return Abwertung": _fmt_pct(down["raw_return"].mean()) if len(down) else "n/a",
                "Stichprobe": _confidence_label(n),
            }
        )
    return pd.DataFrame(rows).sort_values(["Events", label], ascending=[False, True]).reset_index(drop=True)


def _horizon_summary(events):
    rows = []
    episodes = _episode_entries(events)
    for horizon in HORIZONS:
        df = _prepare_calibration_frame(episodes, horizon)
        work = df[df["raw_return"].notna() & df["direction_sign"].notna()].copy()
        up = work[work.get("richtung") == "Aufwertung"]
        down = work[work.get("richtung") == "Abwertung"]
        rows.append(
            {
                "Horizont": f"{horizon}T",
                "Events": len(work),
                "Aufwertungen": len(up),
                "Abwertungen": len(down),
                "\u00d8 Shadow-Edge": _fmt_pct(work["shadow_edge"].mean()) if len(work) else "n/a",
                "Median Edge": _fmt_pct(work["shadow_edge"].median()) if len(work) else "n/a",
                "Trefferquote": _fmt_rate(work["hit"].mean()) if len(work) else "n/a",
                "\u00d8 MFE": _fmt_pct(work["mfe"].mean()) if work["mfe"].notna().any() else "n/a",
                "\u00d8 MAE": _fmt_pct(work["mae"].mean()) if work["mae"].notna().any() else "n/a",
                "Stichprobe": _confidence_label(len(work)),
            }
        )
    return pd.DataFrame(rows)


def _guardrail_backtest(df):
    if df.empty:
        return pd.DataFrame()
    work = df[
        df["raw_return"].notna()
        & df["engine_num"].notna()
        & df["guarded_num"].notna()
        & (df["engine_num"] > df["guarded_num"])
    ].copy()
    if work.empty:
        return pd.DataFrame()
    work["score_reduction"] = work["engine_num"] - work["guarded_num"]
    rows = []
    for bucket, group in work.groupby("guardrail_bucket"):
        n = len(group)
        rows.append(
            {
                "Guardrail": bucket,
                "Events": n,
                "\u00d8 Score-Bremse": f"-{group['score_reduction'].mean():.1f} Pkt",
                "\u00d8 Folge-Return": _fmt_pct(group["raw_return"].mean()),
                "Defensiv best\u00e4tigt (Return <= 0)": _fmt_rate((group["raw_return"] <= 0).mean()),
                "Positiv danach": _fmt_rate((group["raw_return"] > 0).mean()),
                "Stichprobe": _confidence_label(n),
            }
        )
    return pd.DataFrame(rows).sort_values("Events", ascending=False).reset_index(drop=True)


def _metadata_coverage(events):
    if not isinstance(events, pd.DataFrame) or events.empty:
        return pd.DataFrame()
    fields = [
        ("Guarded Score", "guarded_score"),
        ("Raw Engine Score", "engine_score"),
        ("Guardrail", "guardrail"),
        ("RS-Dynamik", "rs_dynamics"),
        ("Marktregime", "market_regime"),
        ("Volatilit\u00e4tsregime", "volatility_regime"),
        ("Trigger-Komponente", "trigger_component"),
        ("Chart-Komponente", "chart_component"),
    ]
    total = max(1, len(events))
    rows = []
    for label, col in fields:
        if col not in events.columns:
            count = 0
        else:
            count = sum(not _blank(v) for v in events[col].tolist())
        rows.append({"Merkmal": label, "Vorhanden": count, "Abdeckung": f"{count / total * 100:.0f}%"})
    return pd.DataFrame(rows)


def _calibration_recommendations(df, score_table, guardrail_table, horizon):
    rows = []
    work = df[df["raw_return"].notna() & df["direction_sign"].notna()].copy()
    n = len(work)
    if n < 5:
        rows.append(
            {
                "Bereich": "Gesamt",
                "Status": "Datensammlung",
                "Beobachtung": f"Nur {n} auswertbare {horizon}T-Ereignisse.",
                "Konsequenz": "Keine Schwellen oder Gewichte \u00e4ndern; weitere echte Shadow-Ereignisse sammeln.",
            }
        )
    else:
        hit = float(work["hit"].mean())
        edge = float(work["shadow_edge"].mean())
        if n < 15:
            status = "Fr\u00fche Evidenz"
            consequence = "Nur beobachten; Stichprobe ist noch zu klein f\u00fcr produktive Anpassungen."
        elif hit >= 0.60 and edge > 0:
            status = "Positives Signal"
            consequence = "Kalibrierungsrichtung ist plausibel; vor Cutover weitere Horizonte/Regime best\u00e4tigen."
        elif hit <= 0.45 or edge <= 0:
            status = "Pr\u00fcfbedarf"
            consequence = "Keine Aufwertung der Engine; Gewichte und Schwellen in v28.8 weiter beobachten."
        else:
            status = "Neutral"
            consequence = "Noch kein klarer Vorteil gegen die Live-Basis; Daten weiter sammeln."
        rows.append(
            {
                "Bereich": "Gesamt",
                "Status": status,
                "Beobachtung": f"{n} Events, Treffer {hit * 100:.0f}%, \u00d8 Edge {edge:+.2f}% auf {horizon}T.",
                "Konsequenz": consequence,
            }
        )

    if isinstance(score_table, pd.DataFrame) and not score_table.empty:
        for _, row in score_table.iterrows():
            band = str(row.get("Guarded Score-Band", ""))
            try:
                events_n = int(row.get("Events", 0) or 0)
            except Exception:
                events_n = 0
            if events_n < 8 or band == "Keine Score-Daten":
                continue
            rate_text = str(row.get("Trefferquote", "n/a")).replace("%", "")
            edge_text = str(row.get("\u00d8 Shadow-Edge", "n/a")).replace("%", "").replace(",", ".")
            try:
                rate = float(rate_text)
                edge = float(edge_text)
            except Exception:
                continue
            if rate >= 60 and edge > 0:
                status = "Band best\u00e4tigt"
                consequence = "Band vorerst beibehalten; erst bei gr\u00f6\u00dferer Stichprobe Schwellen feinjustieren."
            elif rate < 45 or edge <= 0:
                status = "Band pr\u00fcfen"
                consequence = "Schwelle nicht lockern; bei mehr Daten Score-Band bzw. Kontextgewichtung \u00fcberpr\u00fcfen."
            else:
                continue
            rows.append(
                {
                    "Bereich": band,
                    "Status": status,
                    "Beobachtung": f"{events_n} Events, Treffer {rate:.0f}%, \u00d8 Edge {edge:+.2f}%.",
                    "Konsequenz": consequence,
                }
            )

    if isinstance(guardrail_table, pd.DataFrame) and not guardrail_table.empty:
        total_guard = int(pd.to_numeric(guardrail_table.get("Events"), errors="coerce").fillna(0).sum())
        if total_guard >= 8:
            rows.append(
                {
                    "Bereich": "Guardrails",
                    "Status": "Backtest aktiv",
                    "Beobachtung": f"{total_guard} Ereignisse mit messbarer Score-Bremse k\u00f6nnen separat bewertet werden.",
                    "Konsequenz": "Defensiv-best\u00e4tigt-Quote beobachten; Guardrails bleiben bis zu klarer Evidenz unver\u00e4ndert.",
                }
            )

    return pd.DataFrame(rows)


def build_calibration(events, horizon=10):
    """Event-based calibration/backtest for v28.8.

    This is intentionally observational. It uses only recorded Shadow events
    and their later market path; it never changes live scores or thresholds.
    """
    horizon = int(horizon)
    if horizon not in HORIZONS:
        horizon = 10
    if not isinstance(events, pd.DataFrame) or events.empty:
        return {
            "overview": {},
            "horizons": pd.DataFrame(),
            "score_bands": pd.DataFrame(),
            "guardrails": pd.DataFrame(),
            "guardrail_backtest": pd.DataFrame(),
            "rs": pd.DataFrame(),
            "market": pd.DataFrame(),
            "volatility": pd.DataFrame(),
            "recommendations": pd.DataFrame(),
            "coverage": pd.DataFrame(),
        }

    directional_raw = events[events.get("richtung", pd.Series(index=events.index, dtype=object)).isin(["Aufwertung", "Abwertung"])].copy()
    episodes = _episode_entries(events)
    df = _prepare_calibration_frame(episodes, horizon)
    work = df[df["raw_return"].notna() & df["direction_sign"].notna()].copy()
    up = work[work.get("richtung") == "Aufwertung"]
    down = work[work.get("richtung") == "Abwertung"]
    n = len(work)
    overview = {
        "events_raw": len(directional_raw),
        "events_total": len(episodes),
        "events_evaluable": n,
        "up": len(up),
        "down": len(down),
        "hit_rate": None if not n else float(work["hit"].mean()),
        "avg_edge": None if not n else float(work["shadow_edge"].mean()),
        "median_edge": None if not n else float(work["shadow_edge"].median()),
        "avg_mfe": None if not work["mfe"].notna().any() else float(work["mfe"].mean()),
        "avg_mae": None if not work["mae"].notna().any() else float(work["mae"].mean()),
        "sample": _confidence_label(n),
        "horizon": horizon,
    }

    score_table = _segment_summary(df, "score_band", "Guarded Score-Band")
    guardrail_table = _segment_summary(df, "guardrail_bucket", "Guardrail-Segment")
    rs_table = _segment_summary(df, "rs_bucket", "RS-Dynamik")
    market_table = _segment_summary(df, "market_regime", "Marktregime")
    vol_table = _segment_summary(df, "volatility_regime", "Volatilit\u00e4tsregime")
    guardrail_bt = _guardrail_backtest(df)
    recommendations = _calibration_recommendations(df, score_table, guardrail_bt, horizon)

    return {
        "overview": overview,
        "horizons": _horizon_summary(events),
        "score_bands": score_table,
        "guardrails": guardrail_table,
        "guardrail_backtest": guardrail_bt,
        "rs": rs_table,
        "market": market_table,
        "volatility": vol_table,
        "recommendations": recommendations,
        "coverage": _metadata_coverage(episodes),
    }


def best_mature_horizon(events, minimum=5, default=5):
    if not isinstance(events, pd.DataFrame) or events.empty:
        return int(default)
    episodes = _episode_entries(events)
    for horizon in reversed(HORIZONS):
        try:
            count = pd.to_numeric(episodes.get(f"r{horizon}"), errors="coerce").notna().sum()
        except Exception:
            count = 0
        if int(count) >= int(minimum):
            return int(horizon)
    for horizon in HORIZONS:
        try:
            count = pd.to_numeric(episodes.get(f"r{horizon}"), errors="coerce").notna().sum()
        except Exception:
            count = 0
        if int(count) > 0:
            return int(horizon)
    return int(default)
