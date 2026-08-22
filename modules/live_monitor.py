"""Live-Screener, Ampel-Hysterese, Trade-State und Statushistorie.

Dieses Modul wird von app.py ueber configure_context mit den bestehenden
Analyse-Callbacks verbunden. Dadurch bleibt die Analyse-Engine kompatibel,
waehrend der Live-Monitor separat wartbar ist.
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from .live_change_explainer import build_change_explanation

_CONTEXT = {}

def configure_context(**kwargs):
    global _CONTEXT
    _CONTEXT.update(kwargs)
    globals().update(kwargs)

def _v212_monitor_status_from_decision(result, decision, style_name="Ausgewogen", watchlist_meta=None, live_horizon="Swing / 1-4 Wochen"):
    """Verdichtet Radar-/Alert-Logik zu einer Live-Watchlist-Ampel.

    Läuft nur in der geöffneten App und nutzt dieselben Bausteine wie Radar und Setup-Alerts.
    """
    r = result or {}
    d = decision or build_professional_radar_decision_v18(r, style_name)
    # v24.0: live_short_term muss im Scope der Status-/Scorefunktion definiert sein.
    # In v23.11 wurde die Variable zwar spaeter verwendet, aber nicht gesetzt; dadurch
    # brachen nahezu alle Watchlist-Ticker mit NameError ab.
    live_horizon_text = str(live_horizon or "").strip().lower()
    live_short_term = bool(
        live_horizon_text.startswith("kurzfrist")
        or "trading" in live_horizon_text
        or "short" in live_horizon_text
    )
    ticker = str(r.get("ticker") or r.get("Ticker") or "").strip().upper()
    # v22.2: In der Live-Watchlist nicht den Ticker als Namen anzeigen.
    # analyze_stock liefert die Firmendaten meist im Result-/info-Objekt; der robuste
    # Radar-Name-Resolver vermeidet Fallbacks wie Name = Ticker.
    try:
        name = radar_company_display_name_v15237(r, ticker, 36)
    except Exception:
        name = str(r.get("company_name") or r.get("longName") or r.get("shortName") or r.get("name") or r.get("Name") or ticker or "-").strip()
        if name.strip().upper() == ticker:
            name = str((r.get("info", {}) or {}).get("longName") or (r.get("info", {}) or {}).get("shortName") or ticker).strip()
    price = _v210_alert_price(r)
    bucket = str(d.get("bucket") or "-").strip()
    grade = str(d.get("grade") or "-").strip().upper()
    crv = d.get("crv")
    try:
        if crv in {"", "-", "n/a", None}:
            crv_float = None
        else:
            crv_float = float(str(crv).replace(",", "."))
    except Exception:
        crv_float = None

    alerts = build_setup_alerts_v210(r, style_name=style_name, decision=d)
    alert_types = [str(a.get("Alert-Typ") or "") for a in alerts]
    alert_text = " · ".join(alert_types[:2]) if alert_types else "-"

    gate = str(d.get("gate_reasons") or "")
    gate_low = gate.lower().strip()
    hard_gate = bool(gate_low and gate_low not in {"keine harten gates", "keine harten gate", "-", "nan", "none"})

    # v22.1: Exit-/Schutzampel darf den Live-Kauftrigger nicht allein rot faerben.
    # Sie ist Positions-/Stop-Risikohinweis, aber kein Einstiegsgate. Rot bleibt nur
    # bei echter Invalidierung oder einem nicht-exitbezogenen harten Gate.
    exit_gate_terms = [
        "exit", "schutz", "stop prüfen", "stop pruefen", "gewinnschutz",
        "tactical", "de-risk", "derisk", "teilverkauf", "risikoabbau"
    ]
    non_entry_exit_gate = bool(hard_gate and gate_low and any(t in gate_low for t in exit_gate_terms))

    # v22.11: Der Live-Monitor ist ein kurzfristiger Chart-Trigger-Monitor.
    # Fundamentale Warnungen wie Cashflow/Bilanz/Profitabilitaet sollen sichtbar
    # bleiben, aber nicht als hartes charttechnisches Einstiegsgate zaehlen.
    fundamental_gate_terms = [
        "cashflow", "cash flow", "free cash", "fcf", "bilanz", "liquiditaet",
        "liquidität", "verschuld", "debt", "verwässer", "verwaesser",
        "profitabil", "marge", "margen", "earnings", "guidance", "umsatz",
        "bewertung", "valuation", "fundamental", "qualität", "qualitaet"
    ]
    non_entry_fundamental_gate = bool(hard_gate and gate_low and any(t in gate_low for t in fundamental_gate_terms))
    entry_hard_gate = bool(hard_gate and not non_entry_exit_gate and not non_entry_fundamental_gate)

    def _v2211_fundamental_warning_text(src):
        rr = src or {}
        candidates = []
        for key in [
            "top_red_flag", "red_flags", "red_flag", "fundamental_warning",
            "fundamental_red_flags", "quality_red_flags", "risk_flags"
        ]:
            val = rr.get(key)
            if isinstance(val, (list, tuple)):
                candidates.extend([str(x) for x in val if str(x or '').strip()])
            elif val not in [None, "", "-", "nan"]:
                candidates.append(str(val))
        for item in rr.get("red_flag_items", []) or []:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("label") or item.get("reason") or item.get("title")
            else:
                txt = item
            if txt not in [None, "", "-", "nan"]:
                candidates.append(str(txt))
        # Auch ein Gate kann fundamental sein und soll dann nur als Warnhinweis erscheinen.
        if non_entry_fundamental_gate and gate:
            candidates.append(str(gate))
        cleaned = []
        seen = set()
        for txt in candidates:
            low = txt.lower()
            if not any(t in low for t in fundamental_gate_terms):
                continue
            short = shorten_text(txt, 90) if 'shorten_text' in globals() else txt[:90]
            if short.lower() in seen:
                continue
            seen.add(short.lower())
            cleaned.append(short)
        return "-" if not cleaned else "⚠️ " + " · ".join(cleaned[:2])

    fundamental_warning = _v2211_fundamental_warning_text(r)

    # v28.4.5d: Data quality is deliberately separate from the trading score.
    # It describes how complete the technical history is, not whether the setup is good.
    def _v2845d_data_quality(src):
        rr = src or {}
        try:
            history_days = int(rr.get("history_days") or len(rr.get("df")) or 0)
        except Exception:
            history_days = 0
        mode = str(rr.get("history_mode") or "").strip()
        dfq = rr.get("df")
        def _finite(v):
            try:
                return bool(np.isfinite(float(v)) and not pd.isna(float(v)))
            except Exception:
                return False
        ma20_ok = _finite(rr.get("ma20")) or history_days >= 20
        ma50_ok = _finite(rr.get("ma50")) or history_days >= 50
        ma200_ok = _finite(rr.get("ma200")) or history_days >= 200
        atr_ok = _finite(rr.get("atr")) or _finite(rr.get("atr_pct")) or history_days >= 14
        volume_ok = False
        try:
            if isinstance(dfq, pd.DataFrame) and not dfq.empty and "Volume" in dfq.columns:
                vq = pd.to_numeric(dfq["Volume"], errors="coerce").dropna()
                volume_ok = bool(len(vq) >= min(10, max(1, history_days)) and (vq > 0).any())
        except Exception:
            pass
        provider = str(rr.get("data_provider") or rr.get("provider") or "Yahoo").strip()
        fallback = bool(rr.get("provider_fallback") or rr.get("fallback_provider_used"))

        if history_days >= 250 and ma20_ok and ma50_ok and ma200_ok and atr_ok and volume_ok and not fallback:
            stars, label = 5, "Vollständig"
        elif history_days >= 120 and ma20_ok and ma50_ok and atr_ok and volume_ok:
            stars, label = 4, "Reduzierte Historie"
        elif history_days >= 20 and ma20_ok and atr_ok:
            stars, label = 3, "New Listing" if history_days < 120 else "Reduziert"
        elif history_days >= 10:
            stars, label = 2, "Eingeschränkt"
        else:
            stars, label = 1, "Unzureichend"
        if fallback and stars > 1:
            stars = min(stars, 3)
            label = "Fallback-Quelle"
        missing = []
        if not ma20_ok: missing.append("MA20")
        if not ma50_ok: missing.append("MA50")
        if not ma200_ok: missing.append("MA200")
        if not atr_ok: missing.append("ATR")
        if not volume_ok: missing.append("Volumen")
        text = "★" * stars + "☆" * (5 - stars) + " " + label
        detail = f"{history_days} Handelstage · {mode or label} · Quelle {provider}"
        if missing:
            detail += " · eingeschränkt: " + ", ".join(missing)
        return text, detail, stars

    data_quality_text, data_quality_detail, data_quality_stars = _v2845d_data_quality(r)
    wave_dist = _v210_alert_num(d.get("wave_trigger_distance_pct"), default=None)
    entry_distance = _v210_alert_num(d.get("entry_distance_pct"), default=None)
    next_step = str(d.get("next_step") or "-").strip()
    brake = str(d.get("brake") or "-").strip()

    # v22.1: Gruen muss mit der Sofortanalyse konsistent sein.
    # Entry/Wave allein reicht nicht, wenn Timing, valides Setup oder finaler Trigger noch bremsen.
    status_icon = "⚪"
    status = "Beobachten"
    priority = 4
    reason = "Noch kein aktiver Trigger."
    monitor_action = next_step

    grade_ok = grade in {"A", "B", "C"}
    crv_ok = crv_float is not None and crv_float >= 1.5
    entry_reached = "Entry-Zone erreicht" in alert_types
    wave_active = "Wave-Trigger aktiv" in alert_types
    bucket_active = bucket == "Jetzt prüfbar" or "Bucket: Jetzt prüfbar" in alert_types
    bucket_near = bucket == "Nahe am Trigger" or "Wave-Trigger nahe" in alert_types
    final_release_ok, final_blockers = _v214_monitor_final_release_check(r, d)
    final_blocker_text = "; ".join(final_blockers)

    # v22.9: Trendfolge-Gruen getrennt von Pullback-/Entry-Zonen bewerten.
    # Leader/Momentum-Aktien duerfen nicht nur deshalb blockiert werden, weil der
    # Kurs weit ueber einer alten Pullback-Entry-Zone liegt. Entscheidend ist dann
    # Trend-/Timing-Qualitaet plus kontrollierbare Ueberdehnung zum MA20.
    def _v229_num_any(*vals, default=None):
        for v in vals:
            try:
                if v in {None, "", "-", "n/a", "nan"}:
                    continue
                f = float(str(v).replace(",", "."))
                if np.isfinite(f) and not pd.isna(f) and f > 0:
                    return f
            except Exception:
                continue
        return default

    info_obj = r.get("info") if isinstance(r.get("info"), dict) else {}
    fast_info_obj = r.get("fast_info") if isinstance(r.get("fast_info"), dict) else {}
    ma20_val = _v229_num_any(
        r.get("ma20"), r.get("MA20"), r.get("SMA20"), r.get("EMA20"),
        r.get("ma_20"), r.get("sma_20"), info_obj.get("fiftyDayAverage")
    )
    ma50_val = _v229_num_any(
        r.get("ma50"), r.get("MA50"), r.get("SMA50"), r.get("EMA50"),
        r.get("ma_50"), r.get("sma_50"), info_obj.get("fiftyDayAverage")
    )
    price_float = _v229_num_any(price)

    # v24.14: ATR-basierte Volatilität für den Live-Screener.
    # ATR in % ist für kurzfristige Trades aussagekräftiger als eine abstrakte
    # annualisierte Volatilität, weil sie die typische tägliche Handelsspanne
    # relativ zum aktuellen Kurs zeigt.
    def _v2412_atr_pct(src, current_price=None):
        rr = src or {}
        direct_keys = [
            "atr_pct", "ATR_pct", "ATR_Pct", "atr_percent", "ATR_Percent",
            "ATR in %", "ATR%", "volatility_atr_pct"
        ]
        for key in direct_keys:
            try:
                val = rr.get(key)
                if val in [None, "", "-", "n/a", "nan"]:
                    continue
                f = float(str(val).replace(",", ".").replace("%", "").strip())
                if np.isfinite(f) and not pd.isna(f) and f >= 0:
                    return f
            except Exception:
                pass

        # Fallback: ATR-Wert durch aktuellen Kurs teilen.
        atr_abs = None
        for key in ["atr", "ATR", "atr14", "ATR14"]:
            try:
                val = rr.get(key)
                if val in [None, "", "-", "n/a", "nan"]:
                    continue
                f = float(str(val).replace(",", "."))
                if np.isfinite(f) and not pd.isna(f) and f > 0:
                    atr_abs = f
                    break
            except Exception:
                pass
        if atr_abs is not None and current_price is not None and current_price > 0:
            return atr_abs / current_price * 100.0

        # Letzter Fallback: ATR(14) direkt aus OHLC-Daten berechnen.
        try:
            dfv = rr.get("df")
            if isinstance(dfv, pd.DataFrame) and not dfv.empty:
                high_col = next((c for c in ["High", "high"] if c in dfv.columns), None)
                low_col = next((c for c in ["Low", "low"] if c in dfv.columns), None)
                close_col = next((c for c in ["Close", "close", "Adj Close", "Adj_Close"] if c in dfv.columns), None)
                if high_col and low_col and close_col:
                    tmp = dfv[[high_col, low_col, close_col]].copy()
                    for c in [high_col, low_col, close_col]:
                        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                    tmp = tmp.dropna()
                    if len(tmp) >= 15:
                        prev_close = tmp[close_col].shift(1)
                        tr = pd.concat([
                            (tmp[high_col] - tmp[low_col]).abs(),
                            (tmp[high_col] - prev_close).abs(),
                            (tmp[low_col] - prev_close).abs(),
                        ], axis=1).max(axis=1)
                        atr14 = tr.rolling(14, min_periods=14).mean().iloc[-1]
                        last_close = tmp[close_col].iloc[-1]
                        if pd.notna(atr14) and pd.notna(last_close) and last_close > 0:
                            return float(atr14 / last_close * 100.0)
        except Exception:
            pass
        return None

    atr_pct_live = _v2412_atr_pct(r, price_float)
    if atr_pct_live is None:
        volatility_text = "n/a"
    elif atr_pct_live < 2.8:
        volatility_text = f"{atr_pct_live:.1f}% · niedrig"
    elif atr_pct_live < 5.5:
        volatility_text = f"{atr_pct_live:.1f}% · normal"
    elif atr_pct_live < 8.0:
        volatility_text = f"{atr_pct_live:.1f}% · erhöht"
    else:
        volatility_text = f"{atr_pct_live:.1f}% · hoch"

    # v22.13: Performance-Kontext seit Watchlist-Aufnahme.
    # Der Live-Score bleibt ein aktueller Chart-/Trigger-Score, aber stark gelaufene
    # Watchlist-Werte sollen sichtbar sein, auch wenn aktuell kein frischer Entry aktiv ist.
    def _v2212_parse_dt(val):
        if val in [None, "", "-", "nan"]:
            return None
        try:
            return pd.to_datetime(val, dayfirst=True, errors="coerce")
        except Exception:
            return None

    def _v2212_valid_price(val):
        try:
            if val in [None, "", "-", "n/a", "nan"]:
                return None
            f = float(str(val).replace(",", "."))
            if np.isfinite(f) and not pd.isna(f) and f > 0:
                return f
        except Exception:
            pass
        return None

    def _v2212_chart_start_price(src, added_dt):
        df0 = src.get("df") if isinstance(src, dict) else None
        if df0 is None or not isinstance(df0, pd.DataFrame) or df0.empty:
            return None
        close_col = None
        for c in ["Close", "close", "Adj Close", "Adj_Close"]:
            if c in df0.columns:
                close_col = c
                break
        if close_col is None:
            return None
        try:
            tmp = df0.copy()
            if "Date" in tmp.columns:
                tmp["__date"] = pd.to_datetime(tmp["Date"], errors="coerce")
            else:
                tmp["__date"] = pd.to_datetime(tmp.index, errors="coerce")
            tmp = tmp.dropna(subset=["__date", close_col]).sort_values("__date")
            if tmp.empty:
                return None
            if added_dt is not None and not pd.isna(added_dt):
                # Ersten Schlusskurs am/ab Aufnahmedatum nutzen; wenn zu neu, letzten davor.
                after = tmp[tmp["__date"] >= added_dt]
                if not after.empty:
                    return _v2212_valid_price(after.iloc[0][close_col])
                before = tmp[tmp["__date"] <= added_dt]
                if not before.empty:
                    return _v2212_valid_price(before.iloc[-1][close_col])
            return None
        except Exception:
            return None

    meta = watchlist_meta if isinstance(watchlist_meta, dict) else {}
    start_date_raw = None
    for k in ["Added_At", "added_at", "Vorgemerkt_Am", "Created_At", "created_at", "Aufnahme", "Aufnahmedatum"]:
        if str(meta.get(k, "")).strip() not in {"", "-", "nan", "None"}:
            start_date_raw = meta.get(k)
            break
    added_dt = _v2212_parse_dt(start_date_raw)

    start_price = None
    start_price_source = str(meta.get("Startkurs_Quelle") or meta.get("Start_Source") or meta.get("Quelle") or "").strip()
    for k in ["Startkurs", "Start_Kurs", "Start_Price", "start_price", "Added_Price", "Einstand", "Einstandskurs"]:
        start_price = _v2212_valid_price(meta.get(k))
        if start_price is not None:
            if not start_price_source:
                start_price_source = k
            break
    if start_price is None:
        start_price = _v2212_chart_start_price(r, added_dt)
        if start_price is not None and not start_price_source:
            start_price_source = "Chart-Historie"

    # v23.6: Im Live-Monitor KEINE automatische Session-Baseline aus dem aktuellen Kurs setzen.
    # Sonst entstehen irrefuehrende Kombinationen wie Startkurs n/a / Seit Aufnahme n/a /
    # Quelle "Baseline ab jetzt" oder Startkurs = aktueller Kurs = +0.0%.
    # Startkurse duerfen hier nur aus echten Watchlist-Metadaten, manuellem Store
    # oder historischem Backfill kommen. Neue Watchlist-Eintraege setzen ihren
    # Startkurs beim Hinzufuegen; alte Werte muessen manuell/historisch nachgetragen werden.
    try:
        perf_state = st.session_state.get("live_watchlist_start_prices_v2212", {})
        if isinstance(perf_state, dict):
            wl_key = str(meta.get("Watchlist_Name") or meta.get("watchlist_name") or "default").strip() or "default"
            baseline_key = f"{wl_key}::{ticker}"
            stored_baseline = _v2212_valid_price(perf_state.get(baseline_key))
            # Nur noch alte, echte Session-Werte verwenden, wenn sie NICHT exakt dem aktuellen Kurs
            # entsprechen und damit plausibel eine vorherige Baseline darstellen.
            if start_price is None and stored_baseline is not None and price_float is not None:
                try:
                    if abs(float(stored_baseline) - float(price_float)) / max(float(price_float), 1e-9) > 0.002:
                        start_price = stored_baseline
                        if not start_price_source:
                            start_price_source = "Session-Baseline"
                except Exception:
                    pass
    except Exception:
        pass

    perf_pct = None
    if start_price is not None and price_float is not None and start_price > 0:
        perf_pct = (price_float / start_price - 1.0) * 100.0

    # v23.2: Eine automatisch gesetzte "Baseline ab jetzt" ist kein echter Einstands-/
    # Aufnahmekurs. Wenn sie exakt dem aktuellen Kurs entspricht, nicht als +0.0%-
    # Performance ausgeben, weil das wie ein Fehler bzw. falscher Einstand wirkt.
    is_current_baseline = _v232_is_current_baseline_source(start_price_source)
    if start_price is None or (is_current_baseline and perf_pct is not None and abs(float(perf_pct)) < 0.05):
        start_price_text = "n/a"
        perf_text = "n/a"
        # Wenn kein echter Startkurs angezeigt wird, darf auch keine scheinbare
        # Quelle wie "Baseline ab jetzt" stehen. Quelle/Startkurs/Performance muessen konsistent sein.
        start_price_source = "n/a"
    else:
        start_price_text = round(float(start_price), 4)
        if perf_pct is None:
            perf_text = "n/a"
        else:
            perf_text = f"{perf_pct:+.1f}%"

    ma20_stretch_pct = None
    if price_float is not None and ma20_val is not None and ma20_val > 0:
        ma20_stretch_pct = (price_float / ma20_val - 1.0) * 100.0

    timing_pkg_lm = r.get("timing_action_confidence_pkg") if isinstance(r.get("timing_action_confidence_pkg"), dict) else {}
    conf_pkg_lm = r.get("trigger_confluence_pkg") if isinstance(r.get("trigger_confluence_pkg"), dict) else {}
    timing_score_lm = _v210_alert_num(timing_pkg_lm.get("score"), default=None)
    conf_score_lm = _v210_alert_num(conf_pkg_lm.get("score"), default=None)
    chart_pkg_lm = r.get("charttechnik_setup_pkg") if isinstance(r.get("charttechnik_setup_pkg"), dict) else {}
    chart_score_lm = _v210_alert_num(chart_pkg_lm.get("score"), default=None)
    chart_text_lm = " ".join([
        str(chart_pkg_lm.get("label") or ""),
        str(chart_pkg_lm.get("summary") or ""),
        str(chart_pkg_lm.get("trigger") or ""),
        str(chart_pkg_lm.get("invalid") or ""),
    ]).lower()

    trend_structure_ok = bool(
        price_float is not None
        and ma20_val is not None
        and price_float > ma20_val
        and (ma50_val is None or ma20_val >= ma50_val)
    )
    trend_timing_ok = bool((timing_score_lm is None or timing_score_lm >= 70) and (conf_score_lm is None or conf_score_lm >= 65))
    trend_not_too_extended = bool(ma20_stretch_pct is None or ma20_stretch_pct <= 12.0)
    trend_extended_but_interesting = bool(ma20_stretch_pct is not None and 12.0 < ma20_stretch_pct <= 20.0)
    trend_quality_ok = grade in {"A", "B"} and (crv_float is None or crv_float >= 1.3)
    trend_reason_tail = ""
    if ma20_stretch_pct is not None:
        trend_reason_tail = f" Abstand zu MA20 ca. {ma20_stretch_pct:.1f}%."

    if "Invalidierung gebrochen" in alert_types:
        status_icon, status, priority = "🔴", "Invalidiert / meiden", 1
        reason = "Invalidierung aktiv."
        monitor_action = "Kein Kauf: These/Setup zuerst neu prüfen; Stop-/Invalidierungsbruch beachten."
    elif bucket == "Warnsignale / meiden" and grade in {"A", "B"} and crv_ok:
        # v22.1: Warnbucket ist nur ein Diagnose-/Vorsichtssignal, wenn Grade A/B und CRV passen.
        # entry_hard_gate wird hier bewusst nicht mehr als Ausschluss verwendet,
        # weil es in MRVL-artigen Fällen aus Exit-/Schutz-/Warntexten entstehen kann.
        # Rot bleibt nur fuer echte Invalidierung oder fehlende Qualitaets-/CRV-Entlastung.
        status_icon, status, priority = "🟡", "Selektiv prüfen", 2
        if final_release_ok:
            reason = "Sofortanalyse/Freigabe ist konstruktiv; Radar-Bucket bzw. Exit-/Schutzhinweise verlangen nur defensive Ausführung."
            monitor_action = "Selektiv prüfen: Einstieg ist grundsätzlich möglich; Positionsgröße defensiv wählen und Stop/Invalidierung sauber festlegen."
        else:
            reason = "Radar-Bucket warnt, aber Grade und CRV entlasten; kein roter Status ohne echte Invalidierung."
            monitor_action = "Selektiv prüfen: Warnbucket-Grund kontrollieren, Stop/Invalidierung festlegen und nur mit defensiver Positionsgröße handeln."
    elif bucket == "Warnsignale / meiden" and not final_release_ok:
        status_icon, status, priority = "🔴", "Warnsignal / meiden", 1
        reason = "Radar-Bucket warnt und es gibt keine ausreichende Qualitäts-/CRV-Entlastung."
        monitor_action = "Kein Kauf: Bremse zuerst klären und Sofortanalyse erneut prüfen."
    elif bucket == "Warnsignale / meiden" and final_release_ok:
        status_icon, status, priority = "🟡", "Selektiv prüfen", 2
        reason = "Sofortanalyse gibt den Einstieg frei, aber der Radar-Bucket enthält noch Warnhinweise."
        monitor_action = "Selektiv prüfen: Einstieg ist freigegeben, aber Positionsgröße defensiv wählen und Stop/Invalidierung eng beachten."
    elif entry_hard_gate:
        status_icon, status, priority = "🔴", "Setup blockiert", 1
        reason = brake if brake and brake != "-" else "Hartes Einstiegsgate aktiv."
        monitor_action = "Kein Kauf: Einstiegsgate zuerst klären."
    elif (
        live_short_term
        and trend_quality_ok
        and trend_structure_ok
        and trend_timing_ok
        and trend_not_too_extended
        and not entry_hard_gate
        and "Invalidierung gebrochen" not in alert_types
        and (
            bucket_active
            or wave_active
            or entry_reached
            or (timing_score_lm is not None and timing_score_lm >= 74 and (conf_score_lm is None or conf_score_lm >= 68))
        )
    ):
        # v24.0: Kurzfrist-/Trading-Modus nicht durch die breite Swing-Freigabe
        # blockieren lassen. Wenn Trendstruktur, Timing/Konfluenz und ein operativer
        # Trigger-/Trendfolge-Kontext passen, darf der Live-Monitor gruen werden,
        # auch wenn die laengerfristige Sofortanalyse noch weiche Text-Hinweise enthaelt.
        status_icon, status, priority = "🟢", "Kurzfrist-Trigger aktiv", 0
        reason = "Kurzfrist-Setup ist charttechnisch aktiv: Trendstruktur, Timing/Konfluenz und Trigger-/Trendfolge-Kontext passen." + trend_reason_tail
        monitor_action = "Kurzfrist aktiv. Entry/Breakout nur mit klarer Invalidierung, Positionsgröße und Stop-Plan handeln."
    elif (bucket_active or entry_reached or wave_active) and not final_release_ok:
        status_icon, status, priority = "🟡", "Trigger offen / Abwarten", 2
        reason = "Sofortanalyse bestätigt Grün noch nicht" + (f": {final_blocker_text}." if final_blocker_text else ".")
        monitor_action = "Noch kein grünes Kaufsignal: erst finale Trigger-/Timing-Bestätigung abwarten. Risiko/Stückzahl erst festlegen, wenn die Sofortanalyse den Einstieg freigibt."
    elif final_release_ok and trend_quality_ok and trend_structure_ok and trend_timing_ok and trend_not_too_extended:
        status_icon, status, priority = "🟢", "Trendfolge aktiv", 0
        reason = "Trendfolge-Setup ist aktiv: Kurs hält oberhalb MA20, MA20 liegt über/nahe MA50 und Timing/Konfluenz passen." + trend_reason_tail
        monitor_action = "Trendfolge aktiv. Nicht auf alte Entry-Zone fixieren; Positionsgröße über Abstand zu MA20/Stop begrenzen und nur mit klarer Invalidierung handeln."
    elif final_release_ok and trend_quality_ok and trend_structure_ok and trend_timing_ok and trend_extended_but_interesting:
        status_icon, status, priority = "🟡", "Trend stark / Pullback bevorzugt", 2
        reason = "Trend und Timing sind stark, aber der Kurs ist bereits deutlich über MA20 gelaufen." + trend_reason_tail
        monitor_action = "Trend bleibt interessant, aber kein aggressives Hinterherlaufen: kleine Startposition nur bei engem Risiko-Plan oder Pullback/neue Base abwarten."
    elif bucket_active and final_release_ok and grade_ok and (crv_ok or "CRV attraktiv" in alert_types):
        status_icon, status, priority = "🟢", "Kauftrigger aktiv", 0
        reason = "Setup ist operativ aktiv: Bucket ist Jetzt prüfbar und CRV ist attraktiv."
        monitor_action = "Setup aktiv. Jetzt Risiko pro Trade, Stückzahl und Stop/Invalidierung festlegen."
    elif wave_active and final_release_ok and grade_ok and crv_ok and bucket in {"Jetzt prüfbar", "Nahe am Trigger"}:
        status_icon, status, priority = "🟢", "Kauftrigger aktiv", 0
        reason = "Wave-Trigger ist aktiv und CRV/Setup passen."
        monitor_action = "Setup aktiv. Jetzt Risiko pro Trade, Stückzahl und Stop/Invalidierung festlegen."
    elif entry_reached and final_release_ok and grade_ok and crv_ok and bucket == "Jetzt prüfbar":
        status_icon, status, priority = "🟢", "Kauftrigger aktiv", 0
        reason = "Entry-Zone ist erreicht, Bucket ist aktiv und CRV ist attraktiv."
        monitor_action = "Setup aktiv. Jetzt Risiko pro Trade, Stückzahl und Stop/Invalidierung festlegen."
    elif entry_reached and grade_ok and crv_ok and bucket == "Nahe am Trigger":
        status_icon, status, priority = "🟡", "Entry erreicht, Trigger offen", 2
        reason = "Entry-Zone ist erreicht und CRV attraktiv, aber finale Trigger-/Volumenbestätigung fehlt noch."
        monitor_action = "Entry-Zone erreicht; finale Trigger-/Volumenbestätigung abwarten. Noch nicht wie ein grünes Kaufsignal behandeln."
    elif bucket_near:
        status_icon, status, priority = "🟡", "Nahe am Trigger", 2
        reason = "Setup ist interessant, Aktivierung fehlt noch."
        monitor_action = "Nahe am Trigger: finale Aktivierung/Reclaim/Volumenbestätigung abwarten."
    elif "CRV attraktiv" in alert_types or (crv_float is not None and crv_float >= 1.8 and bucket in {"Nahe am Trigger", "Starke Watchlist"}):
        status_icon, status, priority = "🔵", "CRV attraktiv", 3
        reason = "Chance/Risiko ist interessant, aber Entry/Trigger ist noch nicht aktiv."
        monitor_action = "CRV attraktiv, aber noch kein aktiver Kauftrigger. Entry/Trigger weiter beobachten."
    elif bucket in {"Pullback bevorzugt / nicht hinterherlaufen", "Später beobachten"}:
        status_icon, status, priority = "⚪", "Nur beobachten", 5
        reason = brake if brake and brake != "-" else "Bremse/Gate noch aktiv."
        monitor_action = "Nur beobachten; erst bei besserem Entry, Trigger oder geklärter Bremse neu prüfen."

    # v22.13: Starke Watchlist-Performance sichtbar machen.
    # Das ist KEIN automatisches Kaufsignal. Es verhindert nur, dass ein +20%-Lauf
    # als irrelevanter weisser Wert wirkt, wenn aktuell kein frischer Entry erkannt wird.
    if (perf_pct is not None and perf_pct >= 12.0 and status_icon in {"⚪", "🔵"}
            and "Invalidierung gebrochen" not in alert_types and not entry_hard_gate):
        status_icon, status, priority = "🟡", "Läuft stark / Einstieg offen", 2
        reason = f"Seit Watchlist-Aufnahme {perf_pct:+.1f}% gelaufen; aktuell aber noch kein sauberer frischer Einstiegstrigger."
        monitor_action = "Trend/Performance anerkennen, aber Neueinstieg nur bei frischem Trigger, Pullback oder neuer Base planen."
    elif (perf_pct is not None and perf_pct >= 6.0 and status_icon == "⚪" and trend_structure_ok
            and "Invalidierung gebrochen" not in alert_types and not entry_hard_gate):
        status_icon, status, priority = "🟡", "Läuft positiv / Trigger offen", 2
        reason = f"Seit Watchlist-Aufnahme {perf_pct:+.1f}% im Plus; Trend positiv, frischer Einstiegstrigger noch offen."
        monitor_action = "Weiter beobachten: positive Entwicklung, aber Kauf erst bei klarem Trigger oder kontrollierbarem Pullback."

    # v22.11: Chart-Weighted Live-Score x/100.
    # Der Live-Monitor bewertet kurzfristige charttechnische Handlungsreife.
    # Fundamental-/Qualitaets-Red-Flags werden separat als Warnhinweis gezeigt,
    # aber nicht mehr stark in den Live-Score oder die Ampel eingerechnet.
    def _v2210_clip(val, lo=0.0, hi=100.0):
        try:
            f = float(val)
            if not np.isfinite(f) or pd.isna(f):
                return lo
            return max(lo, min(hi, f))
        except Exception:
            return lo

    grade_component = {"A": 92.0, "B": 80.0, "C": 66.0, "D": 48.0, "E": 35.0, "F": 22.0, "G": 12.0}.get(grade, 50.0)
    timing_component = _v2210_clip(timing_score_lm if timing_score_lm is not None else (82.0 if final_release_ok else 50.0))
    conf_component = _v2210_clip(conf_score_lm if conf_score_lm is not None else (78.0 if final_release_ok else 50.0))
    chart_component = _v2210_clip(chart_score_lm if chart_score_lm is not None else (72.0 if final_release_ok else 48.0))
    if any(t in chart_text_lm for t in ["abwarten", "noch nicht", "kein", "fehlt", "nicht reif"]):
        chart_component = min(chart_component, 54.0)
    if any(t in chart_text_lm for t in ["reclaim", "breakout", "ausbruch", "trigger", "entry-zone", "stabilisierung", "bullische reaktion"]):
        chart_component = max(chart_component, 62.0)

    if crv_float is None:
        crv_component = 56.0 if final_release_ok else 48.0
    else:
        # CRV ist wichtig, aber fuer den Live-Monitor nur Nebenfilter.
        crv_component = _v2210_clip(44.0 + (crv_float - 1.0) * 22.0, 25.0, 92.0)

    trigger_component = 42.0
    if bucket_active or wave_active:
        trigger_component = 86.0
    elif entry_reached:
        trigger_component = 76.0
    elif bucket_near:
        trigger_component = 64.0
    elif "CRV attraktiv" in alert_types:
        trigger_component = 54.0
    if trend_structure_ok and trend_timing_ok:
        trigger_component = max(trigger_component, 82.0 if trend_not_too_extended else 66.0)

    trend_component = 50.0
    if trend_structure_ok and trend_timing_ok and trend_not_too_extended:
        trend_component = 82.0
    elif trend_structure_ok and trend_timing_ok:
        trend_component = 66.0
    elif trend_structure_ok:
        trend_component = 58.0

    # v23.9: Short-Term-Live-Engine.
    # Im Kurzfrist-/Trading-Modus wird der Score noch staerker an operativer
    # Chart-Handlungsreife ausgerichtet. Swing bleibt als bisheriger Modus erhalten.
    if live_short_term:
        live_score_raw = (
            0.34 * trigger_component
            + 0.26 * timing_component
            + 0.18 * conf_component
            + 0.14 * chart_component
            + 0.05 * trend_component
            + 0.02 * crv_component
            + 0.01 * grade_component
        )
        # Kurzfristige Trading-Risiken: kein frischer Trigger, schwaches Timing
        # oder Chart-Abwarten duerfen deutlich bremsen.
        if trigger_component < 58.0 and not trend_structure_ok:
            live_score_raw -= 8.0
        if timing_component < 45.0:
            live_score_raw -= 10.0
        if conf_component < 50.0:
            live_score_raw -= 6.0
        if any(t in chart_text_lm for t in ["abwarten", "noch nicht", "nicht reif", "fehlt"]):
            live_score_raw -= 8.0
    else:
        live_score_raw = (
            0.30 * trigger_component
            + 0.24 * timing_component
            + 0.20 * conf_component
            + 0.14 * chart_component
            + 0.06 * trend_component
            + 0.04 * crv_component
            + 0.02 * grade_component
        )

    # Abzuege nur fuer chart-/entry-relevante Bremsen. Fundamentale Hinweise bleiben
    # in der Warnhinweis-Spalte sichtbar, zaehlen hier aber nicht als harter Score-Abzug.
    if bucket == "Warnsignale / meiden" and not (grade in {"A", "B"} and crv_ok):
        live_score_raw -= 6.0
    if entry_hard_gate:
        live_score_raw -= 24.0
    if "Invalidierung gebrochen" in alert_types:
        live_score_raw -= 45.0
    if ma20_stretch_pct is not None and ma20_stretch_pct > 12.0:
        live_score_raw -= min(18.0, (ma20_stretch_pct - 12.0) * 1.6)
    if not final_release_ok and status_icon == "🟢":
        live_score_raw -= 18.0
    # Performance seit Aufnahme ist Kontext, kein primaerer Entry-Trigger:
    # leichter Bonus fuer starke Entwicklung, aber nur bis gelb/stabil, nicht automatisch gruen.
    if perf_pct is not None and perf_pct >= 12.0 and status_icon in {"🟡", "⚪", "🔵"}:
        live_score_raw += min(10.0, (perf_pct - 8.0) * 0.35)

    # v23.6: Score/Ampel-Alignment.
    # Bisher wurde der numerische Score nachtraeglich an die vorab bestimmte Ampel
    # geklemmt. Dadurch konnte 50/100 gelb sein, waehrend 58/100 weiss blieb.
    # Jetzt ist der Rohscore zuerst massgeblich; harte Gruen-/Rot-Gates bleiben
    # erhalten, aber Weiss/Gelb/Blau werden logisch an den Score angeglichen.
    score_candidate = _v2210_clip(live_score_raw, 0.0, 100.0)

    immutable_red = bool("Invalidierung gebrochen" in alert_types or (status_icon == "🔴" and (entry_hard_gate or bucket == "Warnsignale / meiden")))
    immutable_green = bool(status_icon == "🟢")
    weak_yellow_without_chart_trigger = bool(
        status_icon == "🟡"
        and score_candidate < 55.0
        and not (bucket_near or entry_reached or wave_active or bucket_active)
        and not (perf_pct is not None and perf_pct >= 6.0)
        and status in {"Selektiv prüfen", "Nahe am Trigger"}
    )

    if live_short_term:
        # Kurzfrist-Modus: Gruen darf nur dann bleiben, wenn der Trade wirklich
        # operativ planbar ist. Sonst wird auf Gelb zurueckgestuft.
        has_short_term_trigger = bool(bucket_active or wave_active or entry_reached or (trend_structure_ok and trend_timing_ok and trend_not_too_extended))
        if status_icon == "🟢" and not has_short_term_trigger:
            status_icon, status, priority = "🟡", "Setup interessant / Trigger prüfen", 2
            reason = "Kurzfrist-Modus: Qualität reicht nicht; es fehlt ein klarer aktueller Charttrigger."
            monitor_action = "Kurzfrist nur vorbereiten: Trigger, Volumenbestaetigung und engen Stop abwarten."
            immutable_green = False
        # Sehr schwache Kurzfrist-Konfluenz soll auch bei guter Aktie nicht gelb bleiben.
        if status_icon in {"🟡", "🔵"} and timing_component < 42.0 and conf_component < 48.0 and not entry_reached:
            status_icon, status, priority = "⚪", "Kein kurzfristiger Trigger", 4
            reason = "Kurzfrist-Modus: Timing/Konfluenz reichen aktuell nicht fuer einen Trading-Trigger."
            monitor_action = "Kein kurzfristiger Trade: erst bei Reclaim, Breakout oder klarer Entry-Naehe neu prüfen."

    if immutable_red:
        live_score = _v2210_clip(score_candidate, 0.0, 39.0)
    elif immutable_green:
        live_score = _v2210_clip(score_candidate, 75.0, 98.0)
    else:
        live_score = _v2210_clip(score_candidate, 0.0, 74.0)

        # Niedrige selektive Warnbucket-Faelle nicht kuenstlich gelb halten.
        if weak_yellow_without_chart_trigger:
            status_icon, status, priority = "⚪", "Beobachten / Vorsicht", 4
            reason = "Score noch zu niedrig fuer gelb; Warn-/CRV-Kontext vorhanden, aber kein klarer chartnaher Trigger."
            monitor_action = "Beobachten: erst bei verbessertem Charttrigger, Reclaim oder klarer Triggernaehe selektiv pruefen."

        # Mittlere bis hohe Scores duerfen nicht weiss bleiben, wenn keine rote Bremse aktiv ist.
        if status_icon == "⚪" and live_score >= 55.0:
            status_icon, status, priority = "🟡", "Beobachten / nahe dran", 2
            reason = "Live-Score liegt im gelben Bereich, aber ein klarer Kauftrigger fehlt noch."
            monitor_action = "Nahe dran beobachten: Trigger-/Volumenbestätigung, Reclaim oder Pullback mit sauberem Stop abwarten."

        # Blau war semantisch verwirrend: CRV ist Kontext, aber die Ampel soll nach
        # Handlungsnaehe sortieren. Attraktives CRV ohne Trigger wird als Gelb gefuehrt.
        if status_icon == "🔵" and live_score >= 55.0:
            status_icon, status, priority = "🟡", "CRV attraktiv / Trigger offen", 2
            reason = "CRV ist attraktiv und der Live-Score ist gelb, aber ein aktiver Charttrigger fehlt noch."
            monitor_action = "CRV attraktiv, aber noch kein aktiver Kauftrigger. Entry/Trigger weiter beobachten."
        elif status_icon == "🔵" and live_score < 55.0:
            status_icon, status, priority = "⚪", "CRV beobachten", 4
            reason = "CRV ist interessant, aber der charttechnische Live-Score reicht noch nicht fuer gelb."
            monitor_action = "Beobachten: CRV bleibt interessant, aber erst bei Triggernaehe oder Chartfreigabe handeln."

        # Nach einer Status-Anhebung/-Abstufung den Score in einen konsistenten Bereich ziehen.
        if status_icon == "🟡":
            live_score = _v2210_clip(live_score, 55.0, 74.0)
        elif status_icon == "⚪":
            live_score = _v2210_clip(live_score, 0.0, 54.0)

    live_score_int = int(round(live_score))

    # v28.4.6: Explainable Trading. Die Komponenten werden nicht nur intern
    # gespeichert, sondern in eine kompakte, handlungsorientierte Erklaerung
    # uebersetzt. Das aendert den Score nicht.
    _explain_components = [
        ("Trigger", trigger_component),
        ("Timing", timing_component),
        ("Konfluenz", conf_component),
        ("Chart", chart_component),
        ("Trend", trend_component),
        ("CRV", crv_component),
    ]
    _explain_sorted = sorted(_explain_components, key=lambda x: float(x[1]), reverse=True)
    _score_drivers = "; ".join(f"{label} {int(round(float(value)))}/100" for label, value in _explain_sorted[:3])
    _score_brakes_list = [(label, value) for label, value in sorted(_explain_components, key=lambda x: float(x[1])) if float(value) < 60.0]
    _score_brakes = "; ".join(f"{label} {int(round(float(value)))}/100" for label, value in _score_brakes_list[:3]) or "Keine deutliche Komponenten-Bremse"
    if entry_hard_gate:
        _score_brakes = "Hartes Einstiegsgate aktiv; " + _score_brakes
    _score_explanation = f"Treiber: {_score_drivers}. Bremsen: {_score_brakes}."

    return {
        "Ampel": status_icon,
        "Status": status,
        "Live-Score": f"{live_score_int}/100",
        "Live-Horizont": "Kurzfrist" if live_short_term else "Swing",
        "Ticker": ticker,
        "Name": name,
        "Kurs": "n/a" if (price is None or not np.isfinite(float(price)) or pd.isna(price)) else round(float(price), 4),
        "Volatilität": volatility_text,
        "Datenqualität": data_quality_text,
        "Datenbasis": data_quality_detail,
        "Score-Treiber": _score_drivers,
        "Score-Bremsen": _score_brakes,
        "Warum dieser Score?": _score_explanation,
        "ATR-%": None if atr_pct_live is None else round(float(atr_pct_live), 2),
        "Startkurs": start_price_text,
        "Seit Aufnahme": perf_text,
        "Startquelle": "n/a" if str(start_price_text).lower() == "n/a" else (start_price_source or "n/a"),
        "Grade": grade,
        "Radar-Bucket": bucket,
        "CRV": "n/a" if crv_float is None else round(float(crv_float), 2),
        "Entry-Abstand": d.get("entry_distance_text") or ("n/a" if entry_distance is None else f"{entry_distance:+.1f}%"),
        "Wann aktiv?": d.get("wave_trigger") or "-",
        "Setup-Alert": alert_text,
        "Warnhinweis": fundamental_warning,
        "Grund": reason,
        "Nächste Handlung": monitor_action,
        "Letztes Update": get_current_berlin_time().strftime("%d.%m.%Y %H:%M:%S"),
        "__prio": priority,
        "__score": live_score_int,
        # v28.4.3: Diagnosewerte fuer nachvollziehbare Statuswechsel. Diese
        # internen Felder werden nicht direkt angezeigt, aber im vorherigen
        # Snapshot gespeichert und beim naechsten Scan verglichen.
        "__timing_component": round(float(timing_component), 2),
        "__conf_component": round(float(conf_component), 2),
        "__chart_component": round(float(chart_component), 2),
        "__trigger_component": round(float(trigger_component), 2),
        "__trend_component": round(float(trend_component), 2),
        "__crv_component": round(float(crv_component), 2),
        "__entry_hard_gate": bool(entry_hard_gate),
        "__invalidated": bool("Invalidierung gebrochen" in alert_types),
        "__final_release_ok": bool(final_release_ok),
        "__bucket_active": bool(bucket_active),
        "__bucket_near": bool(bucket_near),
        "__entry_reached": bool(entry_reached),
        "__wave_active": bool(wave_active),
        "__ma20_stretch_pct": None if ma20_stretch_pct is None else round(float(ma20_stretch_pct), 2),
        "__gate": gate,
        "__final_blockers": final_blocker_text,
    }


def build_live_watchlist_monitor_v212(tickers, *, style_name="Ausgewogen", max_items=None, watchlist_meta_by_ticker=None, live_horizon="Swing / 1-4 Wochen"):
    """Analyze a normalized ticker sequence without silently truncating it.

    ``max_items`` remains available for explicit caller-side limits, but ``None``
    means all unique values. v28.4.4 moves scope selection into the visible UI.
    """
    rows = []
    errors = []
    unique = []
    for t in tickers or []:
        tt = str(t or "").strip().upper()
        if tt and tt not in unique:
            unique.append(tt)
    meta_by_ticker = watchlist_meta_by_ticker or {}
    if max_items is None:
        selected = unique
    else:
        try:
            selected = unique[:max(0, int(max_items))]
        except (TypeError, ValueError):
            selected = unique
    for ticker in selected:
        try:
            # v24.0: Die Kernanalyse kennt aktuell nur Swing/Langfrist als stabile
            # Horizon-Werte. v23.9 uebergab hier "Kurzfrist (1-7 Tage)" und
            # dadurch brachen viele/alle Ticker ab -> leere Live-Watchlist.
            # Kurzfrist wird deshalb als Live-Monitor-Modus in der Status-/Scorelogik
            # angewendet, die robuste technische Datenbasis bleibt Swing.
            analysis_horizon = "Swing (1-4 Wochen)"
            result = analyze_stock_live_cached_v2414(
                ticker=ticker,
                horizon=analysis_horizon,
                depot=10000,
                risk_pct=1.0,
                override=0.0,
                buy_in_override=0.0,
                smart_money_default=True,
                strict_mode=True,
                market_bucket=_v2414_market_bucket(15),
            )
            decision = build_professional_radar_decision_v18(result, style_name)
            rows.append(_v212_monitor_status_from_decision(result, decision, style_name=style_name, watchlist_meta=meta_by_ticker.get(ticker, {}), live_horizon=live_horizon))
        except Exception as exc:
            _err_text = str(exc or "")[:180]
            _err_lower = _err_text.lower()
            _is_rate_limit = any(token in _err_lower for token in (
                "too many requests", "rate limit", "rate-limited", "ratelimit", "http 429", "status code 429"
            ))
            errors.append({
                "Ticker": ticker,
                "Fehler": _err_text,
                "Status": "Temporär ausstehend (Rate-Limit)" if _is_rate_limit else "Nicht analysierbar",
                "Temporär": bool(_is_rate_limit),
            })
    if rows:
        df = pd.DataFrame(rows)
        # v23.6: Anzeige-Sortierung im Live-Monitor nach Ampel, nicht nach interner
        # Prioritaet. Gewuenschte Reihenfolge: gruen, gelb, weiss, rot.
        # Blau wird als informativer Zwischenstatus nach gelb und vor weiss einsortiert.
        def _v235_live_ampel_sort(val):
            icon = str(val or "").strip()[:1]
            return {"🟢": 0, "🟡": 1, "🔵": 2, "⚪": 3, "🔴": 4}.get(icon, 5)
        df["__ampel_sort"] = df.get("Ampel", "").apply(_v235_live_ampel_sort)
        df = (
            df.sort_values(["__ampel_sort", "__score", "Ticker"], ascending=[True, False, True])
              .drop(columns=["__prio", "__score", "__ampel_sort"], errors="ignore")
              .reset_index(drop=True)
        )
    else:
        df = pd.DataFrame(columns=["Ampel", "Status", "Live-Score", "Live-Horizont", "Ticker", "Name", "Kurs", "Volatilität", "Datenqualität", "Datenbasis", "ATR-%", "Startkurs", "Seit Aufnahme", "Startquelle", "Grade", "Radar-Bucket", "CRV", "Entry-Abstand", "Wann aktiv?", "Setup-Alert", "Warnhinweis", "Grund", "Nächste Handlung", "Letztes Update"])
    # v22.7: Keine NaN-Kurswerte in der Anzeige. Falls Pandas beim Zusammenbau
    # doch NaN erzeugt, sauber als n/a ausgeben.
    if not df.empty and "Kurs" in df.columns:
        df["Kurs"] = df["Kurs"].apply(lambda x: "n/a" if pd.isna(x) else x)
    return df, pd.DataFrame(errors)


def _v220_live_status_rank(ampel, status):
    """Ordnet Live-Status fuer Verbesserungs-/Verschlechterungslogik.
    Niedriger = handlungsnaeher/positiver, hoeher = defensiver.
    """
    a = str(ampel or "").strip()
    stx = str(status or "").strip().lower()
    if a == "🟢" or "kauftrigger" in stx:
        return 0
    if a == "🟡" or "selektiv" in stx or "nahe" in stx or "trigger offen" in stx or "entry erreicht" in stx:
        return 1
    if a == "🔵" or "crv" in stx:
        return 2
    if a == "⚪" or "beobachten" in stx:
        return 3
    if a == "🔴" or "invalid" in stx or "meiden" in stx or "blockiert" in stx:
        return 4
    return 3


def _v220_live_change_label(prev_ampel, prev_status, new_ampel, new_status):
    if prev_status is None:
        return "Neu"
    prev_key = f"{prev_ampel} {prev_status}".strip()
    new_key = f"{new_ampel} {new_status}".strip()
    if prev_key == new_key:
        return "Unverändert"
    old_rank = _v220_live_status_rank(prev_ampel, prev_status)
    new_rank = _v220_live_status_rank(new_ampel, new_status)
    if new_rank < old_rank:
        return "Verbessert"
    if new_rank > old_rank:
        return "Verschlechtert"
    return "Geändert"


def _v237_parse_live_score(value, default=0):
    """Robustes Parsen von Live-Score-Werten wie '72/100'."""
    try:
        txt = str(value or "").strip().replace(",", ".")
        if "/" in txt:
            txt = txt.split("/", 1)[0]
        f = float(txt)
        if pd.isna(f) or not np.isfinite(f):
            return default
        return int(round(max(0, min(100, f))))
    except Exception:
        return default


def _v237_set_live_row(row, *, ampel=None, status=None, stability=None, reason_prefix=None, action_prefix=None, prio=None):
    """Kleine Hilfsfunktion fuer die Signal-Hysterese."""
    if ampel is not None:
        row["Ampel"] = ampel
    if status is not None:
        row["Status"] = status
    if stability is not None:
        row["Signal-Stabilität"] = stability
    if prio is not None:
        row["__prio"] = prio
    if reason_prefix:
        old_reason = str(row.get("Grund") or "").strip()
        if old_reason and old_reason != "-" and reason_prefix not in old_reason:
            row["Grund"] = f"{reason_prefix} {old_reason}"
        else:
            row["Grund"] = reason_prefix.rstrip()
    if action_prefix:
        old_action = str(row.get("Nächste Handlung") or "").strip()
        if old_action and old_action != "-" and action_prefix not in old_action:
            row["Nächste Handlung"] = f"{action_prefix} {old_action}"
        else:
            row["Nächste Handlung"] = action_prefix.rstrip()
    return row


def _v237_apply_live_signal_hysteresis(row, prev):
    """Stabilisiert die Live-Ampel gegen Hin-und-her-Springen.

    Ziel: kleine Score-/Trigger-Schwankungen duerfen nicht sofort Gruen/Gelb/Weiss
    wechseln und damit Overtrading erzeugen. Rot/Invalidierung bleibt sofort wirksam.
    """
    row = row.copy()
    raw_ampel = str(row.get("Ampel") or "").strip()
    raw_status = str(row.get("Status") or "").strip()
    score = _v237_parse_live_score(row.get("Live-Score"), default=0)
    prev = prev if isinstance(prev, dict) else {}
    prev_ampel = str(prev.get("ampel") or "").strip()
    prev_status = str(prev.get("status") or "").strip()
    prev_raw_ampel = str(prev.get("raw_ampel") or "").strip()
    prev_score = _v237_parse_live_score(prev.get("live_score"), default=score)

    # Default, falls nichts angepasst wird.
    row["Signal-Stabilität"] = "Bestätigt" if prev_ampel == raw_ampel and prev_status == raw_status and prev_status else "Frisch"
    row["__raw_ampel"] = raw_ampel
    row["__raw_status"] = raw_status

    # Harte rote Signale nicht weichzeichnen.
    if raw_ampel == "🔴" or "invalid" in raw_status.lower() or "blockiert" in raw_status.lower() or "meiden" in raw_status.lower():
        return _v237_set_live_row(row, stability="Defensiv", prio=5)

    prev_was_green = prev_ampel == "🟢"
    prev_was_yellow = prev_ampel == "🟡"
    prev_was_white = prev_ampel == "⚪"

    # Gruen werden: entweder sehr klarer Score oder zwei Checks hintereinander roh gruen.
    # Erster knapper Gruen-Impuls wird als Gelb/Fast gruen gezeigt.
    if raw_ampel == "🟢":
        # v24.0: Sehr klare Kurzfrist-Signale sollen nicht komplett gelb versteckt
        # werden. Knappe Gruensignale brauchen weiter Bestaetigung, aber ab ca. 78/100
        # darf Gruen sofort sichtbar sein.
        if prev_was_green or prev_raw_ampel == "🟢" or score >= 78:
            return _v237_set_live_row(row, stability="Bestätigt" if (prev_was_green or prev_raw_ampel == "🟢") else "Frisch", prio=0)
        return _v237_set_live_row(
            row,
            ampel="🟡",
            status="Fast grün / Bestätigung abwarten",
            stability="Frisch",
            reason_prefix="Hysterese: erstes/knappes grünes Signal wird erst nach Bestätigung voll grün.",
            action_prefix="Nicht sofort aggressiv handeln; nächsten Check bzw. Triggerbestätigung abwarten.",
            prio=2,
        )

    # Gruen halten: wenn ein bestehendes gruenes Signal nur leicht auf gelb/weiss faellt,
    # nicht sofort rauswerfen. Erst bei klarer Verschlechterung unter ca. 68/100 abwerten.
    if prev_was_green and raw_ampel in {"🟡", "🔵", "⚪"}:
        if score >= 68 and prev_score >= 72:
            return _v237_set_live_row(
                row,
                ampel="🟢",
                status="Aktiv, aber wackelig",
                stability="Wackelig",
                reason_prefix="Hysterese: vorher grün und nur leicht abgeschwächt; kein hektisches Rein/Raus.",
                action_prefix="Bestehende Idee prüfen/halten, aber neue Käufe defensiver planen.",
                prio=0,
            )
        if score >= 58:
            return _v237_set_live_row(
                row,
                ampel="🟡",
                status="Signal abgeschwächt",
                stability="Abgeschwächt",
                reason_prefix="Hysterese: grünes Signal ist schwächer geworden, aber noch nicht klar gebrochen.",
                action_prefix="Nicht hektisch drehen; Trigger, Stop und Volumen im nächsten Check bestätigen lassen.",
                prio=2,
            )

    # Gelb werden: knappe gelbe Signale brauchen ebenfalls etwas Bestätigung.
    if raw_ampel == "🟡":
        if prev_was_yellow or prev_raw_ampel == "🟡" or score >= 62:
            return _v237_set_live_row(row, stability="Bestätigt" if prev_was_yellow else "Frisch", prio=2)
        if prev_was_white and score < 62:
            return _v237_set_live_row(
                row,
                ampel="⚪",
                status="Fast gelb / beobachten",
                stability="Frisch",
                reason_prefix="Hysterese: erstes knappes gelbes Signal, noch keine bestätigte Handlungsnähe.",
                action_prefix="Weiter beobachten; erst bei zweitem Check oder stärkerem Score gelb behandeln.",
                prio=4,
            )

    # Weiss werden: von Gelb nicht sofort auf Weiss kippen, solange Score noch in der Naehe ist.
    if raw_ampel == "⚪" and prev_was_yellow and score >= 50:
        return _v237_set_live_row(
            row,
            ampel="🟡",
            status="Weiter beobachten / wackelig",
            stability="Wackelig",
            reason_prefix="Hysterese: vorher gelb, aktuell nur leicht schwächer; Signal noch nicht vollständig verwerfen.",
            action_prefix="Keine neuen aggressiven Käufe; auf erneute Triggernähe oder klare Schwäche warten.",
            prio=2,
        )

    # Blau wird im Live-Monitor nur als Kontext verstanden; nicht springen lassen.
    if raw_ampel == "🔵":
        if score >= 55:
            return _v237_set_live_row(row, ampel="🟡", status="CRV attraktiv / Trigger offen", stability="Bestätigt" if prev_was_yellow else "Frisch", prio=2)
        return _v237_set_live_row(row, ampel="⚪", status="CRV beobachten", stability="Frisch", prio=4)

    return row


def _v240_live_trade_state(row):
    """v24.0: Operative Trade-State-Machine fuer kurzfristiges Trading.

    Die Ampel bleibt die schnelle Farbe, der Trade-State beschreibt den Workflow:
    Beobachten -> Vorbereiten -> Armed/Bereit -> Trigger aktiv -> Abgeschwaecht/Invalidiert.
    Ein echter "Trade aktiv" wird bewusst noch nicht automatisch gesetzt, weil dafuer
    ein dokumentierter Einstieg/Positions-Tracker noetig ist.
    """
    try:
        ampel = str(row.get("Ampel") or "").strip()
        status = str(row.get("Status") or "").strip()
        stability = str(row.get("Signal-Stabilität") or row.get("Signal-Stabilitaet") or "").strip()
        reason = str(row.get("Grund") or "").strip()
        action = str(row.get("Nächste Handlung") or row.get("Naechste Handlung") or "").strip()
        score = _v237_parse_live_score(row.get("Live-Score"), default=0)
        try:
            conf_txt = str(row.get("Bestätigungen") or "1x").lower().replace("x", "").strip()
            confirmations = int(float(conf_txt)) if conf_txt else 1
        except Exception:
            confirmations = 1
        low = " ".join([status, stability, reason, action]).lower()

        if ampel == "🔴" or any(x in low for x in ["invalidiert", "meiden", "blockiert", "kein kauf"]):
            return "Invalidiert / kein Trade", "Kein neuer Trade. These, Trigger und Invalidierung zuerst neu prüfen."

        if "abgeschw" in low:
            return "Abgeschwächt", "Signal hat nachgelassen: keine aggressiven Neueinstiege; Stop/Trigger erneut prüfen."
        if "wackelig" in low:
            return "Aktiv, aber wackelig", "Signal ist noch nicht stabil: Positionsgröße defensiv und nächsten Check abwarten."

        if ampel == "🟢":
            if confirmations >= 2 and stability == "Bestätigt":
                return "Trigger aktiv", "Trade planbar: Entry, Stop, Risiko und Stückzahl festlegen."
            if score >= 82:
                return "Armed / bereit", "Sehr starkes frisches Signal: Entry/Stop jetzt konkret planen, aber Ausführung sauber bestätigen."
            return "Armed / Bestätigung offen", "Grünes Signal ist frisch: nicht blind hinterherlaufen; Bestätigung/Volumen und Stop prüfen."

        if ampel == "🟡":
            if "fast gr" in low:
                return "Fast armed", "Noch nicht voll aktiv: nächsten Check bzw. Triggerbestätigung abwarten."
            if any(x in low for x in ["nahe", "trigger offen", "crv attraktiv", "selektiv", "pullback", "vorbereiten"]):
                return "Vorbereiten", "Setup vorbereiten: Alarm-/Triggerlevel, Stop und Risikoplan festlegen; noch kein Vollsignal."
            return "Beobachten+", "Interessant, aber noch nicht aktiv genug."

        if ampel == "🔵":
            return "CRV-Kontext", "Chance/Risiko beobachten, aber erst bei Charttrigger handeln."

        return "Beobachten", "Kein kurzfristiger Trade-State. Weiter beobachten."
    except Exception:
        return "Beobachten", "Trade-State konnte nicht eindeutig bestimmt werden."


def _v227_live_history_file_path():
    """Persistente Status-Historie fuer Live-Monitor-Reloads.

    Der Auto-Refresh kann auf Streamlit Cloud eine neue Session erzeugen. Deshalb reicht
    st.session_state allein nicht: ohne persistente Ablage wuerde jeder Refresh wieder
    als "Neu" erscheinen. Die Datei liegt bewusst lokal zur App/Instanz und wird nur
    fuer den geoeffneten Live-Monitor genutzt.
    """
    try:
        base_dir = Path(os.environ.get("LIVE_WATCHLIST_HISTORY_DIR", "."))
        if not base_dir.is_absolute():
            base_dir = Path.cwd() / base_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / ".live_watchlist_status_history_v227.json"
    except Exception:
        try:
            return Path("/tmp/.live_watchlist_status_history_v227.json")
        except Exception:
            return None


def _v227_load_persistent_live_history():
    storage = _CONTEXT.get("storage")
    if storage is not None:
        try:
            payload = storage.load_namespace("live_history", default=None)
            if isinstance(payload, dict):
                state = payload.get("state", {})
                events = payload.get("events", [])
                return (state if isinstance(state, dict) else {}), (events if isinstance(events, list) else [])[-500:]
        except Exception:
            pass
    path = _v227_live_history_file_path()
    if path is None or not path.exists():
        return {}, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = payload.get("state", {}) if isinstance(payload, dict) else {}
        events = payload.get("events", []) if isinstance(payload, dict) else []
        if not isinstance(state, dict):
            state = {}
        if not isinstance(events, list):
            events = []
        return state, events[-500:]
    except Exception:
        return {}, []


def _v227_save_persistent_live_history(state, events):
    payload = {"state": state if isinstance(state, dict) else {}, "events": (events or [])[-500:]}
    storage = _CONTEXT.get("storage")
    if storage is not None:
        try:
            storage.save_namespace("live_history", payload)
        except Exception:
            pass
    path = _v227_live_history_file_path()
    if path is None:
        return
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception:
        pass


def reset_live_watchlist_status_history_v227():
    st.session_state.live_watchlist_status_state_v220 = {}
    st.session_state.live_watchlist_status_events_v220 = []
    storage = _CONTEXT.get("storage")
    if storage is not None:
        try:
            storage.delete_namespace("live_history")
        except Exception:
            pass
    path = _v227_live_history_file_path()
    try:
        if path is not None and path.exists():
            path.unlink()
    except Exception:
        pass


def apply_live_watchlist_status_history_v220(live_df, *, watchlist_name="", style_name=""):
    """Ergaenzt Live-Watchlist um Statuswechsel und schreibt Session-Historie.

    v22.7: Die Historie wird zusaetzlich lokal persistiert, damit ein Auto-Refresh
    nicht wieder alle Zeilen als "Neu" markiert.
    """
    if live_df is None or live_df.empty:
        return live_df, pd.DataFrame()

    # Session-State initialisieren; falls durch Browser-/Streamlit-Reload leer,
    # aus der persistenten Datei laden.
    persistent_state, persistent_events = _v227_load_persistent_live_history()
    if "live_watchlist_status_state_v220" not in st.session_state or not isinstance(st.session_state.get("live_watchlist_status_state_v220"), dict) or not st.session_state.get("live_watchlist_status_state_v220"):
        st.session_state.live_watchlist_status_state_v220 = persistent_state
    if "live_watchlist_status_events_v220" not in st.session_state or not isinstance(st.session_state.get("live_watchlist_status_events_v220"), list) or not st.session_state.get("live_watchlist_status_events_v220"):
        st.session_state.live_watchlist_status_events_v220 = persistent_events

    state = st.session_state.live_watchlist_status_state_v220
    events = st.session_state.live_watchlist_status_events_v220
    now = get_current_berlin_time().strftime("%d.%m.%Y %H:%M:%S")
    enriched = live_df.copy()
    if "Signal-Stabilität" not in enriched.columns:
        enriched["Signal-Stabilität"] = ""
    changes = []
    prev_labels = []

    for idx, row in enriched.iterrows():
        ticker = str(row.get("Ticker") or "").strip().upper()
        if not ticker:
            changes.append("-")
            prev_labels.append("-")
            continue
        key = f"{watchlist_name or 'default'}::{style_name or ''}::{ticker}"
        prev = state.get(key, {}) if isinstance(state.get(key, {}), dict) else {}
        prev_ampel = prev.get("ampel")
        prev_status = prev.get("status")
        # v23.8: Hysterese vor der Statuswechsel-Bewertung anwenden.
        # Dadurch werden kleine Score-/Trigger-Schwankungen nicht als harte
        # Gruen/Gelb/Weiss-Wechsel angezeigt.
        row2 = _v237_apply_live_signal_hysteresis(row, prev)
        for col, val in row2.items():
            enriched.at[idx, col] = val

        new_ampel = str(row2.get("Ampel") or "").strip()
        new_status = str(row2.get("Status") or "").strip()

        # v23.8: Stabilität nicht schon beim ersten gleich aussehenden Scan als
        # "Bestätigt" behandeln. Alte persistierte States hatten noch keinen
        # Confirmation-Counter; diese werden bewusst wie 0 Vorbestätigungen
        # behandelt. Erst der zweite aufeinanderfolgende gleiche finale Status
        # wird als bestätigt markiert.
        try:
            prev_confirmations = int(prev.get("confirmations") or 0)
        except Exception:
            prev_confirmations = 0
        same_final_signal = bool(prev_ampel == new_ampel and prev_status == new_status and prev_status)
        confirmations = (prev_confirmations + 1) if same_final_signal else 1

        current_stability = str(row2.get("Signal-Stabilität") or "").strip()
        if new_ampel == "🔴" or current_stability == "Defensiv":
            current_stability = "Defensiv"
        elif current_stability in {"Wackelig", "Abgeschwächt"}:
            # Diese Zustände sind bewusst warnender als eine reine Bestätigung.
            pass
        elif confirmations >= 2:
            current_stability = "Bestätigt"
        else:
            current_stability = "Frisch"
        row2["Signal-Stabilität"] = current_stability
        row2["Bestätigungen"] = f"{confirmations}x"

        # v24.0: Aus der finalen Ampel + Stabilitaet einen operativen Trade-State ableiten.
        # Das reduziert Overtrading: Gruen ist nicht automatisch "rein", sondern ein
        # Workflow-Zustand wie Armed, Trigger aktiv oder wackelig.
        trade_state, trade_action = _v240_live_trade_state(row2)
        row2["Trade-State"] = trade_state
        row2["Trade-Aktion"] = trade_action

        enriched.at[idx, "Signal-Stabilität"] = current_stability
        enriched.at[idx, "Bestätigungen"] = f"{confirmations}x"
        enriched.at[idx, "Trade-State"] = trade_state
        enriched.at[idx, "Trade-Aktion"] = trade_action

        change = _v220_live_change_label(prev_ampel, prev_status, new_ampel, new_status)
        prev_label = "-" if prev_status is None else f"{prev_ampel} {prev_status}".strip()
        changes.append(change)
        prev_labels.append(prev_label)

        current_snapshot = {
            "ampel": new_ampel,
            "status": new_status,
            "raw_ampel": str(row2.get("__raw_ampel") or str(row.get("Ampel") or "")),
            "raw_status": str(row2.get("__raw_status") or str(row.get("Status") or "")),
            "stability": current_stability,
            "confirmations": confirmations,
            "radar_bucket": str(row2.get("Radar-Bucket") or ""),
            "live_score": str(row2.get("Live-Score") or ""),
            "grade": str(row2.get("Grade") or ""),
            "crv": str(row2.get("CRV") or ""),
            "price": row2.get("Kurs"),
            "updated": now,
            "reason": str(row2.get("Grund") or ""),
            "trade_state": str(row2.get("Trade-State") or ""),
            # v28.4.3: Die wichtigsten Score-/Gate-Bausteine werden je Ticker
            # gespeichert. So kann ein Wechsel auch bei gleichem Kurs konkret
            # auf Trigger, Timing, Konfluenz oder ein neues Gate zurueckgefuehrt werden.
            "timing_component": row2.get("__timing_component"),
            "conf_component": row2.get("__conf_component"),
            "chart_component": row2.get("__chart_component"),
            "trigger_component": row2.get("__trigger_component"),
            "trend_component": row2.get("__trend_component"),
            "crv_component": row2.get("__crv_component"),
            "entry_hard_gate": bool(row2.get("__entry_hard_gate")),
            "invalidated": bool(row2.get("__invalidated")),
            "final_release_ok": bool(row2.get("__final_release_ok")),
            "bucket_active": bool(row2.get("__bucket_active")),
            "bucket_near": bool(row2.get("__bucket_near")),
            "entry_reached": bool(row2.get("__entry_reached")),
            "wave_active": bool(row2.get("__wave_active")),
            "ma20_stretch_pct": row2.get("__ma20_stretch_pct"),
            "gate": str(row2.get("__gate") or ""),
            "final_blockers": str(row2.get("__final_blockers") or ""),
        }
        change_explanation = build_change_explanation(prev, current_snapshot, change)
        row2["Warum geändert?"] = change_explanation
        enriched.at[idx, "Warum geändert?"] = change_explanation
        current_snapshot["change_explanation"] = change_explanation
        if change != "Unverändert":
            events.append({
                "Zeit": now,
                "Ticker": ticker,
                "Änderung": change,
                "Von": prev_label,
                "Zu": f"{new_ampel} {new_status}".strip(),
                "Kurs": row2.get("Kurs"),
                "Live-Score": row2.get("Live-Score"),
                "Grade": row2.get("Grade"),
                "Radar-Bucket": row2.get("Radar-Bucket"),
                "CRV": row2.get("CRV"),
                "Signal-Stabilität": row2.get("Signal-Stabilität"),
                "Bestätigungen": row2.get("Bestätigungen"),
                "Trade-State": row2.get("Trade-State"),
                "Trade-Aktion": row2.get("Trade-Aktion"),
                "Warum geändert?": change_explanation,
                "Grund": row2.get("Grund"),
                "Nächste Handlung": row2.get("Nächste Handlung"),
            })
        # v25.1: Statuswechsel dauerhaft als Event protokollieren.
        if change != "Unverändert":
            event_type = "Statuswechsel"
            if new_ampel == "🟢" and prev_ampel != "🟢":
                event_type = "Neues Grünsignal"
            elif new_ampel == "🔴":
                event_type = "Invalidierung / Rot"
            elif change == "Verbessert":
                event_type = "Signal verbessert"
            elif change == "Verschlechtert":
                event_type = "Signal verschlechtert"
            _v2416_log_event(
                event_type=event_type,
                ticker=ticker,
                watchlist_name=watchlist_name,
                source="Live-Screener",
                status=f"{new_ampel} {new_status}".strip(),
                price=row2.get("Kurs"),
                score=row2.get("Live-Score"),
                trade_state=row2.get("Trade-State"),
                details=str(change_explanation or row2.get("Grund") or row2.get("Nächste Handlung") or ""),
                payload={
                    "Vorher": prev_label,
                    "Änderung": change,
                    "CRV": row2.get("CRV"),
                    "Warum geändert?": change_explanation,
                },
                signature=f"{prev_ampel}|{prev_status}->{new_ampel}|{new_status}|{row2.get('Trade-State')}",
            )
        state[key] = current_snapshot

    # Interne Roh-/Sortier-Spalten nicht anzeigen.
    # v24.1: Hysterese kann __prio temporaer wieder einfuegen; solche
    # Hilfsspalten duerfen nicht in der Live-Monitor-Tabelle landen.
    for _internal_col in [c for c in enriched.columns if str(c).startswith("__")]:
        if _internal_col in enriched.columns:
            enriched = enriched.drop(columns=[_internal_col])
    # Signal-Stabilität und Bestätigungen direkt neben Status platzieren, falls Pandas sie ans Ende gesetzt hat.
    if "Signal-Stabilität" in enriched.columns:
        col_vals = enriched.pop("Signal-Stabilität")
        insert_pos = 2 if "Status" in enriched.columns else min(3, len(enriched.columns))
        enriched.insert(insert_pos, "Signal-Stabilität", col_vals)
    if "Bestätigungen" in enriched.columns:
        col_vals = enriched.pop("Bestätigungen")
        insert_pos = 3 if "Signal-Stabilität" in enriched.columns else (2 if "Status" in enriched.columns else min(4, len(enriched.columns)))
        enriched.insert(insert_pos, "Bestätigungen", col_vals)
    if "Trade-State" in enriched.columns:
        col_vals = enriched.pop("Trade-State")
        insert_pos = 4 if "Bestätigungen" in enriched.columns else (3 if "Signal-Stabilität" in enriched.columns else min(5, len(enriched.columns)))
        enriched.insert(insert_pos, "Trade-State", col_vals)
    if "Trade-Aktion" in enriched.columns:
        col_vals = enriched.pop("Trade-Aktion")
        insert_pos = 5 if "Trade-State" in enriched.columns else min(6, len(enriched.columns))
        enriched.insert(insert_pos, "Trade-Aktion", col_vals)
    enriched.insert(1, "Änderung", changes)
    enriched.insert(2, "Vorher", prev_labels)
    st.session_state.live_watchlist_status_state_v220 = state
    st.session_state.live_watchlist_status_events_v220 = events[-500:]
    _v227_save_persistent_live_history(st.session_state.live_watchlist_status_state_v220, st.session_state.live_watchlist_status_events_v220)

    events_df = pd.DataFrame(st.session_state.live_watchlist_status_events_v220)
    if not events_df.empty:
        events_df = events_df.iloc[::-1].reset_index(drop=True)
    return enriched, events_df
