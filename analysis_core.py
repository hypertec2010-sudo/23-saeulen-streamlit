# -*- coding: utf-8 -*-
"""Core analysis logic extracted from app.py for headless auto-run stability."""

import os

import re

import json

import base64

import warnings

from pathlib import Path

from datetime import datetime, timezone, date, timedelta

import gspread

import numpy as np

import pandas as pd

import plotly.graph_objects as go

import requests

import streamlit as st

import yfinance as yf

from plotly.subplots import make_subplots

from auth_utils import check_password

from logging_utils import (
    append_auto_run_log,
    append_df_to_gsheet,
    add_entries_to_watchlist,
    build_export_df,
    build_trigger_log_df,
    create_watchlist,
    delete_watchlist,
    get_current_berlin_time,
    get_current_schedule_slot,
    get_due_watchlists_for_slot,
    get_watchlist_alert_mode,
    get_watchlist_catalog_df,
    get_watchlist_check_frequency,
    get_watchlist_tickers,
    load_watchlists_df,
    remove_ticker_from_watchlist,
    update_watchlist_alert_mode,
    update_watchlist_check_frequency,
)

try:
    import telegram_utils as _telegram_utils
except Exception:
    _telegram_utils = None

if _telegram_utils is not None and hasattr(_telegram_utils, "send_telegram_message"):
    send_telegram_message = _telegram_utils.send_telegram_message
else:
    def send_telegram_message(*args, **kwargs):
        return False, "Telegram-Funktion in telegram_utils nicht verfügbar"

if _telegram_utils is not None and hasattr(_telegram_utils, "send_watchlist_alerts"):
    send_watchlist_alerts = _telegram_utils.send_watchlist_alerts
else:
    def send_watchlist_alerts(*args, **kwargs):
        return False, "Watchlist-Telegram-Funktion in telegram_utils nicht verfügbar"

from ui_helpers import show_sheet_result

warnings.filterwarnings("ignore")

APP_VERSION = "v12.6C.14"

if not check_password():
    st.stop()

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

if "search_input" not in st.session_state:
    st.session_state.search_input = "AAPL"

if "selected_search_label" not in st.session_state:
    st.session_state.selected_search_label = None

if "analysis_ticker" not in st.session_state:
    st.session_state.analysis_ticker = "AAPL"

if "analysis_requested" not in st.session_state:
    st.session_state.analysis_requested = False

if "batch_input" not in st.session_state:
    st.session_state.batch_input = ""

if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "Einzelanalyse"

if "analysis_mode_run" not in st.session_state:
    st.session_state.analysis_mode_run = "Einzelanalyse"

if "ranking_df" not in st.session_state:
    st.session_state.ranking_df = pd.DataFrame()

if "ranking_results" not in st.session_state:
    st.session_state.ranking_results = {}

if "selected_ranking_ticker" not in st.session_state:
    st.session_state.selected_ranking_ticker = "AAPL"

if "last_mode_label" not in st.session_state:
    st.session_state.last_mode_label = "Watchlist"

if "selected_watchlist_name" not in st.session_state:
    st.session_state.selected_watchlist_name = ""

if "selected_watchlist_type" not in st.session_state:
    st.session_state.selected_watchlist_type = "Watchlist"

if "watchlist_new_name" not in st.session_state:
    st.session_state.watchlist_new_name = ""

if "watchlist_bulk_add" not in st.session_state:
    st.session_state.watchlist_bulk_add = ""

if "selected_watchlist_alert_mode" not in st.session_state:
    st.session_state.selected_watchlist_alert_mode = "Standard"

if "selected_watchlist_check_frequency" not in st.session_state:
    st.session_state.selected_watchlist_check_frequency = "4x täglich"

if "auto_run_requested" not in st.session_state:
    st.session_state.auto_run_requested = False

if "auto_run_slot_label" not in st.session_state:
    st.session_state.auto_run_slot_label = ""

if "auto_run_summary" not in st.session_state:
    st.session_state.auto_run_summary = ""

if "run_selected_watchlist_name" not in st.session_state:
    st.session_state.run_selected_watchlist_name = ""

if "run_selected_watchlist_type" not in st.session_state:
    st.session_state.run_selected_watchlist_type = "Watchlist"

if "send_watchlist_alerts_after_run" not in st.session_state:
    st.session_state.send_watchlist_alerts_after_run = False

if "workspace_mode" not in st.session_state:
    st.session_state.workspace_mode = ""

if "ui_refresh_nonce" not in st.session_state:
    st.session_state.ui_refresh_nonce = 0

def trigger_ui_refresh(**state_updates):
    for key, value in state_updates.items():
        st.session_state[key] = value
    st.session_state.ui_refresh_nonce += 1
    st.rerun()

def ampel(v, g=65, y=45):
    return "🟢" if v >= g else ("🟡" if v >= y else "🔴")

def ampel_tb(score):
    if score >= 9:
        return "🟢"
    if score >= 5:
        return "🟡"
    if score >= 3:
        return "🟠"
    return "🔴"

def ampel_crv(c):
    return "🟢" if c >= 2.5 else ("🟡" if c >= 1.5 else "🔴")

def card_class(score):
    try:
        s = float(score)
        if np.isnan(s):
            return "yellow"
    except Exception:
        return "yellow"
    return "" if s >= 70 else ("yellow" if s >= 45 else "red")

def safe_last(s, default=np.nan):
    try:
        v = s.iloc[-1]
        return default if pd.isna(v) else float(v)
    except Exception:
        return default

def clamp(v, lo=0, hi=100):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo

def fmt_num(x, digits=2, suffix=""):
    return f"{x:.{digits}f}{suffix}" if pd.notna(x) else "n/a"

def known_ratio(values):
    vals = [v for v in values if pd.notna(v)]
    return len(vals) / len(values) if values else 0

def normalize_missing(v):
    if v is None:
        return np.nan
    if isinstance(v, str) and v.strip().lower() in {"", "none", "nan", "null"}:
        return np.nan
    try:
        if pd.isna(v):
            return np.nan
    except Exception:
        pass
    return v

def is_missing_scalar(v):
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in {"", "none", "nan", "null", "n/a"}:
        return True
    try:
        if np.isscalar(v) and pd.isna(v):
            return True
    except Exception:
        pass
    return False

def merge_info(base, extra):
    base = dict(base or {})
    for k, v in dict(extra or {}).items():
        cur = base.get(k)
        if k not in base or is_missing_scalar(cur):
            nv = normalize_missing(v)
            if not is_missing_scalar(nv):
                base[k] = nv
    return base

def split_batch_input(text):
    parts = re.split(r"[\n,;|]+", str(text or ""))
    cleaned = []
    seen = set()
    for p in parts:
        x = p.strip()
        if not x:
            continue
        key = x.lower()
        if key not in seen:
            cleaned.append(x)
            seen.add(key)
    return cleaned[:25]

def score_badge(score):
    if score >= 80:
        return "Exzellent", "#16a34a"
    if score >= 70:
        return "Stark", "#65a30d"
    if score >= 60:
        return "Solide", "#ca8a04"
    if score >= 50:
        return "Gemischt", "#ea580c"
    return "Schwach", "#dc2626"

def style_ranking_df(df):
    score_cols = [
        "Company Quality",
        "Setup Quality",
        "Investment Score",
        "Investment-Attraktivität",
        "Einstieg jetzt attraktiv?",
        "Trade-Struktur",
        "Setup-Confidence",
        "Kurzfrist-Timing",
        "TradingBoard Score",
        "Fundamental-Confidence",
    ]

    def cell_style(v):
        try:
            _, color = score_badge(float(v))
            return f"background-color: {color}; color: white; font-weight:700;"
        except Exception:
            return ""

    def entry_style(v):
        val = str(v or "").strip().lower()
        if val == "gut":
            return "background-color: rgba(34,197,94,0.28); color: white; font-weight:700;"
        if val == "abwarten":
            return "background-color: rgba(245,158,11,0.28); color: white; font-weight:700;"
        if val == "früh":
            return "background-color: rgba(239,68,68,0.28); color: white; font-weight:700;"
        return ""

    styler = df.style
    for c in score_cols:
        if c in df.columns:
            styler = styler.map(cell_style, subset=[c])

    if "Entry-Lage" in df.columns:
        styler = styler.map(entry_style, subset=["Entry-Lage"])

    def trigger_style(v):
        val = str(v or "").strip().lower()
        if val in {"aktiv", "nahe dran"}:
            return "background-color: rgba(34,197,94,0.28); color: white; font-weight:700;"
        if val in {"frühe beobachtung", "beobachten", "warten"}:
            return "background-color: rgba(245,158,11,0.28); color: white; font-weight:700;"
        if val in {"passiv"}:
            return "background-color: rgba(239,68,68,0.28); color: white; font-weight:700;"
        return ""

    def priority_style(v):
        val = str(v or "").strip().lower()
        if val == "hoch":
            return "background-color: rgba(34,197,94,0.28); color: white; font-weight:700;"
        if val == "mittel":
            return "background-color: rgba(245,158,11,0.28); color: white; font-weight:700;"
        if val == "niedrig":
            return "background-color: rgba(239,68,68,0.28); color: white; font-weight:700;"
        return ""

    if "Trigger-Status" in df.columns:
        styler = styler.map(trigger_style, subset=["Trigger-Status"])
    if "Watchlist-Priorität" in df.columns:
        styler = styler.map(priority_style, subset=["Watchlist-Priorität"])

    return styler

def market_regime_label(regime):
    mapping = {
        "POSITIV": "Positiv",
        "NEGATIV": "Negativ",
        "NEUTRAL": "Neutral",
        "UNBEKANNT": "Keine belastbaren Benchmark-Daten",
    }
    return mapping.get(str(regime or "").upper(), str(regime or "-"))

def shorten_text(value, max_len=42):
    if value is None:
        return "-"
    txt = str(value).strip()
    if not txt:
        return "-"
    if len(txt) <= max_len:
        return txt
    clipped = txt[: max_len - 3]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    clipped = clipped.rstrip(" ,;:-")
    return clipped + "..."

def display_mode_label(mode_label):
    mapping = {
        "Watchlist": "Beobachtung",
        "Position": "Bestehende Position",
    }
    return mapping.get(str(mode_label or ""), str(mode_label or "-"))

def display_emp_label(emp):
    mapping = {
        "BUY / ACCUMULATE": "Kauf / Aufbau",
        "WATCH / EINSTIEG PRÜFEN": "Beobachten / Einstieg prüfen",
        "BEOBACHTEN": "Weiter beobachten",
        "AVOID / WAIT": "Aktuell kein Einstieg",
        "HALTEN / AUSBAUEN": "Halten / ggf. ausbauen",
        "HALTEN / ENGE BEOBACHTUNG": "Halten / eng beobachten",
        "HALTEN / RISIKO PRÜFEN": "Halten / Risiko prüfen",
        "RISIKO REDUZIEREN / STOPP PRÜFEN": "Risiko senken / Stop prüfen",
        "NO TRADE": "Kein valides Setup",
        "NO TRADE / WAIT": "Noch kein valides Setup",
        "VETO - Earnings < 7 Tage": "Kein Trade vor Zahlen",
    }
    return mapping.get(str(emp or ""), str(emp or "-"))

def display_conv_label(conv):
    mapping = {
        "HIGH": "hoch",
        "MEDIUM": "mittel",
        "LOW-MEDIUM": "eher verhalten",
        "LOW": "niedrig",
        "NONE": "keine",
        "-": "-",
    }
    return mapping.get(str(conv or ""), str(conv or "-"))

def display_stb_label(signal):
    mapping = {
        "LONG": "Bullisch",
        "HOLD": "Neutral / Halten",
        "WAIT": "Abwarten",
        "SHORT": "Schwach / defensiv",
    }
    return mapping.get(str(signal or ""), str(signal or "-"))

def horizon_status_meta(score):
    try:
        s = float(score)
    except Exception:
        return ("blue", "n/a", "keine belastbare Einschätzung")
    if s >= 75:
        return ("green", "stark", "spricht klar für diesen Horizont")
    if s >= 60:
        return ("blue", "konstruktiv", "insgesamt brauchbar bis positiv")
    if s >= 45:
        return ("amber", "gemischt", "brauchbar, aber nicht überzeugend")
    return ("red", "schwach", "der Horizont wirkt aktuell belastet")

def horizon_icon(score):
    try:
        s = float(score)
    except Exception:
        return "•"
    if s >= 75:
        return "🟢"
    if s >= 60:
        return "🔵"
    if s >= 45:
        return "🟠"
    return "🔴"

SECTOR_ETF_MAP = {
    "technology": "XLK",
    "information technology": "XLK",
    "financial services": "XLF",
    "financial": "XLF",
    "financials": "XLF",
    "healthcare": "XLV",
    "health care": "XLV",
    "consumer cyclical": "XLY",
    "consumer discretionary": "XLY",
    "consumer defensive": "XLP",
    "consumer staples": "XLP",
    "industrials": "XLI",
    "industrial": "XLI",
    "energy": "XLE",
    "basic materials": "XLB",
    "materials": "XLB",
    "real estate": "XLRE",
    "utilities": "XLU",
    "communication services": "XLC",
    "communication": "XLC",
}

def normalize_sector_name(sector):
    if not sector:
        return ""
    s = str(sector).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\s+", " ", s)
    return s

def get_sector_etf_symbol(sector):
    s = normalize_sector_name(sector)
    if not s:
        return None
    if s in SECTOR_ETF_MAP:
        return SECTOR_ETF_MAP[s]
    if "technology" in s:
        return "XLK"
    if "financial" in s:
        return "XLF"
    if "health" in s:
        return "XLV"
    if "consumer discretionary" in s or "cyclical" in s:
        return "XLY"
    if "consumer staples" in s or "defensive" in s:
        return "XLP"
    if "industrial" in s:
        return "XLI"
    if "energy" in s:
        return "XLE"
    if "material" in s:
        return "XLB"
    if "real estate" in s:
        return "XLRE"
    if "utilit" in s:
        return "XLU"
    if "communication" in s:
        return "XLC"
    return None

def load_sector_context(symbol):
    if not symbol:
        return None
    try:
        df = yf.download(symbol, period="2y", auto_adjust=True, progress=False)
        if df is None or df.empty or len(df) < 120:
            return None
        close = df["Close"]
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        sector_rsi_series = rsi14(close)
        return {
            "symbol": symbol,
            "price": float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else np.nan,
            "ma50": float(ma50.iloc[-1]) if pd.notna(ma50.iloc[-1]) else np.nan,
            "ma200": float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else np.nan,
            "ret21": float(close.pct_change(21).iloc[-1] * 100) if pd.notna(close.pct_change(21).iloc[-1]) else np.nan,
            "ret63": float(close.pct_change(63).iloc[-1] * 100) if pd.notna(close.pct_change(63).iloc[-1]) else np.nan,
            "rsi": float(sector_rsi_series.iloc[-1]) if pd.notna(sector_rsi_series.iloc[-1]) else np.nan,
        }
    except Exception:
        return None

def calc_sector_strength_score(sector_ctx):
    if not sector_ctx:
        return np.nan
    score = 0
    sector_ret21 = sector_ctx.get("ret21", np.nan)
    sector_ret63 = sector_ctx.get("ret63", np.nan)
    sector_price = sector_ctx.get("price", np.nan)
    sector_ma50 = sector_ctx.get("ma50", np.nan)
    sector_ma200 = sector_ctx.get("ma200", np.nan)
    sector_rsi = sector_ctx.get("rsi", np.nan)
    if pd.notna(sector_ret21):
        if sector_ret21 > 5:
            score += 25
        elif sector_ret21 > 0:
            score += 15
        elif sector_ret21 > -5:
            score += 8
    if pd.notna(sector_ret63):
        if sector_ret63 > 10:
            score += 25
        elif sector_ret63 > 0:
            score += 15
        elif sector_ret63 > -8:
            score += 8
    if pd.notna(sector_price) and pd.notna(sector_ma50) and sector_price > sector_ma50:
        score += 20
    if pd.notna(sector_price) and pd.notna(sector_ma200) and sector_price > sector_ma200:
        score += 20
    if pd.notna(sector_rsi):
        if 50 <= sector_rsi <= 70:
            score += 10
        elif 40 <= sector_rsi < 50:
            score += 5
    return round(min(100, score))

def calc_rs_benchmark_score(rs21, rs63, rs126):
    score = 0
    if pd.notna(rs21):
        if rs21 > 8:
            score += 30
        elif rs21 > 3:
            score += 20
        elif rs21 > 0:
            score += 12
        elif rs21 > -5:
            score += 6
    if pd.notna(rs63):
        if rs63 > 12:
            score += 35
        elif rs63 > 5:
            score += 24
        elif rs63 > 0:
            score += 14
        elif rs63 > -6:
            score += 7
    if pd.notna(rs126):
        if rs126 > 15:
            score += 35
        elif rs126 > 5:
            score += 22
        elif rs126 > 0:
            score += 12
        elif rs126 > -8:
            score += 6
    return round(min(100, score))

def calc_rs_acceleration_score(rs21, rs63, rs126):
    score = 50
    if pd.notna(rs21) and pd.notna(rs63):
        if rs21 > rs63 + 3:
            score += 20
        elif rs21 > rs63:
            score += 10
        elif rs21 < rs63 - 3:
            score -= 18
        elif rs21 < rs63:
            score -= 8
    if pd.notna(rs63) and pd.notna(rs126):
        if rs63 > rs126 + 4:
            score += 18
        elif rs63 < rs126 - 4:
            score -= 15
    return round(clamp(score))

def calc_industry_strength_score(sector_strength_score, rs_score, company_score):
    if pd.isna(sector_strength_score):
        sector_strength_score = 50
    if pd.isna(rs_score):
        rs_score = 50
    if pd.isna(company_score):
        company_score = 50
    return round(clamp(
        sector_strength_score * 0.60
        + rs_score * 0.25
        + company_score * 0.15
    ))

def calc_leadership_score(sector_strength_score, industry_strength_score, rs_benchmark_score, rs_acceleration_score, rs_score):
    return round(clamp(
        sector_strength_score * 0.22
        + industry_strength_score * 0.13
        + rs_benchmark_score * 0.40
        + rs_acceleration_score * 0.15
        + rs_score * 0.10
    ))

def get_leadership_status(score, accel_score):
    if pd.isna(score):
        return "-"
    if score >= 80 and accel_score >= 60:
        return "Leader"
    if score >= 68:
        return "Stark"
    if score >= 55:
        return "Solide"
    if score >= 40:
        return "Mitläufer"
    return "Schwach"

def strength_text(score):
    if pd.isna(score):
        return "nicht belastbar"
    if score >= 75:
        return "stark"
    if score >= 60:
        return "konstruktiv"
    if score >= 45:
        return "gemischt"
    return "schwach"

def score_or_unavailable_text(score):
    if pd.isna(score):
        return "nicht verfügbar"
    return f"{int(round(float(score)))}/100"

def calc_slope_pct(series, lookback=20):
    try:
        series = series.dropna()
        if series is None or len(series) < lookback + 1:
            return np.nan
        latest = float(series.iloc[-1])
        prev = float(series.iloc[-lookback])
        if prev == 0:
            return np.nan
        return ((latest / prev) - 1) * 100
    except Exception:
        return np.nan

def calc_higher_lows_score(close, low):
    try:
        if len(close) < 80 or len(low) < 80:
            return np.nan
        recent_low_1 = float(low.iloc[-20:].min())
        recent_low_2 = float(low.iloc[-40:-20].min())
        recent_low_3 = float(low.iloc[-60:-40].min())
        score = 50
        if recent_low_1 > recent_low_2:
            score += 20
        else:
            score -= 12
        if recent_low_2 > recent_low_3:
            score += 20
        else:
            score -= 12
        if float(close.iloc[-1]) > recent_low_1:
            score += 10
        return round(clamp(score))
    except Exception:
        return np.nan

def calc_trend_quality_score(price, ma20, ma50, ma200, ma20_slope, ma50_slope, ma200_slope, higher_lows_score):
    score = 0
    if pd.notna(price) and pd.notna(ma20) and price > ma20:
        score += 10
    if pd.notna(price) and pd.notna(ma50) and price > ma50:
        score += 15
    if pd.notna(price) and pd.notna(ma200) and price > ma200:
        score += 15
    if pd.notna(ma20) and pd.notna(ma50) and ma20 > ma50:
        score += 12
    if pd.notna(ma50) and pd.notna(ma200) and ma50 > ma200:
        score += 12
    if pd.notna(ma20_slope) and ma20_slope > 0:
        score += 10
    if pd.notna(ma50_slope) and ma50_slope > 0:
        score += 10
    if pd.notna(ma200_slope) and ma200_slope > 0:
        score += 6
    if pd.notna(higher_lows_score):
        if higher_lows_score >= 70:
            score += 10
        elif higher_lows_score >= 55:
            score += 6
    if pd.notna(price) and pd.notna(ma20):
        dist_to_ma20 = abs((price / ma20 - 1) * 100)
        if dist_to_ma20 < 8:
            score += 5
    if pd.notna(price) and pd.notna(ma50):
        dist_to_ma50 = abs((price / ma50 - 1) * 100)
        if dist_to_ma50 < 15:
            score += 5
    return round(clamp(score))

def calc_base_length_days(close, ma20):
    try:
        if len(close) < 80 or pd.isna(ma20):
            return np.nan
        recent = close.tail(60)
        recent_max = float(recent.max())
        recent_min = float(recent.min())
        if recent_min <= 0:
            return np.nan
        range_pct = ((recent_max / recent_min) - 1) * 100
        if range_pct <= 18:
            return 60
        if range_pct <= 25:
            return 40
        if range_pct <= 35:
            return 20
        return 10
    except Exception:
        return np.nan

def calc_correction_depth_pct(close):
    try:
        if len(close) < 80:
            return np.nan
        recent = close.tail(60)
        peak = float(recent.max())
        trough = float(recent.min())
        if peak <= 0:
            return np.nan
        return ((peak - trough) / peak) * 100
    except Exception:
        return np.nan

def calc_range_tightness_score(close):
    try:
        if len(close) < 40:
            return np.nan
        recent = close.tail(20)
        rmax = float(recent.max())
        rmin = float(recent.min())
        if rmin <= 0:
            return np.nan
        range_pct = ((rmax / rmin) - 1) * 100
        if range_pct <= 4:
            return 90
        if range_pct <= 7:
            return 75
        if range_pct <= 10:
            return 60
        if range_pct <= 15:
            return 45
        return 25
    except Exception:
        return np.nan

def calc_volatility_contraction_score(atr_pct_series, bb_width_s):
    try:
        score = 50
        if atr_pct_series is not None and len(atr_pct_series.dropna()) >= 40:
            atr_recent = float(atr_pct_series.tail(10).mean())
            atr_past = float(atr_pct_series.iloc[-40:-20].mean())
            if pd.notna(atr_recent) and pd.notna(atr_past):
                if atr_recent < atr_past * 0.75:
                    score += 25
                elif atr_recent < atr_past * 0.90:
                    score += 12
                elif atr_recent > atr_past * 1.15:
                    score -= 15
        if bb_width_s is not None and len(bb_width_s.dropna()) >= 40:
            bb_recent = float(bb_width_s.tail(10).mean())
            bb_past = float(bb_width_s.iloc[-40:-20].mean())
            if pd.notna(bb_recent) and pd.notna(bb_past):
                if bb_recent < bb_past * 0.75:
                    score += 25
                elif bb_recent < bb_past * 0.90:
                    score += 12
                elif bb_recent > bb_past * 1.15:
                    score -= 15
        return round(clamp(score))
    except Exception:
        return np.nan

def calc_pullback_quality_score(price, ma20, ma50, rsi, atr_pct, ret20):
    score = 50
    if pd.notna(price) and pd.notna(ma20):
        dist20 = abs((price / ma20 - 1) * 100)
        if dist20 <= 4:
            score += 18
        elif dist20 <= 8:
            score += 10
    if pd.notna(price) and pd.notna(ma50):
        dist50 = abs((price / ma50 - 1) * 100)
        if dist50 <= 6:
            score += 14
        elif dist50 <= 12:
            score += 7
    if pd.notna(rsi):
        if 45 <= rsi <= 60:
            score += 10
        elif 38 <= rsi < 45:
            score += 6
        elif rsi < 30:
            score -= 10
    if pd.notna(atr_pct):
        if atr_pct <= 4:
            score += 10
        elif atr_pct > 8:
            score -= 10
    if pd.notna(ret20):
        if -8 <= ret20 <= 8:
            score += 8
        elif ret20 < -15:
            score -= 12
    return round(clamp(score))

def calc_base_quality_score(base_length_days, correction_depth_pct, range_tightness_score, volatility_contraction_score, pullback_quality_score):
    score = 0
    if pd.notna(base_length_days):
        if 15 <= base_length_days <= 60:
            score += 20
        elif 8 <= base_length_days < 15:
            score += 10
    if pd.notna(correction_depth_pct):
        if 8 <= correction_depth_pct <= 22:
            score += 20
        elif 5 <= correction_depth_pct < 8 or 22 < correction_depth_pct <= 30:
            score += 10
    if pd.notna(range_tightness_score):
        if range_tightness_score >= 70:
            score += 20
        elif range_tightness_score >= 55:
            score += 12
    if pd.notna(volatility_contraction_score):
        if volatility_contraction_score >= 65:
            score += 20
        elif volatility_contraction_score >= 50:
            score += 10
    if pd.notna(pullback_quality_score):
        if pullback_quality_score >= 65:
            score += 20
        elif pullback_quality_score >= 50:
            score += 10
    return round(clamp(score))

def calc_volume_quality_proxy(vol_ratio, obv_trend):
    score = 50
    if pd.notna(vol_ratio):
        if 0.9 <= vol_ratio <= 1.4:
            score += 15
        elif vol_ratio > 1.4:
            score += 10
        elif vol_ratio < 0.7:
            score -= 10
    if str(obv_trend).strip().lower() == "steigend":
        score += 15
    elif str(obv_trend).strip().lower() == "fallend":
        score -= 10
    return round(clamp(score))

def calc_setup_type_quality_score(setup_type, base_quality_score, volume_quality_proxy, rs_score, trend_quality_score, setup_confidence, pullback_quality_score):
    setup_type = str(setup_type or "").strip()
    if setup_type in ["Breakout", "Range-Breakout", "Breakout-Retest"]:
        score = (
            base_quality_score * 0.35
            + volume_quality_proxy * 0.20
            + rs_score * 0.20
            + trend_quality_score * 0.15
            + setup_confidence * 0.10
        )
    elif setup_type in ["Pullback an MA20", "Pullback an MA50", "Trendfolge"]:
        score = (
            trend_quality_score * 0.35
            + pullback_quality_score * 0.25
            + rs_score * 0.15
            + setup_confidence * 0.15
            + base_quality_score * 0.10
        )
    elif setup_type == "Rebound":
        score = (
            pullback_quality_score * 0.30
            + trend_quality_score * 0.20
            + setup_confidence * 0.20
            + rs_score * 0.15
            + base_quality_score * 0.15
        )
    else:
        score = (
            setup_confidence * 0.50
            + trend_quality_score * 0.25
            + base_quality_score * 0.25
        )
    return round(clamp(score))

def calc_setup_priority_score(setup_type_quality_score, leadership_score, trend_quality_score, base_quality_score, trading_case_score):
    return round(clamp(
        setup_type_quality_score * 0.45
        + leadership_score * 0.20
        + trend_quality_score * 0.15
        + base_quality_score * 0.10
        + trading_case_score * 0.10
    ))

def calc_up_down_volume_ratio(close, vol, lookback=20):
    try:
        if len(close) < lookback + 2 or len(vol) < lookback + 2:
            return np.nan
        recent_close = close.tail(lookback + 1)
        recent_vol = vol.tail(lookback + 1)
        delta = recent_close.diff()
        up_mask = delta > 0
        down_mask = delta < 0
        up_volume = recent_vol[up_mask].mean()
        down_volume = recent_vol[down_mask].mean()
        if pd.isna(down_volume) or down_volume == 0 or pd.isna(up_volume):
            return np.nan
        return float(up_volume / down_volume)
    except Exception:
        return np.nan

def calc_accumulation_distribution_days(close, vol, lookback=20):
    try:
        if len(close) < lookback + 2 or len(vol) < lookback + 2:
            return 0, 0
        ret1 = close.pct_change(1) * 100
        vol_ma20 = vol.rolling(20).mean()
        recent_ret = ret1.tail(lookback)
        recent_vol = vol.tail(lookback)
        recent_vol_ma = vol_ma20.tail(lookback)
        accumulation_mask = (recent_ret >= 1.0) & (recent_vol > 1.2 * recent_vol_ma)
        distribution_mask = (recent_ret <= -1.5) & (recent_vol > 1.2 * recent_vol_ma)
        return int(accumulation_mask.fillna(False).sum()), int(distribution_mask.fillna(False).sum())
    except Exception:
        return 0, 0

def calc_volume_trend_score(vol, close):
    try:
        if len(vol) < 60 or len(close) < 60:
            return np.nan
        vol_recent = float(vol.tail(10).mean())
        vol_mid = float(vol.iloc[-30:-10].mean())
        ret20_s = close.pct_change(20) * 100
        ret20 = float(ret20_s.iloc[-1]) if pd.notna(ret20_s.iloc[-1]) else np.nan
        score = 50
        if pd.notna(vol_recent) and pd.notna(vol_mid):
            if vol_recent > vol_mid * 1.15 and pd.notna(ret20) and ret20 > 0:
                score += 20
            elif vol_recent < vol_mid * 0.90 and pd.notna(ret20) and ret20 < 0:
                score -= 15
            elif vol_recent > vol_mid * 1.10:
                score += 8
        return round(clamp(score))
    except Exception:
        return np.nan

def calc_recent_pullback_volume_ratio(close, vol, lookback=10):
    try:
        if len(close) < lookback + 5 or len(vol) < lookback + 20:
            return np.nan
        ret1 = close.pct_change(1)
        pullback_mask = ret1.tail(lookback) < 0
        if pullback_mask.fillna(False).sum() == 0:
            return np.nan
        pullback_vol = vol.tail(lookback)[pullback_mask].mean()
        vol_ma20 = vol.rolling(20).mean().iloc[-1]
        if pd.isna(pullback_vol) or pd.isna(vol_ma20) or vol_ma20 == 0:
            return np.nan
        return float(pullback_vol / vol_ma20)
    except Exception:
        return np.nan

def calc_breakout_day_volume_ratio(close, vol, lookback=20):
    try:
        if len(close) < lookback + 5 or len(vol) < lookback + 20:
            return np.nan
        prev_high = float(close.shift(1).rolling(lookback).max().iloc[-1])
        current_close = float(close.iloc[-1])
        current_vol = float(vol.iloc[-1])
        vol_ma20 = float(vol.rolling(20).mean().iloc[-1])
        if pd.isna(prev_high) or pd.isna(current_close) or pd.isna(current_vol) or pd.isna(vol_ma20) or vol_ma20 == 0:
            return np.nan
        if not (current_close >= prev_high * 0.995):
            return np.nan
        return float(current_vol / vol_ma20)
    except Exception:
        return np.nan

def calc_close_near_day_high(close, high, low):
    try:
        c = float(close.iloc[-1]); h = float(high.iloc[-1]); l = float(low.iloc[-1])
        if pd.isna(c) or pd.isna(h) or pd.isna(l) or h <= l:
            return False
        return c >= (l + 0.75 * (h - l))
    except Exception:
        return False

def calc_accumulation_score(up_down_volume_ratio, obv_trend, accumulation_day_count, strong_up_volume_day):
    score = 50
    if pd.notna(up_down_volume_ratio):
        if up_down_volume_ratio >= 1.25:
            score += 18
        elif up_down_volume_ratio >= 1.10:
            score += 10
        elif up_down_volume_ratio < 0.90:
            score -= 12
    if str(obv_trend).strip().lower() == "steigend":
        score += 15
    elif str(obv_trend).strip().lower() == "fallend":
        score -= 10
    if accumulation_day_count >= 4:
        score += 18
    elif accumulation_day_count >= 2:
        score += 10
    if strong_up_volume_day:
        score += 10
    return round(clamp(score))

def calc_distribution_pressure_score(distribution_day_count, obv_trend, down_volume_heavy, weak_rebound_on_volume):
    score = 0
    if distribution_day_count >= 1:
        score += 18
    if distribution_day_count >= 2:
        score += 20
    if distribution_day_count >= 3:
        score += 20
    if str(obv_trend).strip().lower() == "fallend":
        score += 15
    if down_volume_heavy:
        score += 12
    if weak_rebound_on_volume:
        score += 10
    return round(clamp(score))

def calc_pullback_dryup_score(pullback_active, recent_pullback_volume_ratio, pullback_quality_score, volatility_contraction_score):
    score = 40
    if pullback_active:
        if pd.notna(recent_pullback_volume_ratio):
            if recent_pullback_volume_ratio <= 0.75:
                score += 30
            elif recent_pullback_volume_ratio <= 0.90:
                score += 18
            elif recent_pullback_volume_ratio > 1.10:
                score -= 12
        if pd.notna(pullback_quality_score):
            if pullback_quality_score >= 65:
                score += 15
            elif pullback_quality_score >= 50:
                score += 8
        if pd.notna(volatility_contraction_score):
            if volatility_contraction_score >= 60:
                score += 15
            elif volatility_contraction_score >= 50:
                score += 8
    else:
        score = 50
    return round(clamp(score))

def calc_breakout_volume_score(breakout_context, breakout_day_volume_ratio, rs_score, close_near_day_high, breakout_failure_risk_low=True):
    score = 45
    if breakout_context:
        if pd.notna(breakout_day_volume_ratio):
            if breakout_day_volume_ratio >= 1.8:
                score += 28
            elif breakout_day_volume_ratio >= 1.4:
                score += 18
            elif breakout_day_volume_ratio >= 1.1:
                score += 10
            else:
                score -= 10
        if pd.notna(rs_score) and rs_score >= 65:
            score += 10
        if close_near_day_high:
            score += 10
        if breakout_failure_risk_low:
            score += 8
    else:
        score = 50
    return round(clamp(score))

def calc_volume_quality_score(accumulation_score, distribution_pressure_score, pullback_dryup_score, breakout_volume_score, volume_trend_score):
    if pd.isna(volume_trend_score):
        volume_trend_score = 50
    return round(clamp(
        accumulation_score * 0.30
        + (100 - distribution_pressure_score) * 0.22
        + pullback_dryup_score * 0.18
        + breakout_volume_score * 0.20
        + volume_trend_score * 0.10
    ))

def classify_event_phase(days_earn):
    if pd.isna(days_earn):
        return "kein Eventfenster"
    if days_earn >= 0:
        if days_earn <= 7:
            return "earnings_unmittelbar"
        if days_earn <= 21:
            return "earnings_bald"
        return "earnings_spaeter"
    else:
        if abs(days_earn) <= 7:
            return "frisch_nach_earnings"
        if abs(days_earn) <= 30:
            return "post_earnings_fenster"
        return "kein Eventfenster"

def calc_post_earnings_reaction(close, days_earn):
    try:
        if pd.isna(days_earn) or days_earn >= 0:
            return np.nan, np.nan
        days_since = int(abs(days_earn))
        if days_since < 1 or len(close) < 15:
            return np.nan, np.nan
        lookback_5 = min(days_since, 5)
        lookback_10 = min(days_since, 10)
        reaction_5d_s = close.pct_change(lookback_5) * 100
        reaction_10d_s = close.pct_change(lookback_10) * 100
        reaction_5d = float(reaction_5d_s.iloc[-1]) if pd.notna(reaction_5d_s.iloc[-1]) else np.nan
        reaction_10d = float(reaction_10d_s.iloc[-1]) if pd.notna(reaction_10d_s.iloc[-1]) else np.nan
        return reaction_5d, reaction_10d
    except Exception:
        return np.nan, np.nan

def calc_post_earnings_reaction_score(reaction_5d, reaction_10d, rs_score):
    score = 50
    if pd.notna(reaction_5d):
        if reaction_5d >= 6:
            score += 22
        elif reaction_5d >= 3:
            score += 14
        elif reaction_5d <= -6:
            score -= 24
        elif reaction_5d <= -3:
            score -= 14
    if pd.notna(reaction_10d):
        if reaction_10d >= 8:
            score += 18
        elif reaction_10d >= 4:
            score += 10
        elif reaction_10d <= -8:
            score -= 18
        elif reaction_10d <= -4:
            score -= 10
    if pd.notna(rs_score):
        if rs_score >= 65:
            score += 8
        elif rs_score <= 40:
            score -= 8
    return round(clamp(score))

def calc_event_risk_score(days_earn, has_upcoming_earnings, atr_pct, breakout_context):
    score = 20
    if has_upcoming_earnings and pd.notna(days_earn):
        if days_earn <= 3:
            score += 45
        elif days_earn <= 7:
            score += 35
        elif days_earn <= 14:
            score += 20
        elif days_earn <= 21:
            score += 10
    if pd.notna(atr_pct):
        if atr_pct >= 6:
            score += 12
        elif atr_pct >= 4:
            score += 6
    if breakout_context and has_upcoming_earnings and pd.notna(days_earn) and days_earn <= 7:
        score += 12
    return round(clamp(score))

def calc_revision_momentum_score(upside, revenue_growth, earnings_growth, ret21, rs_score):
    score = 50
    if pd.notna(upside):
        if upside >= 20:
            score += 14
        elif upside >= 10:
            score += 8
        elif upside <= -10:
            score -= 12
    if pd.notna(revenue_growth):
        if revenue_growth >= 0.15:
            score += 12
        elif revenue_growth >= 0.05:
            score += 6
        elif revenue_growth < 0:
            score -= 10
    if pd.notna(earnings_growth):
        if earnings_growth >= 0.15:
            score += 14
        elif earnings_growth >= 0.05:
            score += 8
        elif earnings_growth < 0:
            score -= 12
    if pd.notna(ret21):
        if ret21 >= 8:
            score += 10
        elif ret21 <= -8:
            score -= 10
    if pd.notna(rs_score):
        if rs_score >= 65:
            score += 10
        elif rs_score <= 40:
            score -= 10
    return round(clamp(score))

def calc_earnings_event_score(event_phase_label, event_risk_score, post_earnings_reaction_score):
    if event_phase_label in ["earnings_unmittelbar", "earnings_bald"]:
        return round(clamp(100 - event_risk_score))
    if event_phase_label in ["frisch_nach_earnings", "post_earnings_fenster"]:
        if pd.notna(post_earnings_reaction_score):
            return round(clamp(post_earnings_reaction_score))
        return 50
    return 55

def calc_catalyst_score(earnings_event_score, revision_momentum_score, post_earnings_reaction_score, event_phase_label):
    post_score = 50 if pd.isna(post_earnings_reaction_score) else post_earnings_reaction_score
    score = (
        earnings_event_score * 0.40
        + revision_momentum_score * 0.35
        + post_score * 0.25
    )
    if event_phase_label == "earnings_unmittelbar":
        score -= 8
    elif event_phase_label == "frisch_nach_earnings" and post_score >= 65:
        score += 6
    return round(clamp(score))

def event_phase_text(label):
    mapping = {
        "earnings_unmittelbar": "Earnings unmittelbar bevorstehend",
        "earnings_bald": "Earnings in Kürze",
        "earnings_spaeter": "Earnings später",
        "frisch_nach_earnings": "Frisch nach Earnings",
        "post_earnings_fenster": "Post-Earnings-Fenster",
        "kein Eventfenster": "Kein relevantes Eventfenster",
    }
    return mapping.get(label, "Kein relevantes Eventfenster")

def catalyst_label(score):
    if pd.isna(score):
        return "-"
    if score >= 80:
        return "stark"
    if score >= 65:
        return "konstruktiv"
    if score >= 50:
        return "neutral"
    if score >= 35:
        return "sensibel"
    return "schwach"

def calc_cashflow_stability_score(fcf, op_cf, market_cap, revenue_growth, earnings_growth):
    score = 50
    if pd.notna(fcf):
        if fcf > 0:
            score += 18
        else:
            score -= 15
    if pd.notna(op_cf):
        if op_cf > 0:
            score += 16
        else:
            score -= 12
    if pd.notna(market_cap) and market_cap > 0 and pd.notna(fcf):
        fcf_yield_pct = (fcf / market_cap) * 100
        if fcf_yield_pct >= 4:
            score += 10
        elif fcf_yield_pct >= 1:
            score += 5
        elif fcf_yield_pct < 0:
            score -= 8
    if pd.notna(revenue_growth):
        if revenue_growth >= 0.05:
            score += 4
        elif revenue_growth < 0:
            score -= 5
    if pd.notna(earnings_growth):
        if earnings_growth >= 0.05:
            score += 4
        elif earnings_growth < 0:
            score -= 6
    return round(clamp(score))

def calc_margin_stability_score(profit_margin, oper_margin, gross_margin, roe, roa):
    score = 50
    if pd.notna(gross_margin):
        if gross_margin >= 0.45:
            score += 12
        elif gross_margin >= 0.25:
            score += 6
        elif gross_margin < 0.10:
            score -= 8
    if pd.notna(oper_margin):
        if oper_margin >= 0.20:
            score += 14
        elif oper_margin >= 0.10:
            score += 8
        elif oper_margin < 0:
            score -= 15
    if pd.notna(profit_margin):
        if profit_margin >= 0.15:
            score += 12
        elif profit_margin >= 0.07:
            score += 6
        elif profit_margin < 0:
            score -= 14
    if pd.notna(roe):
        if roe >= 0.18:
            score += 8
        elif roe >= 0.10:
            score += 4
    if pd.notna(roa):
        if roa >= 0.08:
            score += 4
        elif roa < 0:
            score -= 6
    return round(clamp(score))

def calc_institutional_quality_score(cashflow_stability_score, margin_stability_score, balance_score, quality_score, risk_score):
    risk_component = 100 - risk_score if pd.notna(risk_score) else 50
    return round(clamp(
        cashflow_stability_score * 0.30
        + margin_stability_score * 0.28
        + balance_score * 0.18
        + quality_score * 0.14
        + risk_component * 0.10
    ))

def institutional_quality_label(score):
    if pd.isna(score):
        return "-"
    if score >= 80:
        return "sehr stark"
    if score >= 65:
        return "stark"
    if score >= 50:
        return "solide"
    if score >= 35:
        return "gemischt"
    return "schwach"

def diag_direction_from_score(score, inverse=False):
    if pd.isna(score):
        return "neutral"
    s = float(score)
    if not inverse:
        if s >= 65:
            return "positiv"
        if s >= 45:
            return "neutral"
        return "negativ"
    else:
        if s < 35:
            return "positiv"
        if s < 60:
            return "neutral"
        return "negativ"

def diag_direction_class(direction):
    direction = str(direction).strip().lower()
    if direction == "positiv":
        return "diag-pos"
    if direction == "negativ":
        return "diag-neg"
    return "diag-neu"

def make_diag_item(section, label, value, affects, impact="mittel", inverse=False, note=""):
    direction = diag_direction_from_score(value, inverse=inverse)
    return {
        "section": section,
        "label": label,
        "value": value,
        "direction": direction,
        "impact": impact,
        "affects": affects or [],
        "inverse": inverse,
        "note": note,
    }

def affects_text(affects):
    if not affects:
        return "-"
    return ", ".join([str(x) for x in affects if str(x).strip()])

def build_diagnostic_impacts(result):
    items = []
    items.append(make_diag_item("Setup & Timing", "Trendqualität", result.get("trend_quality_score", np.nan), ["Trading-Case", "Setup-Priorität"], impact="mittel"))
    items.append(make_diag_item("Setup & Timing", "Base-Qualität", result.get("base_quality_score", np.nan), ["Trading-Case", "Setup-Priorität"], impact="hoch"))
    items.append(make_diag_item("Setup & Timing", "Setup-Typ-Qualität", result.get("setup_type_quality_score", np.nan), ["Trading-Case", "Setup-Priorität"], impact="hoch"))
    items.append(make_diag_item("Setup & Timing", "Setup-Priorität", result.get("setup_priority_score", np.nan), ["Trading-Case"], impact="hoch"))

    items.append(make_diag_item("Volumen & Akkumulation", "Volumenqualität", result.get("volume_quality_score", np.nan), ["Trading-Case", "Setup-Priorität"], impact="mittel"))
    items.append(make_diag_item("Volumen & Akkumulation", "Akkumulation", result.get("accumulation_score", np.nan), ["Trading-Case"], impact="mittel"))
    items.append(make_diag_item("Volumen & Akkumulation", "Distribution", result.get("distribution_pressure_score", np.nan), ["Trading-Case", "Exit"], impact="hoch", inverse=True))
    items.append(make_diag_item("Volumen & Akkumulation", "Pullback-Dry-up", result.get("pullback_dryup_score", np.nan), ["Trading-Case"], impact="niedrig"))
    items.append(make_diag_item("Volumen & Akkumulation", "Breakout-Volumen", result.get("breakout_volume_score", np.nan), ["Trading-Case", "Setup-Priorität"], impact="niedrig"))

    items.append(make_diag_item("Event & Katalysator", "Katalysator", result.get("catalyst_score", np.nan), ["Investment-Case", "Setup-Priorität"], impact="mittel"))
    items.append(make_diag_item("Event & Katalysator", "Event-Score", result.get("earnings_event_score", np.nan), ["Trading-Case"], impact="mittel"))
    items.append(make_diag_item("Event & Katalysator", "Event-Risiko", result.get("event_risk_score", np.nan), ["Trading-Case", "Exit"], impact="hoch", inverse=True))
    items.append(make_diag_item("Event & Katalysator", "Revision/Momentum", result.get("revision_momentum_score", np.nan), ["Investment-Case", "Katalysator"], impact="mittel"))

    items.append(make_diag_item("Qualität & Fundamentals", "Institutionelle Qualität", result.get("institutional_quality_score", np.nan), ["Investment-Case"], impact="hoch"))
    items.append(make_diag_item("Qualität & Fundamentals", "Cashflow-Stabilität", result.get("cashflow_stability_score", np.nan), ["Investment-Case"], impact="mittel"))
    items.append(make_diag_item("Qualität & Fundamentals", "Margenstabilität", result.get("margin_stability_score", np.nan), ["Investment-Case"], impact="mittel"))
    items.append(make_diag_item("Qualität & Fundamentals", "Leadership", result.get("leadership_score", np.nan), ["Investment-Case", "Trading-Case", "Setup-Priorität"], impact="hoch"))

    items.append(make_diag_item("Marktregime", "Regime-Fit", result.get("regime_fit_score", np.nan), ["Trading-Case", "Setup-Priorität", "Investment-Case"], impact="hoch"))
    items.append(make_diag_item("Marktregime", "Regime-Anpassung", result.get("regime_adjustment_score", np.nan), ["Trading-Case", "Investment-Case"], impact="hoch"))

    items.append(make_diag_item("Exit & Risiko", "Exit-Score", result.get("exit_score", np.nan), ["Exit"], impact="hoch", inverse=True))
    return items

def build_driver_summary(result, max_pos=5, max_neg=4):
    items = build_diagnostic_impacts(result)
    positives, negatives = [], []
    for item in items:
        val = item.get("value", np.nan)
        if pd.isna(val):
            continue
        entry = {
            "label": item.get("label", "-"),
            "value": val,
            "impact": item.get("impact", "mittel"),
            "affects": item.get("affects", []),
            "section": item.get("section", "Diagnose"),
            "note": item.get("note", ""),
        }
        if item.get("direction") == "positiv":
            positives.append(entry)
        elif item.get("direction") == "negativ":
            negatives.append(entry)

    impact_rank = {"hoch": 3, "mittel": 2, "niedrig": 1}
    positives = sorted(positives, key=lambda x: (impact_rank.get(x["impact"], 0), float(x["value"])), reverse=True)[:max_pos]
    negatives = sorted(negatives, key=lambda x: (impact_rank.get(x["impact"], 0), float(x["value"])), reverse=True)[:max_neg]
    return {"positives": positives, "negatives": negatives}

def render_reason_box(title, items, empty_text="Keine klaren Treiber erkannt."):
    st.markdown(f'<div class="reason-box"><div class="reason-title">{title}</div>', unsafe_allow_html=True)
    if not items:
        st.markdown(f'<div class="reason-item"><div class="reason-meta">{empty_text}</div></div></div>', unsafe_allow_html=True)
        return
    for item in items:
        label = item.get("label", "-")
        value = item.get("value", np.nan)
        impact = item.get("impact", "-")
        affects = affects_text(item.get("affects", []))
        st.markdown(
            f"""
            <div class="reason-item">
                <div class="reason-top">
                    <div class="reason-label">{label}</div>
                    <div class="reason-value">{fmt_num(value,0)}/100</div>
                </div>
                <div class="reason-meta">Einfluss: {impact}</div>
                <div class="affects-line">Beeinflusst: {affects}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def render_diagnostic_row(item):
    label = item.get("label", "-")
    value = item.get("value", np.nan)
    direction = item.get("direction", "neutral")
    impact = item.get("impact", "mittel")
    affects = affects_text(item.get("affects", []))
    note = item.get("note", "")
    chip_class = diag_direction_class(direction)
    st.markdown(
        f"""
        <div class="diag-row">
            <div class="diag-head">
                <div class="diag-label">{label}</div>
                <div class="diag-value">{fmt_num(value,0)}/100</div>
            </div>
            <div class="diag-sub">
                <span class="diag-chip {chip_class}">{direction}</span>
                <span class="diag-chip">{impact}</span>
            </div>
            <div class="affects-line">Beeinflusst: {affects}</div>
            {f'<div class="diag-sub">{note}</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_diagnostic_section(title, items):
    if not items:
        return
    st.markdown(f"**{title}**")
    for item in items:
        render_diagnostic_row(item)

def render_score_card(label, value, subtitle="", variant="company", tooltip=""):
    title_attr = f' title="{tooltip}"' if tooltip else ""
    st.markdown(
        f"""
        <div class="score-card {variant}"{title_attr}>
            <div class="score-label">{label}</div>
            <div class="score-value">{value}</div>
            <div class="score-delta">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def trading_timing_label(score):
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 9:
        return "stark"
    if s >= 5:
        return "konstruktiv"
    if s >= 3:
        return "abwartend"
    return "schwach"

def normalize_tb_score_100(score):
    try:
        s = float(score)
    except Exception:
        return np.nan
    return int(round(clamp((s + 3) / 12 * 100, 0, 100)))

def tradeability_label(score):
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 78:
        return "hoch handelbar"
    if s >= 62:
        return "brauchbar"
    if s >= 45:
        return "eingeschränkt"
    return "schwach"

def investment_case_label(score):
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 80:
        return "sehr attraktiv"
    if s >= 70:
        return "attraktiv"
    if s >= 55:
        return "okay"
    if s >= 40:
        return "schwach"
    return "unattraktiv"

def trading_case_label(score):
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 80:
        return "sehr attraktiv"
    if s >= 68:
        return "attraktiv"
    if s >= 52:
        return "brauchbar"
    if s >= 40:
        return "eher abwarten"
    return "nicht attraktiv"

def setup_confidence_label(score):
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 80:
        return "hoch"
    if s >= 60:
        return "mittel"
    if s >= 40:
        return "moderat"
    return "niedrig"

def linear_score(value, low, high, floor=10, ceiling=95):
    if pd.isna(value):
        return np.nan
    if high == low:
        return ceiling
    scaled = (float(value) - low) / (high - low)
    return clamp(floor + scaled * (ceiling - floor), floor, ceiling)

def ideal_range_score(value, ideal_low, ideal_high, hard_low, hard_high, floor=15, ceiling=92):
    if pd.isna(value):
        return np.nan
    v = float(value)
    if ideal_low <= v <= ideal_high:
        return ceiling
    if v < ideal_low:
        if v <= hard_low:
            return floor
        scaled = (v - hard_low) / (ideal_low - hard_low)
        return clamp(floor + scaled * (ceiling - floor), floor, ceiling)
    if v >= hard_high:
        return floor
    scaled = (hard_high - v) / (hard_high - ideal_high)
    return clamp(floor + scaled * (ceiling - floor), floor, ceiling)

def entry_quality_score(entry_quality, price, entry_low, entry_high):
    quality = str(entry_quality or "").lower()
    if quality == "gut":
        return 88
    if quality == "abwarten":
        if pd.notna(entry_high) and entry_high > 0 and pd.notna(price):
            gap_pct = (float(price) / float(entry_high) - 1) * 100
            return clamp(62 - gap_pct * 4.5, 20, 62)
        return 45
    if quality == "früh":
        if pd.notna(entry_low) and entry_low > 0 and pd.notna(price):
            gap_pct = (float(entry_low) / float(price) - 1) * 100
            return clamp(55 - max(gap_pct, 0) * 2.5, 25, 55)
        return 40
    return 45

def format_price_zone(low, high, ccy):
    if pd.isna(low) and pd.isna(high):
        return "-"
    if pd.notna(low) and pd.notna(high):
        if abs(float(low) - float(high)) < 0.01:
            return f"{float(low):.2f} {ccy}"
        return f"{float(low):.2f} - {float(high):.2f} {ccy}"
    if pd.notna(low):
        return f"ab {float(low):.2f} {ccy}"
    return f"bis {float(high):.2f} {ccy}"

def rsi14(close):
    d = close.diff()
    g = d.where(d > 0, 0.0).rolling(14).mean()
    l = (-d.where(d < 0, 0.0)).rolling(14).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def adx14(h, l, c):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up = h.diff()
    dn = -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0).rolling(14).mean()
    ndm = dn.where((dn > up) & (dn > 0), 0.0).rolling(14).mean()
    pdi = 100 * pdm / atr.replace(0, np.nan)
    ndi = 100 * ndm / atr.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.rolling(14).mean()

def true_range(high, low, close):
    return pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1
    ).max(axis=1)

def stoch14(high, low, close, k_period=14, d_period=3):
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d

def williams_r(high, low, close, period=14):
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

def bollinger_bands(close, period=20, num_std=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan) * 100
    return mid, upper, lower, width

def infer_display_currency(ticker, info, fallback="USD"):
    suffix = ticker.split(".")[-1].upper() if "." in ticker else ""
    exchange = str(info.get("exchange", "") or "").upper()
    info_ccy = str(info.get("currency", fallback) or fallback).upper()

    eur_suffixes = {"DE", "PA", "AS", "BR", "MI", "MC", "HE", "VI", "LS", "F"}
    if suffix in eur_suffixes:
        return "EUR"
    if suffix == "L":
        return "GBP"
    if suffix == "SW":
        return "CHF"
    if suffix == "ST":
        return "SEK"
    if suffix == "OL":
        return "NOK"
    if suffix == "CO":
        return "DKK"
    if suffix == "WA":
        return "PLN"
    if suffix == "PR":
        return "CZK"

    if exchange in {"XETRA", "GER", "PAR", "AMS", "MIL", "MAD", "HEL", "VIE", "BRU", "EURONEXT"}:
        return "EUR"
    if exchange in {"LSE"}:
        return "GBP"
    if exchange in {"SIX"}:
        return "CHF"

    return info_ccy if info_ccy else fallback

def analyst_label(rec_key):
    mapping = {
        "strong_buy": "Starker Kauf",
        "buy": "Kauf",
        "hold": "Halten",
        "underperform": "Unterdurchschnittlich",
        "sell": "Verkaufen",
        "strong_sell": "Starker Verkauf",
    }
    return mapping.get(str(rec_key).lower(), str(rec_key))

def tb_signal_label(score):
    if score >= 9:
        return "LONG", "AKTIV HALTEN"
    if score >= 5:
        return "HOLD", "HALTEN"
    if score >= 3:
        return "WAIT", "ABWARTEN"
    return "SHORT", "STOPP PRÜFEN"

def first_existing_row(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for name in names:
        if name in df.index:
            row = df.loc[name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return pd.to_numeric(row, errors="coerce")
    return None

def latest_valid(series_like):
    if series_like is None:
        return np.nan
    s = pd.to_numeric(pd.Series(series_like), errors="coerce").dropna()
    return float(s.iloc[0]) if len(s) else np.nan

def previous_valid(series_like):
    if series_like is None:
        return np.nan
    s = pd.to_numeric(pd.Series(series_like), errors="coerce").dropna()
    return float(s.iloc[1]) if len(s) > 1 else np.nan

def select_benchmark(ticker, info=None):
    suffix = ticker.split(".")[-1].upper() if "." in ticker else ""
    exchange = str((info or {}).get("exchange", "") or "").upper()

    german_suffixes = {"DE", "F", "HM", "BE", "DU", "MU", "SG"}
    europe_suffixes = {"PA", "AS", "BR", "MI", "MC", "HE", "VI", "LS"}
    swiss_suffixes = {"SW"}
    uk_suffixes = {"L"}
    nordic_suffixes = {"ST", "OL", "CO"}

    if suffix in german_suffixes or exchange in {"XETRA", "GER"}:
        return "^GDAXI", "DAX"
    if suffix in europe_suffixes or exchange in {"PAR", "AMS", "MIL", "MAD", "HEL", "VIE", "BRU", "EURONEXT"}:
        return "^STOXX50E", "STOXX 50"
    if suffix in swiss_suffixes or exchange in {"SIX"}:
        return "^SSMI", "SMI"
    if suffix in uk_suffixes or exchange in {"LSE"}:
        return "^FTSE", "FTSE 100"
    if suffix in nordic_suffixes:
        return "^STOXX50E", "STOXX 50"

    return "SPY", "S&P 500"

def calc_return_metrics(close_series):
    return {
        "ret21": safe_last(close_series.pct_change(21) * 100, np.nan),
        "ret63": safe_last(close_series.pct_change(63) * 100, np.nan),
        "ret126": safe_last(close_series.pct_change(126) * 100, np.nan),
    }

def evaluate_market_filter(benchmark_df):
    if benchmark_df is None or benchmark_df.empty or "Close" not in benchmark_df.columns:
        return {
            "price": np.nan,
            "ma50": np.nan,
            "ma200": np.nan,
            "ret21": np.nan,
            "ret63": np.nan,
            "ret126": np.nan,
            "regime": "UNBEKANNT",
            "ampel": "⚪",
            "score": 50
        }

    close = benchmark_df["Close"]
    price = safe_last(close, np.nan)
    ma50 = safe_last(close.rolling(50).mean(), np.nan)
    ma200 = safe_last(close.rolling(200).mean(), np.nan)

    rets = calc_return_metrics(close)

    if pd.notna(price) and pd.notna(ma50) and pd.notna(ma200):
        if price > ma50 and price > ma200 and ma50 > ma200:
            regime, ampel_icon, score = "POSITIV", "🟢", 100
        elif price < ma50 and price < ma200 and ma50 < ma200:
            regime, ampel_icon, score = "NEGATIV", "🔴", 30
        else:
            regime, ampel_icon, score = "NEUTRAL", "🟡", 60
    else:
        regime, ampel_icon, score = "UNBEKANNT", "⚪", 50

    return {
        "price": price,
        "ma50": ma50,
        "ma200": ma200,
        "ret21": rets["ret21"],
        "ret63": rets["ret63"],
        "ret126": rets["ret126"],
        "regime": regime,
        "ampel": ampel_icon,
        "score": score
    }

def infer_market_bucket(ticker, info):
    suffix = ticker.split(".")[-1].upper() if "." in ticker else ""
    exchange = str((info or {}).get("exchange", "") or "").upper()
    ccy = str((info or {}).get("currency", "") or "").upper()

    us_ex = {"NASDAQ", "NASDAQGS", "NYQ", "NYSE", "AMEX", "PCX"}
    eu_suffixes = {"DE", "PA", "AS", "BR", "MI", "MC", "HE", "VI", "LS", "F", "SW", "L", "ST", "OL", "CO"}

    if exchange in us_ex or ccy == "USD" and suffix == "":
        return "USA"
    if suffix in eu_suffixes or ccy in {"EUR", "GBP", "CHF", "SEK", "NOK", "DKK"}:
        return "Europa"
    return "Andere"

def infer_stock_style_advanced(revenue_growth, earnings_growth, pe, pb, beta, debt_to_equity, roe, profit_margin, sector):
    sector = str(sector or "").lower()
    if pd.notna(revenue_growth) and revenue_growth > 0.15 and pd.notna(pe) and pe > 25:
        return "Growth"
    if pd.notna(roe) and roe > 0.18 and pd.notna(profit_margin) and profit_margin > 0.15:
        return "Quality"
    if pd.notna(pe) and pe < 15 and pd.notna(pb) and pb < 2:
        return "Value"
    if pd.notna(beta) and beta > 1.5:
        return "Cyclical"
    if pd.notna(earnings_growth) and earnings_growth < 0 and pd.notna(debt_to_equity) and debt_to_equity > 150:
        return "Turnaround / Risk"
    if "utility" in sector or "consumer defensive" in sector:
        return "Defensive"
    return "Neutral"

def get_style_sector_adjustment(style, sector):
    sector = str(sector or "").lower()
    weights = {
        "quality": 1.0,
        "growth": 1.0,
        "valuation": 1.0,
        "balance": 1.0,
        "momentum": 1.0,
        "trend": 1.0,
    }

    if style == "Growth":
        weights["growth"] = 1.18
        weights["momentum"] = 1.10
        weights["valuation"] = 0.82
    elif style == "Quality":
        weights["quality"] = 1.15
        weights["balance"] = 1.08
    elif style == "Value":
        weights["valuation"] = 1.20
        weights["balance"] = 1.08
        weights["growth"] = 0.88
    elif style == "Defensive":
        weights["balance"] = 1.16
        weights["momentum"] = 0.90
    elif style == "Cyclical":
        weights["trend"] = 1.12
        weights["momentum"] = 1.12

    if "technology" in sector or "software" in sector or "semiconductor" in sector:
        weights["growth"] *= 1.10
        weights["trend"] *= 1.07
        weights["valuation"] *= 0.86
    elif "financial" in sector or "bank" in sector or "insurance" in sector:
        weights["balance"] *= 1.15
        weights["valuation"] *= 1.12
        weights["growth"] *= 0.90
    elif "health" in sector or "pharma" in sector or "biotech" in sector:
        weights["quality"] *= 1.08
        weights["balance"] *= 1.05
    elif "energy" in sector or "materials" in sector:
        weights["trend"] *= 1.12
        weights["momentum"] *= 1.05
    elif "consumer defensive" in sector or "utilities" in sector:
        weights["balance"] *= 1.18
        weights["momentum"] *= 0.88

    return weights

def build_red_flags(
    revenue_growth,
    earnings_growth,
    profit_margin,
    fcf,
    op_cf,
    debt_to_equity,
    current_ratio,
    quick_ratio,
    has_upcoming_earnings,
    days_earn
):
    items = []

    def add_item(category, detail, penalty):
        items.append({
            "Kategorie": category,
            "Status": "🔴" if penalty >= 6 else "🟡",
            "Detail": detail,
            "Penalty": penalty
        })

    if pd.notna(earnings_growth) and earnings_growth < -0.15:
        add_item("Ertrags-Risiko", "Gewinnwachstum stark negativ", 8)
    if pd.notna(profit_margin) and profit_margin < 0:
        add_item("Ertrags-Risiko", "Gewinnmarge negativ", 8)
    if pd.notna(revenue_growth) and revenue_growth < -0.10:
        add_item("Umsatz-/Geschäfts-Risiko", "Umsatzwachstum negativ", 6)
    if pd.notna(fcf) and fcf < 0:
        add_item("Cashflow-Risiko", "Freier Cashflow negativ", 6)
    if pd.notna(op_cf) and op_cf < 0:
        add_item("Cashflow-Risiko", "Operativer Cashflow negativ", 5)
    if pd.notna(debt_to_equity) and debt_to_equity > 180:
        add_item("Bilanz-Risiko", "Verschuldung sehr hoch", 8)

    current_ratio_weak = pd.notna(current_ratio) and current_ratio < 1.0
    quick_ratio_weak = pd.notna(quick_ratio) and quick_ratio < 0.8
    cashflow_weak = (
        (pd.notna(op_cf) and op_cf < 0)
        or (pd.notna(fcf) and fcf < 0)
    )
    cashflow_strong = (
        (pd.notna(op_cf) and op_cf > 0)
        and (pd.notna(fcf) and fcf > 0)
    )

    if current_ratio_weak and (quick_ratio_weak or cashflow_weak):
        add_item("Bilanz-Risiko", "Liquidität schwach (Current Ratio < 1.0)", 5)

    if quick_ratio_weak and not cashflow_strong:
        add_item("Bilanz-Risiko", "Quick Ratio schwach (< 0.8)", 4)
    if has_upcoming_earnings and pd.notna(days_earn) and days_earn <= 7:
        add_item("Event-Risiko", f"Earnings in {int(days_earn)} Tagen", 6)

    total_penalty = sum(x["Penalty"] for x in items)
    return items, total_penalty

def build_decision_explanation(
    setup,
    company,
    investment,
    market_regime,
    rs_vs_benchmark_63,
    quality_score,
    growth_score,
    valuation_score,
    balance_score,
    red_flag_items,
    earnings_warning,
    kb,
    position_mode
):
    strengths = []
    weaknesses = []

    if setup >= 75:
        strengths.append("Technisches Setup ist stark.")
    elif setup < 50:
        weaknesses.append("Technisches Setup ist derzeit schwach.")

    if company >= 70:
        strengths.append("Unternehmensqualität ist solide bis stark.")
    elif company < 50:
        weaknesses.append("Unternehmensqualität ist eher schwach.")

    if pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 > 5:
        strengths.append("Die Aktie zeigt klare Outperformance gegenüber dem Benchmark.")
    elif pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 < -5:
        weaknesses.append("Die Aktie underperformt ihren Benchmark spürbar.")

    if quality_score >= 75:
        strengths.append("Profitabilität und Kapitalrendite sind überzeugend.")
    if growth_score >= 75:
        strengths.append("Wachstum ist aktuell stark.")
    if valuation_score >= 72:
        strengths.append("Bewertung wirkt noch akzeptabel bis attraktiv.")
    elif valuation_score < 50:
        weaknesses.append("Bewertung wirkt anspruchsvoll.")

    if balance_score >= 72:
        strengths.append("Bilanzqualität ist ordentlich.")
    elif balance_score < 50:
        weaknesses.append("Bilanzqualität ist belastet.")

    if market_regime == "POSITIV":
        strengths.append("Das Marktumfeld unterstützt Long-Setups.")
    elif market_regime == "NEGATIV":
        weaknesses.append("Das Marktumfeld ist aktuell klar negativ.")

    if kb >= 3:
        strengths.append("Hohe Konfluenz der Kernblöcke.")
    elif kb <= 1:
        weaknesses.append("Zu wenig Konfluenz der Kernblöcke.")

    if earnings_warning:
        weaknesses.append("Kurzfristiges Earnings-Risiko erhöht die Unsicherheit.")

    for item in red_flag_items[:3]:
        weaknesses.append(f"{item['Kategorie']}: {item['Detail']}")

    strengths = strengths[:5]
    weaknesses = weaknesses[:5]

    if position_mode:
        if investment >= 75 and market_regime != "NEGATIV":
            summary = "Bestehende Position wirkt insgesamt stark. Halten oder selektiv ausbauen ist plausibel."
        elif investment >= 60:
            summary = "Bestehende Position ist grundsätzlich okay, aber nicht frei von Risiken. Eng beobachten."
        else:
            summary = "Bestehende Position wirkt anfällig. Risiko-Management und Stopps prüfen."
    else:
        if investment >= 75 and market_regime == "POSITIV":
            summary = "Guter Watchlist-Kandidat mit unterstützendem Marktumfeld."
        elif investment >= 60:
            summary = "Interessanter Kandidat, aber Timing oder Teilbereiche sind noch nicht ideal."
        else:
            summary = "Aktuell eher Beobachtung statt Einstieg."

    return strengths, weaknesses, summary

def infer_data_source_flags(info):
    direct_fields = [
        "profitMargins", "operatingMargins", "grossMargins", "returnOnEquity", "returnOnAssets",
        "revenueGrowth", "earningsGrowth", "currentRatio", "quickRatio", "debtToEquity",
        "freeCashflow", "operatingCashflow", "forwardPE", "pegRatio", "priceToSalesTrailing12Months",
        "priceToBook", "beta", "shortPercentOfFloat", "recommendationMean",
        "numberOfAnalystOpinions", "targetMeanPrice"
    ]
    loaded = int(info.get("_fund_fields_loaded", 0) or 0)
    total = len(direct_fields)
    derived = 0
    coverage = loaded / total if total else 0
    if coverage >= 0.75:
        confidence = "Hoch"
        confidence_icon = "🟢"
    elif coverage >= 0.50:
        confidence = "Mittel"
        confidence_icon = "🟡"
    else:
        confidence = "Niedrig"
        confidence_icon = "🔴"
    return {
        "loaded": loaded,
        "total": total,
        "coverage": coverage,
        "derived_estimate": derived,
        "confidence": confidence,
        "confidence_icon": confidence_icon
    }

def build_candlestick_chart(chart_df, ticker, ccy):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name=ticker
        ),
        row=1,
        col=1
    )

    if "MA20" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["MA20"], mode="lines", name="MA20"),
            row=1,
            col=1
        )
    if "MA50" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["MA50"], mode="lines", name="MA50"),
            row=1,
            col=1
        )
    if "MA200" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["MA200"], mode="lines", name="MA200"),
            row=1,
            col=1
        )

    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"],
            name="Volumen"
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        title="",
        xaxis_rangeslider_visible=False,
        height=650,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    fig.update_yaxes(title_text=f"Kurs ({ccy})", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)
    return fig

def extract_analyst_data(ticker_obj, info):
    info = dict(info or {})

    try:
        recs = getattr(ticker_obj, "recommendations", None)
    except Exception:
        recs = None

    if recs is not None and not getattr(recs, "empty", True):
        cols = {str(c).lower(): c for c in recs.columns}
        strong_buy_col = cols.get("strongbuy") or cols.get("strong_buy")
        buy_col = cols.get("buy")
        hold_col = cols.get("hold")
        sell_col = cols.get("sell")
        strong_sell_col = cols.get("strongsell") or cols.get("strong_sell")

        if all(c is not None for c in [strong_buy_col, buy_col, hold_col, sell_col, strong_sell_col]):
            row = recs.iloc[-1]
            sb = pd.to_numeric(row.get(strong_buy_col), errors="coerce")
            b = pd.to_numeric(row.get(buy_col), errors="coerce")
            h = pd.to_numeric(row.get(hold_col), errors="coerce")
            s = pd.to_numeric(row.get(sell_col), errors="coerce")
            ss = pd.to_numeric(row.get(strong_sell_col), errors="coerce")
            vals = [sb, b, h, s, ss]
            if sum(pd.notna(v) for v in vals) >= 3:
                sb = 0 if pd.isna(sb) else float(sb)
                b = 0 if pd.isna(b) else float(b)
                h = 0 if pd.isna(h) else float(h)
                s = 0 if pd.isna(s) else float(s)
                ss = 0 if pd.isna(ss) else float(ss)
                total = sb + b + h + s + ss
                if total > 0:
                    mean = (1 * sb + 2 * b + 3 * h + 4 * s + 5 * ss) / total
                    if pd.isna(info.get("recommendationMean")):
                        info["recommendationMean"] = mean
                    if pd.isna(info.get("numberOfAnalystOpinions")):
                        info["numberOfAnalystOpinions"] = int(total)
                    if pd.isna(info.get("recommendationKey")) or str(info.get("recommendationKey", "")).lower() in {"", "none", "nan"}:
                        if mean <= 1.5:
                            info["recommendationKey"] = "strong_buy"
                        elif mean <= 2.5:
                            info["recommendationKey"] = "buy"
                        elif mean <= 3.5:
                            info["recommendationKey"] = "hold"
                        elif mean <= 4.5:
                            info["recommendationKey"] = "sell"
                        else:
                            info["recommendationKey"] = "strong_sell"

    if pd.isna(info.get("targetMeanPrice")):
        try:
            tinfo = ticker_obj.get_info() or {}
            v = normalize_missing(tinfo.get("targetMeanPrice"))
            if not pd.isna(v):
                info["targetMeanPrice"] = v
        except Exception:
            pass

    return info

def extract_earnings_data(ticker_obj, info):
    info = dict(info or {})

    ts = normalize_missing(info.get("earningsTimestamp"))
    if not pd.isna(ts):
        return info

    try:
        cal = ticker_obj.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal:
            dates = cal["Earnings Date"]
            if isinstance(dates, list) and len(dates) > 0:
                dt = pd.to_datetime(dates[0], errors="coerce", utc=True)
                if pd.notna(dt):
                    info["earningsTimestamp"] = int(dt.timestamp())
                    return info
    except Exception:
        pass

    try:
        ed = ticker_obj.get_earnings_dates(limit=8)
        if ed is not None and not getattr(ed, "empty", True):
            idx = ed.index
            idx = pd.to_datetime(pd.Series(idx), errors="coerce", utc=True)
            now_utc = pd.Timestamp.now(tz="UTC")
            future_idx = [x for x in idx if pd.notna(x) and x >= now_utc]
            chosen = min(future_idx) if future_idx else None
            if chosen is None:
                past_idx = [x for x in idx if pd.notna(x)]
                chosen = max(past_idx) if past_idx else None
            if chosen is not None and pd.notna(chosen):
                info["earningsTimestamp"] = int(chosen.timestamp())
    except Exception:
        pass

    return info

def derive_fundamentals_from_statements(ticker_obj, info):
    info = dict(info or {})

    try:
        income = getattr(ticker_obj, "income_stmt", None)
    except Exception:
        income = None
    try:
        q_income = getattr(ticker_obj, "quarterly_income_stmt", None)
    except Exception:
        q_income = None
    try:
        balance = getattr(ticker_obj, "balance_sheet", None)
    except Exception:
        balance = None
    try:
        q_balance = getattr(ticker_obj, "quarterly_balance_sheet", None)
    except Exception:
        q_balance = None
    try:
        cashflow = getattr(ticker_obj, "cashflow", None)
    except Exception:
        cashflow = None
    try:
        q_cashflow = getattr(ticker_obj, "quarterly_cashflow", None)
    except Exception:
        q_cashflow = None

    revenue_row = first_existing_row(income, ["Total Revenue", "Operating Revenue", "Revenue"])
    op_income_row = first_existing_row(income, ["Operating Income", "EBIT", "OperatingIncome"])
    net_income_row = first_existing_row(income, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
    gross_profit_row = first_existing_row(income, ["Gross Profit", "GrossProfit"])
    diluted_eps_row = first_existing_row(income, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"])

    q_revenue_row = first_existing_row(q_income, ["Total Revenue", "Operating Revenue", "Revenue"])
    q_net_income_row = first_existing_row(q_income, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
    q_diluted_eps_row = first_existing_row(q_income, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"])

    total_assets_row = first_existing_row(balance, ["Total Assets", "TotalAssets"])
    total_equity_row = first_existing_row(balance, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    current_assets_row = first_existing_row(balance, ["Current Assets", "Total Current Assets", "CurrentAssets"])
    current_liab_row = first_existing_row(balance, ["Current Liabilities", "Total Current Liabilities", "CurrentLiabilities"])
    inventory_row = first_existing_row(balance, ["Inventory", "Inventories"])
    debt_row = first_existing_row(balance, ["Total Debt", "TotalDebt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"])

    q_current_assets_row = first_existing_row(q_balance, ["Current Assets", "Total Current Assets", "CurrentAssets"])
    q_current_liab_row = first_existing_row(q_balance, ["Current Liabilities", "Total Current Liabilities", "CurrentLiabilities"])
    q_inventory_row = first_existing_row(q_balance, ["Inventory", "Inventories"])
    q_total_equity_row = first_existing_row(q_balance, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    q_total_assets_row = first_existing_row(q_balance, ["Total Assets", "TotalAssets"])
    q_debt_row = first_existing_row(q_balance, ["Total Debt", "TotalDebt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"])

    op_cf_row = first_existing_row(cashflow, ["Operating Cash Flow", "OperatingCashFlow", "Cash Flow From Continuing Operating Activities"])
    fcf_row = first_existing_row(cashflow, ["Free Cash Flow", "FreeCashFlow"])

    revenue = latest_valid(revenue_row)
    op_income = latest_valid(op_income_row)
    net_income = latest_valid(net_income_row)
    gross_profit = latest_valid(gross_profit_row)
    diluted_eps = latest_valid(diluted_eps_row)

    prev_revenue = previous_valid(revenue_row)
    prev_net_income = previous_valid(net_income_row)
    prev_eps = previous_valid(diluted_eps_row)

    q_revenue = latest_valid(q_revenue_row)
    q_prev_revenue = previous_valid(q_revenue_row)
    q_net_income = latest_valid(q_net_income_row)
    q_prev_net_income = previous_valid(q_net_income_row)
    q_eps = latest_valid(q_diluted_eps_row)
    q_prev_eps = previous_valid(q_diluted_eps_row)

    total_assets = latest_valid(total_assets_row)
    total_equity = latest_valid(total_equity_row)
    current_assets = latest_valid(q_current_assets_row if q_current_assets_row is not None else current_assets_row)
    current_liab = latest_valid(q_current_liab_row if q_current_liab_row is not None else current_liab_row)
    inventory = latest_valid(q_inventory_row if q_inventory_row is not None else inventory_row)
    debt = latest_valid(q_debt_row if q_debt_row is not None else debt_row)
    q_total_assets = latest_valid(q_total_assets_row)
    q_total_equity = latest_valid(q_total_equity_row)

    operating_cf = latest_valid(op_cf_row)
    free_cf = latest_valid(fcf_row)

    if pd.isna(info.get("profitMargins")) and pd.notna(net_income) and pd.notna(revenue) and revenue != 0:
        info["profitMargins"] = net_income / revenue
    if pd.isna(info.get("operatingMargins")) and pd.notna(op_income) and pd.notna(revenue) and revenue != 0:
        info["operatingMargins"] = op_income / revenue
    if pd.isna(info.get("grossMargins")) and pd.notna(gross_profit) and pd.notna(revenue) and revenue != 0:
        info["grossMargins"] = gross_profit / revenue

    eq_for_roe = q_total_equity if pd.notna(q_total_equity) else total_equity
    assets_for_roa = q_total_assets if pd.notna(q_total_assets) else total_assets
    if pd.isna(info.get("returnOnEquity")) and pd.notna(net_income) and pd.notna(eq_for_roe) and eq_for_roe != 0:
        info["returnOnEquity"] = net_income / eq_for_roe
    if pd.isna(info.get("returnOnAssets")) and pd.notna(net_income) and pd.notna(assets_for_roa) and assets_for_roa != 0:
        info["returnOnAssets"] = net_income / assets_for_roa

    rev_growth = np.nan
    if pd.notna(q_revenue) and pd.notna(q_prev_revenue) and q_prev_revenue != 0:
        rev_growth = q_revenue / q_prev_revenue - 1
    elif pd.notna(revenue) and pd.notna(prev_revenue) and prev_revenue != 0:
        rev_growth = revenue / prev_revenue - 1
    if pd.isna(info.get("revenueGrowth")) and pd.notna(rev_growth):
        info["revenueGrowth"] = rev_growth

    earn_growth = np.nan
    if pd.notna(q_eps) and pd.notna(q_prev_eps) and q_prev_eps != 0:
        earn_growth = q_eps / q_prev_eps - 1
    elif pd.notna(q_net_income) and pd.notna(q_prev_net_income) and q_prev_net_income != 0:
        earn_growth = q_net_income / q_prev_net_income - 1
    elif pd.notna(diluted_eps) and pd.notna(prev_eps) and prev_eps != 0:
        earn_growth = diluted_eps / prev_eps - 1
    elif pd.notna(net_income) and pd.notna(prev_net_income) and prev_net_income != 0:
        earn_growth = net_income / prev_net_income - 1
    if pd.isna(info.get("earningsGrowth")) and pd.notna(earn_growth):
        info["earningsGrowth"] = earn_growth

    if pd.isna(info.get("currentRatio")) and pd.notna(current_assets) and pd.notna(current_liab) and current_liab != 0:
        info["currentRatio"] = current_assets / current_liab
    if pd.isna(info.get("quickRatio")) and pd.notna(current_assets) and pd.notna(current_liab) and current_liab != 0:
        inv = inventory if pd.notna(inventory) else 0
        info["quickRatio"] = (current_assets - inv) / current_liab
    if pd.isna(info.get("debtToEquity")) and pd.notna(debt) and pd.notna(eq_for_roe) and eq_for_roe != 0:
        info["debtToEquity"] = debt / eq_for_roe * 100

    if pd.isna(info.get("operatingCashflow")) and pd.notna(operating_cf):
        info["operatingCashflow"] = operating_cf
    if pd.isna(info.get("freeCashflow")) and pd.notna(free_cf):
        info["freeCashflow"] = free_cf

    return info

def build_company_summary(info, ticker):
    candidates = [
        info.get("longBusinessSummary"),
        info.get("businessSummary"),
        info.get("description"),
        info.get("longDescription"),
    ]

    for candidate in candidates:
        if isinstance(candidate, str):
            cleaned = " ".join(candidate.strip().split())
            if cleaned:
                return cleaned

    company_name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector")
    industry = info.get("industry")
    exchange = info.get("exchange")
    country = info.get("country")

    parts = []
    if sector and sector != "-":
        parts.append(f"Sektor: {sector}")
    if industry and industry != "-":
        parts.append(f"Branche: {industry}")
    if country and country != "-":
        parts.append(f"Land: {country}")
    if exchange and exchange != "-":
        parts.append(f"Börse: {exchange}")

    if parts:
        return f"{company_name} | " + " | ".join(parts)

    return "Keine Unternehmensbeschreibung verfügbar."

def load_data(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="3y", auto_adjust=True)

    info = {}
    try:
        info = merge_info(info, getattr(t, "fast_info", {}) or {})
    except Exception:
        pass
    try:
        info = merge_info(info, t.get_info() or {})
    except Exception:
        pass
    try:
        info = merge_info(info, t.info or {})
    except Exception:
        pass

    info = derive_fundamentals_from_statements(t, info)
    info = extract_analyst_data(t, info)
    info = extract_earnings_data(t, info)

    try:
        info["_fund_fields_loaded"] = int(sum(pd.notna(normalize_missing(info.get(k))) for k in [
            "profitMargins", "operatingMargins", "grossMargins", "returnOnEquity", "returnOnAssets",
            "revenueGrowth", "earningsGrowth", "currentRatio", "quickRatio", "debtToEquity",
            "freeCashflow", "operatingCashflow", "forwardPE", "pegRatio", "priceToSalesTrailing12Months",
            "priceToBook", "beta", "shortPercentOfFloat", "recommendationMean",
            "numberOfAnalystOpinions", "targetMeanPrice"
        ]))
    except Exception:
        info["_fund_fields_loaded"] = 0

    return hist, info

def load_benchmark_data(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y", auto_adjust=True)
        return hist
    except Exception:
        return pd.DataFrame()

def search_tickers(query, max_results=8):
    query = str(query or "").strip()
    if not query:
        return []

    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": max_results, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    results = []
    for item in data.get("quotes", []):
        symbol = item.get("symbol")
        name = item.get("shortname") or item.get("longname") or "-"
        exchange = item.get("exchange") or "-"
        quote_type = item.get("quoteType") or "-"
        exch_disp = item.get("exchDisp") or exchange

        if not symbol:
            continue
        if quote_type not in {"EQUITY", "ETF"}:
            continue

        results.append({
            "symbol": symbol,
            "name": name,
            "shortname": item.get("shortname"),
            "longname": item.get("longname"),
            "exchange": exch_disp,
            "type": quote_type,
            "label": f"{name} ({symbol}) - {exch_disp}"
        })

    seen = set()
    clean = []
    for r in results:
        if r["symbol"] not in seen:
            clean.append(r)
            seen.add(r["symbol"])
    return clean

def looks_like_real_ticker(user_input):
    raw = str(user_input or "").strip()
    if not raw:
        return False

    upper = raw.upper()

    if re.fullmatch(r"[A-Z0-9]{1,5}([.\-][A-Z0-9]{1,5})?", upper):
        # Wörter wie Apple, Siemens, Nvidia nicht blind als Ticker behandeln,
        # außer der Nutzer hat sie bewusst in echter Ticker-Schreibweise eingegeben.
        if raw.isalpha() and raw != upper:
            return False
        return True

    return False

def score_search_result(query, item):
    query = str(query or "").strip().lower()

    symbol = str(item.get("symbol", "") or "").strip().lower()
    name = str(item.get("name", "") or item.get("shortname", "") or item.get("longname", "") or "").strip().lower()
    exchange = str(item.get("exchange", "") or "").strip().lower()
    quote_type = str(item.get("type", "") or item.get("quoteType", "") or "").strip().upper()

    score = 0

    if query == symbol:
        score += 120
    if query == name:
        score += 110
    if query in name and name:
        score += 55
    if query in symbol and symbol:
        score += 35

    q_tokens = {t for t in re.split(r"\W+", query) if t}
    n_tokens = {t for t in re.split(r"\W+", name) if t}
    overlap = len(q_tokens & n_tokens)
    score += overlap * 12

    if quote_type == "EQUITY":
        score += 12
    elif quote_type == "ETF":
        score += 6

    if exchange in {"nasdaq", "nasdaqgs", "nyq", "nyse", "xetra", "par", "ams", "mil", "six", "lse"}:
        score += 4

    if not name:
        score -= 10

    return score

def resolve_input_to_ticker(user_input, fallback=None):
    user_input = str(user_input or "").strip()
    if not user_input:
        return fallback

    raw = user_input.strip()
    upper = raw.upper()

    if looks_like_real_ticker(raw):
        return upper

    results = search_tickers(raw, max_results=8)
    if results:
        ranked = sorted(results, key=lambda x: score_search_result(raw, x), reverse=True)
        best = ranked[0]
        symbol = best.get("symbol")
        if symbol:
            return str(symbol).upper()

    return fallback if fallback else None

def build_short_thesis(investment, tb_score, market_regime, top_red_flag, position_mode):
    if position_mode:
        if investment >= 78 and market_regime != "NEGATIV":
            txt = "Starke Position mit solidem Halte-/Ausbauprofil."
        elif investment >= 60:
            txt = "Ordentliche Position, aber mit Beobachtungsbedarf."
        else:
            txt = "Position aktuell verletzlich; aktives Risikomanagement wichtig."
    else:
        if investment >= 78 and market_regime == "POSITIV":
            txt = "Attraktiver Watchlist-/Einstiegskandidat."
        elif investment >= 60:
            txt = "Interessanter Kandidat, aber Timing noch nicht perfekt."
        else:
            txt = "Derzeit eher Beobachtung statt Aktion."

    if tb_score >= 9:
        txt += " TradingBoard bestätigt Stärke."
    elif tb_score < 3:
        txt += " TradingBoard ist klar defensiv."

    if isinstance(top_red_flag, str) and top_red_flag and top_red_flag != "-":
        txt += f" Wichtigste Bremse: {top_red_flag}."
    return txt

def build_ranking_table(results):
    rows = []
    for r in results:
        market_info = r.get("market_info", {}) or {}
        confidence_info = r.get("confidence_info", {}) or {}
        full_red_flag = r.get("top_red_flag", "-")
        full_thesis = r.get("short_thesis", r.get("decision_summary", "-"))

        rows.append({
            "Ticker": r.get("ticker", "-"),
            "Name": shorten_text(r.get("name", r.get("ticker", "-")), 28),
            "Setup-Typ": r.get("setup_type", "-"),
            "Benchmark": r.get("benchmark_label", "-"),
            "Marktregime": market_regime_label(market_info.get("regime", "UNBEKANNT")),
            "Company Quality": r.get("company", np.nan),
            "Setup Quality": r.get("setup_adj", np.nan),
            "Investment Score": r.get("investment", np.nan),
            "Investment-Attraktivität": r.get("investment_case_score", np.nan),
            "Einstieg jetzt attraktiv?": r.get("trading_case_score", np.nan),
            "Trade-Struktur": r.get("tradeability_score", np.nan),
            "Setup-Confidence": r.get("setup_confidence", np.nan),
            "Entry-Lage": r.get("entry_quality", "-"),
            "Valides Setup": "Ja" if r.get("valid_trade_setup", False) else "Nein",
            "Trigger-Status": r.get("trigger_status", "-"),
            "Watchlist-Priorität": r.get("watchlist_priority", "-"),
            "Kurzfrist-Timing": r.get("tb_score_100", normalize_tb_score_100(r.get("tb_score", np.nan))),
            "Fundamental-Confidence": round(confidence_info.get("coverage", 0) * 100),
            "Top Red Flag": shorten_text(full_red_flag, 34),
            "Kurzfazit": shorten_text(full_thesis, 52),
            "_Top Red Flag Full": full_red_flag if full_red_flag else "-",
            "_Kurzfazit Full": full_thesis if full_thesis else "-",
        })

    df = pd.DataFrame(rows)

    if "_Top Red Flag Full" not in df.columns:
        df["_Top Red Flag Full"] = df.get("Top Red Flag", "-")
    if "_Kurzfazit Full" not in df.columns:
        df["_Kurzfazit Full"] = df.get("Kurzfazit", "-")

    if not df.empty:
        df = df.sort_values(
            by=["Investment-Attraktivität", "Einstieg jetzt attraktiv?", "Trade-Struktur"],
            ascending=False
        ).reset_index(drop=True)
        df.index = df.index + 1
    return df

def compute_chart_df(df, chart_range):
    if chart_range == "3 Monate":
        chart_df = df.tail(63).copy()
    elif chart_range == "6 Monate":
        chart_df = df.tail(126).copy()
    elif chart_range == "1 Jahr":
        chart_df = df.tail(252).copy()
    else:
        chart_df = df.copy()
    chart_df["MA20"] = chart_df["Close"].rolling(20).mean()
    chart_df["MA50"] = chart_df["Close"].rolling(50).mean()
    chart_df["MA200"] = chart_df["Close"].rolling(200).mean()
    return chart_df

def analyze_stock(
    ticker,
    horizon,
    depot,
    risk_pct,
    override,
    buy_in_override,
    smart_money_default,
    strict_mode
):
    df, info = load_data(ticker)

    if df.empty or len(df) < 220:
        raise ValueError("Nicht genug Kursdaten fuer belastbare Analyse. Prüfe den ausgewählten Ticker.")

    benchmark_symbol, benchmark_label = select_benchmark(ticker, info)
    benchmark_df = load_benchmark_data(benchmark_symbol)
    market_info = evaluate_market_filter(benchmark_df)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    price = float(override) if override > 0 else float(close.iloc[-1])
    name = info.get("longName", ticker)
    raw_ccy = info.get("currency", "USD")
    ccy = infer_display_currency(ticker, info, raw_ccy)
    exch = info.get("exchange", "-")
    ts = df.index[-1].strftime("%d.%m.%Y")
    sector = info.get("sector", "-")
    industry = info.get("industry", "-")

    company_summary = build_company_summary(info, ticker)

    confidence_info = infer_data_source_flags(info)

    # ---------- Technicals ----------
    ma20_series = close.rolling(20).mean()
    ma50_series = close.rolling(50).mean()
    ma150_series = close.rolling(150).mean()
    ma200_series = close.rolling(200).mean()

    ma20 = safe_last(ma20_series)
    ma50 = safe_last(ma50_series)
    ma150 = safe_last(ma150_series)
    ma200 = safe_last(ma200_series)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_v = safe_last(macd)
    signal_v = safe_last(signal)
    macd_up = macd_v > signal_v

    rsi = safe_last(rsi14(close))
    adx = safe_last(adx14(high, low, close))
    tr = true_range(high, low, close)
    atr = safe_last(tr.rolling(14).mean())
    atr_pct = atr / price * 100 if price else 0

    ret_metrics = calc_return_metrics(close)
    ret21 = ret_metrics["ret21"]
    ret63 = ret_metrics["ret63"]
    ret126 = ret_metrics["ret126"]
    roc20 = safe_last(close.pct_change(20) * 100)
    roc60 = safe_last(close.pct_change(60) * 100)
    ret5 = safe_last(close.pct_change(5) * 100, 0)
    ret20 = safe_last(close.pct_change(20) * 100, 0)

    vol20 = safe_last(vol.rolling(20).mean(), 1)
    vol5 = safe_last(vol.rolling(5).mean(), 1)
    vol_ratio = vol5 / vol20 if vol20 else 1

    high52 = safe_last(close.rolling(252).max(), float(close.max()))
    low52 = safe_last(close.rolling(252).min(), float(close.min()))
    dist52 = price / high52 * 100 if high52 else 50

    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    obv_trend = "steigend" if float(obv.iloc[-1]) > float(obv.iloc[-20]) else "fallend"

    stoch_k, stoch_d = stoch14(high, low, close)
    stoch_k_v = safe_last(stoch_k, 50)
    stoch_d_v = safe_last(stoch_d, 50)
    willr_v = safe_last(williams_r(high, low, close), -50)

    bb_mid, bb_upper_s, bb_lower_s, bb_width_s = bollinger_bands(close)
    bb_upper = safe_last(bb_upper_s, np.nan)
    bb_lower = safe_last(bb_lower_s, np.nan)
    bb_width = safe_last(bb_width_s, np.nan)
    bb_width_thresh = safe_last(bb_width_s.rolling(60).quantile(0.2), np.nan)
    bb_squeeze = pd.notna(bb_width) and pd.notna(bb_width_thresh) and bb_width <= bb_width_thresh
    prev20_high = safe_last(close.shift(1).rolling(20).max(), np.nan)
    prev20_low = safe_last(close.shift(1).rolling(20).min(), np.nan)

    macd_hist_series = macd - signal
    macd_hist_current = safe_last(macd_hist_series, 0)
    macd_hist_prev = safe_last(macd_hist_series.shift(1), 0)
    macd_bull_cross = macd_v > signal_v and macd_hist_current > 0 and macd_hist_prev < 0

    # ---------- Trendqualität / Base ----------
    ma20_slope = calc_slope_pct(ma20_series, lookback=20)
    ma50_slope = calc_slope_pct(ma50_series, lookback=20)
    ma200_slope = calc_slope_pct(ma200_series, lookback=20)
    higher_lows_score = calc_higher_lows_score(close, low)
    trend_quality_score = calc_trend_quality_score(
        price, ma20, ma50, ma200, ma20_slope, ma50_slope, ma200_slope, higher_lows_score
    )
    base_length_days = calc_base_length_days(close, ma20)
    correction_depth_pct = calc_correction_depth_pct(close)
    range_tightness_score = calc_range_tightness_score(close)
    atr_pct_series = (tr.rolling(14).mean() / close) * 100
    volatility_contraction_score = calc_volatility_contraction_score(atr_pct_series, bb_width_s)
    pullback_quality_score = calc_pullback_quality_score(price, ma20, ma50, rsi, atr_pct, ret20)
    base_quality_score = calc_base_quality_score(
        base_length_days,
        correction_depth_pct,
        range_tightness_score,
        volatility_contraction_score,
        pullback_quality_score
    )
    volume_quality_proxy = calc_volume_quality_proxy(vol_ratio, obv_trend)

    # ---------- Volumenqualität / Akkumulation ----------
    up_down_volume_ratio = calc_up_down_volume_ratio(close, vol, lookback=20)
    accumulation_day_count, distribution_day_count = calc_accumulation_distribution_days(close, vol, lookback=20)
    volume_trend_score = calc_volume_trend_score(vol, close)
    recent_pullback_volume_ratio = calc_recent_pullback_volume_ratio(close, vol, lookback=10)
    breakout_day_volume_ratio = calc_breakout_day_volume_ratio(close, vol, lookback=20)
    close_near_day_high = calc_close_near_day_high(close, high, low)

    ret1_series = close.pct_change(1) * 100
    ret1_last = float(ret1_series.iloc[-1]) if pd.notna(ret1_series.iloc[-1]) else np.nan
    vol_ma20_last = float(vol.rolling(20).mean().iloc[-1]) if pd.notna(vol.rolling(20).mean().iloc[-1]) else np.nan
    vol_last = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else np.nan

    strong_up_volume_day = (
        pd.notna(ret1_last) and ret1_last >= 1.5
        and pd.notna(vol_last) and pd.notna(vol_ma20_last) and vol_last > 1.2 * vol_ma20_last
    )
    down_volume_heavy = (
        pd.notna(ret1_last) and ret1_last <= -1.5
        and pd.notna(vol_last) and pd.notna(vol_ma20_last) and vol_last > 1.2 * vol_ma20_last
    )
    weak_rebound_on_volume = (
        pd.notna(ret1_last) and 0 < ret1_last < 0.8
        and pd.notna(vol_last) and pd.notna(vol_ma20_last) and vol_last > 1.15 * vol_ma20_last
    )

    pullback_active = bool(pd.notna(ret20) and ret20 < 0 and ((pd.notna(ma50) and pd.notna(price) and price >= ma50 * 0.95) or not pd.notna(ma50)))
    prev20_high_local = float(close.shift(1).rolling(20).max().iloc[-1]) if pd.notna(close.shift(1).rolling(20).max().iloc[-1]) else np.nan
    breakout_context = bool(pd.notna(prev20_high_local) and pd.notna(price) and price >= prev20_high_local * 0.995)

    accumulation_score = calc_accumulation_score(up_down_volume_ratio, obv_trend, accumulation_day_count, strong_up_volume_day)
    distribution_pressure_score = calc_distribution_pressure_score(distribution_day_count, obv_trend, down_volume_heavy, weak_rebound_on_volume)
    pullback_dryup_score = calc_pullback_dryup_score(pullback_active, recent_pullback_volume_ratio, pullback_quality_score, volatility_contraction_score)
    breakout_volume_score = calc_breakout_volume_score(breakout_context, breakout_day_volume_ratio, 50, close_near_day_high, breakout_failure_risk_low=True)
    volume_quality_score = calc_volume_quality_score(accumulation_score, distribution_pressure_score, pullback_dryup_score, breakout_volume_score, volume_trend_score)

    # ---------- Fundamentals ----------
    target = info.get("targetMeanPrice", np.nan)
    upside = ((target / price - 1) * 100) if pd.notna(target) and price else np.nan
    pe = info.get("forwardPE", np.nan)
    peg = info.get("pegRatio", np.nan)
    ps = info.get("priceToSalesTrailing12Months", np.nan)
    pb = info.get("priceToBook", np.nan)
    beta = info.get("beta", np.nan)
    market_cap = info.get("marketCap", np.nan)

    profit_margin = info.get("profitMargins", np.nan)
    oper_margin = info.get("operatingMargins", np.nan)
    gross_margin = info.get("grossMargins", np.nan)
    roe = info.get("returnOnEquity", np.nan)
    roa = info.get("returnOnAssets", np.nan)
    revenue_growth = info.get("revenueGrowth", np.nan)
    earnings_growth = info.get("earningsGrowth", np.nan)
    current_ratio = info.get("currentRatio", np.nan)
    quick_ratio = info.get("quickRatio", np.nan)
    debt_to_equity = info.get("debtToEquity", np.nan)
    fcf = info.get("freeCashflow", np.nan)
    op_cf = info.get("operatingCashflow", np.nan)
    short_pct = info.get("shortPercentOfFloat", np.nan)

    # ---------- Benchmark / Market ----------
    bench_ret21 = market_info["ret21"]
    bench_ret63 = market_info["ret63"]
    bench_ret126 = market_info["ret126"]

    rs_vs_benchmark_21 = ret21 - bench_ret21 if pd.notna(bench_ret21) else np.nan
    rs_vs_benchmark_63 = ret63 - bench_ret63 if pd.notna(bench_ret63) else np.nan
    rs_vs_benchmark_126 = ret126 - bench_ret126 if pd.notna(bench_ret126) else np.nan

    rs_terms = []
    if pd.notna(rs_vs_benchmark_21):
        rs_terms.append(rs_vs_benchmark_21 * 0.25)
    if pd.notna(rs_vs_benchmark_63):
        rs_terms.append(rs_vs_benchmark_63 * 0.45)
    if pd.notna(rs_vs_benchmark_126):
        rs_terms.append(rs_vs_benchmark_126 * 0.30)
    rs_composite = sum(rs_terms) if rs_terms else np.nan

    # ---------- Stock Style ----------
    stock_style = infer_stock_style_advanced(
        revenue_growth, earnings_growth, pe, pb, beta, debt_to_equity, roe, profit_margin, sector
    )
    style_adj = get_style_sector_adjustment(stock_style, sector)

    # ---------- Horizon ----------
    if "1-7" in horizon:
        hd, ws, wc = 7, 0.82, 0.18
    elif "1-4" in horizon:
        hd, ws, wc = 21, 0.68, 0.32
    elif "1-3" in horizon:
        hd, ws, wc = 60, 0.52, 0.48
    elif "1-2" in horizon:
        hd, ws, wc = 365, 0.30, 0.70
    else:
        hd, ws, wc = 730, 0.15, 0.85

    # ---------- Earnings ----------
    earnings_ts = normalize_missing(info.get("earningsTimestamp"))
    if pd.notna(earnings_ts):
        days_earn = (float(earnings_ts) - datetime.now(timezone.utc).timestamp()) / 86400
    else:
        days_earn = np.nan

    has_upcoming_earnings = pd.notna(days_earn) and days_earn >= 0
    has_past_earnings = pd.notna(days_earn) and days_earn < 0

    if has_upcoming_earnings:
        sg_earn = "🟢" if days_earn > 30 else ("🟡" if days_earn > 7 else "🔴")
    elif has_past_earnings:
        sg_earn = "🟡"
    else:
        sg_earn = "⚪"

    if pd.notna(earnings_ts):
        earnings_dt = datetime.fromtimestamp(float(earnings_ts), tz=timezone.utc)
        if has_upcoming_earnings:
            sg_earn_txt = earnings_dt.strftime("%d.%m.%Y")
        else:
            sg_earn_txt = f"Letzte Earnings: {earnings_dt.strftime('%d.%m.%Y')}"
    else:
        sg_earn_txt = "kein Datum"

    earnings_warning = has_upcoming_earnings and days_earn <= 7

    # ---------- Technical Scores ----------
    if price > ma50 > ma150 > ma200:
        regime, reg_amp = "UPTREND", "🟢"
    elif price < ma50 < ma150 < ma200:
        regime, reg_amp = "DOWNTREND", "🔴"
    else:
        regime, reg_amp = "SIDEWAYS", "🟡"

    s3 = 100 if price > ma20 > ma50 > ma150 else (15 if price < ma20 < ma50 < ma150 else 52)
    s3a = ampel(s3)
    s3t = "Trend-Stack sauber" if s3 >= 80 else ("Trend gemischt" if s3 >= 45 else "Trend schwach")

    rsi_s = 100 if 55 <= rsi <= 72 else (70 if 48 <= rsi < 55 or 72 < rsi <= 78 else 25)
    macd_s = 100 if (macd_v > 0 and macd_up) else (68 if macd_up else 22)
    adx_s = 100 if adx > 25 else (65 if adx > 18 else 30)
    roc_s = 100 if roc20 > 4 else (72 if roc20 > 0 else (45 if roc20 > -4 else 20))
    s4 = round(rsi_s * 0.25 + macd_s * 0.30 + adx_s * 0.20 + roc_s * 0.25)
    s4a = ampel(s4)
    s4t = f"RSI {rsi:.1f} | MACD {'up' if macd_up else 'dn'} | ADX {adx:.1f} | ROC20 {roc20:.1f}%"

    if ret5 > 0 and vol_ratio > 1.12 and obv_trend == "steigend":
        s5, s5a, s5t = 100, "🟢", f"Vol {vol_ratio:.2f}x | OBV steigend"
    elif ret20 > 0 and obv_trend == "steigend":
        s5, s5a, s5t = 68, "🟡", f"Vol {vol_ratio:.2f}x | Nachfrage ok"
    elif ret20 > 0:
        s5, s5a, s5t = 52, "🟡", f"Momentum ok | OBV {obv_trend}"
    else:
        s5, s5a, s5t = 28, "🔴", f"Momentum/Volumen schwach | OBV {obv_trend}"

    if atr_pct < 2.8:
        s6, s6a, s6t = 92, "🟢", f"ATR {atr_pct:.1f}% niedrig"
    elif atr_pct < 5.5:
        s6, s6a, s6t = 66, "🟡", f"ATR {atr_pct:.1f}% normal"
    elif atr_pct < 8.0:
        s6, s6a, s6t = 44, "🟡", f"ATR {atr_pct:.1f}% erhoeht"
    else:
        s6, s6a, s6t = 20, "🔴", f"ATR {atr_pct:.1f}% hoch"

    w52 = 100 if 80 <= dist52 <= 98 else (72 if 70 <= dist52 < 80 else (55 if 98 < dist52 <= 101 else (35 if dist52 >= 55 else 15)))

    if pd.notna(rs_composite):
        if rs_composite > 8:
            rs_score = 100
        elif rs_composite > 3:
            rs_score = 78
        elif rs_composite > -3:
            rs_score = 55
        else:
            rs_score = 22
    else:
        rs_score = 100 if ret63 > 12 else (78 if ret63 > 4 else (55 if ret63 > -5 else 22))

    kb = sum([s3 >= 65, s4 >= 65, s5 >= 65, s6 >= 65])

    setup_raw = (
        s3 * (0.22 * style_adj["trend"])
        + s4 * (0.24 * style_adj["momentum"])
        + s5 * 0.18
        + s6 * 0.10
        + rs_score * 0.16
        + w52 * 0.10
    ) / (
        0.22 * style_adj["trend"] + 0.24 * style_adj["momentum"] + 0.18 + 0.10 + 0.16 + 0.10
    )

    if strict_mode:
        if kb < 2:
            setup_raw = min(setup_raw, 44)
        elif kb == 2:
            setup_raw = min(setup_raw, 58)
    setup = round(clamp(setup_raw))
    setup_adj = round(clamp(setup * 0.88 + market_info["score"] * 0.12))

    # ---------- Fundamental Scores ----------
    fundamental_fields = [
        profit_margin, oper_margin, gross_margin, roe, roa,
        revenue_growth, earnings_growth, current_ratio, quick_ratio,
        debt_to_equity, fcf, op_cf, pe, peg, ps, pb,
        beta, short_pct, info.get("recommendationMean", np.nan),
        info.get("numberOfAnalystOpinions", np.nan), target
    ]
    fund_cov = known_ratio(fundamental_fields)
    fund_fields_loaded = int(info.get("_fund_fields_loaded", 0) or 0)
    fund_data_warning = fund_cov < 0.35

    quality_parts = []
    quality_parts.append(90 if pd.notna(profit_margin) and profit_margin > 0.20 else (75 if pd.notna(profit_margin) and profit_margin > 0.10 else (55 if pd.notna(profit_margin) and profit_margin > 0 else 40)))
    quality_parts.append(90 if pd.notna(oper_margin) and oper_margin > 0.25 else (75 if pd.notna(oper_margin) and oper_margin > 0.15 else (55 if pd.notna(oper_margin) and oper_margin > 0.08 else 40)))
    quality_parts.append(92 if pd.notna(roe) and roe > 0.25 else (78 if pd.notna(roe) and roe > 0.15 else (58 if pd.notna(roe) and roe > 0.08 else 42)))
    quality_parts.append(85 if pd.notna(fcf) and fcf > 0 else (60 if pd.notna(fcf) else 45))
    quality_score = round(np.mean(quality_parts))

    growth_parts = []
    growth_parts.append(90 if pd.notna(revenue_growth) and revenue_growth > 0.15 else (75 if pd.notna(revenue_growth) and revenue_growth > 0.05 else (55 if pd.notna(revenue_growth) and revenue_growth > 0 else 35)))
    growth_parts.append(92 if pd.notna(earnings_growth) and earnings_growth > 0.20 else (76 if pd.notna(earnings_growth) and earnings_growth > 0.08 else (56 if pd.notna(earnings_growth) and earnings_growth > 0 else 34)))
    growth_parts.append(88 if ret126 > 20 else (72 if ret126 > 5 else (55 if ret126 > -8 else 35)))
    growth_score = round(np.mean(growth_parts))

    growth_quality = 50
    if pd.notna(revenue_growth) and revenue_growth > 0.08:
        growth_quality += 10
    if pd.notna(earnings_growth) and pd.notna(revenue_growth) and earnings_growth > revenue_growth:
        growth_quality += 10
    if pd.notna(fcf) and fcf > 0:
        growth_quality += 10
    if pd.notna(profit_margin) and profit_margin > 0.10:
        growth_quality += 10
    if pd.notna(oper_margin) and oper_margin > 0.12:
        growth_quality += 10
    growth_quality = round(clamp(growth_quality))

    valuation_parts = []
    if pd.notna(pe):
        valuation_parts.append(86 if 0 < pe < 20 else (72 if pe < 28 else (58 if pe < 38 else 42)))
    if pd.notna(peg):
        valuation_parts.append(84 if 0 < peg < 1.5 else (70 if peg < 2.2 else (55 if peg < 3.0 else 42)))
    if pd.notna(ps):
        if pd.notna(revenue_growth) and revenue_growth > 0.15:
            valuation_parts.append(78 if ps < 8 else (62 if ps < 12 else 45))
        else:
            valuation_parts.append(82 if ps < 4 else (68 if ps < 8 else 42))
    if pd.notna(pb):
        valuation_parts.append(80 if pb < 3 else (65 if pb < 6 else 45))
    valuation_parts.append(82 if pd.notna(upside) and upside > 20 else (70 if pd.notna(upside) and upside > 10 else (55 if pd.notna(upside) and upside > 0 else 40)))
    valuation_score = round(np.mean(valuation_parts)) if valuation_parts else 50

    balance_parts = []
    balance_parts.append(88 if pd.notna(current_ratio) and current_ratio >= 1.5 else (72 if pd.notna(current_ratio) and current_ratio >= 1.1 else 48))
    balance_parts.append(88 if pd.notna(quick_ratio) and quick_ratio >= 1.0 else (70 if pd.notna(quick_ratio) and quick_ratio >= 0.8 else 48))
    balance_parts.append(90 if pd.notna(debt_to_equity) and debt_to_equity < 60 else (72 if pd.notna(debt_to_equity) and debt_to_equity < 120 else 45))
    balance_score = round(np.mean(balance_parts))

    rec = info.get("recommendationKey", "hold")
    rec_label = analyst_label(rec)
    rec_mean = info.get("recommendationMean", np.nan)
    analysts = info.get("numberOfAnalystOpinions", np.nan)

    sentiment_parts = []
    sentiment_parts.append(88 if rec in ["strong_buy", "buy"] else (65 if rec in ["hold"] else 50))
    sentiment_parts.append(84 if pd.notna(analysts) and analysts >= 20 else (72 if pd.notna(analysts) and analysts >= 10 else (58 if pd.notna(analysts) and analysts >= 5 else (52 if pd.notna(target) else 48))))
    sentiment_parts.append(84 if pd.notna(rec_mean) and rec_mean <= 2.0 else (68 if pd.notna(rec_mean) and rec_mean <= 2.5 else (55 if pd.notna(rec_mean) and rec_mean <= 3.0 else (50 if pd.notna(target) else 42))))
    sentiment_score = round(np.mean(sentiment_parts))

    risk_parts = []
    risk_parts.append(80 if pd.notna(beta) and beta < 1.2 else (62 if pd.notna(beta) and beta < 1.6 else 45))
    risk_parts.append(78 if pd.notna(short_pct) and short_pct < 0.03 else (62 if pd.notna(short_pct) and short_pct < 0.07 else 45))
    risk_parts.append(82 if atr_pct < 3.5 else (65 if atr_pct < 6 else 45))
    risk_score = round(np.mean(risk_parts))

    base_company = round(
        quality_score * (0.24 * style_adj["quality"])
        + growth_score * (0.18 * style_adj["growth"])
        + growth_quality * 0.12
        + valuation_score * (0.18 * style_adj["valuation"])
        + balance_score * (0.16 * style_adj["balance"])
        + sentiment_score * 0.06
        + risk_score * 0.06
    )
    base_company = round(base_company / (
        0.24 * style_adj["quality"]
        + 0.18 * style_adj["growth"]
        + 0.12
        + 0.18 * style_adj["valuation"]
        + 0.16 * style_adj["balance"]
        + 0.06
        + 0.06
    ))

    red_flag_items, red_flag_penalty_total = build_red_flags(
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        profit_margin=profit_margin,
        fcf=fcf,
        op_cf=op_cf,
        debt_to_equity=debt_to_equity,
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        has_upcoming_earnings=has_upcoming_earnings,
        days_earn=days_earn
    )
    red_flag_notes = [f"{x['Kategorie']}: {x['Detail']}" for x in red_flag_items]

    coverage_penalty = 0
    if fund_cov < 0.35:
        coverage_penalty = 12
    elif fund_cov < 0.55:
        coverage_penalty = 6

    base_company = max(25, round(base_company - red_flag_penalty_total - coverage_penalty))

    if hd < 30:
        company = round(base_company * 0.55 + 50 * 0.45)
    else:
        company = base_company
    company = int(clamp(company))

    # ---------- Institutionelle Qualität ----------
    cashflow_stability_score = calc_cashflow_stability_score(
        fcf, op_cf, market_cap, revenue_growth, earnings_growth
    )
    margin_stability_score = calc_margin_stability_score(
        profit_margin, oper_margin, gross_margin, roe, roa
    )
    institutional_quality_score = calc_institutional_quality_score(
        cashflow_stability_score,
        margin_stability_score,
        balance_score,
        quality_score,
        risk_score
    )
    institutional_quality_text = institutional_quality_label(institutional_quality_score)

    company = round(clamp(company * 0.90 + institutional_quality_score * 0.10))
    investment = round(clamp(setup_adj * ws + company * wc))

    if bb_squeeze and pd.notna(prev20_high) and price > prev20_high and vol_ratio >= 1.0:
        setup_type = "Range-Breakout"
        preferred_entry = "Ausbruch über Range-Oberkante"
    elif pd.notna(prev20_high) and price > prev20_high and vol_ratio >= 1.05 and rsi < 78:
        setup_type = "Breakout"
        preferred_entry = "Breakout über 20T-Hoch"
    elif (
        pd.notna(prev20_high)
        and price > ma20 > ma50
        and abs(price - prev20_high) / prev20_high <= 0.02
        and -3 <= ret5 <= 4
    ):
        setup_type = "Breakout-Retest"
        preferred_entry = "Retest des Ausbruchsniveaus"
    elif price > ma50 and ma20 > ma50 and pd.notna(ma20) and abs(price - ma20) / price <= 0.025:
        setup_type = "Pullback an MA20"
        preferred_entry = "Pullback nahe MA20"
    elif price > ma200 and pd.notna(ma50) and abs(price - ma50) / price <= 0.03:
        setup_type = "Pullback an MA50"
        preferred_entry = "Pullback nahe MA50"
    elif price > ma200 and rsi < 42 and macd_hist_current > macd_hist_prev:
        setup_type = "Rebound"
        preferred_entry = "Rebound nach Stabilisierung"
    elif price > ma50 and price > ma200 and rs_score >= 55:
        setup_type = "Trendfolge"
        preferred_entry = "Trendfolge bei Rücksetzer"
    else:
        setup_type = "Kein sauberes Setup"
        preferred_entry = "Aktuell kein sauberer Einstieg"

    setup_base_score = {
        "Breakout": 88,
        "Breakout-Retest": 86,
        "Pullback an MA20": 84,
        "Pullback an MA50": 78,
        "Trendfolge": 76,
        "Rebound": 68,
        "Range-Breakout": 82,
        "Kein sauberes Setup": 35,
    }.get(setup_type, 40)

    setup_confidence = round(clamp(
        setup_base_score * 0.38
        + s3 * 0.22
        + s4 * 0.22
        + min(kb / 4 * 100, 100) * 0.10
        + (85 if market_info["regime"] == "POSITIV" else 60 if market_info["regime"] == "NEUTRAL" else 35) * 0.08
    ))
    setup_confidence_text = setup_confidence_label(setup_confidence)

    valid_trade_setup = (
        investment >= 60
        and setup_adj >= 55
        and kb >= 2
        and setup_type != "Kein sauberes Setup"
        and market_info["regime"] != "NEGATIV"
        and not (has_upcoming_earnings and pd.notna(days_earn) and days_earn < 7)
    )

    # ---------- Trade Setup ----------
    if valid_trade_setup:
        # Einstieg je Setup-Typ
        if setup_type == "Breakout":
            anchor_price = prev20_high if pd.notna(prev20_high) else price
            entry_low = max(anchor_price, price * 0.995)
            entry_high = max(anchor_price * 1.015, price * 1.005)
            entry_source = "Breakout-Zone über dem Ausbruchsniveau"
        elif setup_type == "Breakout-Retest":
            anchor_price = prev20_high if pd.notna(prev20_high) else price
            entry_low = anchor_price * 0.99 if pd.notna(anchor_price) else price * 0.98
            entry_high = anchor_price * 1.01 if pd.notna(anchor_price) else price * 1.00
            entry_source = "Retest-Zone am früheren Breakout-Level"
        elif setup_type == "Pullback an MA20":
            anchor_price = ma20 if pd.notna(ma20) else ma50
            entry_low = anchor_price * 0.99 if pd.notna(anchor_price) else price * 0.98
            entry_high = anchor_price * 1.01 if pd.notna(anchor_price) else price
            entry_source = "Pullback-Zone nahe MA20"
        elif setup_type == "Pullback an MA50":
            anchor_price = ma50 if pd.notna(ma50) else ma20
            entry_low = anchor_price * 0.985 if pd.notna(anchor_price) else price * 0.97
            entry_high = anchor_price * 1.015 if pd.notna(anchor_price) else price
            entry_source = "Pullback-Zone nahe MA50"
        elif setup_type == "Rebound":
            anchor_price = prev20_low if pd.notna(prev20_low) else ma20
            entry_low = anchor_price * 1.00 if pd.notna(anchor_price) else price * 0.98
            entry_high = anchor_price * 1.03 if pd.notna(anchor_price) else price
            entry_source = "Rebound-Zone nach Stabilisierung"
        elif setup_type == "Range-Breakout":
            anchor_price = prev20_high if pd.notna(prev20_high) else price
            entry_low = anchor_price * 1.000 if pd.notna(anchor_price) else price * 0.995
            entry_high = anchor_price * 1.012 if pd.notna(anchor_price) else price * 1.005
            entry_source = "Ausbruchszone über der Range"
        elif setup_type == "Trendfolge":
            anchor_price = ma20 if pd.notna(ma20) else price
            entry_low = anchor_price * 0.995 if pd.notna(anchor_price) else price * 0.99
            entry_high = anchor_price * 1.015 if pd.notna(anchor_price) else price * 1.01
            entry_source = "Trendfolge-Zone bei Rücksetzer"
        else:
            entry_low = price
            entry_high = price
            entry_source = "Aktueller Kurs / kein sauberes Setup"

        suggested_entry_zone = format_price_zone(entry_low, entry_high, ccy)
        entry_quality = (
            "gut" if pd.notna(entry_low) and pd.notna(entry_high) and entry_low <= price <= entry_high
            else ("abwarten" if pd.notna(entry_high) and price > entry_high else "früh")
        )

        # Stop-Logik je Setup-Typ
        generic_atr_stop = price - 1.8 * atr if pd.notna(atr) else np.nan
        generic_struct_stop = ma50 * 0.965 if pd.notna(ma50) else np.nan
        setup_stop_candidates = []
        stop_source = "Standard-Stop"

        if setup_type == "Breakout":
            breakout_level = prev20_high if pd.notna(prev20_high) else price
            setup_stop_candidates = [
                breakout_level * 0.975 if pd.notna(breakout_level) else np.nan,
                price - 1.6 * atr if pd.notna(atr) else np.nan,
                ma20 * 0.985 if pd.notna(ma20) else np.nan,
            ]
            stop_source = "Unter Breakout-Level / ATR"
        elif setup_type == "Breakout-Retest":
            retest_level = prev20_high if pd.notna(prev20_high) else price
            setup_stop_candidates = [
                retest_level * 0.985 if pd.notna(retest_level) else np.nan,
                prev20_low * 0.995 if pd.notna(prev20_low) else np.nan,
                price - 1.4 * atr if pd.notna(atr) else np.nan,
            ]
            stop_source = "Unter Retest-Niveau"
        elif setup_type == "Pullback an MA20":
            setup_stop_candidates = [
                ma20 * 0.985 if pd.notna(ma20) else np.nan,
                prev20_low * 0.995 if pd.notna(prev20_low) else np.nan,
                price - 1.4 * atr if pd.notna(atr) else np.nan,
            ]
            stop_source = "Unter MA20 / Pullback-Tief"
        elif setup_type == "Pullback an MA50":
            setup_stop_candidates = [
                ma50 * 0.985 if pd.notna(ma50) else np.nan,
                prev20_low * 0.99 if pd.notna(prev20_low) else np.nan,
                price - 1.5 * atr if pd.notna(atr) else np.nan,
            ]
            stop_source = "Unter MA50 / Pullback-Tief"
        elif setup_type == "Rebound":
            setup_stop_candidates = [
                prev20_low * 0.99 if pd.notna(prev20_low) else np.nan,
                price - 1.3 * atr if pd.notna(atr) else np.nan,
                ma20 * 0.98 if pd.notna(ma20) else np.nan,
            ]
            stop_source = "Unter Rebound-Tief"
        elif setup_type == "Range-Breakout":
            range_top = prev20_high if pd.notna(prev20_high) else price
            setup_stop_candidates = [
                range_top * 0.985 if pd.notna(range_top) else np.nan,
                price - 1.5 * atr if pd.notna(atr) else np.nan,
                ma20 * 0.985 if pd.notna(ma20) else np.nan,
            ]
            stop_source = "Unter Range-Oberkante"
        elif setup_type == "Trendfolge":
            setup_stop_candidates = [
                ma20 * 0.985 if pd.notna(ma20) else np.nan,
                ma50 * 0.985 if pd.notna(ma50) else np.nan,
                price - 1.8 * atr if pd.notna(atr) else np.nan,
            ]
            stop_source = "Unter Trendzone / Higher Low"

        stop_candidates = [
            x for x in setup_stop_candidates + [generic_atr_stop, generic_struct_stop]
            if pd.notna(x) and x > 0 and x < price
        ]

        if stop_candidates:
            stop_used = round(max(stop_candidates), 2)
        else:
            stop_used = round(price - max(price * 0.08, (atr * 1.8 if pd.notna(atr) else price * 0.06)), 2)
            stop_source = "Fallback-Stop"

        atr_stop = round(generic_atr_stop, 2) if pd.notna(generic_atr_stop) else np.nan
        stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0
        if stop_used <= 0 or stop_used >= price:
            stop_used = round(price - max(price * 0.08, (atr * 1.8 if pd.notna(atr) else price * 0.06)), 2)
            stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0
            stop_source = "Fallback-Stop"

        practical_min_stop_dist_pct = 3.5
        if price > 0 and stop_dist < practical_min_stop_dist_pct:
            stop_used = round(price * (1 - practical_min_stop_dist_pct / 100), 2)
            stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0
            stop_source = "Praxis-Mindestabstand"

        risk_per_share = price - stop_used

        tp1 = round(price + 1 * risk_per_share, 2)
        tp1_source = "1R vom Stop"

        # Setup-spezifische Ziel-Logik
        technical_target_1 = np.nan
        technical_target_2 = np.nan

        if setup_type in {"Breakout", "Range-Breakout"}:
            technical_target_1 = prev20_high * 1.03 if pd.notna(prev20_high) and prev20_high > price else np.nan
            technical_target_2 = high52 if pd.notna(high52) and high52 > price else np.nan
        elif setup_type == "Breakout-Retest":
            technical_target_1 = prev20_high * 1.02 if pd.notna(prev20_high) and prev20_high > price else np.nan
            technical_target_2 = high52 if pd.notna(high52) and high52 > price else np.nan
        elif setup_type == "Pullback an MA20":
            technical_target_1 = prev20_high if pd.notna(prev20_high) and prev20_high > price else np.nan
            technical_target_2 = high52 if pd.notna(high52) and high52 > price else np.nan
        elif setup_type == "Pullback an MA50":
            technical_target_1 = ma20 * 1.03 if pd.notna(ma20) and ma20 > price else prev20_high if pd.notna(prev20_high) and prev20_high > price else np.nan
            technical_target_2 = high52 if pd.notna(high52) and high52 > price else np.nan
        elif setup_type == "Rebound":
            technical_target_1 = ma50 if pd.notna(ma50) and ma50 > price else ma20 if pd.notna(ma20) and ma20 > price else np.nan
            technical_target_2 = prev20_high if pd.notna(prev20_high) and prev20_high > price else high52 if pd.notna(high52) and high52 > price else np.nan
        elif setup_type == "Trendfolge":
            technical_target_1 = prev20_high if pd.notna(prev20_high) and prev20_high > price else np.nan
            technical_target_2 = high52 if pd.notna(high52) and high52 > price else np.nan

        tp2_floor = price + 1.8 * risk_per_share
        if pd.notna(technical_target_1) and technical_target_1 > price:
            tp2 = round(max(float(technical_target_1), tp2_floor), 2)
            tp2_source = f"Primärziel aus Setup ({setup_type})"
        elif pd.notna(target) and target > price:
            tp2 = round(max(float(target), tp2_floor), 2)
            tp2_source = "Analysten-Target"
        elif pd.notna(high52) and high52 > price:
            tp2 = round(max(float(high52), tp2_floor), 2)
            tp2_source = "52W-Hoch"
        else:
            tp2 = round(price + 2 * risk_per_share, 2)
            tp2_source = "2R-Fallback"

        tp3_floor = max(price + 2.8 * risk_per_share, tp2 + 0.8 * risk_per_share)
        if pd.notna(technical_target_2) and technical_target_2 > tp2:
            tp3 = round(max(float(technical_target_2), tp3_floor), 2)
            tp3_source = f"Sekundärziel aus Setup ({setup_type})"
        elif pd.notna(target) and target > tp2:
            tp3 = round(max(float(target), tp3_floor), 2)
            tp3_source = "Analysten-Target"
        elif pd.notna(high52) and high52 > tp2:
            tp3 = round(max(float(high52), tp3_floor), 2)
            tp3_source = "52W-Hoch"
        else:
            tp3 = round(max(price + 3 * risk_per_share, tp2 + risk_per_share), 2)
            tp3_source = "3R-Ziel"

        crv = (tp2 - price) / (price - stop_used) if (price - stop_used) > 0 else 0
        timing_trade_score = round(clamp(s4 * 0.45 + s5 * 0.25 + rs_score * 0.20 + s6 * 0.10))
        stop_score = ideal_range_score(stop_dist, ideal_low=3.0, ideal_high=7.5, hard_low=1.0, hard_high=13.0)
        crv_score = linear_score(crv, low=0.9, high=3.0, floor=15, ceiling=95)
        market_trade_score = 85 if market_info["regime"] == "POSITIV" else (60 if market_info["regime"] == "NEUTRAL" else 25)
        entry_score = entry_quality_score(entry_quality, price, entry_low, entry_high)

        tradeability_score = round(clamp(
            crv_score * 0.34
            + stop_score * 0.18
            + timing_trade_score * 0.22
            + market_trade_score * 0.10
            + entry_score * 0.16
        ))
        tradeability_text = tradeability_label(tradeability_score)

        setup_confidence = round(clamp(
            (88 if setup_type in {"Breakout", "Pullback im Aufwärtstrend", "Trendfolge"} else 72 if setup_type in {"Rebound im Aufwärtstrend"} else 35) * 0.35
            + s3 * 0.20
            + s4 * 0.20
            + min(kb / 4 * 100, 100) * 0.15
            + (100 if entry_quality == "gut" else 60 if entry_quality == "abwarten" else 45) * 0.10
        ))
        setup_confidence_text = setup_confidence_label(setup_confidence)

        confidence_numeric = round(confidence_info.get("coverage", 0) * 100) if isinstance(confidence_info, dict) else 50
        market_long_score = 85 if market_info["regime"] == "POSITIV" else (60 if market_info["regime"] == "NEUTRAL" else 30)
        red_flag_adjustment = clamp(100 - min(red_flag_penalty_total * 4, 55), 35, 100)

        investment_case_score = round(clamp(
            company * 0.35
            + investment * 0.35
            + confidence_numeric * 0.10
            + market_long_score * 0.10
            + red_flag_adjustment * 0.10
        ))
        investment_case_text = investment_case_label(investment_case_score)

        entry_location_score = 90 if entry_quality == "gut" else (58 if entry_quality == "abwarten" else 44)
        trading_case_score = round(clamp(
            tradeability_score * 0.35
            + timing_trade_score * 0.25
            + setup_adj * 0.20
            + entry_location_score * 0.10
            + market_trade_score * 0.10
        ))

        # Konsistenz-Deckel: ein Einstiegs-Case darf nicht sehr hoch werden,
        # wenn Setup-Typ, Timing oder Setup-Confidence dagegen sprechen.
        if setup_type == "Kein sauberes Setup":
            trading_case_score = min(trading_case_score, 55)
        if pd.notna(timing_trade_score) and timing_trade_score < 50:
            trading_case_score = min(trading_case_score, 52)
        if pd.notna(setup_confidence) and setup_confidence < 60:
            trading_case_score = min(trading_case_score, 58)
        if entry_quality == "abwarten":
            trading_case_score = min(trading_case_score, 60)
        elif entry_quality == "früh":
            trading_case_score = min(trading_case_score, 56)

        trading_case_score = round(clamp(trading_case_score))
        trading_case_text = trading_case_label(trading_case_score)

        risk_eur = depot * (risk_pct / 100)
        pos_size = int(risk_eur / risk_per_share) if risk_per_share > 0 else 0
        time_stop = (date.today() + timedelta(days=hd)).strftime("%d.%m.%Y")
    else:
        atr_stop = np.nan
        stop_used = np.nan
        stop_dist = np.nan
        tp1 = np.nan
        tp2 = np.nan
        tp3 = np.nan
        tp1_source = "-"
        tp2_source = "-"
        tp3_source = "-"
        technical_target_1 = np.nan
        technical_target_2 = np.nan
        stop_source = "-"
        suggested_entry_zone = "-"
        entry_source = "-"
        entry_quality = "-"
        crv = np.nan
        timing_trade_score = round(clamp(s4 * 0.45 + s5 * 0.25 + rs_score * 0.20 + s6 * 0.10))
        market_trade_score = 85 if market_info["regime"] == "POSITIV" else (60 if market_info["regime"] == "NEUTRAL" else 25)
        tradeability_score = round(clamp(
            20 * 0.34
            + 30 * 0.18
            + timing_trade_score * 0.22
            + market_trade_score * 0.10
            + 35 * 0.16
        ))
        tradeability_text = tradeability_label(tradeability_score)
        crv_score = np.nan
        stop_score = np.nan
        entry_score = np.nan
        confidence_numeric = round(confidence_info.get("coverage", 0) * 100) if isinstance(confidence_info, dict) else 50
        market_long_score = 85 if market_info["regime"] == "POSITIV" else (60 if market_info["regime"] == "NEUTRAL" else 30)
        red_flag_adjustment = clamp(100 - min(red_flag_penalty_total * 4, 55), 35, 100)
        investment_case_score = round(clamp(
            company * 0.35
            + investment * 0.35
            + confidence_numeric * 0.10
            + market_long_score * 0.10
            + red_flag_adjustment * 0.10
        ))
        investment_case_text = investment_case_label(investment_case_score)
        trading_case_score = round(clamp(
            (tradeability_score if pd.notna(tradeability_score) else 20) * 0.35
            + timing_trade_score * 0.25
            + setup_adj * 0.20
            + 40 * 0.10
            + (85 if market_info["regime"] == "POSITIV" else 60 if market_info["regime"] == "NEUTRAL" else 25) * 0.10
        ))
        if setup_type == "Kein sauberes Setup":
            trading_case_score = min(trading_case_score, 55)
        if pd.notna(timing_trade_score) and timing_trade_score < 50:
            trading_case_score = min(trading_case_score, 52)
        if pd.notna(setup_confidence) and setup_confidence < 60:
            trading_case_score = min(trading_case_score, 58)
        trading_case_score = round(clamp(trading_case_score))
        trading_case_text = trading_case_label(trading_case_score)
        risk_eur = depot * (risk_pct / 100)
        pos_size = 0
        time_stop = "-"

    short_term_raw = round(clamp(s4 * 0.45 + s5 * 0.28 + s6 * 0.17 + rs_score * 0.10))
    swing_raw = round(clamp(s3 * 0.26 + s4 * 0.28 + s5 * 0.16 + s6 * 0.10 + rs_score * 0.12 + w52 * 0.08))
    mid_term_raw = round(clamp(setup_adj * 0.55 + company * 0.45))
    long_term_raw = round(clamp(company * 0.50 + growth_score * 0.15 + growth_quality * 0.10 + quality_score * 0.15 + valuation_score * 0.10))
    very_long_term_raw = round(clamp(company * 0.40 + quality_score * 0.20 + growth_score * 0.15 + growth_quality * 0.10 + valuation_score * 0.15))

    market_hmap_adj = 6 if market_info["regime"] == "POSITIV" else (-8 if market_info["regime"] == "NEGATIV" else 0)
    red_flag_hmap_penalty = min(red_flag_penalty_total * 2, 18)

    short_term_score = round(clamp(
        short_term_raw * 0.55
        + trading_case_score * 0.35
        + setup_confidence * 0.10
        + market_hmap_adj
        - red_flag_hmap_penalty * 0.35
    ))
    swing_score = round(clamp(
        swing_raw * 0.45
        + trading_case_score * 0.35
        + investment_case_score * 0.10
        + setup_confidence * 0.10
        + market_hmap_adj
        - red_flag_hmap_penalty * 0.30
    ))
    mid_term_score = round(clamp(
        mid_term_raw * 0.40
        + investment_case_score * 0.35
        + trading_case_score * 0.15
        + company * 0.10
        + market_hmap_adj * 0.7
        - red_flag_hmap_penalty * 0.45
    ))
    long_term_score = round(clamp(
        long_term_raw * 0.45
        + investment_case_score * 0.35
        + company * 0.10
        + quality_score * 0.10
        + market_hmap_adj * 0.35
        - red_flag_hmap_penalty * 0.55
    ))
    very_long_term_score = round(clamp(
        very_long_term_raw * 0.45
        + investment_case_score * 0.30
        + company * 0.15
        + quality_score * 0.10
        + market_hmap_adj * 0.20
        - red_flag_hmap_penalty * 0.60
    ))

    # Zeithorizonte sollen nicht deutlich positiver wirken als die Gesamtsicht
    short_term_score = min(short_term_score, max(trading_case_score + 8, 0))
    swing_score = min(swing_score, max(trading_case_score + 10, 0))
    mid_term_score = min(mid_term_score, max(investment_case_score + 10, 0))
    long_term_score = min(long_term_score, max(investment_case_score + 8, 0))
    very_long_term_score = min(very_long_term_score, max(investment_case_score + 6, 0))

    hmap = {
        "Kurzfrist": short_term_score,
        "Swing": swing_score,
        "Mittelfrist": mid_term_score,
        "Langfrist": long_term_score,
        "Sehr langfristig": very_long_term_score,
    }

    # ---------- Katalysatoren / Event-Kontext ----------
    event_phase_label = classify_event_phase(days_earn)
    earnings_reaction_5d, earnings_reaction_10d = calc_post_earnings_reaction(close, days_earn)

    event_risk_score = calc_event_risk_score(
        days_earn,
        has_upcoming_earnings,
        atr_pct,
        breakout_context if 'breakout_context' in locals() else False
    )

    post_earnings_reaction_score = calc_post_earnings_reaction_score(
        earnings_reaction_5d,
        earnings_reaction_10d,
        rs_score if pd.notna(rs_score) else 50
    )

    revision_momentum_score = calc_revision_momentum_score(
        upside,
        revenue_growth,
        earnings_growth,
        ret21,
        rs_score if pd.notna(rs_score) else 50
    )

    earnings_event_score = calc_earnings_event_score(
        event_phase_label,
        event_risk_score,
        post_earnings_reaction_score
    )

    catalyst_score = calc_catalyst_score(
        earnings_event_score,
        revision_momentum_score,
        post_earnings_reaction_score,
        event_phase_label
    )

    catalyst_text = catalyst_label(catalyst_score)
    post_earnings_text = event_phase_text(event_phase_label)

    # ---------- Leadership / Marktbreite ----------
    sector_label = sector if pd.notna(sector) and sector not in ["", "-", None] else "Unbekannt"
    industry_label = industry if pd.notna(industry) and industry not in ["", "-", None] else "Unbekannt"
    sector_etf_symbol = get_sector_etf_symbol(sector_label)
    sector_ctx = load_sector_context(sector_etf_symbol) if sector_etf_symbol else None

    sector_strength_score = calc_sector_strength_score(sector_ctx)
    sector_strength_available = pd.notna(sector_strength_score)
    rs_benchmark_score = calc_rs_benchmark_score(rs_vs_benchmark_21, rs_vs_benchmark_63, rs_vs_benchmark_126)
    rs_acceleration_score = calc_rs_acceleration_score(rs_vs_benchmark_21, rs_vs_benchmark_63, rs_vs_benchmark_126)
    industry_strength_score = calc_industry_strength_score(
        sector_strength_score if pd.notna(sector_strength_score) else 50,
        rs_score if pd.notna(rs_score) else 50,
        company if pd.notna(company) else 50,
    )
    leadership_score = calc_leadership_score(
        sector_strength_score if pd.notna(sector_strength_score) else 50,
        industry_strength_score,
        rs_benchmark_score,
        rs_acceleration_score,
        rs_score if pd.notna(rs_score) else 50,
    )
    leadership_status = get_leadership_status(leadership_score, rs_acceleration_score)
    sector_trend_text = strength_text(sector_strength_score)
    industry_trend_text = strength_text(industry_strength_score)

    setup_type_quality_score = calc_setup_type_quality_score(
        setup_type,
        base_quality_score,
        volume_quality_proxy,
        rs_score if pd.notna(rs_score) else 50,
        trend_quality_score,
        setup_confidence if pd.notna(setup_confidence) else 50,
        pullback_quality_score
    )
    setup_priority_score = calc_setup_priority_score(
        setup_type_quality_score,
        leadership_score,
        trend_quality_score,
        base_quality_score,
        trading_case_score
    )

    trading_case_score = round(clamp(trading_case_score * 0.78 + trend_quality_score * 0.08 + base_quality_score * 0.07 + setup_type_quality_score * 0.07))
    investment_case_score = round(clamp(investment_case_score * 0.90 + leadership_score * 0.10))
    tradeability_score = round(clamp(tradeability_score * 0.82 + trend_quality_score * 0.08 + base_quality_score * 0.05 + setup_priority_score * 0.05))
    trading_case_score = round(clamp(
        trading_case_score * 0.82
        + volume_quality_score * 0.10
        + breakout_volume_score * 0.04
        + pullback_dryup_score * 0.04
    ))
    setup_type_quality_score = round(clamp(
        setup_type_quality_score * 0.82
        + volume_quality_score * 0.10
        + breakout_volume_score * 0.04
        + pullback_dryup_score * 0.04
    ))
    setup_priority_score = round(clamp(
        setup_priority_score * 0.88
        + volume_quality_score * 0.12
    ))

    investment_case_score = round(clamp(
        investment_case_score * 0.80
        + catalyst_score * 0.10
        + institutional_quality_score * 0.10
    ))
    setup_priority_score = round(clamp(
        setup_priority_score * 0.92
        + catalyst_score * 0.08
    ))
    tradeability_score = round(clamp(
        tradeability_score * 0.96
        + earnings_event_score * 0.04
    ))

    trading_case_text = trading_case_label(trading_case_score)
    investment_case_text = investment_case_label(investment_case_score)
    tradeability_text = tradeability_label(tradeability_score)

    position_mode = buy_in_override > 0

    # ---------- Recommendations ----------
    if has_upcoming_earnings and days_earn < 7:
        emp, conv = ("VETO - Earnings < 7 Tage", "-")
    elif position_mode:
        if investment >= 78 and kb >= 3 and market_info["regime"] == "POSITIV":
            emp, conv = ("HALTEN / AUSBAUEN", "HIGH")
        elif investment >= 65 and market_info["regime"] != "NEGATIV":
            emp, conv = ("HALTEN / ENGE BEOBACHTUNG", "MEDIUM")
        elif investment >= 52:
            emp, conv = ("HALTEN / RISIKO PRÜFEN", "LOW-MEDIUM")
        else:
            emp, conv = ("RISIKO REDUZIEREN / STOPP PRÜFEN", "LOW")
    else:
        if investment >= 78 and kb >= 3 and market_info["regime"] == "POSITIV":
            emp, conv = ("BUY / ACCUMULATE", "HIGH")
        elif investment >= 68 and market_info["regime"] != "NEGATIV":
            emp, conv = ("WATCH / EINSTIEG PRÜFEN", "MEDIUM")
        elif investment >= 52:
            emp, conv = ("BEOBACHTEN", "LOW-MEDIUM")
        else:
            emp, conv = ("AVOID / WAIT", "LOW")

    # ---------- Trading Board ----------
    tb_score = 0
    tb_details = []
    tb_context = []

    tb_buy = buy_in_override if buy_in_override > 0 else 0.0
    tb_basispreis = tb_buy if tb_buy > 0 else price
    tb_perf = ((price - tb_buy) / tb_buy) * 100 if tb_buy > 0 else 0.0
    tb_stop = price - (2.5 * atr)
    tb_tp1 = tb_basispreis + (2.5 * atr)
    tb_tp2 = target if pd.notna(target) and target > tb_tp1 else tb_basispreis + (5.0 * atr)

    tb_details.append(f"S0: {price:.2f} {ccy}")

    if pd.notna(earnings_ts):
        if has_past_earnings:
            tb_details.append(f"S1 Earnings: letzte Earnings am {earnings_dt.strftime('%d.%m.%Y')}")
        else:
            tb_details.append(f"S1 Earnings: in {int(days_earn)}d ({sg_earn_txt})")
    else:
        tb_details.append("S1 Earnings: kein Datum")

    if price > ma200:
        tb_score += 1
        tb_details.append("S2: Über MA200 ✓")
    else:
        tb_details.append("S2: Unter MA200 ❌")

    if price > ma50:
        tb_score += 1
        tb_details.append("S3: Über MA50 (+1) ✓")
    else:
        tb_score -= 1
        tb_details.append("S3: Unter MA50 (-1) ❌")

    if ma50 > ma200:
        tb_score += 1
        tb_details.append("S4: Golden Cross ✓")
    else:
        tb_details.append("S4: Trendstruktur schwach ❌")

    if 40 < rsi < 60 or rsi < 30:
        tb_score += 1
        tb_details.append("S5: RSI konstruktiv ✓")
    else:
        tb_details.append("S5: RSI hoch/niedrig ❌")

    if position_mode:
        if tb_perf > 5:
            tb_score += 1
            tb_details.append(f"S6: +{tb_perf:.1f}% seit Einstieg ✓")
        else:
            tb_details.append(f"S6: {tb_perf:.1f}% seit Einstieg ❌")
    else:
        tb_details.append("S6: Watchlist-Modus (neutral)")

    if macd_hist_current > macd_hist_prev:
        tb_score += 1
        tb_details.append("S7: Momentum steigt ✓")
    else:
        tb_details.append("S7: Momentum fällt ❌")

    if earnings_warning:
        tb_score -= 3
        tb_details.insert(0, "⚠️ EARNINGS IN <7 TAGEN (Vorsicht!)")

    if 20 < rsi < 80:
        tb_context.append("S8: Vola ok ✓")

    if macd_bull_cross:
        tb_context.append("S9: MACD Bull-Cross! 🚀")

    if smart_money_default:
        tb_context.append("S10: Smart Money sammelt ein ✓")
    else:
        tb_context.append("S10: Smart Money verkauft ❌")

    if adx > 25:
        tb_context.append("S11: ADX>25 starker Trend ✓")
    else:
        tb_context.append("S11: ADX<25 Seitwärts ❌")

    if stoch_k_v < 20 and stoch_d_v < 20 and stoch_k_v > stoch_d_v:
        tb_context.append("S12: Stoch Oversold Cross ✓")
    elif stoch_k_v > 80:
        tb_context.append("S12: Stoch überkauft ❌")
    else:
        tb_context.append("S12: Stoch neutral ❌")

    if willr_v < -80:
        tb_context.append("S13: Williams%R extrem Oversold ✓")
    elif willr_v > -20:
        tb_context.append("S13: Williams%R überkauft ❌")
    else:
        tb_context.append("S13: Williams%R neutral ❌")

    if obv_trend == "steigend" and vol_ratio >= 1.0:
        tb_context.append("S14: OBV/Volumen bestätigt ✓")
    else:
        tb_context.append("S14: OBV/Volumen schwach ❌")

    if pd.notna(prev20_high) and price > prev20_high:
        tb_context.append("S15: 20D Breakout ✓")
    elif pd.notna(prev20_low) and price < prev20_low:
        tb_context.append("S15: 20D Breakdown ❌")
    else:
        tb_context.append("S15: Range intakt ❌")

    if pd.notna(bb_upper) and price > bb_upper:
        tb_context.append("S16: BB Breakout UP ✓")
    elif bb_squeeze:
        tb_context.append("S16: BB Squeeze Achtung ✓")
    elif pd.notna(bb_lower) and price < bb_lower:
        tb_context.append("S16: BB Breakout DOWN ❌")
    else:
        tb_context.append("S16: BB neutral ❌")

    if pd.notna(target) and target > 0 and price > 0:
        tb_potenzial = ((target - price) / price) * 100
        if tb_potenzial > 15:
            tb_context.append(f"S17: Target +{tb_potenzial:.1f}% ✓")
        elif tb_potenzial < 0:
            tb_context.append(f"S17: Target -{abs(tb_potenzial):.1f}% ❌")
        else:
            tb_context.append(f"S17: Target +{tb_potenzial:.1f}% neutral ❌")
    else:
        tb_context.append("S17: Kein valides Target ❌")

    current_month = datetime.now().month
    if current_month in [8, 9]:
        tb_context.append("S18: Seasonality schlecht (-1) ❌")
    elif current_month in [11, 12, 1]:
        tb_context.append("S18: Seasonality stark (+1) ✓")
    else:
        tb_context.append("S18: Seasonality neutral ❌")

    if crv >= 2.0:
        tb_context.append("S19: CRV >= 2.0 ✓")
    elif crv < 1.5:
        tb_context.append("S19: CRV schwach ❌")
    else:
        tb_context.append("S19: CRV ok/neutral ❌")

    short_squeeze = pd.notna(short_pct) and short_pct > 0.12 and ret5 > 0 and vol_ratio > 1.2
    if short_squeeze:
        tb_context.append("S20: 🚀 SHORT SQUEEZE POTENZIAL ✓")
    else:
        tb_context.append("S20: kein Short-Squeeze-Signal ❌")

    if pd.notna(pe) and 0 < pe < 15:
        tb_context.append(f"S21: 🟢 VALUE KGV ({pe:.1f}) ✓")
    elif pd.notna(pe) and pe > 50:
        tb_context.append(f"S21: 🔴 TEUER KGV>50 ({pe:.1f}) ❌")
    else:
        tb_context.append(f"S21: Value neutral ({fmt_num(pe,1)}) ❌")

    if market_info["regime"] == "POSITIV":
        tb_context.append(f"S22: Marktfilter positiv ({benchmark_label}) ✓")
    elif market_info["regime"] == "NEGATIV":
        tb_context.append(f"S22: Marktfilter negativ ({benchmark_label}) ❌")
    else:
        tb_context.append(f"S22: Marktfilter neutral ({benchmark_label}) ❌")

    if pd.notna(rs_vs_benchmark_63):
        if rs_vs_benchmark_63 > 0:
            tb_context.append(f"S23: Outperformance vs {benchmark_label} +{rs_vs_benchmark_63:.1f}% ✓")
        else:
            tb_context.append(f"S23: Underperformance vs {benchmark_label} {rs_vs_benchmark_63:.1f}% ❌")
    else:
        tb_context.append("S23: Benchmark-Vergleich n/a ❌")

    tb_signal, tb_empf = tb_signal_label(tb_score)
    tb_score_100 = normalize_tb_score_100(tb_score)
    tb_timing_text = trading_timing_label(tb_score)

    # ---------- Positionsmanagement 2.0 ----------
    if position_mode:
        add_on_action = "Nein"
        partial_profit_action = "Nein"
        stop_action = "Beibehalten"
        risk_note = "Keine Auffälligkeit"
        position_action = "Halten"
        legacy_position_action = "Halten"

        if has_upcoming_earnings and pd.notna(days_earn) and days_earn < 7:
            risk_note = "Earnings-Risiko kurzfristig erhöht"

        if pd.notna(tb_perf) and tb_perf >= 12 and pd.notna(tp1) and price >= tp1 * 0.98:
            partial_profit_action = "Ja, Teilgewinn prüfen"

        if (
            investment_case_score >= 74
            and trading_case_score >= 70
            and valid_trade_setup
            and setup_confidence >= 62
            and market_info["regime"] == "POSITIV"
        ):
            add_on_action = "Ja, selektiv möglich"

        if trading_case_score < 48 or market_info["regime"] == "NEGATIV" or setup_confidence < 45:
            legacy_position_action = "Risiko reduzieren"
        elif trading_case_score < 60 or setup_confidence < 60 or entry_quality == "früh":
            legacy_position_action = "Halten / eng beobachten"
        elif investment_case_score >= 75 and trading_case_score >= 68 and setup_confidence >= 60:
            legacy_position_action = "Halten / ggf. ausbauen"
        else:
            legacy_position_action = "Halten"

        if pd.notna(stop_used) and price > 0:
            if pd.notna(tb_perf) and tb_perf >= 15 and pd.notna(tp1) and price >= tp1:
                stop_action = f"Stop auf {max(stop_used, tb_basispreis):.2f} {ccy} anheben"
            elif pd.notna(tb_perf) and tb_perf >= 8:
                stop_action = f"Stop enger nachziehen auf {stop_used:.2f} {ccy}"
            else:
                stop_action = f"Stop aktuell bei {stop_used:.2f} {ccy}"

        if pd.notna(tb_perf) and tb_perf < -6 and trading_case_score < 55:
            risk_note = "Verlustposition mit schwächerem Setup"
        elif pd.notna(tb_perf) and tb_perf > 18:
            risk_note = "Gewinnposition, aktives Management sinnvoll"
    else:
        position_action = "Nicht anwendbar"
        add_on_action = "Nicht anwendbar"
        partial_profit_action = "Nicht anwendbar"
        stop_action = "Nicht anwendbar"
        risk_note = "Watchlist-Modus"

    # ---------- Watchlist / Trigger-Logik ----------
    if not position_mode:
        if has_upcoming_earnings and pd.notna(days_earn) and days_earn < 7:
            trigger_status = "Warten"
            watchlist_priority = "Niedrig"
            watchlist_priority_score = 30
            next_trigger = "Nach den Zahlen neu prüfen"
            trigger_reason = "Earnings-Veto kurzfristig"
        elif market_info["regime"] == "NEGATIV":
            trigger_status = "Passiv"
            watchlist_priority = "Niedrig"
            watchlist_priority_score = 28
            next_trigger = "Auf besseres Marktumfeld warten"
            trigger_reason = "Marktumfeld aktuell zu schwach"
        elif valid_trade_setup and entry_quality == "gut" and trading_case_score >= 68 and setup_confidence >= 60:
            trigger_status = "Aktiv"
            watchlist_priority = "Hoch"
            watchlist_priority_score = 85
            next_trigger = "Jetzt prüfbar"
            trigger_reason = "Setup valide, Timing stimmig und in Entry-Zone"
        elif valid_trade_setup and entry_quality == "abwarten" and trading_case_score >= 60 and setup_confidence >= 55:
            trigger_status = "Nahe dran"
            watchlist_priority = "Hoch"
            watchlist_priority_score = 76
            next_trigger = "Rücksetzer in Entry-Zone abwarten"
            trigger_reason = "Setup valide, aber Kurs aktuell über Entry-Zone"
        elif setup_type != "Kein sauberes Setup" and entry_quality == "früh" and trading_case_score >= 55 and setup_confidence >= 50:
            trigger_status = "Frühe Beobachtung"
            watchlist_priority = "Mittel"
            watchlist_priority_score = 60
            next_trigger = "Setup-Confirmation oder bessere Entry-Lage"
            trigger_reason = "Setup vorhanden, aber noch zu früh für einen sauberen Einstieg"
        elif investment_case_score >= 70:
            trigger_status = "Beobachten"
            watchlist_priority = "Mittel"
            watchlist_priority_score = 52
            next_trigger = "Trading-Case verbessern"
            trigger_reason = "Guter Investment-Case, aber noch kein sauberer Trigger"
        else:
            trigger_status = "Passiv"
            watchlist_priority = "Niedrig"
            watchlist_priority_score = 35
            next_trigger = "Auf klareres Setup warten"
            trigger_reason = "Noch kein priorisierter Watchlist-Kandidat"

        if setup_type == "Kein sauberes Setup":
            trigger_status = "Passiv"
            watchlist_priority = "Niedrig" if investment_case_score < 75 else "Mittel"
            watchlist_priority_score = 38 if investment_case_score < 75 else 48
            next_trigger = "Auf neues Setup warten"
            trigger_reason = "Ohne sauberes Setup kein aktiver Trigger"
    else:
        trigger_status = "Nicht anwendbar"
        watchlist_priority = "Nicht anwendbar"
        watchlist_priority_score = np.nan
        next_trigger = "Nicht anwendbar"
        trigger_reason = "Positionsmodus"

    # ---------- Finale Konsistenzregeln ----------
    if not position_mode:
        if setup_type == "Kein sauberes Setup":
            emp = "BEOBACHTEN" if investment_case_score >= 70 else "AVOID / WAIT"
            conv = "LOW-MEDIUM" if investment_case_score >= 70 else "LOW"
        elif trigger_status == "Aktiv" and trading_case_score >= 68:
            emp = "WATCH / EINSTIEG PRÜFEN"
            conv = "MEDIUM" if conv == "-" else conv
        elif trigger_status in {"Passiv", "Warten"}:
            emp = "AVOID / WAIT" if investment_case_score < 70 else "BEOBACHTEN"

    if position_mode and position_action == "Risiko reduzieren":
        emp = "RISIKO REDUZIEREN / STOPP PRÜFEN"
        conv = "LOW"

    # ---------- Short-term helper board ----------
    stb_score = 0
    stb_items = []

    if price > ma50:
        stb_score += 2
        stb_items.append("MA50 +2")
    elif price > ma200:
        stb_score += 1
        stb_items.append("MA200 +1")
    else:
        stb_score -= 1
        stb_items.append("Trend -1")

    if 40 < rsi < 60 or rsi < 30:
        stb_score += 1
        stb_items.append("RSI +1")

    if 20 < rsi < 80:
        stb_score += 1
        stb_items.append("Vola +1")

    if macd_hist_current > macd_hist_prev:
        stb_score += 1
        stb_items.append("Momentum +1")

    if macd_bull_cross:
        stb_score += 1
        stb_items.append("Bull-Cross +1")

    if smart_money_default:
        stb_score += 1
        stb_items.append("Smart Money +1")

    if adx > 25:
        stb_score += 1
        stb_items.append("ADX +1")

    if stoch_k_v < 20 and stoch_d_v < 20 and stoch_k_v > stoch_d_v:
        stb_score += 1
        stb_items.append("Stoch +1")

    if willr_v < -80:
        stb_score += 1
        stb_items.append("Williams +1")

    stb_signal, stb_empf = tb_signal_label(stb_score)
    stb_text = ", ".join(stb_items) if stb_items else "keine positiven Kurzfrist-Signale"

    # ---------- Explanations ----------
    strengths, weaknesses, decision_summary = build_decision_explanation(
        setup=setup_adj,
        company=company,
        investment=investment,
        market_regime=market_info["regime"],
        rs_vs_benchmark_63=rs_vs_benchmark_63,
        quality_score=quality_score,
        growth_score=growth_score,
        valuation_score=valuation_score,
        balance_score=balance_score,
        red_flag_items=red_flag_items,
        earnings_warning=earnings_warning,
        kb=kb,
        position_mode=position_mode
    )

    rows = []
    for line in tb_details:
        if ": " in line:
            k, v = line.split(": ", 1)
            rows.append({"Punkt": k, "Detail": v})
        else:
            rows.append({"Punkt": "Info", "Detail": line})
    tb_df = pd.DataFrame(rows)

    red_flags_df = pd.DataFrame(red_flag_items) if red_flag_items else pd.DataFrame(
        [{"Kategorie": "-", "Status": "🟢", "Detail": "Keine relevanten Red Flags erkannt", "Penalty": 0}]
    )

    top_red_flag = red_flag_items[0]["Detail"] if red_flag_items else "-"
    short_thesis = build_short_thesis(investment, tb_score, market_info["regime"], top_red_flag, position_mode)


    # ---------- Exit / Verkaufssystem ----------
    avg_cost = normalize_missing(override) if "override" in locals() else np.nan
    if not pd.notna(avg_cost) or avg_cost <= 0:
        avg_cost = np.nan

    position_pnl_pct = ((price / avg_cost) - 1) * 100 if pd.notna(avg_cost) and avg_cost > 0 and pd.notna(price) else np.nan

    # Kontext für Gewinner / Verlierer / Korrektur
    if pd.notna(position_pnl_pct):
        if position_pnl_pct >= 20:
            pnl_bucket = "starker Gewinner"
        elif position_pnl_pct >= 8:
            pnl_bucket = "Gewinner"
        elif position_pnl_pct <= -10:
            pnl_bucket = "klarer Verlierer"
        elif position_pnl_pct < 0:
            pnl_bucket = "leichter Verlierer"
        else:
            pnl_bucket = "nahe Einstand"
    else:
        pnl_bucket = "ohne Einstandsdaten"
    horizon_label = str(horizon or "").strip() or "unbekannt"

    healthy_trend_context = (
        pd.notna(price) and pd.notna(ma50) and price >= ma50
        and market_info["regime"] == "POSITIV"
        and pd.notna(setup_confidence) and setup_confidence >= 55
        and pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 >= 0
    )

    winner_context = pd.notna(position_pnl_pct) and position_pnl_pct >= 8
    strong_winner_context = pd.notna(position_pnl_pct) and position_pnl_pct >= 18
    loser_context = pd.notna(position_pnl_pct) and position_pnl_pct <= -5

    trend_break_score = 0
    if pd.notna(price) and pd.notna(ma20) and price < ma20:
        trend_break_score += 5 if healthy_trend_context else 8
    if pd.notna(price) and pd.notna(ma50) and price < ma50:
        trend_break_score += 16
    if pd.notna(price) and pd.notna(ma200) and price < ma200:
        trend_break_score += 22
    if pd.notna(ma20) and pd.notna(ma50) and ma20 < ma50:
        trend_break_score += 10
    if pd.notna(ma50) and pd.notna(ma200) and ma50 < ma200:
        trend_break_score += 12
    swing_low_20 = safe_last(close.shift(1).rolling(20).min(), np.nan)
    if pd.notna(price) and pd.notna(swing_low_20) and price < swing_low_20:
        trend_break_score += 10
    if healthy_trend_context and pd.notna(price) and pd.notna(ma50) and price >= ma50:
        trend_break_score = max(0, trend_break_score - 6)
    trend_break_score = min(100, trend_break_score)

    momentum_collapse_score = 0
    if pd.notna(rsi) and rsi < 50:
        momentum_collapse_score += 5
    if pd.notna(rsi) and rsi < 45:
        momentum_collapse_score += 9
    if pd.notna(rsi) and rsi < 40:
        momentum_collapse_score += 14
    if pd.notna(macd_v) and pd.notna(signal_v) and macd_v < signal_v:
        momentum_collapse_score += 8
    if pd.notna(macd_hist_current) and macd_hist_current < 0:
        momentum_collapse_score += 6
    if pd.notna(roc20) and roc20 < 0:
        momentum_collapse_score += 7
    if pd.notna(roc20) and roc20 < -5:
        momentum_collapse_score += 12
    if pd.notna(adx) and adx < 18 and pd.notna(roc20) and roc20 < 0:
        momentum_collapse_score += 5
    if healthy_trend_context and pd.notna(rsi) and rsi >= 45:
        momentum_collapse_score = max(0, momentum_collapse_score - 5)
    momentum_collapse_score = min(100, momentum_collapse_score)

    relative_weakness_score = 0
    if pd.notna(rs_score) and rs_score < 50:
        relative_weakness_score += 7
    if pd.notna(rs_score) and rs_score < 40:
        relative_weakness_score += 11
    if pd.notna(rs_vs_benchmark_21) and rs_vs_benchmark_21 < 0:
        relative_weakness_score += 6
    if pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 < 0:
        relative_weakness_score += 10
    if pd.notna(rs_vs_benchmark_126) and rs_vs_benchmark_126 < 0:
        relative_weakness_score += 7
    if pd.notna(rs_composite) and rs_composite < 45:
        relative_weakness_score += 10
    if healthy_trend_context and pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 > 3:
        relative_weakness_score = max(0, relative_weakness_score - 6)
    relative_weakness_score = min(100, relative_weakness_score)

    vol_ma20 = safe_last(vol.rolling(20).mean(), np.nan)
    prev_close = safe_last(close.shift(1), np.nan)
    ret1 = safe_last(close.pct_change(1) * 100, np.nan)
    ret2 = safe_last(close.shift(1).pct_change(1) * 100, np.nan)
    vol_now = safe_last(vol, np.nan)
    vol_prev = safe_last(vol.shift(1), np.nan)
    down_day = pd.notna(price) and pd.notna(prev_close) and price < prev_close
    high_volume = pd.notna(vol_now) and pd.notna(vol_ma20) and vol_now > 1.3 * vol_ma20

    distribution_score = 0
    if down_day and high_volume:
        distribution_score += 10
    dist_day_1 = (
        pd.notna(ret1) and ret1 < -1.5 and
        pd.notna(vol_now) and pd.notna(vol_ma20) and vol_now > 1.2 * vol_ma20
    )
    dist_day_prev = (
        pd.notna(ret2) and ret2 < -1.5 and
        pd.notna(vol_prev) and pd.notna(vol_ma20) and vol_prev > 1.2 * vol_ma20
    )
    if dist_day_1:
        distribution_score += 8
    if dist_day_1 and dist_day_prev:
        distribution_score += 12
    if pd.notna(ret21) and ret21 < 0 and down_day and high_volume:
        distribution_score += 8
    if healthy_trend_context and not dist_day_1:
        distribution_score = max(0, distribution_score - 4)
    distribution_score = min(100, distribution_score)

    exit_trigger_score = 0
    stop_broken = pd.notna(stop_used) and pd.notna(price) and price < stop_used
    if stop_broken:
        exit_trigger_score += 32
    if pd.notna(days_earn) and days_earn <= 7 and pd.notna(trading_case_score) and trading_case_score < 55:
        exit_trigger_score += 10
    if pd.notna(setup_confidence) and setup_confidence < 40:
        exit_trigger_score += 8
    if red_flag_penalty_total >= 12:
        exit_trigger_score += 8
    gap_down_pct = ((price / prev_close) - 1) * 100 if pd.notna(price) and pd.notna(prev_close) and prev_close != 0 else np.nan
    if pd.notna(gap_down_pct) and gap_down_pct <= -4:
        exit_trigger_score += 15
    if stop_broken and market_info["regime"] == "NEGATIV":
        exit_trigger_score += 8
    exit_trigger_score = min(100, exit_trigger_score)

    exit_score = round(clamp(
        trend_break_score * 0.31
        + momentum_collapse_score * 0.20
        + relative_weakness_score * 0.18
        + distribution_score * 0.13
        + exit_trigger_score * 0.18
    ))

    if strong_winner_context and healthy_trend_context:
        exit_score = max(0, exit_score - 10)
    elif winner_context and healthy_trend_context:
        exit_score = max(0, exit_score - 6)
    elif loser_context:
        exit_score = min(100, exit_score + 8)

    if horizon == "Swing (1-4 Wochen)":
        exit_score = min(100, exit_score + 5)
    elif horizon == "Langfristig (6-24 Monate)":
        exit_score = max(0, exit_score - 4)

    # Mindest-Score, wenn bereits echte Exit-Gründe vorliegen
    structural_exit_reasons = 0
    if pd.notna(ma20) and pd.notna(ma50) and ma20 < ma50:
        structural_exit_reasons += 1
    if pd.notna(rs_vs_benchmark_21) and rs_vs_benchmark_21 < 0:
        structural_exit_reasons += 1
    if pd.notna(price) and pd.notna(ma50) and price < ma50:
        structural_exit_reasons += 1
    if pd.notna(rsi) and rsi < 45:
        structural_exit_reasons += 1
    if pd.notna(macd_v) and pd.notna(signal_v) and macd_v < signal_v:
        structural_exit_reasons += 1
    if stop_broken:
        structural_exit_reasons += 2

    if structural_exit_reasons >= 3:
        exit_score = max(exit_score, 35)
    elif structural_exit_reasons == 2:
        exit_score = max(exit_score, 24)
    elif structural_exit_reasons == 1:
        exit_score = max(exit_score, 12)

    near_tp1 = pd.notna(tp1) and pd.notna(price) and price >= tp1 * 0.98
    near_tp2 = pd.notna(tp2) and pd.notna(price) and price >= tp2 * 0.96
    de_risk_gain_zone = winner_context and (near_tp1 or near_tp2)

    if exit_score >= 80:
        exit_score_text = "klarer Exit-Druck"
    elif exit_score >= 65:
        exit_score_text = "Verkaufsdruck erhöht"
    elif exit_score >= 45:
        exit_score_text = "Gewinne absichern"
    elif exit_score >= 25:
        exit_score_text = "erste Schwäche"
    else:
        exit_score_text = "stabil"

    if stop_broken:
        exit_action = "Verkaufen"
    elif exit_score >= 80:
        exit_action = "Verkaufen"
    elif exit_score >= 65:
        exit_action = "Risiko reduzieren"
    elif exit_score >= 45:
        if winner_context or de_risk_gain_zone:
            exit_action = "Teilgewinn prüfen"
        else:
            exit_action = "Risiko reduzieren"
    elif exit_score >= 25:
        exit_action = "Beobachten"
    else:
        exit_action = "Halten"

    if position_mode:
        legacy_action_for_merge = legacy_position_action if "legacy_position_action" in locals() else position_action
        if exit_action in {"Verkaufen", "Risiko reduzieren"}:
            position_action = exit_action
        elif exit_action == "Teilgewinn prüfen":
            position_action = "Teilgewinn prüfen"
        elif exit_action == "Beobachten":
            position_action = "Halten / eng beobachten"
        elif str(add_on_action).lower().startswith("ja") and exit_score < 25:
            position_action = "Halten / ggf. ausbauen"
        elif str(partial_profit_action).lower().startswith("ja") and winner_context:
            position_action = "Teilgewinn prüfen"
        else:
            position_action = legacy_action_for_merge

        if de_risk_gain_zone and exit_action in {"Teilgewinn prüfen", "Beobachten"}:
            partial_profit_action = "Ja, Teilgewinn prüfen"
        elif exit_action == "Verkaufen":
            partial_profit_action = "Nein"
            add_on_action = "Nein"
            risk_note = f"Exit-Modell: klarer Verkaufsdruck | {pnl_bucket}"
        elif exit_action == "Risiko reduzieren":
            add_on_action = "Nein"
            risk_note = f"Exit-Modell: Risikoabbau sinnvoll | {pnl_bucket}"
        elif exit_action == "Teilgewinn prüfen":
            partial_profit_action = "Ja, Teilgewinn prüfen"
            risk_note = f"Gewinnsicherung sinnvoll | {pnl_bucket}"
        elif exit_action == "Beobachten":
            risk_note = f"Erste Exit-Schwäche | {pnl_bucket}"
        elif str(add_on_action).lower().startswith("ja"):
            risk_note = f"Konstruktive Lage trotz Positionsmodus | {pnl_bucket}"

        if pd.notna(days_earn) and days_earn <= 7 and exit_score >= 45:
            risk_note = f"Earnings-Risiko bei erhöhter Exit-Schwäche | {pnl_bucket}"

    exit_reason_list = []
    if pd.notna(price) and pd.notna(ma50) and price < ma50:
        exit_reason_list.append("Kurs unter MA50")
    if pd.notna(price) and pd.notna(ma200) and price < ma200:
        exit_reason_list.append("Kurs unter MA200")
    if pd.notna(ma20) and pd.notna(ma50) and ma20 < ma50:
        exit_reason_list.append("MA20 unter MA50")
    if pd.notna(rsi) and rsi < 45:
        exit_reason_list.append("RSI unter 45")
    if pd.notna(macd_v) and pd.notna(signal_v) and macd_v < signal_v:
        exit_reason_list.append("MACD unter Signal")
    if pd.notna(roc20) and roc20 < 0:
        exit_reason_list.append("ROC20 negativ")
    if pd.notna(rs_vs_benchmark_21) and rs_vs_benchmark_21 < 0:
        exit_reason_list.append("Relative Schwäche vs Benchmark")
    if stop_broken:
        exit_reason_list.append("Stop unterschritten")
    if pd.notna(gap_down_pct) and gap_down_pct <= -4:
        exit_reason_list.append("deutlicher Gap-down")
    if dist_day_1:
        exit_reason_list.append("Distributionstag")
    if de_risk_gain_zone and not exit_reason_list:
        exit_reason_list.append("Gewinnzone erreicht, Teilgewinn sinnvoll")

    # Doppelte Exit-Gründe entfernen, Reihenfolge aber beibehalten
    deduped_exit_reason_list = []
    seen_exit_reasons = set()
    for reason in exit_reason_list:
        reason_key = str(reason).strip()
        if reason_key and reason_key not in seen_exit_reasons:
            deduped_exit_reason_list.append(reason_key)
            seen_exit_reasons.add(reason_key)
    exit_reason_list = deduped_exit_reason_list

    exit_reason_top = exit_reason_list[0] if exit_reason_list else "kein akuter Exit-Grund"

    if position_mode:
        if exit_score >= 80:
            hmap["Kurzfrist"] = min(hmap["Kurzfrist"], 25)
            hmap["Swing"] = min(hmap["Swing"], 22)
            hmap["Mittelfrist"] = min(hmap["Mittelfrist"], 28)
        elif exit_score >= 65:
            hmap["Kurzfrist"] = min(hmap["Kurzfrist"], 35)
            hmap["Swing"] = min(hmap["Swing"], 32)
            hmap["Mittelfrist"] = min(hmap["Mittelfrist"], 40)
        elif exit_score >= 45:
            hmap["Kurzfrist"] = min(hmap["Kurzfrist"], 48)
            hmap["Swing"] = min(hmap["Swing"], 50)

    return {
        "ticker": ticker,
        "df": df,
        "info": info,
        "name": name,
        "ccy": ccy,
        "exch": exch,
        "ts": ts,
        "sector": sector,
        "industry": industry,
        "company_summary": company_summary,
        "confidence_info": confidence_info,
        "market_info": market_info,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_label": benchmark_label,
        "price": price,
        "target": target,
        "upside": upside,
        "regime": regime,
        "reg_amp": reg_amp,
        "sg_earn": sg_earn,
        "sg_earn_txt": sg_earn_txt,
        "days_earn": days_earn,
        "has_upcoming_earnings": has_upcoming_earnings,
        "has_past_earnings": has_past_earnings,
        "fund_cov": fund_cov,
        "fund_fields_loaded": fund_fields_loaded,
        "fund_data_warning": fund_data_warning,
        "red_flag_items": red_flag_items,
        "red_flags_df": red_flags_df,
        "red_flag_notes": red_flag_notes,
        "red_flag_penalty_total": red_flag_penalty_total,
        "top_red_flag": top_red_flag,
        "quality_score": quality_score,
        "growth_score": growth_score,
        "growth_quality": growth_quality,
        "valuation_score": valuation_score,
        "balance_score": balance_score,
        "sentiment_score": sentiment_score,
        "risk_score": risk_score,
        "company": company,
        "setup": setup,
        "setup_adj": setup_adj,
        "investment": investment,
        "tb_score": tb_score,
        "tb_score_100": tb_score_100,
        "tb_timing_text": tb_timing_text,
        "position_action": position_action,
        "exit_score": exit_score,
        "exit_score_text": exit_score_text,
        "trend_break_score": trend_break_score,
        "momentum_collapse_score": momentum_collapse_score,
        "relative_weakness_score": relative_weakness_score,
        "distribution_score": distribution_score,
        "exit_trigger_score": exit_trigger_score,
        "exit_action": exit_action,
        "exit_reason_top": exit_reason_top,
        "exit_reason_list": exit_reason_list,
        "position_pnl_pct": position_pnl_pct,
        "pnl_bucket": pnl_bucket,
        "horizon_label": horizon_label,
        "add_on_action": add_on_action,
        "partial_profit_action": partial_profit_action,
        "stop_action": stop_action,
        "risk_note": risk_note,
        "trigger_status": trigger_status,
        "watchlist_priority": watchlist_priority,
        "watchlist_priority_score": watchlist_priority_score,
        "sector_strength_score": sector_strength_score,
        "industry_strength_score": industry_strength_score,
        "rs_benchmark_score": rs_benchmark_score,
        "rs_acceleration_score": rs_acceleration_score,
        "leadership_score": leadership_score,
        "leadership_status": leadership_status,
        "sector_label": sector_label,
        "industry_label": industry_label,
        "sector_trend_text": sector_trend_text,
        "industry_trend_text": industry_trend_text,
        "trend_quality_score": trend_quality_score,
        "ma20_slope": ma20_slope,
        "ma50_slope": ma50_slope,
        "ma200_slope": ma200_slope,
        "higher_lows_score": higher_lows_score,
        "base_quality_score": base_quality_score,
        "base_length_days": base_length_days,
        "correction_depth_pct": correction_depth_pct,
        "range_tightness_score": range_tightness_score,
        "volatility_contraction_score": volatility_contraction_score,
        "pullback_quality_score": pullback_quality_score,
        "volume_quality_proxy": volume_quality_proxy,
        "setup_type_quality_score": setup_type_quality_score,
        "setup_priority_score": setup_priority_score,
        "sector_strength_available": sector_strength_available,
        "volume_quality_score": volume_quality_score,
        "accumulation_score": accumulation_score,
        "distribution_pressure_score": distribution_pressure_score,
        "pullback_dryup_score": pullback_dryup_score,
        "breakout_volume_score": breakout_volume_score,
        "up_down_volume_ratio": up_down_volume_ratio,
        "volume_trend_score": volume_trend_score,
        "accumulation_day_count": accumulation_day_count,
        "distribution_day_count": distribution_day_count,
        "recent_pullback_volume_ratio": recent_pullback_volume_ratio,
        "breakout_day_volume_ratio": breakout_day_volume_ratio,
        "catalyst_score": catalyst_score,
        "earnings_event_score": earnings_event_score,
        "post_earnings_reaction_score": post_earnings_reaction_score,
        "revision_momentum_score": revision_momentum_score,
        "event_risk_score": event_risk_score,
        "catalyst_text": catalyst_text,
        "post_earnings_text": post_earnings_text,
        "event_phase_label": event_phase_label,
        "earnings_reaction_5d": earnings_reaction_5d,
        "earnings_reaction_10d": earnings_reaction_10d,
        "cashflow_stability_score": cashflow_stability_score,
        "margin_stability_score": margin_stability_score,
        "institutional_quality_score": institutional_quality_score,
        "institutional_quality_text": institutional_quality_text,
        "sector_etf_symbol": sector_etf_symbol if sector_etf_symbol else "-",
        "next_trigger": next_trigger,
        "trigger_reason": trigger_reason,
        "tb_signal": tb_signal,
        "tb_empf": tb_empf,
        "tb_df": tb_df,
        "tb_details": tb_details,
        "tb_context": tb_context,
        "stb_score": stb_score,
        "stb_signal": stb_signal,
        "stb_empf": stb_empf,
        "stb_text": stb_text,
        "kb": kb,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "decision_summary": decision_summary,
        "mode_label": "Position" if position_mode else "Watchlist",
        "stock_style": stock_style,
        "market_bucket": infer_market_bucket(ticker, info),
        "hmap": hmap,
        "atr_stop": atr_stop,
        "stop_used": stop_used,
        "stop_dist": stop_dist,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp1_source": tp1_source,
        "tp2_source": tp2_source,
        "tp3_source": tp3_source,
        "technical_target_1": technical_target_1,
        "technical_target_2": technical_target_2,
        "stop_source": stop_source,
        "suggested_entry_zone": suggested_entry_zone,
        "entry_source": entry_source,
        "entry_quality": entry_quality,
        "tradeability_score": tradeability_score,
        "tradeability_text": tradeability_text,
        "setup_type": setup_type,
        "preferred_entry": preferred_entry,
        "investment_case_score": investment_case_score,
        "investment_case_text": investment_case_text,
        "trading_case_score": trading_case_score,
        "trading_case_text": trading_case_text,
        "setup_confidence": setup_confidence,
        "setup_confidence_text": setup_confidence_text,
        "trade_crv_score": crv_score,
        "trade_stop_score": stop_score,
        "trade_entry_score": entry_score,
        "trade_timing_score": timing_trade_score,
        "trade_market_score": market_trade_score,
        "crv": crv,
        "pos_size": pos_size,
        "risk_eur": risk_eur,
        "risk_pct": risk_pct,
        "time_stop": time_stop,
        "valid_trade_setup": valid_trade_setup,
        "short_term_score": short_term_score,
        "s3": s3,
        "s3a": s3a,
        "s3t": s3t,
        "s4": s4,
        "s4a": s4a,
        "s4t": s4t,
        "s5": s5,
        "s5a": s5a,
        "s5t": s5t,
        "s6": s6,
        "s6a": s6a,
        "s6t": s6t,
        "w52": w52,
        "dist52": dist52,
        "rs_score": rs_score,
        "rs_composite": rs_composite,
        "ret21": ret21,
        "ret63": ret63,
        "ret126": ret126,
        "bench_ret21": bench_ret21,
        "bench_ret63": bench_ret63,
        "bench_ret126": bench_ret126,
        "rs_vs_benchmark_21": rs_vs_benchmark_21,
        "rs_vs_benchmark_63": rs_vs_benchmark_63,
        "rs_vs_benchmark_126": rs_vs_benchmark_126,
        "ma20": ma20,
        "ma50": ma50,
        "ma150": ma150,
        "ma200": ma200,
        "rsi": rsi,
        "macd_v": macd_v,
        "signal_v": signal_v,
        "macd_hist_current": macd_hist_current,
        "adx": adx,
        "atr": atr,
        "atr_pct": atr_pct,
        "stoch_k_v": stoch_k_v,
        "stoch_d_v": stoch_d_v,
        "willr_v": willr_v,
        "roc20": roc20,
        "roc60": roc60,
        "high52": high52,
        "low52": low52,
        "profit_margin": profit_margin,
        "oper_margin": oper_margin,
        "gross_margin": gross_margin,
        "roe": roe,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "debt_to_equity": debt_to_equity,
        "pe": pe,
        "peg": peg,
        "ps": ps,
        "pb": pb,
        "rec_label": rec_label,
        "analysts": analysts,
        "rec_mean": rec_mean,
        "beta": beta,
        "short_pct": short_pct,
        "market_cap": market_cap,
        "short_thesis": short_thesis,
        "emp": emp,
        "conv": conv,
    }