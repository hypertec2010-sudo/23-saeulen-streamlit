
# -*- coding: utf-8 -*-
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
from analysis_core import analyze_stock
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

APP_VERSION = "v13.1A.1"

st.set_page_config(
    page_title=f"Capital-Hill-Score-Modell {APP_VERSION}",
    page_icon="📊",
    layout="wide"
)


if not check_password():
    st.stop()

# ---------- Session State / App Defaults ----------
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

if "radar_universe" not in st.session_state:
    st.session_state.radar_universe = "US Basisliste"

if "radar_max_candidates" not in st.session_state:
    st.session_state.radar_max_candidates = 15

if "radar_custom_input" not in st.session_state:
    st.session_state.radar_custom_input = ""

if "radar_requested" not in st.session_state:
    st.session_state.radar_requested = False

if "radar_screening_style" not in st.session_state:
    st.session_state.radar_screening_style = "Leader"


def trigger_ui_refresh(**state_updates):
    for key, value in state_updates.items():
        st.session_state[key] = value
    st.session_state.ui_refresh_nonce += 1
    st.rerun()




# ---------- UI Theme / CSS ----------
st.markdown("""
<style>

/* Bessere Lesbarkeit für Formulare, Labels und Radio-Gruppen */
label, .stRadio label, .stSelectbox label, .stTextInput label, .stTextArea label,
[data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] p,
[data-baseweb="radio"] label, [role="radiogroup"] label,
div[role="radiogroup"] label, div[role="radiogroup"] p,
div[data-testid="stRadio"] label, div[data-testid="stRadio"] p {
    color:#f1f5f9 !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] > div {
    color:#f8fafc !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] label > div,
div[data-testid="stRadio"] [data-baseweb="radio"] label span,
div[data-testid="stRadio"] [data-baseweb="radio"] label p {
    color:#f8fafc !important;
    font-weight:700 !important;
}
div[data-testid="stRadio"] [aria-checked="true"] {
    color:#ffffff !important;
}
.small-note, .mobile-note, .compact-help, .muted-meta, .panel-caption, .workspace-sub {
    color:#dbeafe !important;
}
.stCaptionContainer, [data-testid="stCaptionContainer"] {
    color:#d1d5db !important;
}


:root{
    color-scheme: dark;
}
html, body, [class*="css"]{
    background-color:#0b1220;
    color:#f8fafc;
}
body{
    background:linear-gradient(180deg,#0b1220 0%, #0f172a 35%, #111827 100%);
}
[data-testid="stAppViewContainer"]{
    background:linear-gradient(180deg,#0b1220 0%, #0f172a 35%, #111827 100%) !important;
    color:#f8fafc !important;
}
[data-testid="stHeader"]{
    background:rgba(11,18,32,0.88) !important;
}
[data-testid="stMainBlockContainer"]{
    color:#f8fafc !important;
}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b1220 0%, #0f172a 100%) !important;
}
[data-testid="stMarkdownContainer"], [data-testid="stText"], p, label, span, div{
    color:inherit;
}

.metric-card{
    background:linear-gradient(180deg,#111827 0%, #0f172a 100%);
    border-radius:16px;
    padding:16px 20px;
    margin:8px 0;
    border:1px solid #243042;
    border-left:4px solid #4CAF50;
    box-shadow:0 10px 24px rgba(0,0,0,0.18);
}
.metric-card.red{border-left-color:#f44336;}
.metric-card.yellow{border-left-color:#FFC107;}
.small-note{color:#9aa4b2;font-size:0.88rem;}
pre{white-space:pre-wrap !important;}
.score-card{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:16px 16px 14px 16px;
    min-height:124px;
    box-shadow:0 10px 24px rgba(0,0,0,0.18);
}
.score-label{
    color:#94a3b8;
    font-size:0.80rem;
    text-transform:uppercase;
    letter-spacing:0.03em;
    line-height:1.2;
    margin-bottom:10px;
}
.score-value{
    color:#f8fafc;
    font-size:1.55rem;
    font-weight:800;
    line-height:1.15;
}
.score-delta{
    color:#cbd5e1;
    font-size:0.88rem;
    line-height:1.2;
    margin-top:10px;
}
.score-card.company{border-left:4px solid #3b82f6;}
.score-card.setup{border-left:4px solid #22c55e;}
.score-card.short{border-left:4px solid #f59e0b;}
.score-card.helper{border-left:4px solid #a78bfa;}
.score-card.investment{border-left:4px solid #14b8a6; box-shadow:0 12px 28px rgba(20,184,166,0.24); background:linear-gradient(180deg,#0b1f20 0%, #111827 100%);}
.score-card.board{border-left:4px solid #ef4444; box-shadow:0 12px 28px rgba(239,68,68,0.24); background:linear-gradient(180deg,#221113 0%, #111827 100%);}
.score-card.kb{border-left:4px solid #14b8a6; box-shadow:0 12px 28px rgba(20,184,166,0.22);}
.section-card{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:16px 18px;
    box-shadow:0 10px 24px rgba(0,0,0,0.18);
    margin:10px 0;
}
.premium-card{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:16px 18px;
    box-shadow:0 10px 24px rgba(0,0,0,0.18);
}
.premium-title{
    color:#94a3b8;
    font-size:0.80rem;
    text-transform:uppercase;
    letter-spacing:0.03em;
    margin-bottom:8px;
}
.premium-value{
    color:#f8fafc;
    font-size:1.15rem;
    font-weight:700;
    line-height:1.25;
}
.premium-sub{
    color:#cbd5e1;
    font-size:0.88rem;
    margin-top:8px;
    line-height:1.2;
}
.wrap-metric-card{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:14px 16px;
    min-height:112px;
    box-shadow:0 10px 24px rgba(0,0,0,0.18);
}
.wrap-metric-label{
    color:#94a3b8;
    font-size:0.80rem;
    text-transform:uppercase;
    letter-spacing:0.03em;
    line-height:1.2;
    margin-bottom:10px;
}
.wrap-metric-value{
    color:#f8fafc;
    font-size:clamp(1.0rem, 1.2vw, 1.18rem);
    font-weight:800;
    line-height:1.2;
    white-space:normal;
    word-break:break-word;
    overflow-wrap:anywhere;
}
.wrap-metric-sub{
    color:#cbd5e1;
    font-size:0.90rem;
    line-height:1.3;
    margin-top:8px;
    white-space:normal;
    word-break:break-word;
    overflow-wrap:anywhere;
}
.hero-shell{
    background:linear-gradient(135deg,#0f172a 0%, #111827 55%, #0b1220 100%);
    border:1px solid #243042;
    border-radius:24px;
    padding:18px 20px;
    box-shadow:0 18px 40px rgba(0,0,0,0.22);
    margin:14px 0 18px 0;
}
.hero-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;}
.hero-kicker{font-size:0.82rem;color:#93c5fd;font-weight:800;letter-spacing:0.04em;text-transform:uppercase;}
.hero-title{font-size:1.7rem;font-weight:900;color:#f8fafc;line-height:1.1;margin-top:6px;}
.hero-sub{font-size:0.96rem;color:#cbd5e1;margin-top:8px;line-height:1.45;max-width:920px;}
.hero-chip-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;}
.hero-chip{
    padding:7px 11px;border-radius:999px;background:#0b1220;border:1px solid #334155;
    color:#e5e7eb;font-size:0.85rem;font-weight:700;
}
.hero-score-pill{
    min-width:150px;padding:10px 12px;border-radius:16px;background:#0b1220;border:1px solid #334155;
    color:#f8fafc;text-align:center;
}
.hero-score-pill .pill-label{font-size:0.76rem;color:#9fb1c8;font-weight:800;text-transform:uppercase;}
.hero-score-pill .pill-value{font-size:clamp(1.0rem, 1.25vw, 1.2rem);font-weight:900;margin-top:4px;white-space:normal;word-break:break-word;overflow-wrap:anywhere;}
.exec-shell{
    background:linear-gradient(135deg,#0b1220 0%, #111827 55%, #172033 100%);
    border:1px solid #2c3a50;
    border-radius:26px;
    padding:18px 20px;
    box-shadow:0 18px 42px rgba(0,0,0,0.24);
    margin:14px 0 18px 0;
}
.exec-top{
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:18px;
    flex-wrap:wrap;
}
.exec-kicker{
    font-size:0.78rem;
    color:#93c5fd;
    font-weight:800;
    letter-spacing:0.06em;
    text-transform:uppercase;
}
.exec-title{
    font-size:1.85rem;
    color:#f8fafc;
    font-weight:900;
    line-height:1.08;
    margin-top:6px;
}
.exec-sub{
    color:#cbd5e1;
    font-size:0.95rem;
    line-height:1.45;
    margin-top:8px;
    max-width:920px;
}
.exec-meta{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    margin-top:14px;
}
.status-chip{
    display:inline-flex;
    align-items:center;
    gap:6px;
    padding:7px 11px;
    border-radius:999px;
    border:1px solid #334155;
    background:#0b1220;
    color:#e5e7eb;
    font-size:0.84rem;
    font-weight:800;
    line-height:1.2;
}
.status-chip.green{background:rgba(34,197,94,0.16); color:#dcfce7; border-color:rgba(34,197,94,0.28);}
.status-chip.amber{background:rgba(245,158,11,0.16); color:#fef3c7; border-color:rgba(245,158,11,0.26);}
.status-chip.red{background:rgba(239,68,68,0.16); color:#fee2e2; border-color:rgba(239,68,68,0.26);}
.status-chip.blue{background:rgba(59,130,246,0.16); color:#dbeafe; border-color:rgba(59,130,246,0.26);}
.status-chip.purple{background:rgba(139,92,246,0.16); color:#ede9fe; border-color:rgba(139,92,246,0.26);}
.exec-score-box{
    min-width:180px;
    background:linear-gradient(180deg,#0b1220 0%, #10192a 100%);
    border:1px solid #334155;
    border-radius:18px;
    padding:12px 14px;
    text-align:center;
}
.exec-score-label{
    color:#9fb1c8;
    font-size:0.77rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:0.04em;
}
.exec-score-value{
    color:#f8fafc;
    font-size:clamp(1.0rem, 1.3vw, 1.25rem);
    font-weight:900;
    margin-top:6px;
    line-height:1.2;
    white-space:normal;
    word-break:break-word;
    overflow-wrap:anywhere;
}
.exec-score-sub{
    color:#cbd5e1;
    font-size:0.84rem;
    margin-top:6px;
}
.section-head{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    flex-wrap:wrap;
    margin:6px 0 10px 0;
}
.section-title{
    color:#f8fafc;
    font-size:1.02rem;
    font-weight:900;
    letter-spacing:0.01em;
}
.section-meta-line{
    color:#94a3b8;
    font-size:0.86rem;
    line-height:1.35;
}
.action-row-note{
    color:#94a3b8;
    font-size:0.84rem;
    margin-top:4px;
}

.empty-state{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px dashed #334155;
    border-radius:18px;
    padding:16px 18px;
    margin:10px 0 12px 0;
    box-shadow:0 10px 24px rgba(0,0,0,0.12);
}
.empty-state-title{
    color:#f8fafc;
    font-size:0.96rem;
    font-weight:800;
    margin-bottom:6px;
}
.empty-state-text{
    color:#cbd5e1;
    font-size:0.90rem;
    line-height:1.45;
}
.muted-meta{
    color:#94a3b8;
    font-size:0.84rem;
    line-height:1.35;
}

.secondary-action-row{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    flex-wrap:wrap;
    margin-top:8px;
    margin-bottom:2px;
}
.secondary-action-note{
    color:#94a3b8;
    font-size:0.83rem;
    line-height:1.35;
}
.export-btn-wrap{
    display:block;
    width:100%;
}
.export-btn-wrap div[data-testid="stButton"],
.export-btn-wrap div[data-testid="stDownloadButton"]{
    width:100% !important;
}
.export-btn-wrap div[data-testid="stButton"] > button,
.export-btn-wrap div[data-testid="stDownloadButton"] > button,
.export-btn-wrap button{
    width:100% !important;
    min-width:100% !important;
    max-width:100% !important;
    min-height:2.95rem !important;
    height:2.95rem !important;
    border-radius:14px !important;
    border:1px solid #3b82f6 !important;
    background:linear-gradient(180deg,#1e3a8a 0%, #111827 100%) !important;
    color:#f8fafc !important;
    font-weight:800 !important;
    box-shadow:0 12px 24px rgba(30,58,138,0.24) !important;
    padding:0.48rem 0.95rem !important;
    display:inline-flex !important;
    align-items:center !important;
    justify-content:center !important;
    line-height:1 !important;
}
.export-btn-wrap div[data-testid="stButton"] > button p,
.export-btn-wrap div[data-testid="stButton"] > button span,
.export-btn-wrap div[data-testid="stButton"] > button div,
.export-btn-wrap div[data-testid="stDownloadButton"] > button p,
.export-btn-wrap div[data-testid="stDownloadButton"] > button span,
.export-btn-wrap div[data-testid="stDownloadButton"] > button div,
.export-btn-wrap button p,
.export-btn-wrap button span,
.export-btn-wrap button div{
    color:#f8fafc !important;
    font-weight:800 !important;
    line-height:1 !important;
    margin:0 !important;
}
.export-btn-wrap div[data-testid="stButton"] > button:hover,
.export-btn-wrap div[data-testid="stDownloadButton"] > button:hover,
.export-btn-wrap button:hover{
    border-color:#60a5fa !important;
    box-shadow:0 16px 30px rgba(59,130,246,0.28) !important;
}
div[data-testid="stDownloadButton"]{
    width:100% !important;
}
div[data-testid="stDownloadButton"] > button,
div[data-testid="stDownloadButton"] button,
div[data-testid="stDownloadButton"] a{
    width:100% !important;
    min-width:100% !important;
    max-width:100% !important;
    min-height:2.95rem !important;
    height:2.95rem !important;
    border-radius:14px !important;
    border:1px solid #3b82f6 !important;
    background:linear-gradient(180deg,#1e3a8a 0%, #111827 100%) !important;
    color:#f8fafc !important;
    font-weight:800 !important;
    box-shadow:0 12px 24px rgba(30,58,138,0.24) !important;
    padding:0.48rem 0.95rem !important;
    display:inline-flex !important;
    align-items:center !important;
    justify-content:center !important;
    line-height:1 !important;
}
div[data-testid="stDownloadButton"] > button p,
div[data-testid="stDownloadButton"] > button span,
div[data-testid="stDownloadButton"] > button div,
div[data-testid="stDownloadButton"] button p,
div[data-testid="stDownloadButton"] button span,
div[data-testid="stDownloadButton"] button div,
div[data-testid="stDownloadButton"] a p,
div[data-testid="stDownloadButton"] a span,
div[data-testid="stDownloadButton"] a div{
    color:#f8fafc !important;
    font-weight:800 !important;
    line-height:1 !important;
    margin:0 !important;
}
.export-btn-wrap div[data-testid="stButton"] > button > div,
.export-btn-wrap div[data-testid="stButton"] > button > div > p,
.export-btn-wrap div[data-testid="stDownloadButton"] > button > div,
.export-btn-wrap div[data-testid="stDownloadButton"] > button > div > p,
.export-btn-wrap div[data-testid="stDownloadButton"] button > div,
.export-btn-wrap div[data-testid="stDownloadButton"] button > div > p{
    min-height:2.95rem !important;
    height:2.95rem !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    line-height:1 !important;
    padding:0 !important;
    margin:0 !important;
}
.soft-divider{
    height:1px;
    background:linear-gradient(90deg, rgba(51,65,85,0) 0%, rgba(51,65,85,0.8) 18%, rgba(51,65,85,0.8) 82%, rgba(51,65,85,0) 100%);
    margin:16px 0 14px 0;
}
.panel-caption{
    color:#cbd5e1;
    font-size:0.90rem;
    line-height:1.45;
}
.stTabs [data-baseweb="tab-list"]{
    padding:11px;
    gap:10px;
}
.stTabs [data-baseweb="tab"]{
    font-size:0.88rem;
    letter-spacing:0.01em;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,0.02);
}
.stTabs [aria-selected="true"]{
    box-shadow:0 12px 26px rgba(59,130,246,0.24);
}
.decision-card, .compact-panel, .bullet-card, .mobile-result-card{
    transition:transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.decision-card:hover, .compact-panel:hover, .bullet-card:hover, .mobile-result-card:hover{
    transform:translateY(-1px);
    box-shadow:0 18px 34px rgba(0,0,0,0.22);
}
.decision-card, .compact-panel, .mobile-result-card{
    min-height:auto;
}
.decision-card .dc-value, .compact-panel .cp-value, .mobile-result-card .mobile-result-value{
    max-width:100%;
}
div[data-testid="stButton"] > button{
    letter-spacing:0.01em;
}
div[data-testid="stButton"] > button[kind="secondary"]{
    border-color:#334155;
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
}
div[data-testid="stDownloadButton"] > button{
    border-color:#334155;
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
}
@media (max-width: 768px){
    .secondary-action-row{gap:8px !important; margin-top:6px !important;}
    .section-title{font-size:0.98rem !important;}
    .section-meta-line{font-size:0.82rem !important;}
    .decision-card, .compact-panel, .bullet-card, .mobile-result-card{
        margin-bottom:4px !important;
    }
}
.section-spacer{height:8px;}
.stTabs [data-baseweb="tab-list"]{
    box-shadow:0 10px 24px rgba(0,0,0,0.12);
}
.stTabs [data-baseweb="tab"]{
    transition:all 0.18s ease;
}
.stTabs [data-baseweb="tab"]:hover{
    border-color:#475569;
    transform:translateY(-1px);
}
div[data-testid="stButton"] > button[kind="secondary"]{
    opacity:0.96;
}
div[data-testid="stExpander"] details{
    border-radius:16px;
    border:1px solid #243042;
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
}
div[data-testid="stExpander"] summary{
    font-weight:800;
}
@media (max-width: 768px){
    .empty-state{padding:14px 15px !important;}
}
@media (max-width: 768px){
    .exec-title{font-size:1.45rem !important;}
    .exec-sub{font-size:0.90rem !important;}
    .exec-shell{padding:16px 16px !important;}
    .exec-score-box{min-width:150px !important; width:100%;}
}

.decision-card{
    background:linear-gradient(180deg,#111827 0%,#0b1220 100%);
    border:1px solid #243042;border-radius:22px;padding:16px 16px 14px 16px;min-height:150px;
    box-shadow:0 14px 30px rgba(0,0,0,0.20);
}
.decision-card .dc-label{font-size:0.82rem;color:#9fb1c8;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;}
.decision-card .dc-value{font-size:1.85rem;font-weight:900;color:#f8fafc;line-height:1.05;margin-top:8px;}
.decision-card .dc-sub{font-size:0.98rem;color:#e5e7eb;font-weight:700;margin-top:7px;}
.decision-card .dc-note{font-size:0.88rem;color:#cbd5e1;margin-top:8px;line-height:1.4;}
.decision-card.invest{border-left:5px solid #14b8a6;background:linear-gradient(180deg,#082f2d 0%,#111827 100%);box-shadow:0 16px 32px rgba(20,184,166,0.20);}
.decision-card.entry{border-left:5px solid #f59e0b;background:linear-gradient(180deg,#3a2410 0%,#111827 100%);box-shadow:0 16px 32px rgba(245,158,11,0.20);}
.decision-card.action{border-left:5px solid #8b5cf6;background:linear-gradient(180deg,#24143d 0%,#111827 100%);box-shadow:0 16px 32px rgba(139,92,246,0.20);}
.compact-panel{
    background:linear-gradient(180deg,#111827 0%, #0f172a 100%);
    border:1px solid #243042;border-radius:18px;padding:14px 16px;min-height:118px;
    box-shadow:0 12px 28px rgba(0,0,0,0.16);
}
.compact-panel .cp-label{font-size:0.82rem;color:#9fb1c8;font-weight:800;text-transform:uppercase;}
.compact-panel .cp-value{font-size:1.28rem;font-weight:900;color:#f8fafc;margin-top:6px;}
.compact-panel .cp-sub{font-size:0.9rem;color:#d1d5db;margin-top:6px;line-height:1.35;}
.bullet-card{
    background:linear-gradient(180deg,#111827 0%, #0b1220 100%);
    border:1px solid #243042;border-radius:18px;padding:14px 16px;min-height:165px;
}
.bullet-card h4{margin:0 0 10px 0;color:#f8fafc;font-size:0.98rem;}
.bullet-card ul{margin:0;padding-left:18px;}
.bullet-card li{color:#d1d5db;line-height:1.5;margin-bottom:4px;}
.mobile-form-card{
    background:linear-gradient(180deg,#111827 0%, #0b1220 100%);
    border:1px solid #243042;
    border-radius:22px;
    padding:16px 18px;
    margin:12px 0 18px 0;
    box-shadow:0 14px 30px rgba(0,0,0,0.18);
}
.mobile-form-title{font-size:1.15rem;font-weight:800;color:#f8fafc;margin-bottom:4px;}
.mobile-form-sub{font-size:0.92rem;color:#cbd5e1;margin-bottom:10px;line-height:1.45;}
.mobile-note{font-size:0.84rem;color:#94a3b8;margin-top:6px;}

.workspace-shell{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:24px;
    padding:16px 18px 18px 18px;
    box-shadow:0 16px 32px rgba(0,0,0,0.18);
    margin:12px 0 18px 0;
}
.workspace-title{
    font-size:1.22rem;
    font-weight:900;
    color:#f8fafc;
    margin-bottom:4px;
}
.workspace-sub{
    font-size:0.92rem;
    color:#cbd5e1;
    margin-bottom:14px;
    line-height:1.45;
}
.workspace-card{
    border-radius:20px;
    padding:16px 16px 14px 16px;
    border:1px solid #334155;
    min-height:150px;
    box-shadow:0 12px 24px rgba(0,0,0,0.16);
}
.workspace-card.analysis{
    background:linear-gradient(180deg,#10263f 0%, #111827 100%);
    border-left:5px solid #3b82f6;
}
.workspace-card.watchlist{
    background:linear-gradient(180deg,#24143d 0%, #111827 100%);
    border-left:5px solid #8b5cf6;
}
.workspace-card.position{
    background:linear-gradient(180deg,#3a2410 0%, #111827 100%);
    border-left:5px solid #f59e0b;
}
.workspace-card{
    cursor:pointer;
    transition:transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.workspace-card:hover{
    transform:translateY(-2px);
    box-shadow:0 18px 34px rgba(0,0,0,0.22);
}
.workspace-card.active{
    border-color:#e5e7eb;
    box-shadow:0 20px 36px rgba(255,255,255,0.08);
}
.workspace-card.analysis.active{box-shadow:0 20px 36px rgba(59,130,246,0.20);}
.workspace-card.watchlist.active{box-shadow:0 20px 36px rgba(139,92,246,0.20);}
.workspace-card.position.active{box-shadow:0 20px 36px rgba(245,158,11,0.20);}
.workspace-select-btn{margin-top:10px;}

div.stButton > button[kind="secondary"]{
    border-radius:18px;
}
.mode-button{
    width:100%;
}
.workspace-kicker{
    color:#cbd5e1;
    font-size:0.78rem;
    text-transform:uppercase;
    font-weight:800;
    letter-spacing:0.04em;
}
.workspace-name{
    color:#f8fafc;
    font-size:1.18rem;
    font-weight:900;
    margin-top:8px;
}
.workspace-desc{
    color:#d1d5db;
    font-size:0.92rem;
    margin-top:8px;
    line-height:1.42;
}
.section-accent{
    display:inline-block;
    padding:6px 10px;
    border-radius:999px;
    font-size:0.8rem;
    font-weight:800;
    margin-top:6px;
}
.section-accent.blue{background:rgba(59,130,246,0.16); color:#bfdbfe; border:1px solid rgba(59,130,246,0.24);}
.section-accent.purple{background:rgba(139,92,246,0.16); color:#ddd6fe; border:1px solid rgba(139,92,246,0.24);}
.section-accent.amber{background:rgba(245,158,11,0.16); color:#fde68a; border:1px solid rgba(245,158,11,0.24);}

.stTabs [data-baseweb="tab-list"]{
    gap:10px;
    flex-wrap:wrap;
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:10px;
    margin-bottom:10px;
}
.stTabs [data-baseweb="tab"]{
    height:auto;
    white-space:nowrap;
    border-radius:999px;
    padding:10px 14px;
    background:linear-gradient(180deg,#111827 0%, #0b1220 100%);
    border:1px solid #334155;
    color:#cbd5e1;
    font-weight:700;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(90deg,#2563eb 0%, #7c3aed 100%) !important;
    color:white !important;
    border-color:transparent !important;
    box-shadow:0 10px 24px rgba(59,130,246,0.25);
}

.mobile-result-card{
    background:linear-gradient(180deg,#111827 0%, #0b1220 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:14px 16px;
    box-shadow:0 12px 28px rgba(0,0,0,0.16);
    margin:8px 0;
}
.mobile-result-label{
    font-size:0.8rem;
    color:#9fb1c8;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:0.03em;
}
.mobile-result-value{
    font-size:1.3rem;
    color:#f8fafc;
    font-weight:900;
    margin-top:6px;
    line-height:1.15;
}
.mobile-result-sub{
    font-size:0.9rem;
    color:#d1d5db;
    margin-top:7px;
    line-height:1.35;
}
@media (max-width: 768px){
    .hero-title{font-size:1.35rem !important;}
    .hero-sub{font-size:0.90rem !important;}
    .hero-chip{font-size:0.80rem !important;padding:6px 10px !important;}
    .decision-card{min-height:auto !important;padding:14px 14px 12px 14px !important;}
    .decision-card .dc-value{font-size:1.55rem !important;}
    .bullet-card{min-height:auto !important;}
    .compact-panel{min-height:auto !important;}
    .reco-card{min-height:auto !important;}
    .stTabs [data-baseweb="tab-list"]{gap:8px !important;padding:8px !important;}
    .stTabs [data-baseweb="tab"]{padding:8px 11px !important;font-size:0.84rem !important;}
}
.model-pill{
    display:inline-block;
    margin-top:6px;
    padding:6px 10px;
    border-radius:999px;
    background:linear-gradient(90deg,#0f766e 0%, #1d4ed8 100%);
    color:white;
    font-size:0.84rem;
    font-weight:800;
    letter-spacing:0.02em;
}


.reco-card{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:14px 16px;
    min-height:132px;
    box-shadow:0 10px 24px rgba(0,0,0,0.20);
    display:flex;
    flex-direction:column;
    justify-content:space-between;
}
.reco-top{
    display:flex;
    align-items:flex-start;
    justify-content:space-between;
    gap:10px;
}
.reco-label{
    color:#94a3b8;
    font-size:0.79rem;
    line-height:1.2;
    margin-bottom:8px;
    letter-spacing:0.02em;
    text-transform:uppercase;
}
.reco-icon{
    font-size:1.1rem;
    line-height:1;
    opacity:0.95;
}
.reco-value{
    color:#f8fafc;
    font-size:1.05rem;
    font-weight:700;
    line-height:1.28;
    white-space:normal;
    word-break:break-word;
}
.reco-delta{
    color:#cbd5e1;
    font-size:0.85rem;
    margin-top:10px;
    line-height:1.2;
}
.reco-chip{
    display:inline-block;
    margin-top:10px;
    padding:4px 10px;
    border-radius:999px;
    background:#1e293b;
    color:#e2e8f0;
    font-size:0.78rem;
    font-weight:600;
    width:fit-content;
}
.reco-card.context{border-left:4px solid #60a5fa;}
.reco-card.main{border-left:4px solid #22c55e;}
.reco-card.conviction{border-left:4px solid #a78bfa;}
.reco-card.signal{border-left:4px solid #f59e0b;}
.reco-card.market{border-left:4px solid #14b8a6;}
.score-pill{
    border-radius:999px;
    padding:2px 10px;
    font-size:0.82rem;
    font-weight:600;
    display:inline-block;
}

.horizon-card{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:14px 14px 12px 14px;
    min-height:142px;
    box-shadow:0 10px 24px rgba(0,0,0,0.18);
}
.horizon-card.green{border-left:4px solid #22c55e; box-shadow:0 12px 26px rgba(34,197,94,0.16);}
.horizon-card.amber{border-left:4px solid #f59e0b; box-shadow:0 12px 26px rgba(245,158,11,0.14);}
.horizon-card.red{border-left:4px solid #ef4444; box-shadow:0 12px 26px rgba(239,68,68,0.14);}
.horizon-card.blue{border-left:4px solid #3b82f6; box-shadow:0 12px 26px rgba(59,130,246,0.14);}
.horizon-top{
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:10px;
}
.horizon-label{
    color:#94a3b8;
    font-size:0.78rem;
    text-transform:uppercase;
    letter-spacing:0.03em;
    line-height:1.2;
}
.horizon-icon{
    font-size:1.1rem;
    line-height:1;
}
.horizon-value{
    color:#f8fafc;
    font-size:clamp(1.05rem, 1.25vw, 1.2rem);
    font-weight:900;
    line-height:1.15;
    margin-top:12px;
}
.horizon-score{
    color:#e2e8f0;
    font-size:0.98rem;
    font-weight:800;
    margin-top:6px;
}
.horizon-sub{
    color:#cbd5e1;
    font-size:0.84rem;
    line-height:1.35;
    margin-top:8px;
}
@media (max-width: 768px){
    .horizon-card{min-height:128px !important; padding:12px 12px 10px 12px !important;}
}


/* Finale Angleichung Export-Zeile: Sheets an native CSV-Höhe annähern */
.export-btn-wrap div[data-testid="stButton"] > button{
    min-height:2.30rem !important;
    height:2.30rem !important;
    padding:0.20rem 0.90rem !important;
    line-height:1 !important;
}
.export-btn-wrap div[data-testid="stButton"] > button > div,
.export-btn-wrap div[data-testid="stButton"] > button > div > p,
.export-btn-wrap div[data-testid="stButton"] > button p,
.export-btn-wrap div[data-testid="stButton"] > button span,
.export-btn-wrap div[data-testid="stButton"] > button div{
    min-height:2.30rem !important;
    height:2.30rem !important;
    line-height:1 !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    padding:0 !important;
    margin:0 !important;
}


.export-action-btn{
    display:flex;
    align-items:center;
    justify-content:center;
    width:100%;
    min-height:2.85rem;
    height:2.85rem;
    border-radius:14px;
    border:1px solid #3b82f6;
    background:linear-gradient(180deg,#1e3a8a 0%, #111827 100%);
    color:#f8fafc !important;
    font-weight:800;
    text-decoration:none !important;
    box-shadow:0 12px 24px rgba(30,58,138,0.24);
    padding:0.40rem 0.95rem;
    line-height:1;
    text-align:center;
}
.export-action-btn:hover{
    border-color:#60a5fa;
    box-shadow:0 16px 30px rgba(59,130,246,0.28);
    color:#ffffff !important;
    text-decoration:none !important;
}
.export-action-btn span{
    color:#f8fafc !important;
    font-weight:800;
    line-height:1;
}

</style>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
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


# ---------- Ranking / Table Helpers ----------
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
        if val in {"aktiv", "nahe dran", "jetzt prüfbar", "fast prüfbar"}:
            return "background-color: rgba(34,197,94,0.28); color: white; font-weight:700;"
        if val in {"frühe beobachtung", "beobachten", "warten", "früh interessant", "weiter beobachten", "noch warten"}:
            return "background-color: rgba(245,158,11,0.28); color: white; font-weight:700;"
        if val in {"passiv", "aktuell kein fokus"}:
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
        "BUY CANDIDATE / TIMING PRÜFEN": "Kaufkandidat / Timing prüfen",
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


def display_trigger_status_label(status):
    mapping = {
        "Aktiv": "Jetzt prüfbar",
        "Nahe dran": "Fast prüfbar",
        "Frühe Beobachtung": "Früh interessant",
        "Beobachten": "Weiter beobachten",
        "Passiv": "Aktuell kein Fokus",
        "Warten": "Noch warten",
        "Nicht anwendbar": "Nicht anwendbar",
        "-": "-",
    }
    return mapping.get(str(status or ""), str(status or "-"))


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


@st.cache_data(show_spinner=False, ttl=60 * 60)
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


# ---------- Indicators ----------
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


# ---------- Domain Helpers ----------
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


# ---------- Data Enrichment ----------
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


# ---------- IO ----------
@st.cache_data(ttl=120, show_spinner=False)
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


@st.cache_data(ttl=120, show_spinner=False)
def load_benchmark_data(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y", auto_adjust=True)
        return hist
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
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
            "Trigger-Status": display_trigger_status_label(r.get("trigger_status", "-")),
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


# ---------- Export / Logging Helpers ----------
# ausgelagert nach logging_utils.py

# ---------- Core Analysis Engine ----------
def _legacy_analyze_stock(
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
        if (
            investment >= 78 and kb >= 3 and market_info["regime"] == "POSITIV"
            and valid_trade_setup and entry_quality == "gut"
            and trading_case_score >= 68 and setup_confidence >= 60
        ):
            emp, conv = ("BUY / ACCUMULATE", "HIGH")
        elif investment >= 78 and kb >= 3 and market_info["regime"] == "POSITIV":
            emp, conv = ("BUY CANDIDATE / TIMING PRÜFEN", "HIGH")
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
            if investment >= 78 and kb >= 3 and market_info["regime"] == "POSITIV":
                emp = "BUY / ACCUMULATE"
                conv = "HIGH"
            else:
                emp = "WATCH / EINSTIEG PRÜFEN"
                conv = "MEDIUM" if conv == "-" else conv
        elif trigger_status in {"Nahe dran", "Frühe Beobachtung"}:
            if emp == "BUY / ACCUMULATE":
                emp = "BUY CANDIDATE / TIMING PRÜFEN"
            elif emp not in {"BUY CANDIDATE / TIMING PRÜFEN", "WATCH / EINSTIEG PRÜFEN"}:
                emp = "WATCH / EINSTIEG PRÜFEN"
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


# ---------- Main App Flow ----------
logo_path = Path("a_logo_for_the_capital_hill_score_model_is_promi.png")

top1, top2 = st.columns([0.72, 2.28])
with top1:
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
        st.markdown(
            f"""
            <div style="text-align:left; margin-top:6px; margin-bottom:6px;">
                <span class="model-pill">Release {APP_VERSION} · Premium Dashboard</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title("📊 Capital Hill Score Modell")
        st.markdown(
            f"""
            <div style="text-align:left; margin-top:6px; margin-bottom:6px;">
                <span class="model-pill">Release {APP_VERSION} · Premium Dashboard</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
with top2:
    st.markdown("")
st.caption("Investment- und Trading-Entscheidungen in einer hochwertigen, klar lesbaren Oberfläche. Multi-Screening, Setup-Logik, Trade-Plan, Positionsmanagement, Watchlisten und Telegram-Alerts.")


st.markdown(
    """
    <style>
    .landing-stage{
        position:relative;
        margin: 8px 0 22px 0;
        padding: 26px 26px 24px 26px;
        border-radius: 28px;
        background:
            radial-gradient(circle at top left, rgba(96,165,250,0.20), rgba(0,0,0,0) 32%),
            radial-gradient(circle at top right, rgba(14,165,233,0.12), rgba(0,0,0,0) 24%),
            linear-gradient(180deg, rgba(15,23,42,0.98), rgba(15,23,42,0.82));
        border: 1px solid rgba(96,165,250,0.22);
        box-shadow:
            0 24px 50px rgba(2,6,23,0.30),
            inset 0 1px 0 rgba(255,255,255,0.04);
        overflow:hidden;
    }
    .landing-stage::after{
        content:"";
        position:absolute;
        inset:0;
        background:linear-gradient(120deg, rgba(255,255,255,0.045), rgba(255,255,255,0.00) 42%);
        pointer-events:none;
    }
    .landing-topline{
        display:flex;
        align-items:center;
        gap:10px;
        flex-wrap:wrap;
        margin-bottom:12px;
        position:relative;
        z-index:1;
    }
    .landing-kicker{
        display:inline-block;
        padding:7px 13px;
        border-radius:999px;
        background:rgba(37,99,235,0.18);
        color:#dbeafe;
        border:1px solid rgba(96,165,250,0.36);
        font-size:0.78rem;
        font-weight:900;
        letter-spacing:0.04em;
    }
    .landing-statusdot{
        width:9px;
        height:9px;
        border-radius:999px;
        background:#22c55e;
        box-shadow:0 0 0 4px rgba(34,197,94,0.14);
    }
    .landing-title{
        position:relative;
        z-index:1;
        font-size:2.25rem;
        line-height:1.02;
        font-weight:950;
        color:#f8fafc;
        margin-bottom:10px;
        letter-spacing:-0.03em;
        max-width:900px;
    }
    .landing-sub{
        position:relative;
        z-index:1;
        color:#cbd5e1;
        font-size:1.02rem;
        line-height:1.62;
        max-width:860px;
        margin-bottom:18px;
    }
    .landing-feature-row{
        position:relative;
        z-index:1;
        display:flex;
        flex-wrap:wrap;
        gap:10px;
        margin-bottom:6px;
    }
    .landing-feature{
        display:inline-flex;
        align-items:center;
        gap:8px;
        padding:8px 12px;
        border-radius:999px;
        background:rgba(15,23,42,0.55);
        border:1px solid rgba(148,163,184,0.16);
        color:#dbeafe;
        font-size:0.82rem;
        font-weight:700;
    }
    .landing-cards-wrap{
        margin-top: -4px;
        margin-bottom: 10px;
        padding: 14px;
        border-radius: 30px;
        background:
            linear-gradient(180deg, rgba(15,23,42,0.55), rgba(15,23,42,0.26));
        border:1px solid rgba(96,165,250,0.14);
        box-shadow: 0 18px 40px rgba(2,6,23,0.18);
    }
    .landing-divider{
        position:relative;
        z-index:1;
        height:1px;
        background:linear-gradient(90deg, rgba(96,165,250,0.44), rgba(148,163,184,0.14), rgba(0,0,0,0));
        margin:18px 0 22px 0;
    }
    .landing-mini-note{
        color:#94a3b8;
        font-size:0.82rem;
        margin-top:10px;
        margin-bottom:0;
    }

    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
        white-space: pre-line !important;
        line-height: 1.28 !important;
        font-weight: 900 !important;
        border-radius: 28px !important;
        min-height: 8.2rem !important;
        padding-top: 1.25rem !important;
        padding-bottom: 1.15rem !important;
        background:
            radial-gradient(circle at top left, rgba(191,219,254,0.24), rgba(0,0,0,0) 34%),
            linear-gradient(180deg, #2563eb 0%, #111827 100%) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(191,219,254,0.42) !important;
        box-shadow:
            0 28px 54px rgba(15,23,42,0.34),
            0 10px 24px rgba(37,99,235,0.24),
            inset 0 1px 0 rgba(255,255,255,0.10) !important;
        text-shadow: none !important;
        transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease !important;
        position:relative !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button::before{
        content:"";
        position:absolute;
        inset:0;
        border-radius:26px;
        background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.00) 40%);
        pointer-events:none;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p,
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button span,
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button div {
        color: #f8fafc !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
        border-color: rgba(191,219,254,0.88) !important;
        box-shadow:
            0 30px 56px rgba(29,78,216,0.30),
            0 12px 26px rgba(15,23,42,0.26) !important;
        transform: translateY(-3px);
    }

    .utility-shell{
        border:1px solid rgba(148,163,184,0.12);
        border-radius:18px;
        background:linear-gradient(180deg, rgba(15,23,42,0.40), rgba(15,23,42,0.22));
        padding:12px 14px 14px 14px;
        margin-top:18px;
        box-shadow:0 10px 24px rgba(2,6,23,0.14);
    }
    .utility-title{
        font-size:0.92rem;
        font-weight:800;
        color:#dbeafe;
        margin-bottom:2px;
    }
    .utility-sub{
        font-size:0.81rem;
        color:#94a3b8;
        margin-bottom:6px;
    }

    div[data-testid="stExpander"] {
        border:1px solid rgba(148,163,184,0.12) !important;
        border-radius:18px !important;
        background:rgba(15,23,42,0.34) !important;
        overflow:hidden;
        box-shadow: 0 10px 24px rgba(2,6,23,0.10);
    }
    div[data-testid="stExpander"] summary {
        background:rgba(15,23,42,0.30) !important;
        border-radius:18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="landing-stage">
        <div class="landing-topline">
            <div class="landing-kicker">PREMIUM WORKSPACE</div>
            <div class="landing-statusdot"></div>
        </div>
        <div class="landing-title">Wähle deinen Einstieg.</div>
        <div class="landing-sub">
            Sofortanalyse, Watchlisten und Positionen stehen als Hauptaktionen im Fokus.
            Verwaltung und Hilfen bleiben bewusst darunter und optisch ruhiger.
        </div>
        <div class="landing-divider"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("""<div class="landing-cards-wrap">""", unsafe_allow_html=True)
wc1, wc2, wc3, wc4 = st.columns(4)
with wc1:
    if st.button(
        "🔎 Sofortanalyse\nEinzelaktie oder Vergleich",
        use_container_width=True,
        key="workspace_analysis_btn"
    ):
        st.session_state.workspace_mode = "Sofortanalyse"
with wc2:
    if st.button(
        "📋 Watchlisten\nListen pflegen und prüfen",
        use_container_width=True,
        key="workspace_watchlist_btn"
    ):
        st.session_state.workspace_mode = "Watchlisten"
with wc3:
    if st.button(
        "🛡️ Positionen\nPositionen überwachen",
        use_container_width=True,
        key="workspace_position_btn"
    ):
        st.session_state.workspace_mode = "Positionen"
with wc4:
    if st.button(
        "🎯 Kandidaten-Radar\nInteressante Werte vorsortieren",
        use_container_width=True,
        key="workspace_candidate_radar_btn"
    ):
        st.session_state.workspace_mode = "Kandidaten-Radar"

st.markdown("""</div>""", unsafe_allow_html=True)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

st.markdown("""<div class="landing-mini-note">Tipp: Die vier Hauptkarten sind für den täglichen Schnellzugriff optimiert.</div>""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="utility-shell">
        <div class="utility-title">Hilfen & Verwaltung</div>
        <div class="utility-sub">Kurzanleitung, Auto-Run und Technik bewusst zurückgenommen unter den Hauptaktionen.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.expander("Hilfen & Verwaltung", expanded=False):
    st.caption("Kurzanleitung, Auto-Run Control Center und Technik / Admin sind hier gebündelt.")

    st.markdown("#### Kurzanleitung")
    st.markdown(
        "- **Sofortanalyse**: spontane Einzelanalyse oder Vergleich mehrerer Aktien.\n"
        "- **Watchlisten**: neue Chancen beobachten, priorisieren und bei Bedarf an Telegram senden.\n"
        "- **Positionen**: bestehende Werte mit Fokus auf Risiko, Stop und Teilgewinn überwachen.\n"
        "- **Alert-Modus**: Konservativ = selektiver, Standard = Mittelweg, Früh = mehr Hinweise.\n"
        "- **Prüf-Frequenz**: steuert, in welchen Auto-Run-Slots eine Liste berücksichtigt wird."
    )

    with st.expander("Ausführliche Hilfe", expanded=False):
        st.markdown("### Unterschied: Watchlist vs. Positions-Watchlist")
        st.markdown(
            "- **Watchlist** = Werte, die du noch nicht hältst oder neu beobachten willst. Fokus: Einstieg, Trigger, Priorität.\n"
            "- **Positions-Watchlist** = Werte, die du bereits besitzt. Fokus: Risiko reduzieren, Stop prüfen, Teilgewinn, Maßnahmen."
        )
        st.markdown("### Telegram")
        st.markdown(
            "- **Watchlist analysieren + Telegram** prüft die aktuelle Watchlist sofort.\n"
            "- Telegram sendet nicht jede Analyse erneut, sondern nutzt Alert-History und unterdrückt doppelte Meldungen.\n"
            "- Neue Watchlist-Werte können zusätzlich eine **Erst-Check-Info** auslösen."
        )
        st.markdown("### Auto-Run")
        st.markdown(
            "- Die gespeicherte Prüf-Frequenz bestimmt, in welchen Slots eine Watchlist automatisch fällig ist.\n"
            "- Der Auto-Run-Bereich liegt bewusst im Technik-/Admin-Teil, damit die Hauptoberfläche ruhiger bleibt."
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.markdown("#### Auto-Run Control Center")
    berlin_now = get_current_berlin_time()
    current_slot_label = get_current_schedule_slot(berlin_now)
    slot_options = ["10:30", "15:40", "18:30", "22:10"]

    st.caption(f"Aktuelle Berlin-Zeit: {berlin_now.strftime('%d.%m.%Y %H:%M')} · Aktueller Slot: {current_slot_label if current_slot_label else 'noch keiner'}")
    st.caption("Zeigt, welche Watchlisten im aktuellen Slot automatisch fällig wären.")

    due_df, due_err = get_due_watchlists_for_slot(current_slot_label) if current_slot_label else (pd.DataFrame(), None)
    if due_err:
        st.warning(f"Fällige Watchlisten konnten nicht geladen werden: {due_err}")
    elif current_slot_label and due_df is not None and not due_df.empty:
        st.dataframe(
            due_df[["Watchlist_Name", "Watchlist_Type", "Alert_Mode", "Check_Frequency"]],
            hide_index=True,
            use_container_width=True,
            height=min(260, 45 * len(due_df) + 40),
        )
    elif current_slot_label:
        st.info("Für den aktuellen Slot sind gerade keine Watchlisten fällig.")
    else:
        st.info("Vor dem ersten Slot des Tages ist noch keine Watchlist automatisch fällig.")

    ar1, ar2 = st.columns([1.2, 1.0])
    with ar1:
        selected_auto_slot = st.selectbox(
            "Auto-Run-Testslot",
            options=slot_options,
            index=slot_options.index(current_slot_label) if current_slot_label in slot_options else 0,
            key="selected_auto_run_slot_widget"
        )
    with ar2:
        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        if st.button("Auto-Run-Test für Slot starten", use_container_width=True, key="run_auto_slot_test_btn"):
            st.session_state.auto_run_requested = True
            st.session_state.auto_run_slot_label = selected_auto_slot
            st.rerun()

    st.markdown("#### Technik / Admin")
    test1, test2 = st.columns([1.2, 2.8])
    with test1:
        if st.button("Telegram-Test senden", use_container_width=True, key="telegram_test_button"):
            test_message = (
                f"Capital Hill Test\n"
                f"Version: {APP_VERSION}\n"
                f"Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Status: Telegram-Versand funktioniert."
            )
            ok, msg = send_telegram_message(test_message)
            if ok:
                st.success("Telegram-Testnachricht wurde gesendet.")
            else:
                st.error(f"Telegram-Test fehlgeschlagen: {msg}")
    with test2:
        st.caption("Technischer Test unabhängig von Marktlogik, Alert-History und Watchlist-Regeln.")

st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
        white-space: pre-line !important;
        line-height: 1.35 !important;
        font-weight: 800 !important;
        border-radius: 20px !important;
        min-height: 5.25rem !important;
        background: linear-gradient(180deg, #1e3a8a 0%, #111827 100%) !important;
        color: #f8fafc !important;
        border: 1px solid #3b82f6 !important;
        box-shadow: 0 14px 28px rgba(30,58,138,0.25) !important;
        text-shadow: none !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p,
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button span,
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button div {
        color: #f8fafc !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
        border-color: #60a5fa !important;
        box-shadow: 0 18px 34px rgba(59,130,246,0.28) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stDownloadButton"] > button {
        color: #f8fafc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    :root{
        --bg-main:#0B1220;
        --bg-panel:#111827;
        --bg-panel-2:#0F172A;
        --bg-elev:#162033;
        --border-soft:rgba(148,163,184,0.18);
        --border-strong:rgba(96,165,250,0.32);
        --text-main:#F8FAFC;
        --text-sub:#CBD5E1;
        --text-muted:#94A3B8;
        --blue-1:#1D4ED8;
        --blue-2:#2563EB;
        --blue-3:#3B82F6;
        --green-1:#052E16;
        --green-2:#16A34A;
        --green-3:#22C55E;
        --red-1:#450A0A;
        --red-2:#DC2626;
        --red-3:#EF4444;
        --amber-1:#3F2F0A;
        --amber-2:#D97706;
        --amber-3:#F59E0B;
        --shadow-card:0 10px 24px rgba(2,6,23,0.20);
        --shadow-hero:0 16px 36px rgba(15,23,42,0.28);
        --shadow-blue:0 18px 34px rgba(59,130,246,0.28);
        --radius-card:18px;
        --radius-hero:22px;
    }

    .stApp{
        background:
            radial-gradient(circle at top right, rgba(37,99,235,0.10), transparent 28%),
            radial-gradient(circle at top left, rgba(34,197,94,0.05), transparent 22%),
            linear-gradient(180deg,var(--bg-main) 0%, #0B1220 100%);
        color:var(--text-main);
    }

    html, body, [class*="css"], .stApp, p, li, label, span, div{
        -webkit-font-smoothing:antialiased;
        text-rendering:optimizeLegibility;
    }

    h1,h2,h3,h4,h5,h6, .section-title, .mobile-form-title, .premium-value, .wrap-metric-value, .reco-value{
        color:var(--text-main) !important;
        letter-spacing:0.01em;
    }

    .stCaptionContainer, .stCaptionContainer p, .section-meta-line, .panel-caption, .workspace-sub, .mobile-note,
    .premium-sub, .wrap-metric-sub, .reco-sub, .action-row-note{
        color:var(--text-sub) !important;
    }

    .model-pill{
        display:inline-block;
        margin-top:6px;
        padding:7px 12px;
        border-radius:999px;
        background:linear-gradient(90deg,var(--blue-1) 0%, var(--blue-3) 100%);
        color:white;
        font-size:0.84rem;
        font-weight:800;
        letter-spacing:0.02em;
        box-shadow:0 8px 18px rgba(37,99,235,0.24);
        border:1px solid rgba(147,197,253,0.28);
    }

    .section-head{
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:12px;
        flex-wrap:wrap;
        margin:8px 0 12px 0;
    }
    .section-title{
        font-size:1.08rem !important;
        font-weight:900 !important;
    }
    .section-meta-line{
        font-size:0.88rem !important;
        line-height:1.45 !important;
    }

    .wrap-metric-card, .reco-card, .mobile-form-card, .compact-summary-card, .reason-box, .diag-row,
    .score-card-shell, .premium-card, .section-card{
        background:linear-gradient(180deg,var(--bg-panel-2) 0%, var(--bg-panel) 100%) !important;
        border:1px solid var(--border-soft) !important;
        border-radius:var(--radius-card) !important;
        box-shadow:var(--shadow-card) !important;
    }

    .reco-card, .mobile-form-card{
        border-radius:var(--radius-hero) !important;
        box-shadow:var(--shadow-hero) !important;
    }

    .premium-title, .wrap-metric-label, .reco-label, .compact-summary-title{
        color:var(--text-muted) !important;
        font-size:0.80rem !important;
        text-transform:uppercase;
        letter-spacing:0.04em;
        font-weight:800 !important;
    }

    .premium-value, .wrap-metric-value, .compact-summary-value{
        color:var(--text-main) !important;
        font-size:1.2rem !important;
        font-weight:900 !important;
        line-height:1.2 !important;
    }

    .premium-sub, .wrap-metric-sub, .compact-summary-sub{
        color:var(--text-sub) !important;
        line-height:1.45 !important;
    }

    .section-accent{
        display:inline-block;
        padding:7px 12px;
        border-radius:999px;
        font-size:0.80rem;
        font-weight:900;
        margin-top:6px;
        border:1px solid transparent;
        box-shadow:inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .section-accent.blue{background:rgba(37,99,235,0.16); color:#BFDBFE; border-color:rgba(59,130,246,0.32);}
    .section-accent.purple{background:rgba(139,92,246,0.16); color:#DDD6FE; border-color:rgba(139,92,246,0.28);}
    .section-accent.amber{background:rgba(245,158,11,0.16); color:#FDE68A; border-color:rgba(245,158,11,0.30);}

    .diag-chip, .status-chip, .tag-chip{
        display:inline-block;
        padding:3px 9px;
        border-radius:999px;
        font-size:0.78rem;
        font-weight:900;
        margin-right:6px;
        border:1px solid transparent;
    }
    .diag-pos, .status-pos{background:var(--green-1) !important; color:#86EFAC !important; border-color:rgba(34,197,94,0.34) !important;}
    .diag-neg, .status-neg{background:var(--red-1) !important; color:#FCA5A5 !important; border-color:rgba(239,68,68,0.34) !important;}
    .diag-neu, .status-neu{background:var(--amber-1) !important; color:#FDE68A !important; border-color:rgba(245,158,11,0.34) !important;}

    .affects-line{
        margin-top:6px;
        font-size:0.82rem;
        color:#93C5FD !important;
        font-weight:800;
    }

    div[data-testid="stMetric"]{
        background:linear-gradient(180deg,var(--bg-panel-2) 0%, var(--bg-panel) 100%);
        border:1px solid var(--border-soft);
        border-radius:16px;
        padding:10px 12px 8px 12px;
        box-shadow:var(--shadow-card);
    }
    div[data-testid="stMetricLabel"]{
        color:var(--text-muted) !important;
        font-weight:800 !important;
    }
    div[data-testid="stMetricValue"]{
        color:var(--text-main) !important;
        font-weight:900 !important;
    }

    div[data-testid="stExpander"]{
        border:1px solid var(--border-soft);
        border-radius:16px;
        background:linear-gradient(180deg, rgba(17,24,39,0.82) 0%, rgba(15,23,42,0.82) 100%);
        box-shadow:var(--shadow-card);
    }
    div[data-testid="stExpander"] details summary{
        color:var(--text-main) !important;
        font-weight:800 !important;
    }

    .stTabs [data-baseweb="tab-list"]{
        gap:8px !important;
        padding:8px !important;
        background:rgba(15,23,42,0.65);
        border:1px solid var(--border-soft);
        border-radius:16px;
    }
    .stTabs [data-baseweb="tab"]{
        border-radius:12px !important;
        color:var(--text-sub) !important;
        font-weight:800 !important;
    }
    .stTabs [aria-selected="true"]{
        background:linear-gradient(180deg, rgba(37,99,235,0.20) 0%, rgba(29,78,216,0.12) 100%) !important;
        color:var(--text-main) !important;
        border:1px solid var(--border-strong) !important;
    }

    div[data-testid="stDataFrame"]{
        border:1px solid var(--border-soft);
        border-radius:16px;
        box-shadow:var(--shadow-card);
        overflow:hidden;
    }

    div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button{
        border-radius:16px !important;
        font-weight:800 !important;
        border:1px solid rgba(96,165,250,0.24) !important;
        box-shadow:0 8px 18px rgba(15,23,42,0.18) !important;
    }
    div[data-testid="stButton"] > button:hover, div[data-testid="stDownloadButton"] > button:hover{
        border-color:rgba(96,165,250,0.42) !important;
        transform:translateY(-1px);
    }

    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button{
        white-space:pre-line !important;
        line-height:1.35 !important;
        font-weight:900 !important;
        border-radius:22px !important;
        min-height:5.35rem !important;
        background:linear-gradient(180deg,var(--blue-1) 0%, #111827 100%) !important;
        color:var(--text-main) !important;
        border:1px solid var(--blue-3) !important;
        box-shadow:var(--shadow-blue) !important;
        text-shadow:none !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover{
        border-color:#93C5FD !important;
        box-shadow:0 22px 38px rgba(59,130,246,0.32) !important;
    }

    .pos, .success, .ok{ color:var(--green-3) !important; }
    .neg, .error, .risk{ color:var(--red-3) !important; }
    .warn, .caution{ color:var(--amber-3) !important; }

    @media (max-width: 900px){
        .section-title{font-size:1.02rem !important;}
        .premium-value, .wrap-metric-value, .compact-summary-value{font-size:1.08rem !important;}
        div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button{
            min-height:4.85rem !important;
            border-radius:20px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    workspace_mode = st.session_state.get("workspace_mode", "Expertenmodus")
except Exception:
    workspace_mode = "Expertenmodus"

if not workspace_mode:
    st.info("Wähle oben einen Arbeitsmodus aus. Erst danach werden die passenden Eingaben und Werkzeuge eingeblendet.")
elif workspace_mode == "Sofortanalyse":
    st.markdown("<div class='section-accent blue'>Sofortanalyse aktiv</div>", unsafe_allow_html=True)
    st.caption("Direkter Einstieg für spontane Einzelanalysen oder Multi-Screenings.")
elif workspace_mode == "Watchlisten":
    st.markdown("<div class='section-accent purple'>Watchlisten aktiv</div>", unsafe_allow_html=True)
    st.caption("Hier organisierst du Beobachtungslisten und kannst sie direkt analysieren oder mit Telegram prüfen.")
elif workspace_mode == "Kandidaten-Radar":
    st.markdown("<div class='section-accent blue'>Kandidaten-Radar aktiv</div>", unsafe_allow_html=True)
    st.caption("Neue Werte vorsortieren, bevor du sie tiefer in der Analyse oder Watchlist bearbeitest.")
else:
    st.markdown("<div class='section-accent amber'>Positionen aktiv</div>", unsafe_allow_html=True)
    st.caption("Hier konzentrierst du dich auf bestehende Positionen und führst sie als Positions-Watchlisten.")

# ---------- Watchlisten direkt in der App ----------
if workspace_mode in {"Watchlisten", "Positionen"}:
    with st.expander("Watchlisten", expanded=True):
        st.caption("Oben operativ arbeiten, darunter optional verwalten.")

        watchlists_df, watchlists_err = load_watchlists_df()
        if watchlists_err:
            st.warning(f"Watchlisten konnten noch nicht geladen werden: {watchlists_err}")

        if watchlists_df is None or watchlists_df.empty:
            catalog_df = pd.DataFrame(columns=["Watchlist_Name", "Watchlist_Type"])
        else:
            catalog_df = (
                watchlists_df[["Watchlist_Name", "Watchlist_Type"]]
                .fillna("")
                .astype(str)
                .query("Watchlist_Name != ''")
                .drop_duplicates()
                .sort_values(["Watchlist_Name", "Watchlist_Type"])
                .reset_index(drop=True)
            )

        default_type = "Positions-Watchlist" if workspace_mode == "Positionen" else "Watchlist"

        with st.expander("Watchlist verwalten", expanded=False):
                    wl1, wl2 = st.columns([1.2, 1.8])

                    with wl1:
                        st.markdown("**Neue Watchlist anlegen**")
                        st.markdown('<div class="compact-help">Watchlist = neue Chancen · Positions-Watchlist = bestehende Werte</div>', unsafe_allow_html=True)
                        new_watchlist_name = st.text_input(
                            "Name der Watchlist",
                            value=st.session_state.watchlist_new_name,
                            placeholder="z. B. US Tech, Europa Qualität, Depot Kernpositionen",
                            key="watchlist_new_name_widget"
                        ).strip()
                        st.session_state.watchlist_new_name = new_watchlist_name

                        type_options = ["Watchlist", "Positions-Watchlist"]
                        selected_type_for_create = st.session_state.selected_watchlist_type if st.session_state.selected_watchlist_type in type_options else default_type
                        default_type_idx = type_options.index(selected_type_for_create if workspace_mode == "Watchlisten" else "Positions-Watchlist")
                        new_watchlist_type = st.selectbox(
                            "Typ der Watchlist",
                            type_options,
                            index=default_type_idx,
                            key="watchlist_type_widget"
                        )
                        st.session_state.selected_watchlist_type = new_watchlist_type

                        freq_options = ["Nur manuell", "2x täglich", "3x täglich", "4x täglich"]
                        default_freq = "3x täglich" if new_watchlist_type == "Positions-Watchlist" else "4x täglich"
                        if st.session_state.selected_watchlist_check_frequency not in freq_options:
                            st.session_state.selected_watchlist_check_frequency = default_freq
                        new_check_frequency = st.selectbox(
                            "Prüf-Frequenz",
                            freq_options,
                            index=freq_options.index(st.session_state.selected_watchlist_check_frequency if st.session_state.selected_watchlist_check_frequency in freq_options else default_freq),
                            key="watchlist_check_frequency_widget"
                        )
                        st.session_state.selected_watchlist_check_frequency = new_check_frequency
                        st.markdown('<div class="compact-help">Standard: Watchlist 4x täglich · Positionen 3x täglich</div>', unsafe_allow_html=True)

                        if st.button("Watchlist erstellen", use_container_width=True, key="create_watchlist_btn"):
                            ok, msg = create_watchlist(new_watchlist_name, new_watchlist_type, check_frequency=new_check_frequency)
                            if ok:
                                st.session_state.selected_watchlist_name = new_watchlist_name
                                st.session_state.selected_watchlist_type = new_watchlist_type
                                st.session_state.selected_watchlist_check_frequency = new_check_frequency
                                st.session_state.watchlist_new_name = ""
                                st.session_state.workspace_mode = "Positionen" if new_watchlist_type == "Positions-Watchlist" else "Watchlisten"
                                st.success(msg)
                                trigger_ui_refresh()
                            else:
                                st.error(msg)

                    with wl2:
                        st.markdown("**Bestehende Watchlist auswählen**")
                        watchlist_options = catalog_df["Watchlist_Name"].tolist() if not catalog_df.empty else []

                        if workspace_mode == "Positionen" and not catalog_df.empty:
                            filtered_catalog_df = catalog_df[catalog_df["Watchlist_Type"] == "Positions-Watchlist"].copy()
                            watchlist_options = filtered_catalog_df["Watchlist_Name"].tolist()
                        elif workspace_mode == "Watchlisten" and not catalog_df.empty:
                            filtered_catalog_df = catalog_df[catalog_df["Watchlist_Type"] == "Watchlist"].copy()
                            watchlist_options = filtered_catalog_df["Watchlist_Name"].tolist()
                        else:
                            filtered_catalog_df = catalog_df.copy()

                        if watchlist_options:
                            default_idx = watchlist_options.index(st.session_state.selected_watchlist_name) if st.session_state.selected_watchlist_name in watchlist_options else 0
                            selected_watchlist_name = st.selectbox(
                                "Watchlist",
                                options=watchlist_options,
                                index=default_idx,
                                key="selected_watchlist_name_widget"
                            )
                            st.session_state.selected_watchlist_name = selected_watchlist_name

                            selected_watchlist_type = filtered_catalog_df.loc[filtered_catalog_df["Watchlist_Name"] == selected_watchlist_name, "Watchlist_Type"].iloc[0]
                            current_alert_mode = get_watchlist_alert_mode(selected_watchlist_name)
                            current_check_frequency = get_watchlist_check_frequency(selected_watchlist_name)
                            st.session_state.selected_watchlist_type = selected_watchlist_type
                            st.session_state.selected_watchlist_alert_mode = current_alert_mode
                            st.session_state.selected_watchlist_check_frequency = current_check_frequency
                            st.markdown(f'<div class="compact-help"><strong>Status:</strong> Typ {selected_watchlist_type} | Alert {current_alert_mode} | Frequenz {current_check_frequency}</div>', unsafe_allow_html=True)
                        else:
                            selected_watchlist_name = ""
                            selected_watchlist_type = default_type
                            st.markdown(
                    '<div class="empty-state"><div class="empty-state-title">Noch keine passende Watchlist vorhanden</div><div class="empty-state-text">Lege im Verwaltungsbereich zuerst eine neue Liste an. Danach kannst du sie hier sofort operativ nutzen.</div></div>',
                    unsafe_allow_html=True,
                )

        if selected_watchlist_name:
            watchlists_df, watchlists_err = load_watchlists_df()
            current_watchlist_df = (
                watchlists_df[
                    watchlists_df["Watchlist_Name"].astype(str).str.strip().str.lower() == selected_watchlist_name.strip().lower()
                ].copy()
                if watchlists_err is None and watchlists_df is not None and not watchlists_df.empty
                else pd.DataFrame(columns=["Watchlist_Name", "Watchlist_Type", "Ticker", "Added_At"])
            )
            current_tickers = [
                str(x).strip().upper()
                for x in current_watchlist_df.get("Ticker", pd.Series(dtype=str)).tolist()
                if str(x).strip()
            ]

            st.markdown(
                f'<div class="section-chip"><strong>Ausgewählte Liste:</strong> <span>{selected_watchlist_name} | Typ: {selected_watchlist_type} | Alert: {st.session_state.selected_watchlist_alert_mode} | Frequenz: {st.session_state.selected_watchlist_check_frequency} | Werte: {len(current_tickers)}</span></div>',
                unsafe_allow_html=True
            )

            st.markdown("""
<style>
div[data-testid="stExpander"] div[data-testid="stButton"] > button {
    min-height: 2.1rem !important;
    padding: 0.18rem 0.65rem !important;
    font-size: 0.82rem !important;
    line-height: 1.15 !important;
    border-radius: 0.6rem !important;
}
div[data-testid="stExpander"] div[data-testid="stButton"] > button p {
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)

            with st.expander("Einstellungen und Pflege dieser Watchlist", expanded=False):

                st.markdown("**Alert-Einstellungen für diese Watchlist**")
                am1, am2 = st.columns([1.6, 1.0])
                with am1:
                    alert_mode_options = ["Konservativ", "Standard", "Früh"]
                    selected_alert_mode = st.selectbox(
                        "Alert-Schärfe",
                        options=alert_mode_options,
                        index=alert_mode_options.index(st.session_state.selected_watchlist_alert_mode) if st.session_state.selected_watchlist_alert_mode in alert_mode_options else 1,
                        key="selected_watchlist_alert_mode_widget"
                    )
                    st.markdown('<div class="compact-help">Konservativ = selektiver · Standard = Mittelweg · Früh = mehr Hinweise</div>', unsafe_allow_html=True)
                with am2:
                    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
                    if st.button("Alert-Modus speichern", use_container_width=True, key="save_watchlist_alert_mode_btn"):
                        ok, msg = update_watchlist_alert_mode(selected_watchlist_name, selected_alert_mode)
                        if ok:
                            st.session_state.selected_watchlist_alert_mode = selected_alert_mode
                            st.success("Alert-Modus gespeichert.")
                            trigger_ui_refresh()
                        else:
                            st.error(msg)

                st.markdown("**Prüf-Frequenz für diese Watchlist**")
                fq1, fq2 = st.columns([1.6, 1.0])
                with fq1:
                    frequency_options = ["Nur manuell", "2x täglich", "3x täglich", "4x täglich"]
                    selected_check_frequency = st.selectbox(
                        "Automatische Prüf-Frequenz",
                        options=frequency_options,
                        index=frequency_options.index(st.session_state.selected_watchlist_check_frequency) if st.session_state.selected_watchlist_check_frequency in frequency_options else 3,
                        key="selected_watchlist_check_frequency_widget"
                    )
                    st.markdown('<div class="compact-help">Slots: 2x 10:30/18:30 · 3x 10:30/18:30/22:10 · 4x 10:30/15:40/18:30/22:10</div>', unsafe_allow_html=True)
                with fq2:
                    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
                    if st.button("Frequenz speichern", use_container_width=True, key="save_watchlist_frequency_btn"):
                        ok, msg = update_watchlist_check_frequency(selected_watchlist_name, selected_check_frequency)
                        if ok:
                            st.session_state.selected_watchlist_check_frequency = selected_check_frequency
                            st.success("Prüf-Frequenz gespeichert.")
                            trigger_ui_refresh()
                        else:
                            st.error(msg)

                st.markdown("**Ticker zur Watchlist hinzufügen**")
                add1, add2 = st.columns([2, 1])

                with add1:
                    watchlist_bulk_add = st.text_area(
                        "Ticker oder Firmennamen für diese Watchlist",
                        value=st.session_state.watchlist_bulk_add,
                        placeholder="Ein Wert pro Zeile oder mit Komma trennen",
                        height=100,
                        key="watchlist_bulk_add_widget"
                    ).strip()
                    st.session_state.watchlist_bulk_add = watchlist_bulk_add

                with add2:
                    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
                    if st.button("Aktuellen Ticker hinzufügen", use_container_width=True, key="add_current_ticker_watchlist"):
                        current_to_add = st.session_state.get("selected_ticker", "").strip().upper()
                        if current_to_add:
                            ok, msg = add_entries_to_watchlist(selected_watchlist_name, selected_watchlist_type, [current_to_add], check_frequency=st.session_state.get("selected_watchlist_check_frequency", "4x täglich"))
                            if ok:
                                st.success(msg)
                                trigger_ui_refresh()
                            else:
                                st.error(msg)
                        else:
                            st.markdown(
                            '<div class="empty-state"><div class="empty-state-title">Noch kein aktiver Ticker</div><div class="empty-state-text">Wähle zuerst einen Wert in der Analyse oder füge Eingaben manuell zur Watchlist hinzu.</div></div>',
                            unsafe_allow_html=True,
                        )

                    if st.button("Eingaben hinzufügen", use_container_width=True, key="add_bulk_watchlist"):
                        raw_entries = [x.strip() for x in re.split(r"[\n,;]+", watchlist_bulk_add) if x.strip()]
                        if not raw_entries:
                            st.markdown(
                            '<div class="empty-state"><div class="empty-state-title">Keine Eingaben erkannt</div><div class="empty-state-text">Gib mindestens einen Ticker oder Firmennamen ein, getrennt durch Zeilenumbruch oder Komma.</div></div>',
                            unsafe_allow_html=True,
                        )
                        else:
                            resolved_entries = []
                            for entry in raw_entries:
                                looks_like_ticker = (" " not in entry and len(entry) <= 12 and entry.replace(".", "").replace("-", "").isalnum())
                                if looks_like_ticker:
                                    resolved_entries.append(entry.upper())
                                else:
                                    matches = search_tickers(entry, max_results=1)
                                    if matches:
                                        resolved_entries.append(matches[0]["symbol"])
                            ok, msg = add_entries_to_watchlist(selected_watchlist_name, selected_watchlist_type, resolved_entries, check_frequency=st.session_state.get("selected_watchlist_check_frequency", "4x täglich"))
                            if ok:
                                st.success(msg)
                                st.session_state.watchlist_bulk_add = ""
                                trigger_ui_refresh()
                            else:
                                st.error(msg)

                st.markdown("**Inhalt der aktuellen Watchlist**")
                st.caption(f"Anzahl Werte: {len(current_tickers)}")
                if current_tickers:
                    preview_df = pd.DataFrame({"Ticker": current_tickers})
                    st.dataframe(preview_df, hide_index=True, use_container_width=True, height=min(320, 45 * len(preview_df) + 40))

                    rem1, rem2, rem3 = st.columns([1.7, 0.8, 0.9])
                    with rem1:
                        ticker_to_remove = st.selectbox("Ticker entfernen", options=current_tickers, key="remove_watchlist_ticker_widget")
                    with rem2:
                        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
                        if st.button("Ticker entfernen", use_container_width=True, key="remove_watchlist_ticker_btn"):
                            ok, msg = remove_ticker_from_watchlist(selected_watchlist_name, ticker_to_remove)
                            if ok:
                                st.success(msg)
                                trigger_ui_refresh()
                            else:
                                st.error(msg)
                    with rem3:
                        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
                        if st.button("Watchlist löschen", use_container_width=True, key="delete_watchlist_btn"):
                            ok, msg = delete_watchlist(selected_watchlist_name)
                            if ok:
                                st.success(msg)
                                st.session_state.selected_watchlist_name = ""
                                trigger_ui_refresh()
                            else:
                                st.error(msg)
                else:
                    st.markdown(
                        '<div class="empty-state"><div class="empty-state-title">Diese Watchlist ist noch leer</div><div class="empty-state-text">Füge im Verwaltungsbereich Werte hinzu oder übernimm den aktuell ausgewählten Ticker.</div></div>',
                        unsafe_allow_html=True,
                    )
                    empty_del_left, empty_del_mid, empty_del_right = st.columns([1.7, 0.8, 0.9])
                    with empty_del_right:
                        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                        if st.button("Watchlist löschen", use_container_width=True, key="delete_empty_watchlist_btn"):
                            ok, msg = delete_watchlist(selected_watchlist_name)
                            if ok:
                                st.success(msg)
                                st.session_state.selected_watchlist_name = ""
                                trigger_ui_refresh()
                            else:
                                st.error(msg)

            act1, act2, act3 = st.columns(3)
            with act1:
                if st.button("Watchlist in Analyse laden", use_container_width=True, key="load_watchlist_into_analysis"):
                    joined = "\n".join(current_tickers)
                    st.session_state.batch_input = joined
                    st.session_state.analysis_mode = "Mehrere Aktien vergleichen"
                    st.success(f"Watchlist '{selected_watchlist_name}' wurde in die Analyse geladen.")
            with act2:
                if st.button("Watchlist jetzt analysieren", use_container_width=True, key="run_watchlist_now"):
                    joined = "\n".join(current_tickers)
                    if joined.strip():
                        st.session_state.batch_input = joined
                        st.session_state.analysis_mode = "Mehrere Aktien vergleichen"
                        st.session_state.analysis_mode_run = "Mehrere Aktien vergleichen"
                        st.session_state.analysis_requested = True
                        st.session_state.run_selected_watchlist_name = selected_watchlist_name
                        st.session_state.run_selected_watchlist_type = selected_watchlist_type
                        st.session_state.send_watchlist_alerts_after_run = False
                        st.success(f"Watchlist '{selected_watchlist_name}' wird jetzt analysiert.")
                    else:
                        st.info("In dieser Watchlist sind noch keine Ticker.")
            with act3:
                if st.button("Watchlist analysieren + Telegram", use_container_width=True, key="run_watchlist_telegram"):
                    joined = "\n".join(current_tickers)
                    if joined.strip():
                        st.session_state.batch_input = joined
                        st.session_state.analysis_mode = "Mehrere Aktien vergleichen"
                        st.session_state.analysis_mode_run = "Mehrere Aktien vergleichen"
                        st.session_state.analysis_requested = True
                        st.session_state.run_selected_watchlist_name = selected_watchlist_name
                        st.session_state.run_selected_watchlist_type = selected_watchlist_type
                        st.session_state.send_watchlist_alerts_after_run = True
                        st.success(f"Watchlist '{selected_watchlist_name}' wird jetzt analysiert und danach werden Telegram-Alerts geprüft.")
                    else:
                        st.info("In dieser Watchlist sind noch keine Ticker.")

# ---------- Analyse-Steuerung im Hauptbereich ----------
if workspace_mode:
    if workspace_mode:
        pass

    if workspace_mode == "Sofortanalyse":
        pass
    elif workspace_mode == "Watchlisten":
        st.markdown(
            """
            <div class="mobile-form-card" style="border-left:5px solid #8b5cf6;">
                <div class="mobile-form-title">Watchlist-Analyse</div>
                <div class="mobile-form-sub">
                    Du kannst spontan analysieren oder direkt eine Watchlist in die Analyse laden. Watchlisten-Verwaltung und Alerts stehen darüber bereit.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif workspace_mode == "Kandidaten-Radar":
        st.markdown(
            """
            <div class="mobile-form-card" style="border-left:5px solid #3b82f6;">
                <div class="mobile-form-title">Kandidaten-Radar</div>
                <div class="mobile-form-sub">
                    Dieser Bereich soll dir interessante Kaufkandidaten vorsortieren, bevor du sie tiefer analysierst. Als erster echter Vorschlagsmodus ist jetzt <strong>US Tech</strong> aktiv. Eigene Listen bleiben zusätzlich möglich.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        us_tech_universe = [
            "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "IBM",
            "QCOM", "TXN", "MU", "INTU", "AMAT", "ADI", "LRCX", "KLAC", "INTC", "PANW",
            "CRWD", "SNPS", "CDNS", "ANET", "PLTR", "NOW", "ADSK", "TEAM", "ROP", "DELL",
            "HPQ", "WDAY", "DDOG", "NET", "MDB", "ZS", "OKTA", "HUBS", "SHOP", "SQ",
            "UBER", "ABNB", "META", "GOOGL", "AMZN", "NFLX", "TSM", "ASML", "ARM", "SMCI",
            "APH", "FTNT", "MCHP", "NXPI", "MRVL", "ON", "STM", "MPWR", "GFS", "WDC",
            "STX", "NTAP", "DOCU", "SNOW", "FICO", "TTD", "PINS", "SAP", "PATH", "ESTC",
            "DT", "APP", "RBLX", "GEN", "AKAM"
        ]
        us_basis_universe = [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "LLY", "V",
            "XOM", "UNH", "AVGO", "MA", "COST", "WMT", "JNJ", "PG", "HD", "ABBV",
            "BAC", "KO", "MRK", "PEP", "CVX", "ADBE", "CRM", "NFLX", "AMD", "ORCL",
            "LIN", "TMO", "MCD", "GE", "CAT", "AMAT", "GS", "AXP", "NOW", "PM"
        ]
        europa_quality_universe = [
            "SAP", "ASML", "NESN.SW", "NOVO-B.CO", "MC.PA", "SU.PA", "AIR.PA", "SIE.DE", "DTE.DE", "ALV.DE",
            "MUV2.DE", "RMS.PA", "OR.PA", "SAN.PA", "BN.PA", "DG.PA", "EL.PA", "SAF.PA", "CS.PA", "ULVR.L",
            "AZN.L", "SHEL.L", "REL.L", "LSEG.L", "DGE.L", "GSK.L", "ABBN.SW", "ROG.SW", "SIKA.SW", "UHR.SW",
            "ISP.MI", "UCG.MI", "IBE.MC", "IBE.MC", "FER.MC", "RACE.MI", "NDA-FI.HE", "DSY.PA", "KER.PA", "HEI.DE"
        ]
        semiconductor_universe = [
            "NVDA", "AVGO", "AMD", "QCOM", "TXN", "MU", "ADI", "AMAT", "LRCX", "KLAC",
            "INTC", "MCHP", "NXPI", "MRVL", "ON", "MPWR", "GFS", "SWKS", "QRVO", "TER",
            "TSM", "ASML", "ASM.AS", "BE Semiconductor Industries NV", "ARM"
        ]
        semiconductor_universe = [
            "NVDA", "AVGO", "AMD", "QCOM", "TXN", "MU", "ADI", "AMAT", "LRCX", "KLAC",
            "INTC", "MCHP", "NXPI", "MRVL", "ON", "MPWR", "GFS", "SWKS", "QRVO", "TER",
            "TSM", "ASML", "ARM", "STM", "ENTG"
        ]
        radar_universe_map = {
            "US Tech": (us_tech_universe, "US Tech Fokus", "Breites Tech- und Plattformuniversum mit 75 vordefinierten Werten."),
            "US Basisliste": (us_basis_universe, "US Basisliste", "Große US-Standardwerte als breiter Startscreen für neue Ideen."),
            "Europa Qualität": (europa_quality_universe, "Europa Qualität", "Europäische Qualitäts- und Large-Cap-Werte mit stabiler Marktstellung."),
            "Halbleiter": (semiconductor_universe, "Halbleiter", "Fokusliste für Chipdesigner, Ausrüster und Halbleiterfertiger."),
        }

        rc1, rc2, rc3, rc4 = st.columns([1.3, 1.0, 0.7, 1.0])
        with rc1:
            radar_universe = st.selectbox(
                "Universum",
                options=["US Tech", "US Basisliste", "Europa Qualität", "Halbleiter", "Eigene Liste"],
                index=["US Tech", "US Basisliste", "Europa Qualität", "Halbleiter", "Eigene Liste"].index(st.session_state.radar_universe if st.session_state.radar_universe in ["US Tech", "US Basisliste", "Europa Qualität", "Halbleiter", "Eigene Liste"] else "US Tech"),
                key="radar_universe_widget"
            )
            st.session_state.radar_universe = radar_universe
        with rc2:
            style_options = ["Leader", "Turnaround", "Ausgewogen"]
            radar_screening_style = st.selectbox(
                "Screening-Stil",
                options=style_options,
                index=style_options.index(st.session_state.radar_screening_style if st.session_state.radar_screening_style in style_options else "Leader"),
                key="radar_screening_style_widget"
            )
            st.session_state.radar_screening_style = radar_screening_style
        with rc3:
            radar_max_candidates = st.selectbox(
                "Max. Kandidaten",
                options=[10, 15, 20],
                index=[10, 15, 20].index(st.session_state.radar_max_candidates if st.session_state.radar_max_candidates in [10, 15, 20] else 15),
                key="radar_max_candidates_widget"
            )
            st.session_state.radar_max_candidates = radar_max_candidates
        with rc4:
            st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
            run_candidate_radar = st.button("Kandidaten-Radar starten", use_container_width=True, type="primary", key="run_candidate_radar_btn")
            if run_candidate_radar:
                st.session_state.radar_requested = True

        style_note_map = {
            "Leader": "Bevorzugt bestätigte Stärke, Leadership und saubere Trend-Setups.",
            "Turnaround": "Bevorzugt frühe Drehkandidaten, Rebounds und technische Erholungsfenster.",
            "Ausgewogen": "Mittelweg zwischen bestätigter Stärke und früheren Chancen.",
        }
        st.caption(f"Aktiver Screening-Stil: {st.session_state.radar_screening_style} | {style_note_map.get(st.session_state.radar_screening_style, '')}")

        if st.session_state.radar_universe == "Eigene Liste":
            radar_custom_input = st.text_area(
                "Eigene Kandidatenliste",
                value=st.session_state.radar_custom_input,
                placeholder="Ein Wert pro Zeile oder durch Komma trennen, z. B.\nAAPL\nNVDA\nASML\nSAP",
                key="radar_custom_input_widget"
            ).strip()
            st.session_state.radar_custom_input = radar_custom_input
        elif st.session_state.radar_universe in radar_universe_map:
            active_universe, active_label, active_desc = radar_universe_map[st.session_state.radar_universe]
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="premium-title">Aktives Start-Universum</div>
                    <div class="premium-value">{active_label}</div>
                    <div class="premium-sub">
                        {len(active_universe)} vordefinierte Werte werden mit der bestehenden Analyse-Engine gescannt und anschließend nach Trigger-Nähe, Einstiegsqualität und Investment-Attraktivität sortiert. {active_desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Beispielwerte: " + ", ".join(active_universe[:12]) + " …")
        else:
            st.info("Bitte wähle ein Radar-Universum oder nutze eine eigene Liste.")

        st.markdown(
            """
            <div class="section-card">
                <div class="premium-title">V1 in diesem Schritt</div>
                <div class="premium-value">Vordefinierte Listen oder Eigene Liste → Radar-Lauf → Top-Kandidaten heute</div>
                <div class="premium-sub">
                    Die bestehende Analyse-Logik wird auf ein erstes Start-Universum oder deine eigene Liste angewendet und als kompakte Kandidaten-Tabelle sortiert.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.get("radar_requested", False):
            st.session_state.radar_requested = False
            if st.session_state.radar_universe == "Eigene Liste":
                radar_entries = split_batch_input(st.session_state.radar_custom_input)
                radar_resolution_rows = []
                if not radar_entries:
                    st.warning("Bitte gib mindestens einen Ticker oder Firmennamen für den Radar-Lauf ein.")
                    resolved_radar_entries = []
                else:
                    resolved_radar_entries = []
                    for entry in radar_entries:
                        resolved = resolve_input_to_ticker(entry, fallback=None)
                        radar_resolution_rows.append({
                            "Eingabe": entry,
                            "Aufgelöst zu": resolved if resolved else "Nicht gefunden"
                        })
                        if resolved and resolved not in resolved_radar_entries:
                            resolved_radar_entries.append(resolved)
            elif st.session_state.radar_universe in radar_universe_map:
                resolved_radar_entries = list(radar_universe_map[st.session_state.radar_universe][0])
                radar_resolution_rows = [{"Eingabe": tkr, "Aufgelöst zu": tkr} for tkr in resolved_radar_entries]
            else:
                resolved_radar_entries = []
                radar_resolution_rows = []
                st.warning("Bitte wähle ein gültiges Radar-Universum oder nutze eine eigene Liste.")

            if st.session_state.radar_universe in set(radar_universe_map.keys()) | {"Eigene Liste"}:
                if not resolved_radar_entries:
                    if st.session_state.radar_universe == "Eigene Liste":
                        st.error("Keine der Eingaben konnte in einen auswertbaren Ticker aufgelöst werden.")
                else:
                    radar_results = []
                    radar_errors = []
                    radar_progress = st.progress(0)
                    radar_status = st.empty()
                    for i, tkr in enumerate(resolved_radar_entries, start=1):
                        radar_status.info(f"Radar analysiert {tkr} ({i}/{len(resolved_radar_entries)}) ...")
                        try:
                            radar_result = analyze_stock(
                                ticker=tkr,
                                horizon="Swing (1-4 Wochen)",
                                depot=10000,
                                risk_pct=1.0,
                                override=0.0,
                                buy_in_override=0.0,
                                smart_money_default=True,
                                strict_mode=True
                            )
                            radar_results.append(radar_result)
                        except Exception as e:
                            radar_errors.append((tkr, str(e)))
                        radar_progress.progress(i / len(resolved_radar_entries))
                    radar_progress.empty()
                    radar_status.empty()

                    if not radar_results:
                        st.error("Der Radar-Lauf hat keine belastbaren Kandidaten geliefert.")
                    else:
                        trigger_rank_map = {
                            "Aktiv": 5,
                            "Jetzt prüfbar": 5,
                            "Nahe dran": 4,
                            "Fast prüfbar": 4,
                            "Frühe Beobachtung": 3,
                            "Früh interessant": 3,
                            "Beobachten": 2,
                            "Weiter beobachten": 2,
                            "Passiv": 1,
                            "Aktuell kein Fokus": 1,
                            "Warten": 0,
                            "Noch warten": 0,
                        }

                        def build_radar_reason(r):
                            style_name_local = str(st.session_state.get("radar_screening_style", "Leader") or "Leader")
                            trigger = str(r.get("trigger_status", "") or "").strip()
                            setup_type_local = str(r.get("setup_type", "") or "").strip()
                            entry_quality_local = str(r.get("entry_quality", "") or "").strip().lower()
                            investment_case_local = float(r.get("investment_case_score", np.nan)) if pd.notna(r.get("investment_case_score", np.nan)) else np.nan
                            trading_case_local = float(r.get("trading_case_score", np.nan)) if pd.notna(r.get("trading_case_score", np.nan)) else np.nan
                            leadership_local = str(r.get("leadership_status", "") or "").strip()
                            top_red_flag_local = str(r.get("top_red_flag", "") or "").strip()
                            market_regime_local = str((r.get("market_info", {}) or {}).get("regime", "") or "").strip().upper()
                            catalyst_local = pd.to_numeric(r.get("catalyst_score", np.nan), errors="coerce")
                            short_term_local = pd.to_numeric(r.get("short_term_score", np.nan), errors="coerce")
                            tb_local = pd.to_numeric(r.get("tb_score_100", np.nan), errors="coerce")
                            exit_local = pd.to_numeric(r.get("exit_score", np.nan), errors="coerce")
                            setup_conf_local = pd.to_numeric(r.get("setup_confidence", np.nan), errors="coerce")
                            setup_priority_local = pd.to_numeric(r.get("setup_priority_score", np.nan), errors="coerce")

                            if style_name_local == "Turnaround":
                                if setup_type_local == "Rebound":
                                    return "Frühe technische Drehung mit Rebound-Charakter"
                                if setup_type_local == "Breakout-Retest":
                                    return "Rückeroberung läuft, Bestätigung über Retest möglich"
                                if setup_type_local == "Pullback an MA20":
                                    return "Frischer Stabilisierungsversuch nahe MA20"
                                if setup_type_local == "Pullback an MA50":
                                    return "Pullback an MA50 kann zur Trenddrehung werden"
                                if trigger in {"Aktiv", "Jetzt prüfbar"}:
                                    return "Turnaround-Setup ist jetzt erstmals konkret prüfbar"
                                if trigger in {"Nahe dran", "Fast prüfbar"}:
                                    return "Frühe Drehung sichtbar, aber Bestätigung fehlt noch knapp"
                                if pd.notna(catalyst_local) and catalyst_local >= 65:
                                    return "Katalysator verbessert das Turnaround-Fenster"
                                if pd.notna(short_term_local) and short_term_local >= 60 and pd.notna(tb_local) and tb_local >= 58:
                                    return "Kurzfristbild dreht, obwohl der Titel noch nicht voll bestätigt ist"
                                if pd.notna(exit_local) and exit_local <= 35:
                                    return "Exit-Druck ist begrenzt, Raum für technische Erholung vorhanden"
                                if entry_quality_local == "früh":
                                    return "Noch zu früh, aber erste Drehansätze werden sichtbar"
                                if top_red_flag_local not in {"", "-", "None"}:
                                    return f"Früher Kandidat, aber noch Gegenwind: {top_red_flag_local[:40]}"
                                return "Früher technischer Kandidat vor voller Bestätigung"

                            if style_name_local == "Leader":
                                if leadership_local == "Leader" and setup_type_local and setup_type_local != "Kein sauberes Setup":
                                    return f"Leader mit sauberem {setup_type_local}-Setup"
                                if leadership_local == "Leader":
                                    return "Leader mit bestätigter Stärke"
                                if trigger in {"Aktiv", "Jetzt prüfbar"} and pd.notna(setup_conf_local) and setup_conf_local >= 65:
                                    return "Bestätigtes Setup, Einstieg jetzt konkret prüfbar"
                                if trigger in {"Nahe dran", "Fast prüfbar"} and pd.notna(setup_priority_local) and setup_priority_local >= 65:
                                    return "Starker Leader-Kandidat, Trigger fast vollständig"
                                if setup_type_local == "Breakout":
                                    return "Breakout-Leader mit sauberer Struktur"
                                if setup_type_local == "Trendfolge":
                                    return "Trendführer mit stabiler Fortsetzungsstruktur"
                                if setup_type_local == "Pullback an MA20":
                                    return "Leader im konstruktiven Pullback an MA20"
                                if pd.notna(trading_case_local) and trading_case_local >= 70 and pd.notna(investment_case_local) and investment_case_local >= 72:
                                    return "Bestätigte Stärke bei gutem Timing-Fenster"
                                if top_red_flag_local not in {"", "-", "None"}:
                                    return f"Leader-Qualität vorhanden, aber noch Bremse: {top_red_flag_local[:42]}"
                                return "Bestätigter Stärke-Kandidat mit sauberem Gesamtbild"

                            if trigger in {"Aktiv", "Jetzt prüfbar"}:
                                return "Gutes Gesamtbild mit direkt prüfbarem Einstieg"
                            if trigger in {"Nahe dran", "Fast prüfbar"}:
                                return "Ausgewogener Kandidat, Timing fast vollständig"
                            if pd.notna(investment_case_local) and investment_case_local >= 78 and pd.notna(trading_case_local) and trading_case_local < 60:
                                return "Starker Investment-Case, Timing zieht noch nicht ganz mit"
                            if leadership_local == "Leader" and pd.notna(catalyst_local) and catalyst_local >= 60:
                                return "Leader-Qualität mit zusätzlichem Katalysator"
                            if setup_type_local == "Breakout":
                                return "Starker Kandidat mit Breakout-Charakter"
                            if setup_type_local == "Breakout-Retest":
                                return "Retest-Kandidat mit ausgewogenem Chance-Risiko-Profil"
                            if setup_type_local == "Pullback an MA20":
                                return "Qualitätswert im konstruktiven Pullback an MA20"
                            if setup_type_local == "Pullback an MA50":
                                return "Rücksetzer an MA50, aber Grundbild bleibt intakt"
                            if setup_type_local == "Trendfolge":
                                return "Solider Trendfolger mit brauchbarer Struktur"
                            if trigger in {"Frühe Beobachtung", "Früh interessant"}:
                                if setup_type_local and setup_type_local != "Kein sauberes Setup":
                                    return f"{setup_type_local}-Ansatz vorhanden, aber noch nicht reif genug"
                                return "Früher Kandidat mit gemischtem Timing"
                            if top_red_flag_local not in {"", "-", "None"} and market_regime_local == "NEGATIV":
                                return "Ordentlicher Kandidat, aber Marktumfeld und Risiko bremsen"
                            if entry_quality_local == "abwarten":
                                return "Interessant, aber aktuell noch über der Entry-Zone"
                            if entry_quality_local == "früh":
                                return "Interessant, aber Bestätigung fehlt noch"
                            if top_red_flag_local not in {"", "-", "None"}:
                                return f"Ausgewogener Kandidat, aber {top_red_flag_local[:52]}"
                            return "Ausgewogener Kandidat mit brauchbarem Gesamtbild"

                        radar_df = build_ranking_table(radar_results)
                        radar_reason_map = {str(r.get("ticker", "")): build_radar_reason(r) for r in radar_results}
                        radar_result_map = {str(r.get("ticker", "")): r for r in radar_results}
                        radar_df["Warum heute auffällig"] = radar_df["Ticker"].astype(str).map(radar_reason_map)
                        radar_df["__trigger_sort"] = radar_df.get("Trigger-Status", pd.Series(dtype=str)).map(trigger_rank_map).fillna(0)

                        def compute_radar_style_sort(row):
                            tkr = str(row.get("Ticker", "") or "")
                            r = radar_result_map.get(tkr, {}) or {}
                            trigger_score = float(row.get("__trigger_sort", 0) or 0)
                            entry_score_local = pd.to_numeric(row.get("Einstieg jetzt attraktiv?", np.nan), errors="coerce")
                            invest_score_local = pd.to_numeric(row.get("Investment-Attraktivität", np.nan), errors="coerce")
                            setup_priority_local = pd.to_numeric(row.get("Setup-Priorität", np.nan), errors="coerce")
                            leadership_local = pd.to_numeric(r.get("leadership_score", np.nan), errors="coerce")
                            catalyst_local = pd.to_numeric(r.get("catalyst_score", np.nan), errors="coerce")
                            short_term_local = pd.to_numeric(r.get("short_term_score", np.nan), errors="coerce")
                            tb_local = pd.to_numeric(r.get("tb_score_100", np.nan), errors="coerce")
                            exit_local = pd.to_numeric(r.get("exit_score", np.nan), errors="coerce")
                            setup_conf_local = pd.to_numeric(r.get("setup_confidence", np.nan), errors="coerce")
                            rebound_bonus = 0.0
                            setup_type_local = str(r.get("setup_type", "") or "")
                            if setup_type_local in {"Rebound", "Breakout-Retest", "Pullback an MA20", "Pullback an MA50"}:
                                rebound_bonus = 8.0
                            leader_bonus = 0.0
                            if str(r.get("leadership_status", "") or "") == "Leader":
                                leader_bonus = 8.0
                            safe = lambda v, default=0.0: default if pd.isna(v) else float(v)
                            style_name = str(st.session_state.get("radar_screening_style", "Leader") or "Leader")
                            if style_name == "Leader":
                                return (
                                    trigger_score * 18
                                    + safe(entry_score_local) * 0.32
                                    + safe(setup_priority_local) * 0.24
                                    + safe(leadership_local) * 0.18
                                    + safe(invest_score_local) * 0.08
                                    + safe(setup_conf_local) * 0.08
                                    + leader_bonus
                                )
                            if style_name == "Turnaround":
                                return (
                                    trigger_score * 10
                                    + safe(entry_score_local) * 0.18
                                    + safe(invest_score_local) * 0.12
                                    + safe(catalyst_local) * 0.20
                                    + safe(short_term_local) * 0.18
                                    + safe(tb_local) * 0.12
                                    + (100 - safe(exit_local, 100)) * 0.08
                                    + rebound_bonus
                                )
                            return (
                                trigger_score * 14
                                + safe(entry_score_local) * 0.26
                                + safe(invest_score_local) * 0.18
                                + safe(setup_priority_local) * 0.18
                                + safe(leadership_local) * 0.10
                                + safe(catalyst_local) * 0.08
                                + safe(setup_conf_local) * 0.06
                                + rebound_bonus * 0.5
                                + leader_bonus * 0.5
                            )

                        radar_df["__style_sort"] = radar_df.apply(compute_radar_style_sort, axis=1)

                        radar_sort_cols = []
                        radar_sort_ascending = []
                        if "__style_sort" in radar_df.columns:
                            radar_sort_cols.append("__style_sort")
                            radar_sort_ascending.append(False)
                        if "__trigger_sort" in radar_df.columns:
                            radar_sort_cols.append("__trigger_sort")
                            radar_sort_ascending.append(False)
                        if "Einstieg jetzt attraktiv?" in radar_df.columns:
                            radar_sort_cols.append("Einstieg jetzt attraktiv?")
                            radar_sort_ascending.append(False)
                        if "Investment-Attraktivität" in radar_df.columns:
                            radar_sort_cols.append("Investment-Attraktivität")
                            radar_sort_ascending.append(False)
                        if "Setup-Priorität" in radar_df.columns:
                            radar_sort_cols.append("Setup-Priorität")
                            radar_sort_ascending.append(False)
                        if radar_sort_cols:
                            radar_df = radar_df.sort_values(by=radar_sort_cols, ascending=radar_sort_ascending)

                        radar_df = radar_df.drop(columns=["__trigger_sort", "__style_sort"], errors="ignore").reset_index(drop=True)
                        radar_df = radar_df.head(int(st.session_state.radar_max_candidates))

                        st.markdown("### Kandidaten nach Reifegrad")
                        st.caption("Die Radar-Ergebnisse sind jetzt in sofort nutzbare Kaufkandidaten, vorbereitete Kandidaten und spätere Beobachtungswerte getrennt.")
                        radar_cols = [c for c in [
                            "Ticker", "Name", "Setup-Typ", "Investment-Attraktivität", "Einstieg jetzt attraktiv?",
                            "Setup-Priorität", "Trigger-Status", "Watchlist-Priorität", "Warum heute auffällig", "Top Red Flag"
                        ] if c in radar_df.columns]

                        trigger_series = radar_df["Trigger-Status"].astype(str).fillna("") if "Trigger-Status" in radar_df.columns else pd.Series([""] * len(radar_df))
                        entry_series = pd.to_numeric(radar_df["Einstieg jetzt attraktiv?"], errors="coerce") if "Einstieg jetzt attraktiv?" in radar_df.columns else pd.Series([np.nan] * len(radar_df))
                        invest_series = pd.to_numeric(radar_df["Investment-Attraktivität"], errors="coerce") if "Investment-Attraktivität" in radar_df.columns else pd.Series([np.nan] * len(radar_df))

                        mask_now = (
                            trigger_series.isin(["Aktiv", "Jetzt prüfbar", "Nahe dran", "Fast prüfbar"])
                            | (entry_series >= 68)
                        )
                        mask_later = (
                            trigger_series.isin(["Beobachten", "Weiter beobachten", "Passiv", "Aktuell kein Fokus", "Warten", "Noch warten"])
                            | ((invest_series >= 70) & (entry_series < 60))
                        )
                        mask_near = ~(mask_now | mask_later)

                        radar_now_df = radar_df[mask_now].copy().reset_index(drop=True)
                        radar_near_df = radar_df[mask_near].copy().reset_index(drop=True)
                        radar_later_df = radar_df[mask_later].copy().reset_index(drop=True)

                        section_specs = [
                            ("Jetzt spannend", "Aktive oder fast aktive Kandidaten mit brauchbarer Einstiegsreife.", radar_now_df),
                            ("Nahe dran", "Interessante Kandidaten, bei denen Setup oder Timing noch einen Schritt brauchen.", radar_near_df),
                            ("Später beobachten", "Gute Werte für die engere Watchlist, aber noch nicht reif für einen direkten Einstieg.", radar_later_df),
                        ]

                        for section_title, section_caption, section_df in section_specs:
                            st.markdown(f"#### {section_title}")
                            st.caption(section_caption)
                            if section_df.empty:
                                st.markdown(
                                    '<div class="empty-state"><div class="empty-state-title">Aktuell keine Werte in dieser Stufe</div><div class="empty-state-text">Die momentane Radar-Auswahl liefert in diesem Reifegrad gerade keine Kandidaten.</div></div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.dataframe(
                                    style_ranking_df(section_df[radar_cols].copy()),
                                    hide_index=True,
                                    use_container_width=True,
                                    height=min(340, 45 * len(section_df) + 40)
                                )

                        if radar_resolution_rows and st.session_state.radar_universe == "Eigene Liste":
                            with st.expander("Aufgelöste Radar-Eingaben", expanded=False):
                                st.dataframe(pd.DataFrame(radar_resolution_rows), hide_index=True, use_container_width=True)

                        if radar_errors:
                            with st.expander("Nicht analysierbare Radar-Werte", expanded=False):
                                st.dataframe(pd.DataFrame(radar_errors, columns=["Ticker", "Fehler"]), hide_index=True, use_container_width=True)

                        radar_top_tickers = radar_df["Ticker"].astype(str).tolist() if "Ticker" in radar_df.columns else []
                        radar_batch_text = "\n".join(radar_top_tickers)

                        st.markdown("### Radar-Ergebnisse direkt nutzen")
                        st.caption("Die aktuellen Top-Kandidaten lassen sich direkt in die Sofortanalyse übernehmen oder in eine frei wählbare Watchlist schreiben.")
                        radar_watchlists_df, radar_watchlists_err = load_watchlists_df()
                        if radar_watchlists_err:
                            st.warning(f"Watchlisten konnten für den Radar nicht geladen werden: {radar_watchlists_err}")
                            radar_catalog_df = pd.DataFrame(columns=["Watchlist_Name", "Watchlist_Type"])
                        elif radar_watchlists_df is None or radar_watchlists_df.empty:
                            radar_catalog_df = pd.DataFrame(columns=["Watchlist_Name", "Watchlist_Type"])
                        else:
                            radar_catalog_df = (
                                radar_watchlists_df[["Watchlist_Name", "Watchlist_Type"]]
                                .fillna("")
                                .astype(str)
                                .query("Watchlist_Name != ''")
                                .drop_duplicates()
                                .sort_values(["Watchlist_Name", "Watchlist_Type"])
                                .reset_index(drop=True)
                            )

                        radar_watchlist_options = []
                        radar_watchlist_label_map = {}
                        for _, _row in radar_catalog_df.iterrows():
                            _wl_name = str(_row.get("Watchlist_Name", "") or "").strip()
                            _wl_type = str(_row.get("Watchlist_Type", "Watchlist") or "Watchlist").strip() or "Watchlist"
                            if _wl_name:
                                _label = f"{_wl_name} | {_wl_type}"
                                radar_watchlist_options.append(_label)
                                radar_watchlist_label_map[_label] = (_wl_name, _wl_type)

                        default_radar_watchlist_label = None
                        current_selected_watchlist_name = str(st.session_state.get("selected_watchlist_name", "") or "").strip()
                        current_selected_watchlist_type = str(st.session_state.get("selected_watchlist_type", "Watchlist") or "Watchlist").strip() or "Watchlist"
                        if current_selected_watchlist_name:
                            candidate_label = f"{current_selected_watchlist_name} | {current_selected_watchlist_type}"
                            if candidate_label in radar_watchlist_options:
                                default_radar_watchlist_label = candidate_label
                        if default_radar_watchlist_label is None and radar_watchlist_options:
                            default_radar_watchlist_label = radar_watchlist_options[0]

                        if radar_watchlist_options:
                            radar_target_watchlist_label = st.selectbox(
                                "Ziel-Watchlist für Radar-Kandidaten",
                                options=radar_watchlist_options,
                                index=radar_watchlist_options.index(default_radar_watchlist_label) if default_radar_watchlist_label in radar_watchlist_options else 0,
                                key="radar_target_watchlist_widget"
                            )
                            selected_watchlist_name_for_radar, selected_watchlist_type_for_radar = radar_watchlist_label_map.get(
                                radar_target_watchlist_label,
                                ("", "Watchlist")
                            )
                            st.caption(f"Aktuelles Ziel: {selected_watchlist_name_for_radar} ({selected_watchlist_type_for_radar})")
                        else:
                            selected_watchlist_name_for_radar, selected_watchlist_type_for_radar = "", "Watchlist"
                            st.markdown(
                                '<div class="empty-state"><div class="empty-state-title">Noch keine Watchlist vorhanden</div><div class="empty-state-text">Lege zuerst eine Watchlist oder Positions-Watchlist an. Danach kannst du Radar-Kandidaten direkt dorthin übernehmen.</div></div>',
                                unsafe_allow_html=True,
                            )

                        ra1, ra2 = st.columns(2)
                        with ra1:
                            if st.button("Top-Kandidaten in Sofortanalyse laden", use_container_width=True, key="radar_load_into_analysis_btn"):
                                if radar_top_tickers:
                                    st.session_state.batch_input = radar_batch_text
                                    st.session_state.analysis_mode = "Mehrere Aktien vergleichen"
                                    st.session_state.analysis_mode_run = "Mehrere Aktien vergleichen"
                                    st.session_state.workspace_mode = "Sofortanalyse"
                                    st.success(f"{len(radar_top_tickers)} Radar-Kandidaten wurden in die Sofortanalyse übernommen.")
                                    st.rerun()
                                else:
                                    st.info("Es sind keine Radar-Kandidaten zum Übernehmen vorhanden.")
                        with ra2:
                            add_label = (
                                f"Top-Kandidaten zu {selected_watchlist_name_for_radar} hinzufügen"
                                if selected_watchlist_name_for_radar else
                                "Top-Kandidaten zur Watchlist hinzufügen"
                            )
                            if st.button(add_label, use_container_width=True, key="radar_add_to_watchlist_btn"):
                                if not radar_top_tickers:
                                    st.info("Es sind keine Radar-Kandidaten zum Hinzufügen vorhanden.")
                                elif not selected_watchlist_name_for_radar:
                                    st.info("Bitte lege zuerst eine Watchlist an oder wähle eine Ziel-Watchlist aus.")
                                else:
                                    ok, msg = add_entries_to_watchlist(
                                        selected_watchlist_name_for_radar,
                                        selected_watchlist_type_for_radar,
                                        radar_top_tickers,
                                        check_frequency=st.session_state.get("selected_watchlist_check_frequency", "4x täglich")
                                    )
                                    if ok:
                                        st.success(msg)
                                    else:
                                        st.error(msg)

        st.stop()
    else:
        st.markdown(
            """
            <div class="mobile-form-card" style="border-left:5px solid #f59e0b;">
                <div class="mobile-form-title">Positionsüberwachung</div>
                <div class="mobile-form-sub">
                    Nutze bevorzugt Positions-Watchlisten für bestehende Depotwerte. Zusätzliche Sofortanalysen bleiben weiterhin möglich.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    analysis_mode_default_idx = 0 if st.session_state.analysis_mode == "Einzelanalyse" else 1
    analysis_mode = st.radio(
        "Was möchtest du machen?",
        ["Einzelanalyse", "Mehrere Aktien vergleichen"],
        index=analysis_mode_default_idx,
        horizontal=True,
        key="analysis_mode_widget_main"
    )
    st.session_state.analysis_mode = analysis_mode

    single_input = ""
    batch_input = ""

    if analysis_mode == "Einzelanalyse":
        single_input = st.text_input(
            "Aktie oder Firmenname",
            value=st.session_state.search_input,
            placeholder="z. B. AAPL, Apple, Siemens, BASF",
            key="search_input_widget_main"
        ).strip()
        st.session_state.search_input = single_input
        st.caption("Du kannst einen Ticker oder einfach einen Firmennamen eingeben.")
    else:
        batch_input = st.text_area(
            "Mehrere Ticker oder Firmennamen",
            value=st.session_state.batch_input,
            placeholder="Ein Wert pro Zeile oder durch Komma trennen, z. B.\nAAPL\nMicrosoft\nASML\nNVIDIA",
            height=120,
            key="batch_input_widget_main"
        ).strip()
        st.session_state.batch_input = batch_input
        st.caption("Ein Wert pro Zeile oder mit Komma trennen. Die App löst Firmennamen automatisch auf.")

    with st.expander("Erweiterte Einstellungen", expanded=False):
        horizon = st.selectbox(
            "Zeithorizont",
            [
                "Kurzfrist (1-7 Tage)",
                "Swing (1-4 Wochen)",
                "Mittelfrist (1-3 Monate)",
                "Langfrist (1-2 Jahre)",
                "Sehr langfristig (2+ Jahre)",
            ],
            key="horizon_widget_main"
        )

        adv1, adv2 = st.columns(2)
        with adv1:
            depot = st.number_input("Depotwert EUR", min_value=1000, value=10000, step=1000, key="depot_widget_main")
            override = st.number_input("Kurs-Override (0 = auto)", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="override_widget_main")
            smart_money_default = st.checkbox("TradingBoard: Smart Money = True", value=True, key="smart_money_widget_main")
        with adv2:
            risk_pct = st.slider("Risiko pro Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5, key="risk_pct_widget_main")
            buy_in_override_default = 0.0 if workspace_mode != "Positionen" else 100.0
            buy_in_override = st.number_input("Buy-in für Positionsmodus (0 = Watchlist)", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="buy_in_widget_main")
            strict_mode = st.checkbox("Strenges Mapping", value=True, key="strict_mode_widget_main")

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Cache leeren", use_container_width=True, key="clear_cache_main"):
                st.cache_data.clear()
                st.success("Cache geleert. Bitte Analyse neu starten.")
        with action_col2:
            st.markdown("<div class='mobile-note'>Tippe nur darauf, wenn Daten hängen oder du neue Werte erzwingen willst.</div>", unsafe_allow_html=True)

    position_mode = buy_in_override > 0
    mode_label = "Position" if position_mode else "Watchlist"
    st.caption(f"Aktueller Modus: {mode_label}")
    if st.session_state.get("selected_watchlist_name"):
        st.caption(f"Aktive Watchlist-Auswahl: {st.session_state.get('selected_watchlist_name')}")

    # ---- Eingabe auflösen ----
    search_results = []
    ticker = st.session_state.selected_ticker
    resolved_input_rows = []

    if analysis_mode == "Einzelanalyse":
        search_input = single_input
        if search_input:
            looks_like_ticker = (
                " " not in search_input and len(search_input) <= 12
                and search_input.replace(".", "").replace("-", "").isalnum()
            )

            if looks_like_ticker:
                ticker = search_input.upper()
                st.session_state.selected_ticker = ticker
                st.session_state.selected_search_label = None
                resolved_input_rows = [{"Eingabe": search_input, "Auflösung": ticker, "Typ": "Ticker direkt"}]
            else:
                search_results = search_tickers(search_input)

                if len(search_results) == 1:
                    ticker = search_results[0]["symbol"]
                    st.session_state.selected_ticker = ticker
                    st.session_state.selected_search_label = search_results[0]["label"]
                    resolved_input_rows = [{"Eingabe": search_input, "Auflösung": ticker, "Typ": "Firmenname aufgelöst"}]
                    st.caption(f"Automatisch gefunden: {search_results[0]['label']}")
                elif len(search_results) > 1:
                    labels = [r["label"] for r in search_results]
                    if st.session_state.selected_search_label not in labels:
                        st.session_state.selected_search_label = labels[0]

                    selected_label = st.selectbox(
                        "Passenden Treffer auswählen",
                        options=labels,
                        index=labels.index(st.session_state.selected_search_label),
                        key="search_result_select_main"
                    )

                    st.session_state.selected_search_label = selected_label
                    ticker = next(r["symbol"] for r in search_results if r["label"] == selected_label)
                    st.session_state.selected_ticker = ticker
                    resolved_input_rows = [{"Eingabe": search_input, "Auflösung": ticker, "Typ": "Aus Trefferauswahl"}]
                else:
                    st.warning("Kein passender Ticker gefunden. Bitte Namen präzisieren oder Ticker direkt eingeben.")
                    ticker = st.session_state.selected_ticker
        else:
            ticker = st.session_state.selected_ticker

        analysis_candidates = [ticker] if ticker else []
    else:
        raw_batch_entries = []
        for part in re.split(r"[\n,;]+", batch_input):
            part = str(part).strip()
            if part:
                raw_batch_entries.append(part)

        analysis_candidates = []
        for entry in raw_batch_entries:
            looks_like_ticker = (
                " " not in entry and len(entry) <= 12
                and entry.replace(".", "").replace("-", "").isalnum()
            )
            if looks_like_ticker:
                analysis_candidates.append(entry.upper())
                resolved_input_rows.append({"Eingabe": entry, "Auflösung": entry.upper(), "Typ": "Ticker direkt"})
            else:
                matches = search_tickers(entry, max_results=3)
                if matches:
                    analysis_candidates.append(matches[0]["symbol"])
                    resolved_input_rows.append({"Eingabe": entry, "Auflösung": matches[0]["symbol"], "Typ": "Firmenname aufgelöst"})
                else:
                    resolved_input_rows.append({"Eingabe": entry, "Auflösung": "-", "Typ": "Nicht gefunden"})

        seen = set()
        analysis_candidates = [x for x in analysis_candidates if not (x in seen or seen.add(x))]

    run_analysis_label = "Analyse starten" if workspace_mode == "Sofortanalyse" else ("Zusatzanalyse starten" if workspace_mode == "Positionen" else "Analyse starten / Watchlist ergänzen")
    run_analysis = st.button(run_analysis_label, use_container_width=True, type="primary", key="run_analysis_main")
    if run_analysis:
        if analysis_mode == "Einzelanalyse":
            st.session_state.analysis_ticker = ticker
        st.session_state.analysis_requested = True
        st.session_state.analysis_mode_run = analysis_mode
# ---------- Internal Auto-Run Mode ----------
if st.session_state.get("auto_run_requested", False):
    slot_label = st.session_state.get("auto_run_slot_label", "")
    berlin_now = get_current_berlin_time()
    due_df, due_err = get_due_watchlists_for_slot(slot_label)

    if due_err:
        st.error(f"Auto-Run fehlgeschlagen: {due_err}")
        st.session_state.auto_run_requested = False
        st.stop()

    if due_df is None or due_df.empty:
        st.info(f"Für den Slot {slot_label} sind aktuell keine Watchlisten fällig.")
        st.session_state.auto_run_requested = False
        st.stop()

    st.subheader(f"Auto-Run · Slot {slot_label}")
    st.caption(f"Berlin-Zeit jetzt: {berlin_now.strftime('%d.%m.%Y %H:%M')}")

    auto_run_rows = []
    total_sent = 0

    for _, wl_row in due_df.iterrows():
        wl_name = str(wl_row.get("Watchlist_Name", "")).strip()
        wl_type = str(wl_row.get("Watchlist_Type", "Watchlist")).strip() or "Watchlist"
        wl_alert_mode = str(wl_row.get("Alert_Mode", "Standard")).strip() or "Standard"
        wl_freq = str(wl_row.get("Check_Frequency", "4x täglich")).strip() or "4x täglich"

        tickers, tick_err = get_watchlist_tickers(wl_name)
        if tick_err:
            st.error(f"{wl_name}: {tick_err}")
            auto_run_rows.append({
                "Run_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Berlin_Time": berlin_now.strftime("%Y-%m-%d %H:%M"),
                "Slot": slot_label,
                "Watchlist_Name": wl_name,
                "Watchlist_Type": wl_type,
                "Alert_Mode": wl_alert_mode,
                "Check_Frequency": wl_freq,
                "Ticker_Count": 0,
                "Analyzed_Count": 0,
                "Sent_Count": 0,
                "Status": "Fehler",
                "Message": tick_err,
            })
            continue

        results = []
        analyze_errors = []
        for tkr in tickers:
            try:
                result = analyze_stock(
                    ticker=tkr,
                    horizon="Swing (1-4 Wochen)",
                    depot=10000,
                    risk_pct=1.0,
                    override=0.0,
                    buy_in_override=0.0 if wl_type != "Positions-Watchlist" else 0.0,
                    smart_money_default=True,
                    strict_mode=True
                )
                results.append(result)
            except Exception as e:
                analyze_errors.append(f"{tkr}: {e}")

        ok, msg, sent_count = send_watchlist_alerts(results, wl_name, wl_type, wl_alert_mode) if results else (False, "Keine auswertbaren Ergebnisse", 0)

        if ok:
            st.success(f"{wl_name}: {msg}")
        else:
            if "Keine" in msg or "unterdrückt" in msg:
                st.info(f"{wl_name}: {msg}")
            else:
                st.error(f"{wl_name}: {msg}")

        total_sent += int(sent_count or 0)
        auto_run_rows.append({
            "Run_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Berlin_Time": berlin_now.strftime("%Y-%m-%d %H:%M"),
            "Slot": slot_label,
            "Watchlist_Name": wl_name,
            "Watchlist_Type": wl_type,
            "Alert_Mode": wl_alert_mode,
            "Check_Frequency": wl_freq,
            "Ticker_Count": len(tickers),
            "Analyzed_Count": len(results),
            "Sent_Count": int(sent_count or 0),
            "Status": "OK" if ok else "Info",
            "Message": msg + (f" | Analysefehler: {' ; '.join(analyze_errors[:2])}" if analyze_errors else ""),
        })

    log_ok, log_msg = append_auto_run_log(auto_run_rows)
    if log_ok:
        st.caption(log_msg)
    else:
        st.warning(f"Auto-Run-Logging fehlgeschlagen: {log_msg}")

    if total_sent > 0:
        st.success(f"Auto-Run abgeschlossen. Insgesamt {total_sent} Telegram-Meldungen gesendet.")
    else:
        st.info("Auto-Run abgeschlossen. Es wurden keine neuen Telegram-Meldungen gesendet.")

    st.session_state.auto_run_requested = False
    st.stop()




def style_ranking_table(df):
    if df is None or df.empty:
        return df

    styled = df.style

    def parse_num(v):
        try:
            s = str(v).strip().replace("%", "").replace(",", ".")
            return float(s)
        except Exception:
            return None

    def score_bg(v):
        num = parse_num(v)
        if num is None:
            return ""
        if num >= 85:
            return "background-color: #00c853; color: #ffffff; font-weight: 800;"
        if num >= 75:
            return "background-color: #2eeb70; color: #08130b; font-weight: 800;"
        if num >= 65:
            return "background-color: #b2ff59; color: #18240a; font-weight: 800;"
        if num >= 55:
            return "background-color: #fff176; color: #2b2200; font-weight: 800;"
        if num >= 45:
            return "background-color: #ffd54f; color: #2b1700; font-weight: 800;"
        if num >= 35:
            return "background-color: #ff8a65; color: #2b0f08; font-weight: 800;"
        return "background-color: #ff1744; color: #ffffff; font-weight: 800;"

    score_cols = [c for c in [
        "Investment-Attraktivität",
        "Einstieg jetzt attraktiv?",
        "Trade-Struktur",
        "Kurzfrist-Timing",
        "Setup-Confidence",
        "Exit-Score",
        "Leadership",
        "Sektor-Stärke",
        "Industrie-Stärke",
        "RS-Benchmark-Score",
        "RS-Beschleunigung",
        "Trendqualität",
        "Base-Qualität",
        "Setup-Typ-Qualität",
        "Setup-Priorität",
        "Volumenqualität",
        "Akkumulation",
        "Distribution",
        "Pullback-Dry-up",
        "Breakout-Volumen",
        "Katalysator",
        "Event-Score",
        "Event-Risiko",
        "Institutionelle Qualität",
        "Cashflow-Stabilität",
        "Margenstabilität",
    ] if c in df.columns]

    for col in score_cols:
        styled = styled.map(score_bg, subset=[col])

    def style_valid(v):
        s = str(v).strip().lower()
        if s == "ja":
            return "background-color: #00c853; color: #ffffff; font-weight: 800;"
        if s == "nein":
            return "background-color: #ff1744; color: #ffffff; font-weight: 800;"
        return ""

    def style_priority(v):
        s = str(v).strip().lower()
        if s == "hoch":
            return "background-color: #ff1744; color: #ffffff; font-weight: 800;"
        if s == "mittel":
            return "background-color: #ffd54f; color: #2b1700; font-weight: 800;"
        if s == "niedrig":
            return "background-color: #00c853; color: #ffffff; font-weight: 800;"
        return ""

    def style_action(v):
        s = str(v).strip().lower()
        if any(x in s for x in ["kaufen", "aufbauen", "long", "beobachten"]):
            return "background-color: #00c853; color: #ffffff; font-weight: 800;"
        if any(x in s for x in ["halten", "abwarten"]):
            return "background-color: #ffd54f; color: #2b1700; font-weight: 800;"
        if any(x in s for x in ["reduzieren", "verkaufen", "stop", "risiko"]):
            return "background-color: #ff1744; color: #ffffff; font-weight: 800;"
        return "font-weight: 800;"

    if "Valides Setup" in df.columns:
        styled = styled.map(style_valid, subset=["Valides Setup"])
    if "Watchlist-Priorität" in df.columns:
        styled = styled.map(style_priority, subset=["Watchlist-Priorität"])
    def style_leadership_status(v):
        s = str(v).strip().lower()
        if s == "leader":
            return "background-color: #00c853; color: #ffffff; font-weight: 800;"
        if s == "stark":
            return "background-color: #2eeb70; color: #08130b; font-weight: 800;"
        if s == "solide":
            return "background-color: #ffd54f; color: #2b1700; font-weight: 800;"
        if s == "mitläufer":
            return "background-color: #ff8a65; color: #2b0f08; font-weight: 800;"
        if s == "schwach":
            return "background-color: #ff1744; color: #ffffff; font-weight: 800;"
        return ""

    if "Handlung" in df.columns:
        styled = styled.map(style_action, subset=["Handlung"])
    if "Exit-Aktion" in df.columns:
        styled = styled.map(style_action, subset=["Exit-Aktion"])
    if "Leadership-Status" in df.columns:
        styled = styled.map(style_leadership_status, subset=["Leadership-Status"])

    return styled


# ---------- Batch / Ranking Run ----------
has_cached_results = (
    "ranking_df" in st.session_state
    and st.session_state.ranking_df is not None
    and not st.session_state.ranking_df.empty
    and "ranking_results" in st.session_state
    and isinstance(st.session_state.ranking_results, dict)
    and len(st.session_state.ranking_results) > 0
)

if not st.session_state.get("analysis_requested", False) and not has_cached_results:
    st.stop()

analysis_mode_run = st.session_state.get("analysis_mode_run", "Einzelanalyse")
errors = []
resolved_input_rows = []

if st.session_state.get("analysis_requested", False):
    if analysis_mode_run == "Einzelanalyse":
        entries = [st.session_state.analysis_ticker]
    else:
        entries = split_batch_input(st.session_state.batch_input)

    if not entries:
        entries = [st.session_state.analysis_ticker]

    resolved_entries = []
    resolved_input_rows = []
    for e in entries:
        fallback = st.session_state.analysis_ticker if analysis_mode_run == "Einzelanalyse" else None
        resolved = resolve_input_to_ticker(e, fallback=fallback)
        resolved_input_rows.append({
            "Eingabe": e,
            "Aufgelöst zu": resolved if resolved else "Nicht gefunden"
        })
        if resolved and resolved not in resolved_entries:
            resolved_entries.append(resolved)

    results = []
    errors = []

    st.session_state.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    progress = st.progress(0)
    status = st.empty()

    for i, tkr in enumerate(resolved_entries, start=1):
        status.info(f"Analysiere {tkr} ({i}/{len(resolved_entries)}) …")
        try:
            result = analyze_stock(
                ticker=tkr,
                horizon=horizon,
                depot=depot,
                risk_pct=risk_pct,
                override=override if analysis_mode_run == "Einzelanalyse" and len(resolved_entries) == 1 else 0.0,
                buy_in_override=buy_in_override if analysis_mode_run == "Einzelanalyse" and len(resolved_entries) == 1 else 0.0,
                smart_money_default=smart_money_default,
                strict_mode=strict_mode
            )
            results.append(result)
        except Exception as e:
            errors.append((tkr, str(e)))
        progress.progress(i / len(resolved_entries))

    progress.empty()
    status.empty()
    st.session_state.analysis_requested = False

    if not results:
        st.error("Keine belastbaren Ergebnisse. Prüfe die Eingaben.")
        if errors:
            st.write(pd.DataFrame(errors, columns=["Ticker", "Fehler"]))
        st.stop()

    if st.session_state.get("send_watchlist_alerts_after_run", False):
        ok, msg, sent_count = send_watchlist_alerts(
            results,
            st.session_state.get("run_selected_watchlist_name", "Watchlist"),
            st.session_state.get("run_selected_watchlist_type", "Watchlist"),
            st.session_state.get("selected_watchlist_alert_mode", "Standard"),
        )
        if ok:
            st.success(msg)
        else:
            if "Keine alert-relevanten Werte" in msg or "unterdrückt" in msg:
                st.info(msg)
            else:
                st.error(msg)
        st.session_state.send_watchlist_alerts_after_run = False

    ranking_df = build_ranking_table(results)
    if not ranking_df.empty:
        exit_score_map = {str(r.get("ticker", "")): r.get("exit_score", np.nan) for r in results}
        exit_action_map = {str(r.get("ticker", "")): r.get("exit_action", "-") for r in results}
        leadership_map = {str(r.get("ticker", "")): r.get("leadership_score", np.nan) for r in results}
        leadership_status_map = {str(r.get("ticker", "")): r.get("leadership_status", "-") for r in results}
        sector_strength_map = {str(r.get("ticker", "")): r.get("sector_strength_score", np.nan) for r in results}
        industry_strength_map = {str(r.get("ticker", "")): r.get("industry_strength_score", np.nan) for r in results}
        rs_benchmark_map = {str(r.get("ticker", "")): r.get("rs_benchmark_score", np.nan) for r in results}
        rs_accel_map = {str(r.get("ticker", "")): r.get("rs_acceleration_score", np.nan) for r in results}
        trend_quality_map = {str(r.get("ticker", "")): r.get("trend_quality_score", np.nan) for r in results}
        base_quality_map = {str(r.get("ticker", "")): r.get("base_quality_score", np.nan) for r in results}
        setup_type_quality_map = {str(r.get("ticker", "")): r.get("setup_type_quality_score", np.nan) for r in results}
        setup_priority_map = {str(r.get("ticker", "")): r.get("setup_priority_score", np.nan) for r in results}

        ranking_df["Exit-Score"] = ranking_df["Ticker"].astype(str).map(exit_score_map)
        ranking_df["Exit-Aktion"] = ranking_df["Ticker"].astype(str).map(exit_action_map)
        ranking_df["Leadership"] = ranking_df["Ticker"].astype(str).map(leadership_map)
        ranking_df["Leadership-Status"] = ranking_df["Ticker"].astype(str).map(leadership_status_map)
        ranking_df["Sektor-Stärke"] = ranking_df["Ticker"].astype(str).map(sector_strength_map)
        ranking_df["Industrie-Stärke"] = ranking_df["Ticker"].astype(str).map(industry_strength_map)
        ranking_df["RS-Benchmark-Score"] = ranking_df["Ticker"].astype(str).map(rs_benchmark_map)
        ranking_df["RS-Beschleunigung"] = ranking_df["Ticker"].astype(str).map(rs_accel_map)
        ranking_df["Trendqualität"] = ranking_df["Ticker"].astype(str).map(trend_quality_map)
        ranking_df["Base-Qualität"] = ranking_df["Ticker"].astype(str).map(base_quality_map)
        ranking_df["Setup-Typ-Qualität"] = ranking_df["Ticker"].astype(str).map(setup_type_quality_map)
        ranking_df["Setup-Priorität"] = ranking_df["Ticker"].astype(str).map(setup_priority_map)
        volume_quality_map = {str(r.get("ticker", "")): r.get("volume_quality_score", np.nan) for r in results}
        accumulation_map = {str(r.get("ticker", "")): r.get("accumulation_score", np.nan) for r in results}
        distribution_map = {str(r.get("ticker", "")): r.get("distribution_pressure_score", np.nan) for r in results}
        ranking_df["Volumenqualität"] = ranking_df["Ticker"].astype(str).map(volume_quality_map)
        ranking_df["Akkumulation"] = ranking_df["Ticker"].astype(str).map(accumulation_map)
        ranking_df["Distribution"] = ranking_df["Ticker"].astype(str).map(distribution_map)
        catalyst_map = {str(r.get("ticker", "")): r.get("catalyst_score", np.nan) for r in results}
        event_risk_map = {str(r.get("ticker", "")): r.get("event_risk_score", np.nan) for r in results}
        event_score_map = {str(r.get("ticker", "")): r.get("earnings_event_score", np.nan) for r in results}
        ranking_df["Katalysator"] = ranking_df["Ticker"].astype(str).map(catalyst_map)
        ranking_df["Event-Risiko"] = ranking_df["Ticker"].astype(str).map(event_risk_map)
        ranking_df["Event-Score"] = ranking_df["Ticker"].astype(str).map(event_score_map)
        institutional_quality_map = {str(r.get("ticker", "")): r.get("institutional_quality_score", np.nan) for r in results}
        cashflow_stability_map = {str(r.get("ticker", "")): r.get("cashflow_stability_score", np.nan) for r in results}
        margin_stability_map = {str(r.get("ticker", "")): r.get("margin_stability_score", np.nan) for r in results}
        ranking_df["Institutionelle Qualität"] = ranking_df["Ticker"].astype(str).map(institutional_quality_map)
        ranking_df["Cashflow-Stabilität"] = ranking_df["Ticker"].astype(str).map(cashflow_stability_map)
        ranking_df["Margenstabilität"] = ranking_df["Ticker"].astype(str).map(margin_stability_map)
    results_map = {r["ticker"]: r for r in results}
    st.session_state.ranking_df = ranking_df
    st.session_state.ranking_results = results_map
    st.session_state.last_analysis_errors = errors
    st.session_state.last_resolved_input_rows = resolved_input_rows
else:
    ranking_df = st.session_state.ranking_df.copy()
    results_map = st.session_state.ranking_results.copy()
    results = list(results_map.values())
    errors = st.session_state.get("last_analysis_errors", [])
    resolved_input_rows = st.session_state.get("last_resolved_input_rows", [])
    if not ranking_df.empty and results_map:
        exit_score_map = {str(k): v.get("exit_score", np.nan) for k, v in results_map.items()}
        exit_action_map = {str(k): v.get("exit_action", "-") for k, v in results_map.items()}
        leadership_map = {str(k): v.get("leadership_score", np.nan) for k, v in results_map.items()}
        leadership_status_map = {str(k): v.get("leadership_status", "-") for k, v in results_map.items()}
        sector_strength_map = {str(k): v.get("sector_strength_score", np.nan) for k, v in results_map.items()}
        industry_strength_map = {str(k): v.get("industry_strength_score", np.nan) for k, v in results_map.items()}
        rs_benchmark_map = {str(k): v.get("rs_benchmark_score", np.nan) for k, v in results_map.items()}
        rs_accel_map = {str(k): v.get("rs_acceleration_score", np.nan) for k, v in results_map.items()}
        ranking_df["Exit-Score"] = ranking_df["Ticker"].astype(str).map(exit_score_map)
        ranking_df["Exit-Aktion"] = ranking_df["Ticker"].astype(str).map(exit_action_map)
        ranking_df["Leadership"] = ranking_df["Ticker"].astype(str).map(leadership_map)
        ranking_df["Leadership-Status"] = ranking_df["Ticker"].astype(str).map(leadership_status_map)
        ranking_df["Sektor-Stärke"] = ranking_df["Ticker"].astype(str).map(sector_strength_map)
        ranking_df["Industrie-Stärke"] = ranking_df["Ticker"].astype(str).map(industry_strength_map)
        ranking_df["RS-Benchmark-Score"] = ranking_df["Ticker"].astype(str).map(rs_benchmark_map)
        ranking_df["RS-Beschleunigung"] = ranking_df["Ticker"].astype(str).map(rs_accel_map)

if analysis_mode_run == "Einzelanalyse":
    st.caption("Aktiver Modus: Einzelanalyse")
else:
    st.caption("Aktiver Modus: Mehrfach-Ranking")

if st.session_state.get("run_selected_watchlist_name"):
    st.caption(
        f"Aktiver Watchlist-Run: {st.session_state.get('run_selected_watchlist_name')} "
        f"({st.session_state.get('run_selected_watchlist_type', 'Watchlist')})"
    )

if st.session_state.selected_ranking_ticker not in results_map and not ranking_df.empty:
    st.session_state.selected_ranking_ticker = ranking_df.iloc[0]["Ticker"]

# Prefer explicit single-analysis ticker if present in results
if st.session_state.analysis_ticker in results_map:
    selected_display_ticker = st.session_state.analysis_ticker
else:
    selected_display_ticker = st.session_state.selected_ranking_ticker

if selected_display_ticker not in results_map and not ranking_df.empty:
    selected_display_ticker = ranking_df.iloc[0]["Ticker"]

try:
    if "selected_display_ticker" in locals():
        st.session_state.selected_ranking_ticker = selected_display_ticker
except Exception:
    pass

# ---------- Ranking Section ----------
try:
    ranking_expanded_default = len(ranking_df) > 1
except Exception:
    ranking_expanded_default = False

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
with st.expander("Ranking & Auswahl", expanded=ranking_expanded_default):
    st.subheader("Ranking mehrerer Aktien")
    st.caption(
        "Ranking, Filter und Auswahl der Detailanalyse. Bei Einzelwerten meist nur bei Bedarf öffnen."
    )

    ranking_focus = st.radio(
        "Ranking-Modus",
        ["Investment-Ranking", "Trading-Ranking", "Watchlist-Ranking"],
        horizontal=True,
        key="ranking_focus_mode"
    )

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        only_valid_setup = st.checkbox("Nur valide Setups", value=False, key="ranking_only_valid")
    with filter_col2:
        min_investment_attr = st.slider("Min. Investment-Attraktivität", 0, 100, 0, 5, key="ranking_min_invest")
    with filter_col3:
        min_entry_attr = st.slider("Min. Einstieg jetzt attraktiv?", 0, 100, 0, 5, key="ranking_min_entry")
    with filter_col4:
        available_setup_types = sorted([str(x) for x in ranking_df.get("Setup-Typ", pd.Series(dtype=str)).dropna().unique().tolist()]) if not ranking_df.empty and "Setup-Typ" in ranking_df.columns else []
        selected_setup_types = st.multiselect(
            "Setup-Typen filtern",
            options=available_setup_types,
            default=available_setup_types,
            key="ranking_setup_filter"
        )

    ranking_df = ranking_df.copy()

    if only_valid_setup and "Valides Setup" in ranking_df.columns:
        ranking_df = ranking_df[ranking_df["Valides Setup"] == "Ja"]

    if "Investment-Attraktivität" in ranking_df.columns:
        ranking_df = ranking_df[pd.to_numeric(ranking_df["Investment-Attraktivität"], errors="coerce").fillna(0) >= min_investment_attr]

    if "Einstieg jetzt attraktiv?" in ranking_df.columns:
        ranking_df = ranking_df[pd.to_numeric(ranking_df["Einstieg jetzt attraktiv?"], errors="coerce").fillna(0) >= min_entry_attr]

    if selected_setup_types and "Setup-Typ" in ranking_df.columns:
        ranking_df = ranking_df[ranking_df["Setup-Typ"].isin(selected_setup_types)]

    if ranking_focus == "Trading-Ranking":
        trading_sort_cols = [c for c in ["Einstieg jetzt attraktiv?", "Trade-Struktur", "Kurzfrist-Timing", "Setup-Confidence"] if c in ranking_df.columns]
        if trading_sort_cols:
            ranking_df = ranking_df.sort_values(
                by=trading_sort_cols,
                ascending=[False] * len(trading_sort_cols)
            ).reset_index(drop=True)
    elif ranking_focus == "Watchlist-Ranking":
        if "Watchlist-Priorität" in ranking_df.columns:
            priority_map = {"Hoch": 3, "Mittel": 2, "Niedrig": 1}
            ranking_df["_watchlist_sort"] = ranking_df["Watchlist-Priorität"].map(priority_map).fillna(0)
        else:
            ranking_df["_watchlist_sort"] = 0
        watchlist_sort_cols = [c for c in ["_watchlist_sort", "Einstieg jetzt attraktiv?", "Trade-Struktur", "Investment-Attraktivität"] if c in ranking_df.columns]
        if watchlist_sort_cols:
            ranking_df = ranking_df.sort_values(
                by=watchlist_sort_cols,
                ascending=[False] * len(watchlist_sort_cols)
            ).reset_index(drop=True)
        ranking_df = ranking_df.drop(columns=["_watchlist_sort"], errors="ignore")
    else:
        investment_sort_cols = [c for c in ["Investment-Attraktivität", "Unternehmen", "Sicherheit"] if c in ranking_df.columns]
        if investment_sort_cols:
            ranking_df = ranking_df.sort_values(
                by=investment_sort_cols,
                ascending=[False] * len(investment_sort_cols)
            ).reset_index(drop=True)

    if ranking_df.empty:
        st.markdown(
            '<div class="empty-state"><div class="empty-state-title">Keine Werte nach aktuellem Filter</div><div class="empty-state-text">Reduziere einzelne Filter oder öffne den Ranking-Block erneut, um mehr Ergebnisse sichtbar zu machen.</div></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    ranking_cols = [
        c for c in [
            "Ticker", "Name", "Investment-Attraktivität", "Einstieg jetzt attraktiv?",
            "Leadership", "Leadership-Status", "Sektor-Stärke",
            "Trendqualität", "Base-Qualität", "Setup-Priorität",
            "Volumenqualität",
            "Katalysator",
            "Institutionelle Qualität",
            "Event-Risiko",
            "Distribution",
            "Exit-Score", "Exit-Aktion",
            "Trade-Struktur", "Kurzfrist-Timing", "Setup-Confidence",
            "Valides Setup", "Setup-Typ", "Watchlist-Priorität", "Handlung"
        ] if c in ranking_df.columns
    ]
    ranking_display_df = ranking_df[ranking_cols].copy()
    st.dataframe(
        style_ranking_table(ranking_display_df),
        hide_index=True,
        use_container_width=True,
        height=min(420, 45 * len(ranking_display_df) + 40)
    )

    try:
        resolved_df = pd.DataFrame(resolved_input_rows)
        if not resolved_df.empty:
            with st.expander("Aufgelöste Eingaben", expanded=False):
                st.dataframe(resolved_df, hide_index=True, use_container_width=True)
    except Exception:
        pass

    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        ranking_ticker_options = ranking_df["Ticker"].tolist()
        ranking_select_index = 0
        if ranking_ticker_options and selected_display_ticker in ranking_ticker_options:
            ranking_select_index = ranking_ticker_options.index(selected_display_ticker)
        selected_display_ticker = st.selectbox(
            "Einzelanalyse aus Ranking auswählen",
            options=ranking_ticker_options,
            index=ranking_select_index,
            key="selected_ranking_ticker_widget"
        )
        st.session_state.selected_ranking_ticker = selected_display_ticker

    with sel_col2:
        st.metric("Analysierte Werte", len(results))

    if errors:
        with st.expander("Nicht analysierbare Eingaben", expanded=False):
            st.dataframe(pd.DataFrame(errors, columns=["Ticker", "Fehler"]), hide_index=True, use_container_width=True)

try:
    result = results_map[selected_display_ticker]
except Exception:
    result = None
try:
    single_export_df = build_export_df([result]) if result is not None else pd.DataFrame()
except Exception:
    single_export_df = pd.DataFrame()

sheet_log_triggered = False
try:
    qp = st.query_params
    if str(qp.get("sheet_log", "0")) == "1":
        sheet_log_triggered = True
except Exception:
    sheet_log_triggered = False

if result is not None:
    ticker = result["ticker"]
    csv_filename = f"capital_hill_single_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_payload = single_export_df.to_csv(index=False).encode("utf-8-sig")
    csv_b64 = base64.b64encode(csv_payload).decode("utf-8")
    csv_href = f"data:text/csv;base64,{csv_b64}"

    if sheet_log_triggered:
        ok, msg = append_df_to_gsheet(single_export_df, worksheet_name="Analysis_Log")
        show_sheet_result(ok, msg)
        try:
            st.query_params.clear()
        except Exception:
            pass

    df = result["df"]
    info = result["info"]
    name = result["name"]
    ccy = result["ccy"]
    exch = result["exch"]
    ts = result["ts"]
    sector = result["sector"]
    industry = result["industry"]
    company_summary = result["company_summary"]
    confidence_info = result["confidence_info"]
    benchmark_label = result["benchmark_label"]
    benchmark_symbol = result["benchmark_symbol"]
    market_info = result["market_info"]
    price = result["price"]
    target = result["target"]
    upside = result["upside"]
    regime = result["regime"]
    reg_amp = result["reg_amp"]
    sg_earn = result["sg_earn"]
    sg_earn_txt = result["sg_earn_txt"]
    days_earn = result["days_earn"]
    has_upcoming_earnings = result["has_upcoming_earnings"]
    has_past_earnings = result["has_past_earnings"]
    fund_cov = result["fund_cov"]
    fund_fields_loaded = result["fund_fields_loaded"]
    fund_data_warning = result["fund_data_warning"]
    red_flag_items = result["red_flag_items"]
    red_flags_df = result["red_flags_df"]
    red_flag_notes = result["red_flag_notes"]
    red_flag_penalty_total = result["red_flag_penalty_total"]
    quality_score = result["quality_score"]
    growth_score = result["growth_score"]
    growth_quality = result["growth_quality"]
    valuation_score = result["valuation_score"]
    balance_score = result["balance_score"]
    sentiment_score = result["sentiment_score"]
    risk_score = result["risk_score"]
    company = result["company"]
    setup_adj = result["setup_adj"]
    investment = result["investment"]
    tb_score = result["tb_score"]
    tb_score_100 = result["tb_score_100"]
    tb_timing_text = result["tb_timing_text"]
    position_action = result["position_action"]
    add_on_action = result["add_on_action"]
    partial_profit_action = result["partial_profit_action"]
    stop_action = result["stop_action"]
    risk_note = result["risk_note"]
    trigger_status = result["trigger_status"]
    watchlist_priority = result["watchlist_priority"]
    watchlist_priority_score = result["watchlist_priority_score"]
    next_trigger = result["next_trigger"]
    trigger_reason = result["trigger_reason"]
    tb_signal = result["tb_signal"]
    tb_empf = result["tb_empf"]
    tb_df = result["tb_df"]
    tb_details = result["tb_details"]
    tb_context = result["tb_context"]
    stb_score = result["stb_score"]
    stb_signal = result["stb_signal"]
    stb_empf = result["stb_empf"]
    stb_text = result["stb_text"]
    kb = result["kb"]
    strengths = result["strengths"]
    weaknesses = result["weaknesses"]
    decision_summary = result["decision_summary"]
    mode_label = result["mode_label"]
    stock_style = result["stock_style"]
    hmap = result["hmap"]
    atr_stop = result["atr_stop"]
    stop_used = result["stop_used"]
    stop_dist = result["stop_dist"]
    tp1 = result["tp1"]
    tp2 = result["tp2"]
    tp3 = result["tp3"]
    tp1_source = result["tp1_source"]
    tp2_source = result["tp2_source"]
    tp3_source = result["tp3_source"]
    technical_target_1 = result["technical_target_1"]
    technical_target_2 = result["technical_target_2"]
    stop_source = result["stop_source"]
    suggested_entry_zone = result["suggested_entry_zone"]
    entry_source = result["entry_source"]
    entry_quality = result["entry_quality"]
    tradeability_score = result["tradeability_score"]
    tradeability_text = result["tradeability_text"]
    investment_case_score = result["investment_case_score"]
    investment_case_text = result["investment_case_text"]
    trading_case_score = result["trading_case_score"]
    trading_case_text = result["trading_case_text"]
    setup_type = result["setup_type"]
    preferred_entry = result["preferred_entry"]
    setup_confidence = result["setup_confidence"]
    setup_confidence_text = result["setup_confidence_text"]
    trade_crv_score = result["trade_crv_score"]
    trade_stop_score = result["trade_stop_score"]
    trade_entry_score = result["trade_entry_score"]
    trade_timing_score = result["trade_timing_score"]
    trade_market_score = result["trade_market_score"]
    crv = result["crv"]
    pos_size = result["pos_size"]
    risk_eur = result["risk_eur"]
    time_stop = result["time_stop"]
    valid_trade_setup = result["valid_trade_setup"]
    short_term_score = result["short_term_score"]
    s3 = result["s3"]
    s3a = result["s3a"]
    s3t = result["s3t"]
    s4 = result["s4"]
    s4a = result["s4a"]
    s4t = result["s4t"]
    s5 = result["s5"]
    s5a = result["s5a"]
    s5t = result["s5t"]
    s6 = result["s6"]
    s6a = result["s6a"]
    s6t = result["s6t"]
    w52 = result["w52"]
    dist52 = result["dist52"]
    rs_score = result["rs_score"]
    rs_composite = result["rs_composite"]
    ret21 = result["ret21"]
    ret63 = result["ret63"]
    ret126 = result["ret126"]
    bench_ret21 = result["bench_ret21"]
    bench_ret63 = result["bench_ret63"]
    bench_ret126 = result["bench_ret126"]
    rs_vs_benchmark_21 = result["rs_vs_benchmark_21"]
    rs_vs_benchmark_63 = result["rs_vs_benchmark_63"]
    rs_vs_benchmark_126 = result["rs_vs_benchmark_126"]
    ma20 = result["ma20"]
    ma50 = result["ma50"]
    ma150 = result["ma150"]
    ma200 = result["ma200"]
    rsi = result["rsi"]
    macd_v = result["macd_v"]
    signal_v = result["signal_v"]
    macd_hist_current = result["macd_hist_current"]
    adx = result["adx"]
    atr = result["atr"]
    atr_pct = result["atr_pct"]
    stoch_k_v = result["stoch_k_v"]
    stoch_d_v = result["stoch_d_v"]
    willr_v = result["willr_v"]
    roc20 = result["roc20"]
    roc60 = result["roc60"]
    high52 = result["high52"]
    low52 = result["low52"]
    profit_margin = result["profit_margin"]
    oper_margin = result["oper_margin"]
    gross_margin = result["gross_margin"]
    roe = result["roe"]
    revenue_growth = result["revenue_growth"]
    earnings_growth = result["earnings_growth"]
    current_ratio = result["current_ratio"]
    quick_ratio = result["quick_ratio"]
    debt_to_equity = result["debt_to_equity"]
    pe = result["pe"]
    peg = result["peg"]
    ps = result["ps"]
    pb = result["pb"]
    rec_label = result["rec_label"]
    analysts = result["analysts"]
    rec_mean = result["rec_mean"]
    beta = result["beta"]
    short_pct = result["short_pct"]
    market_cap = result["market_cap"]
    short_thesis = result["short_thesis"]
    top_red_flag = result["top_red_flag"]

    # ---------- Header ----------
    st.markdown(f"## {name} `{ticker}` — {exch} ({ccy})")
    st.markdown(
        f"<div class='small-note'>Sektor: {sector} | Industrie: {industry} | Stil: {stock_style} | "
        f"Kontext: {display_mode_label(mode_label)} | Benchmark: {benchmark_label} | Marktumfeld: {market_regime_label(market_info['regime'])} | "
        f"Top Red Flag: {top_red_flag}</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="premium-card">
                <div class="premium-title">Kurs (Adj. Close)</div>
                <div class="premium-value">{price:.2f} {ccy}</div>
                <div class="premium-sub">{ts}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="premium-card">
                <div class="premium-title">Trend-Regime</div>
                <div class="premium-value">{regime}</div>
                <div class="premium-sub">{reg_amp}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="premium-card">
                <div class="premium-title">Earnings-Datum</div>
                <div class="premium-value">{sg_earn_txt}</div>
                <div class="premium-sub">{sg_earn}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        if has_upcoming_earnings:
            countdown_value = f"{int(days_earn)}d"
        elif has_past_earnings:
            countdown_value = "vorbei"
        else:
            countdown_value = "kein Datum"
        st.markdown(
            f"""
            <div class="premium-card">
                <div class="premium-title">Earnings-Countdown</div>
                <div class="premium-value">{countdown_value}</div>
                <div class="premium-sub">{sg_earn}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="premium-card">
                <div class="premium-title">Analysten-Target</div>
                <div class="premium-value">{fmt_num(target, 2, f" {ccy}")}</div>
                <div class="premium-sub">{fmt_num(upside, 1, "%")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    if fund_data_warning:
        st.warning(
            f"Fundamentaldaten nur teilweise geladen ({fund_cov*100:.0f}% Abdeckung, {fund_fields_loaded}/21 Felder). "
            "Der Company Quality Score kann dadurch zu niedrig ausfallen."
        )
    elif fund_cov < 0.55:
        st.info(
            f"Fundamentaldaten teilweise vorhanden ({fund_cov*100:.0f}% Abdeckung, {fund_fields_loaded}/21 Felder)."
        )

    if red_flag_items:
        st.warning("Red Flags erkannt: " + " | ".join(red_flag_notes[:4]))

    # ---------- Scores ----------
    # ---------- v10.0B Overview ----------

    def ui_chip_class_from_score(score):
        try:
            value = float(score)
        except Exception:
            value = None
        if value is None:
            return "blue"
        if value >= 75:
            return "green"
        if value >= 55:
            return "amber"
        return "red"


    def ui_action_chip_class(action_text):
        s = str(action_text).strip().lower()
        if any(x in s for x in ["kauf", "aufbau", "nachkauf", "long"]):
            return "green"
        if any(x in s for x in ["verkauf", "reduzier", "risiko", "stop"]):
            return "red"
        if any(x in s for x in ["abwarten", "beobacht", "halten"]):
            return "amber"
        return "purple"


    def ui_priority_chip_class(priority_text):
        s = str(priority_text).strip().lower()
        if s == "hoch":
            return "green"
        if s == "mittel":
            return "amber"
        if s == "niedrig":
            return "blue"
        return "blue"


    def ui_safe_metric_text(value, digits=1, suffix=""):
        try:
            if value in [None, "", "-"]:
                return "n/a"
            return fmt_num(float(value), digits, suffix)
        except Exception:
            try:
                return fmt_num(value, digits, suffix)
            except Exception:
                return str(value) if str(value).strip() else "n/a"



    def ui_target_text(value, ccy="", missing_text="kein sauberes Setup-Ziel ableitbar"):
        try:
            if value is None or (isinstance(value, str) and value.strip().lower() in {"", "-", "n/a", "none"}):
                return missing_text
            if pd.isna(value):
                return missing_text
            return f"{float(value):.2f} {ccy}".strip()
        except Exception:
            s = str(value).strip()
            return s if s else missing_text



    def exit_score_label(score):
        if score >= 80:
            return "klarer Exit-Druck"
        if score >= 65:
            return "Verkaufsdruck erhöht"
        if score >= 45:
            return "Gewinne absichern"
        if score >= 25:
            return "erste Schwäche"
        return "stabil"


    def derive_exit_action(exit_score, position_pnl_pct, price, stop_used):
        try:
            if pd.notna(stop_used) and pd.notna(price) and price < stop_used:
                return "Verkaufen"
        except Exception:
            pass
        if exit_score >= 80:
            return "Verkaufen"
        if exit_score >= 65:
            return "Risiko reduzieren"
        if exit_score >= 45:
            try:
                if pd.notna(position_pnl_pct) and position_pnl_pct > 8:
                    return "Teilgewinn prüfen"
            except Exception:
                pass
            return "Risiko reduzieren"
        if exit_score >= 25:
            return "Beobachten"
        return "Halten"



    def derive_position_context(position_pnl_pct, horizon):
        if pd.notna(position_pnl_pct):
            if position_pnl_pct >= 15:
                pnl_bucket = "starker Gewinner"
            elif position_pnl_pct >= 5:
                pnl_bucket = "Gewinner"
            elif position_pnl_pct <= -8:
                pnl_bucket = "klarer Verlierer"
            elif position_pnl_pct < 0:
                pnl_bucket = "leichter Verlierer"
            else:
                pnl_bucket = "nahe Einstand"
        else:
            pnl_bucket = "ohne Einstandsdaten"

        horizon_label = str(horizon or "").strip() or "unbekannt"
        return pnl_bucket, horizon_label


    def combine_position_action(exit_action, legacy_position_action, add_on_action, partial_profit_action, position_pnl_pct):
        if exit_action in {"Verkaufen", "Risiko reduzieren"}:
            return exit_action
        if exit_action == "Teilgewinn prüfen":
            return "Teilgewinn prüfen"
        if exit_action == "Beobachten":
            return "Halten / eng beobachten"

        if str(add_on_action).lower().startswith("ja"):
            return "Halten / ggf. ausbauen"
        if str(partial_profit_action).lower().startswith("ja") and pd.notna(position_pnl_pct) and position_pnl_pct > 10:
            return "Teilgewinn prüfen"
        return legacy_position_action




    main_action_label = position_action if position_mode else display_emp_label(result.get("emp", "-"))
    top_strengths = strengths[:3] if strengths else []
    top_weaknesses = weaknesses[:3] if weaknesses else []

    action_chip_class = ui_action_chip_class(main_action_label)
    entry_chip_class = ui_chip_class_from_score(trading_case_score)
    investment_chip_class = ui_chip_class_from_score(investment_case_score)
    priority_chip_class = ui_priority_chip_class(watchlist_priority)
    crv_value = result.get("crv", rr if "rr" in locals() else result.get("rr", "-"))
    trigger_label = trigger_status if trigger_status not in ["", None] else entry_quality
    exit_score_display = result.get("exit_score", 0)
    exit_action_display = result.get("exit_action", "Halten")
    exit_chip_class = ui_chip_class_from_score(100 - float(exit_score_display)) if str(exit_score_display).strip() not in {"", "-", "n/a"} else "blue"
    exit_score_text_display = result.get("exit_score_text", "stabil")
    exit_reason_top_display = result.get("exit_reason_top", "kein akuter Exit-Grund")
    exit_reason_list_display = result.get("exit_reason_list", []) or []
    exit_reason_extra_display = [x for x in exit_reason_list_display if str(x).strip() and str(x).strip() != str(exit_reason_top_display).strip()]
    leadership_score_display = result.get("leadership_score", np.nan)
    leadership_status_display = result.get("leadership_status", "-")
    sector_strength_display = result.get("sector_strength_score", np.nan)
    industry_strength_display = result.get("industry_strength_score", np.nan)
    rs_acceleration_display = result.get("rs_acceleration_score", np.nan)
    sector_label_display = result.get("sector_label", sector if 'sector' in locals() else "-")
    industry_label_display = result.get("industry_label", industry if 'industry' in locals() else "-")
    sector_trend_text_display = result.get("sector_trend_text", "nicht belastbar")
    industry_trend_text_display = result.get("industry_trend_text", "nicht belastbar")
    sector_strength_text_display = score_or_unavailable_text(sector_strength_display)
    if str(sector_strength_text_display).strip().lower() == "nicht verfügbar":
        sector_strength_text_display = sector_trend_text_display if str(sector_trend_text_display).strip() not in {"", "-", "nicht belastbar"} else "aktuell keine saubere Sektor-Einordnung möglich"
    sector_etf_display = result.get("sector_etf_symbol", "-")
    sector_delta_display = f"{sector_label_display} | ETF: {sector_etf_display}" if sector_etf_display not in ["", "-", None] else sector_label_display
    trend_quality_display = result.get("trend_quality_score", np.nan)
    base_quality_display = result.get("base_quality_score", np.nan)
    setup_type_quality_display = result.get("setup_type_quality_score", np.nan)
    setup_priority_display = result.get("setup_priority_score", np.nan)
    base_length_display = result.get("base_length_days", np.nan)
    correction_depth_display = result.get("correction_depth_pct", np.nan)
    range_tightness_display = result.get("range_tightness_score", np.nan)
    volatility_contraction_display = result.get("volatility_contraction_score", np.nan)
    pullback_quality_display = result.get("pullback_quality_score", np.nan)
    volume_quality_display = result.get("volume_quality_score", np.nan)
    accumulation_display = result.get("accumulation_score", np.nan)
    distribution_display = result.get("distribution_pressure_score", np.nan)
    pullback_dryup_display = result.get("pullback_dryup_score", np.nan)
    breakout_volume_display = result.get("breakout_volume_score", np.nan)
    up_down_volume_ratio_display = result.get("up_down_volume_ratio", np.nan)
    volume_trend_display = result.get("volume_trend_score", np.nan)
    accumulation_day_count_display = result.get("accumulation_day_count", np.nan)
    distribution_day_count_display = result.get("distribution_day_count", np.nan)
    catalyst_score_display = result.get("catalyst_score", np.nan)
    earnings_event_score_display = result.get("earnings_event_score", np.nan)
    post_earnings_reaction_score_display = result.get("post_earnings_reaction_score", np.nan)
    revision_momentum_score_display = result.get("revision_momentum_score", np.nan)
    event_risk_score_display = result.get("event_risk_score", np.nan)
    catalyst_text_display = result.get("catalyst_text", "-")
    event_phase_label_display = result.get("event_phase_label", "kein Eventfenster")
    earnings_reaction_5d_display = result.get("earnings_reaction_5d", np.nan)
    earnings_reaction_10d_display = result.get("earnings_reaction_10d", np.nan)
    cashflow_stability_display = result.get("cashflow_stability_score", np.nan)
    margin_stability_display = result.get("margin_stability_score", np.nan)
    institutional_quality_display = result.get("institutional_quality_score", np.nan)
    institutional_quality_text_display = result.get("institutional_quality_text", "-")


    st.markdown("""
    <style>
    .reason-box{
        border:1px solid rgba(148,163,184,0.18);
        border-radius:18px;
        padding:14px 16px;
        background:rgba(15,23,42,0.55);
        margin:8px 0 12px 0;
    }
    .reason-title{font-size:1.02rem;font-weight:800;margin-bottom:8px;}
    .reason-item{padding:8px 0;border-top:1px solid rgba(148,163,184,0.10);}
    .reason-item:first-child{border-top:none;}
    .reason-top{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;}
    .reason-label{font-weight:700;}
    .reason-value{font-weight:800;}
    .reason-meta{margin-top:4px;font-size:0.88rem;color:#cbd5e1;}
    .diag-row{border:1px solid rgba(148,163,184,0.12);border-radius:16px;padding:10px 12px;margin:8px 0;background:rgba(15,23,42,0.42);}
    .diag-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;}
    .diag-label{font-weight:800;}
    .diag-value{font-weight:800;}
    .diag-sub{margin-top:4px;font-size:0.88rem;color:#cbd5e1;}
    .diag-chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:0.78rem;font-weight:800;margin-right:6px;background:rgba(148,163,184,0.15);color:#e2e8f0;}
    .diag-pos{background:#052e16;color:#86efac;}
    .diag-neu{background:#3f2f0a;color:#fde68a;}
    .diag-neg{background:#450a0a;color:#fca5a5;}
    .affects-line{margin-top:4px;font-size:0.82rem;color:#93c5fd;font-weight:700;}
    .compact-summary-card{border:1px solid rgba(148,163,184,0.14);border-radius:18px;padding:12px 14px;background:rgba(15,23,42,0.48);}
    .compact-summary-title{font-size:0.82rem;color:#cbd5e1;font-weight:700;}
    .compact-summary-value{font-size:1.18rem;font-weight:900;margin-top:2px;}
    .compact-summary-sub{font-size:0.82rem;color:#93c5fd;margin-top:4px;}
    </style>
    """, unsafe_allow_html=True)

    analysis_view_mode = "Expertenmodus"

    is_simple_mode = False
    is_pro_mode = False
    is_expert_mode = True

    driver_summary = build_driver_summary(result)
    diag_items_126a = build_diagnostic_impacts(result)
    diag_sections_126a = {}
    for _item in diag_items_126a:
        _sec = _item.get("section", "Diagnose")
        diag_sections_126a.setdefault(_sec, []).append(_item)

    if str(exit_action_display).strip().lower() == "halten":
        exit_action_sub_display = "Kein akuter Verkaufsdruck"
    elif str(exit_action_display).strip().lower() == "beobachten":
        exit_action_sub_display = "Frühe Schwäche, eng beobachten"
    elif str(exit_action_display).strip().lower() == "teilgewinn prüfen":
        exit_action_sub_display = "Gewinnschutz und De-Risking"
    elif str(exit_action_display).strip().lower() == "risiko reduzieren":
        exit_action_sub_display = "Verkaufsdruck erhöht"
    elif str(exit_action_display).strip().lower() == "verkaufen":
        exit_action_sub_display = "Klarer Exit-Druck"
    else:
        exit_action_sub_display = "Exit-Einordnung der aktuellen Lage"

    if is_expert_mode:
        st.markdown(
            f"""
            <div class="exec-shell">
                <div class="exec-top">
                    <div>
                        <div class="exec-kicker">Capital Hill Analyse</div>
                        <div class="exec-title">{name} <span style="color:#93c5fd;">{ticker}</span></div>
                        <div class="exec-sub">{shorten_text(short_thesis, 210)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    compact_cols = st.columns(6)
    compact_data = [
        ("Hauptsignal", main_action_label, display_mode_label(mode_label)),
        ("Investment-Case", f"{investment_case_score}/100", investment_case_text),
        ("Trading-Case", f"{trading_case_score}/100", trading_case_text),
        ("Exit", exit_action_display, exit_score_text_display),
        ("Setup-Priorität", f"{fmt_num(result.get('setup_priority_score', np.nan),0)}/100", watchlist_priority),
        ("Regime-Fit", f"{fmt_num(result.get('regime_fit_score', np.nan),0)}/100", market_regime_label(market_info["regime"])),
    ]
    for _col, (_title, _value, _sub) in zip(compact_cols, compact_data):
        with _col:
            st.markdown(
                f"""
                <div class="compact-summary-card">
                    <div class="compact-summary-title">{_title}</div>
                    <div class="compact-summary-value">{_value}</div>
                    <div class="compact-summary-sub">{_sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    why_col, risk_col = st.columns(2)
    with why_col:
        render_reason_box("Warum attraktiv", driver_summary.get("positives", []), empty_text="Keine klaren positiven Treiber erkannt.")
    with risk_col:
        render_reason_box("Was bremst", driver_summary.get("negatives", []), empty_text="Keine klaren Bremsfaktoren erkannt.")

    if is_pro_mode or is_expert_mode:
        with st.expander("Setup & Timing", expanded=False):
            render_diagnostic_section("Setup & Timing", diag_sections_126a.get("Setup & Timing", []))
        with st.expander("Volumen & Akkumulation", expanded=False):
            render_diagnostic_section("Volumen & Akkumulation", diag_sections_126a.get("Volumen & Akkumulation", []))
        with st.expander("Event & Katalysator", expanded=False):
            render_diagnostic_section("Event & Katalysator", diag_sections_126a.get("Event & Katalysator", []))
        with st.expander("Qualität & Fundamentals", expanded=False):
            render_diagnostic_section("Qualität & Fundamentals", diag_sections_126a.get("Qualität & Fundamentals", []))
        with st.expander("Marktregime", expanded=False):
            render_diagnostic_section("Marktregime", diag_sections_126a.get("Marktregime", []))
        with st.expander("Exit & Risiko", expanded=False):
            render_diagnostic_section("Exit & Risiko", diag_sections_126a.get("Exit & Risiko", []))

    if is_pro_mode or is_expert_mode:
        decision_meta = (
            f"Positionsmodus | Risiko: {shorten_text(risk_note, 42)}"
            if position_mode
            else f"Marktumfeld: {market_regime_label(market_info['regime'])} | Trigger-Status: {display_trigger_status_label(trigger_status)} | Priorität: {watchlist_priority}"
        )
        st.markdown(
            f"""
            <div class="section-head">
                <div class="section-title">Operative Einordnung auf einen Blick</div>
                <div class="section-meta-line">{decision_meta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        d1, d2, d3 = st.columns(3)
        if position_mode:
            with d1:
                st.markdown(
                    f"""
                    <div class="decision-card action" title="Aktuell sinnvollste Führungsaktion für die bestehende Position.">
                        <div class="dc-label">Positions-Aktion</div>
                        <div class="dc-value" style="font-size:clamp(1.02rem, 1.45vw, 1.26rem); line-height:1.18; word-break:break-word; overflow-wrap:anywhere;">{main_action_label}</div>
                        <div class="dc-sub">{shorten_text(risk_note, 52)}</div>
                        <div class="dc-note">Operative Führung der laufenden Position.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(
                    f"""
                    <div class="decision-card entry" title="Aktuelle Stop- und Führungslogik der Position.">
                        <div class="dc-label">Stop-Führung</div>
                        <div class="dc-value" style="font-size:clamp(1.02rem, 1.45vw, 1.24rem); line-height:1.18; word-break:break-word; overflow-wrap:anywhere;">{stop_action}</div>
                        <div class="dc-sub">{exit_action_display}</div>
                        <div class="dc-note">Stop und Exit-Taktik der Position.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with d3:
                st.markdown(
                    f"""
                    <div class="decision-card invest" title="Worauf bei der Position aktuell besonders zu achten ist.">
                        <div class="dc-label">Jetzt eng beobachten</div>
                        <div class="dc-value" style="font-size:clamp(1.0rem, 1.35vw, 1.18rem); line-height:1.18; word-break:break-word; overflow-wrap:anywhere;">{exit_reason_top_display}</div>
                        <div class="dc-sub">{exit_score_text_display}</div>
                        <div class="dc-note">Wichtigster aktueller Exit- oder Risikotreiber.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            with d1:
                st.markdown(
                    f"""
                    <div class="decision-card invest" title="Grundsätzliche Attraktivität des Werts als Investment, unabhängig vom exakten Einstiegstiming.">
                        <div class="dc-label">Investment-Urteil</div>
                        <div class="dc-value" style="font-size:clamp(0.98rem, 1.35vw, 1.18rem); line-height:1.18; word-break:break-word; overflow-wrap:anywhere;">{investment_case_text}</div>
                        <div class="dc-sub">{investment_case_score}/100</div>
                        <div class="dc-note">Antwort auf die Frage: Ist der Wert grundsätzlich attraktiv?</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(
                    f"""
                    <div class="decision-card entry" title="Qualität des aktuellen Einstiegsfensters und des Setups, unabhängig von der grundsätzlichen Investmentqualität.">
                        <div class="dc-label">Einstiegs-Urteil</div>
                        <div class="dc-value" style="font-size:clamp(0.98rem, 1.35vw, 1.18rem); line-height:1.18; word-break:break-word; overflow-wrap:anywhere;">{trading_case_text}</div>
                        <div class="dc-sub">{trading_case_score}/100 | Entry-Lage: {entry_quality}</div>
                        <div class="dc-note">Antwort auf die Frage: Ist das Timing für einen Einstieg aktuell gut?</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with d3:
                st.markdown(
                    f"""
                    <div class="decision-card action" title="Konkrete Handlungsableitung aus Investmentqualität, Einstiegstiming und Triggerstatus.">
                        <div class="dc-label">Konkrete Aktion</div>
                        <div class="dc-value" style="font-size:clamp(1.0rem, 1.35vw, 1.18rem); line-height:1.18; word-break:break-word; overflow-wrap:anywhere;">{main_action_label}</div>
                        <div class="dc-sub">{trigger_status} | {next_trigger}</div>
                        <div class="dc-note">Antwort auf die Frage: Kaufen, vorbereiten oder weiter warten?</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )



        st.markdown(
            f"""
            <div class="section-head">
                <div class="section-title">Exit-Sicht für Positionen</div>
                <div class="section-meta-line">Frühe Verkaufssignale aus Trendbruch, Momentum, relativer Schwäche, Distributionsdruck und harten Exit-Triggern.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.markdown(
                f"""
                <div class="decision-card action" title="Verdichteter Verkaufsdruck für die aktuelle Situation.">
                    <div class="dc-label">Exit-Score</div>
                    <div class="dc-value">{exit_score_display}/100</div>
                    <div class="dc-sub">{exit_score_text_display}</div>
                    <div class="dc-note">Je höher, desto stärker der Verkaufsdruck.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ex2:
            st.markdown(
                f"""
                <div class="decision-card entry" title="Operative Exit-Aktion für die aktuelle Lage.">
                    <div class="dc-label">Exit-Aktion</div>
                    <div class="dc-value">{exit_action_display}</div>
                    <div class="dc-sub">{exit_action_sub_display}</div>
                    <div class="dc-note">Ergänzt die bestehende Kauf-/Aufbaulogik um ein eigenes Verkaufssystem.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ex3:
            st.markdown(
                f"""
                <div class="decision-card invest" title="Der derzeit stärkste konkrete Exit-Grund.">
                    <div class="dc-label">Hauptgrund</div>
                    <div class="dc-value" style="font-size:clamp(1.0rem, 1.15vw, 1.18rem); line-height:1.2; word-break:break-word; overflow-wrap:anywhere;">{exit_reason_top_display}</div>
                    <div class="dc-sub">{' | '.join(exit_reason_extra_display[:2]) if exit_reason_extra_display else 'keine weiteren Exit-Hinweise'}</div>
                    <div class="dc-note">Hilft, normale Schwäche von echtem Exit-Druck zu trennen.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="section-head">
                <div class="section-title">Die wichtigsten Begründungen</div>
                <div class="section-meta-line">Die App trennt jetzt klarer zwischen Kernaussage, operativer Ausführung und Diagnose.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b1, b2, b3 = st.columns(3)
        with b1:
            strengths_html = "".join([f"<li>{s}</li>" for s in top_strengths]) if top_strengths else "<li>Keine klaren Stärken identifiziert.</li>"
            st.markdown(
                f"""
                <div class="bullet-card">
                    <h4>Pro</h4>
                    <ul>{strengths_html}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b2:
            weaknesses_html = "".join([f"<li>{w}</li>" for w in top_weaknesses]) if top_weaknesses else "<li>Keine wesentlichen Schwächen identifiziert.</li>"
            st.markdown(
                f"""
                <div class="bullet-card">
                    <h4>Contra</h4>
                    <ul>{weaknesses_html}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b3:
            st.markdown(
                f"""
                <div class="bullet-card">
                    <h4>Kurzfazit</h4>
                    <ul>
                        <li>{shorten_text(decision_summary, 170)}</li>
                        <li>Setup-Typ: {setup_type}</li>
                        <li>Trigger / Lage: {entry_quality}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="section-head">
                <div class="section-title">Kernbausteine</div>
                <div class="section-meta-line">Operative Subscores für Struktur, Confidence und Qualitätsbild.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        kb1, kb2, kb3, kb4 = st.columns(4)
        with kb1:
            st.markdown(
                f"""
                <div class="compact-panel" title="Wie gut das Setup grundsätzlich handelbar aufgebaut werden kann.">
                    <div class="cp-label">Trade-Struktur</div>
                    <div class="cp-value">{fmt_num(tradeability_score,0)}/100</div>
                    <div class="cp-sub">{tradeability_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kb2:
            st.markdown(
                f"""
                <div class="compact-panel" title="Wie sauber und belastbar das erkannte Setup aktuell wirkt.">
                    <div class="cp-label">Setup-Confidence</div>
                    <div class="cp-value">{fmt_num(setup_confidence,0)}/100</div>
                    <div class="cp-sub">{setup_confidence_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kb3:
            st.markdown(
                f"""
                <div class="compact-panel" title="Kombiniert Profitabilität, Wachstum, Bilanz, Bewertung, Sentiment und Risiko.">
                    <div class="cp-label">Company Quality</div>
                    <div class="cp-value">{company}/100</div>
                    <div class="cp-sub">{ampel(company)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kb4:
            st.markdown(
                f"""
                <div class="compact-panel" title="Gesamtbewertung aus technischer und fundamentaler Qualität.">
                    <div class="cp-label">Investment Score</div>
                    <div class="cp-value">{investment}/100</div>
                    <div class="cp-sub">{ampel(investment)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="secondary-action-row"><div class="muted-meta">Export und Logging der aktuellen Einzelanalyse</div><div class="secondary-action-note">Diagnose-Scores und Hilfswerte liegen darunter im aufklappbaren Bereich.</div></div>', unsafe_allow_html=True)
        se_outer1, se_outer2, se_outer3 = st.columns([1.0, 1.0, 2.3])
        sheet_href = f"?sheet_log=1&sheet_nonce={datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        with se_outer1:
            st.markdown(
                f'<a class="export-action-btn" href="{csv_href}" download="{csv_filename}"><span>CSV</span></a>',
                unsafe_allow_html=True
            )
        with se_outer2:
            st.markdown(
                f'<a class="export-action-btn" href="{sheet_href}"><span>Sheets</span></a>',
                unsafe_allow_html=True
            )
        with se_outer3:
            st.markdown("", unsafe_allow_html=True)

        with st.expander("Diagnose-Scores und Hilfswerte anzeigen", expanded=False):
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            with c1:
                render_score_card("Company Quality", f"{company}/100", ampel(company), "company", tooltip="Bewertet Profitabilität, Wachstum, Bilanz, Bewertung, Sentiment und Risiko des Unternehmens.")
            with c2:
                render_score_card("Setup Quality", f"{setup_adj}/100", ampel(setup_adj), "setup", tooltip="Bewertet das technische Gesamtbild der Aktie. Enthält Trend, Momentum, Volumen, Volatilität und einen kleinen Marktfilter.")
            with c3:
                render_score_card("Investment Score", f"{investment}/100", ampel(investment), "investment", tooltip="Gesamtbewertung aus technischer und fundamentaler Qualität.")
            with c4:
                render_score_card("Trade-Struktur", f"{fmt_num(tradeability_score,0)}/100", tradeability_text, "kb", tooltip="Wie gut das Setup grundsätzlich handelbar aufgebaut werden kann.")
            with c5:
                render_score_card("Kurzfrist-Timing", f"{tb_score_100}/100", f"{tb_timing_text} | Board: {tb_score} Punkte", "board", tooltip="Schneller Taktik- und Timing-Blick aus dem TradingBoard.")
            with c6:
                render_score_card("Kurzfrist Core", f"{short_term_score}/100", ampel(short_term_score), "short", tooltip="Kurzfristige Kernbewertung aus Momentum, Volumen, Volatilität und relativer Stärke.")
            with c7:
                render_score_card("Setup-Confidence", f"{fmt_num(setup_confidence,0)}/100", setup_confidence_text, "helper", tooltip="Wie sauber und belastbar das erkannte Setup aktuell wirkt.")





    st.markdown(
        """
        <style>
        .exec-v2-shell{
            position:relative;
            margin: 8px 0 18px 0;
            padding: 22px 22px 18px 22px;
            border-radius: 28px;
            background:
                radial-gradient(circle at top left, rgba(96,165,250,0.14), rgba(0,0,0,0) 30%),
                linear-gradient(180deg, rgba(15,23,42,0.98), rgba(15,23,42,0.88));
            border:1px solid rgba(96,165,250,0.20);
            box-shadow:
                0 24px 52px rgba(2,6,23,0.26),
                inset 0 1px 0 rgba(255,255,255,0.05);
            overflow:hidden;
        }
        .exec-v2-shell::after{
            content:"";
            position:absolute;
            inset:0;
            background:linear-gradient(120deg, rgba(255,255,255,0.04), rgba(255,255,255,0.00) 42%);
            pointer-events:none;
        }
        .exec-v2-top{
            position:relative;
            z-index:1;
            display:flex;
            justify-content:space-between;
            gap:16px;
            align-items:flex-start;
            flex-wrap:wrap;
            margin-bottom:16px;
        }
        .exec-v2-eyebrow{
            display:inline-block;
            padding:6px 11px;
            border-radius:999px;
            background:rgba(37,99,235,0.16);
            border:1px solid rgba(96,165,250,0.28);
            color:#dbeafe;
            font-size:0.76rem;
            font-weight:900;
            letter-spacing:0.03em;
            margin-bottom:10px;
        }
        .exec-v2-title{
            font-size:1.95rem;
            line-height:1.02;
            font-weight:950;
            color:#f8fafc;
            letter-spacing:-0.03em;
            margin-bottom:6px;
        }
        .exec-v2-sub{
            color:#cbd5e1;
            font-size:0.95rem;
            line-height:1.55;
            max-width:760px;
        }
        .exec-v2-signal{
            min-width:260px;
            padding:14px 16px;
            border-radius:22px;
            background:linear-gradient(180deg, rgba(30,41,59,0.86), rgba(15,23,42,0.62));
            border:1px solid rgba(148,163,184,0.18);
            box-shadow:0 14px 28px rgba(2,6,23,0.20);
        }
        .exec-v2-signal-label{
            font-size:0.74rem;
            text-transform:uppercase;
            letter-spacing:0.06em;
            color:#93c5fd;
            font-weight:900;
            margin-bottom:6px;
        }
        .exec-v2-signal-value{
            font-size:1.22rem;
            line-height:1.15;
            font-weight:900;
            color:#f8fafc;
            margin-bottom:6px;
        }
        .exec-v2-signal-meta{
            font-size:0.86rem;
            color:#cbd5e1;
            line-height:1.45;
        }
        .exec-v2-grid{
            position:relative;
            z-index:1;
            display:grid;
            grid-template-columns:1.4fr 1fr;
            gap:16px;
            align-items:stretch;
        }
        .exec-v2-panel{
            border-radius:22px;
            padding:16px 16px 14px 16px;
            background:linear-gradient(180deg, rgba(17,24,39,0.88), rgba(15,23,42,0.68));
            border:1px solid rgba(148,163,184,0.16);
            box-shadow:0 12px 24px rgba(2,6,23,0.16);
        }
        .exec-v2-panel-title{
            font-size:0.78rem;
            text-transform:uppercase;
            letter-spacing:0.06em;
            color:#93c5fd;
            font-weight:900;
            margin-bottom:10px;
        }
        .exec-v2-kpi-grid{
            display:grid;
            grid-template-columns:repeat(2, minmax(0,1fr));
            gap:12px;
        }
        .exec-v2-kpi{
            border-radius:18px;
            padding:12px 12px 10px 12px;
            background:rgba(15,23,42,0.52);
            border:1px solid rgba(148,163,184,0.14);
        }
        .exec-v2-kpi-label{
            font-size:0.76rem;
            color:#94a3b8;
            font-weight:800;
            margin-bottom:6px;
            text-transform:uppercase;
            letter-spacing:0.03em;
        }
        .exec-v2-kpi-value{
            font-size:1.26rem;
            line-height:1.1;
            color:#f8fafc;
            font-weight:950;
            margin-bottom:4px;
        }
        .exec-v2-kpi-sub{
            font-size:0.82rem;
            color:#cbd5e1;
            line-height:1.38;
        }
        .exec-v2-list{
            display:flex;
            flex-direction:column;
            gap:10px;
        }
        .exec-v2-list-item{
            border-radius:16px;
            padding:11px 12px;
            border:1px solid rgba(148,163,184,0.14);
            background:rgba(15,23,42,0.44);
        }
        .exec-v2-list-label{
            font-size:0.73rem;
            font-weight:900;
            letter-spacing:0.04em;
            text-transform:uppercase;
            color:#94a3b8;
            margin-bottom:5px;
        }
        .exec-v2-list-value{
            font-size:1.02rem;
            line-height:1.35;
            color:#f8fafc;
            font-weight:800;
        }
        .exec-v2-divider{
            height:1px;
            background:linear-gradient(90deg, rgba(96,165,250,0.34), rgba(148,163,184,0.08), rgba(0,0,0,0));
            margin:16px 0 4px 0;
        }
        @media (max-width: 980px){
            .exec-v2-grid{ grid-template-columns:1fr; }
            .exec-v2-title{ font-size:1.64rem; }
            .exec-v2-signal{ width:100%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    exec_signal_value = (
        main_action_label if 'main_action_label' in locals() else
        signal_label if 'signal_label' in locals() else
        signal_text if 'signal_text' in locals() else
        display_emp_label(emp) if 'emp' in locals() else
        position_action if 'position_action' in locals() else
        "-"
    )
    exec_focus_value = (
        top_red_flag if str(top_red_flag).strip() not in {"", "-", "None"} else
        risk_note if str(risk_note).strip() not in {"", "-", "None", "Watchlist-Modus"} else
        ("Ja" if str(trigger_status).strip() == "Aktiv" else "Noch nicht") if str(top_red_flag).strip() in {"", "-", "None"} and str(risk_note).strip() in {"", "-", "None", "Watchlist-Modus"} else
        main_action_label if str(main_action_label).strip() not in {"", "-", "None", "Nicht anwendbar"} else
        "Lage weiter beobachten"
    )
    exec_focus_label = "Kritischer Kipppunkt" if str(top_red_flag).strip() not in {"", "-", "None"} else "Jetzt ein guter Einstieg?"
    exec_summary_text = company_summary if company_summary not in [None, "", "-"] else "Die Aktie ist oben bereits verdichtet. Hier bleibt nur die zusätzliche Einordnung, was das aktuelle Urteil kippen oder bestätigen würde."
    exec_redflag = top_red_flag if 'top_red_flag' in locals() else red_flag if 'red_flag' in locals() else "-"
    exec_risk_context = risk_note if str(risk_note).strip() not in {"", "-", "None", "Keine Auffälligkeit", "Watchlist-Modus"} else exit_reason_top_display
    exec_confirmation = (
        leadership_status_display if str(leadership_status_display).strip() not in {"", "-", "None"} else
        sector_trend_text_display if str(sector_trend_text_display).strip() not in {"", "-", "None", "nicht belastbar"} else
        display_trigger_status_label(trigger_status) if str(trigger_status).strip() not in {"", "-", "None"} else
        "noch kein zusätzlicher Bestätigungsfaktor"
    )

    st.markdown(
        f"""
        <div class="exec-v2-shell">
            <div class="exec-v2-top">
                <div>
                    <div class="exec-v2-eyebrow">Executive Summary</div>
                    <div class="exec-v2-title">{name} <span style="color:#93c5fd;">| {ticker}</span></div>
                    <div class="exec-v2-sub">
                        Nur die zusätzliche Einordnung, die oberhalb noch nicht als Score oder Kachel gezeigt wird.
                    </div>
                </div>
                <div class="exec-v2-signal">
                    <div class="exec-v2-signal-label">{exec_focus_label}</div>
                    <div class="exec-v2-signal-value" style="font-size:clamp(1.15rem, 1.75vw, 1.55rem); line-height:1.2; word-break:break-word; overflow-wrap:anywhere;">{exec_focus_value}</div>
                    <div class="exec-v2-signal-meta">
                        Dieser Block wiederholt bewusst keine Hauptscores mehr, sondern ordnet nur noch den wichtigsten Kipppunkt oder die Einstiegsreife ein.
                    </div>
                </div>
            </div>
            <div class="exec-v2-grid" style="grid-template-columns:1.1fr 0.9fr;">
                <div class="exec-v2-panel">
                    <div class="exec-v2-panel-title">Zusätzliche Einordnung</div>
                    <div class="exec-v2-list">
                        <div class="exec-v2-list-item">
                            <div class="exec-v2-list-label">Was das Urteil kippen würde</div>
                            <div class="exec-v2-list-value">{exec_redflag if str(exec_redflag).strip() not in {"", "-", "None"} else "derzeit keine dominierende Red Flag"}</div>
                        </div>
                    </div>
                </div>
                <div class="exec-v2-panel">
                    <div class="exec-v2-panel-title">Zusätzlicher Kontext</div>
                    <div class="exec-v2-list">
                        <div class="exec-v2-list-item">
                            <div class="exec-v2-list-label">Wichtig zu beachten</div>
                            <div class="exec-v2-list-value">{exec_risk_context if str(exec_risk_context).strip() not in {"", "-", "None"} else "kein zusätzlicher Risikohinweis"}</div>
                        </div>
                        <div class="exec-v2-list-item">
                            <div class="exec-v2-list-label">Was das Urteil bestätigt</div>
                            <div class="exec-v2-list-value">{exec_confirmation}</div>
                        </div>
                    </div>
                    <div class="exec-v2-divider"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Tabs ----------
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.divider()
    if is_pro_mode or is_expert_mode:
        t0, t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
            "Überblick",
            "Trading-Case",
            "Signalbild",
            "TradingBoard",
            "Investment-Case",
            "Sicherheit & Checks",
            "Trade-Plan",
            "Position",
            "Watchlist"
        ])

        with t0:
            st.subheader("Überblick")
            st.markdown('<div class="panel-caption">Kurzfazit, Kerndaten und Chartverlauf des aktuell ausgewählten Werts.</div>', unsafe_allow_html=True)

            p1, p2, p3 = st.columns(3)
            p1.metric("Unternehmen", name)
            p2.metric("Sektor", sector if sector else "-")
            p3.metric("Industrie", industry if industry else "-")

            if is_expert_mode:
                st.markdown(
                    """
                    <div class="section-head">
                        <div class="section-title">Leadership & Marktbreite</div>
                        <div class="section-meta-line">Bewertet, ob die Aktie in einem starken Umfeld selbst als Leader auftritt oder eher nur mitläuft.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                l1, l2, l3, l4 = st.columns(4)
                l1.metric("Leadership", f"{fmt_num(leadership_score_display,0)}/100", leadership_status_display)
                l2.metric("Sektor-Stärke", sector_strength_text_display, sector_delta_display)
                l3.metric("Industrie-Stärke", f"{fmt_num(industry_strength_display,0)}/100", industry_label_display)
                l4.metric("RS-Beschleunigung", f"{fmt_num(rs_acceleration_display,0)}/100")

            st.markdown(
                    f"""
                        <div class="section-card">
                            <div class="premium-title">Einordnung</div>
                            <div class="premium-value">Leadership-Status: {leadership_status_display}</div>
                            <div class="premium-sub">
                                Sektor wirkt {sector_trend_text_display}, Industrie wirkt {industry_trend_text_display}. Relative Stärke gegenüber dem Benchmark ist ein zentraler Treiber des Long-Urteils.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("Details zwischen Einordnung und Kurzfazit anzeigen", expanded=False):
                    st.markdown(
                        """
                        <div class="section-head">
                            <div class="section-title">Setup-Qualität & Trendstruktur</div>
                            <div class="section-meta-line">Bewertet, wie sauber Trend, Base und konkreter Setup-Typ wirklich aufgebaut sind.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    q1, q2, q3, q4 = st.columns(4)
                    q1.metric("Trendqualität", f"{fmt_num(trend_quality_display,0)}/100")
                    q2.metric("Base-Qualität", f"{fmt_num(base_quality_display,0)}/100")
                    q3.metric("Setup-Typ-Qualität", f"{fmt_num(setup_type_quality_display,0)}/100", setup_type)
                    q4.metric("Setup-Priorität", f"{fmt_num(setup_priority_display,0)}/100")

                    d1, d2, d3, d4, d5 = st.columns(5)
                    d1.metric("Base-Länge", fmt_num(base_length_display, 0))
                    d2.metric("Korrekturtiefe %", fmt_num(correction_depth_display, 1, "%"))
                    d3.metric("Range-Tightness", f"{fmt_num(range_tightness_display,0)}/100")
                    d4.metric("Volatility Contraction", f"{fmt_num(volatility_contraction_display,0)}/100")
                    d5.metric("Pullback-Qualität", f"{fmt_num(pullback_quality_display,0)}/100")

                    st.markdown(
                        """
                        <div class="section-head">
                            <div class="section-title">Volumenqualität & Akkumulation</div>
                            <div class="section-meta-line">Bewertet, ob Nachfrage, Pullback-Volumen und Breakout-Bestätigung das Long-Setup wirklich tragen.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    v1, v2, v3, v4, v5 = st.columns(5)
                    v1.metric("Volumenqualität", f"{fmt_num(volume_quality_display,0)}/100")
                    v2.metric("Akkumulation", f"{fmt_num(accumulation_display,0)}/100")
                    v3.metric("Distribution", f"{fmt_num(distribution_display,0)}/100")
                    v4.metric("Pullback-Dry-up", f"{fmt_num(pullback_dryup_display,0)}/100")
                    v5.metric("Breakout-Volumen", f"{fmt_num(breakout_volume_display,0)}/100")

                    dv1, dv2, dv3, dv4 = st.columns(4)
                    dv1.metric("Up/Down-Vol.-Ratio", fmt_num(up_down_volume_ratio_display, 2))
                    dv2.metric("Akkumulationstage", fmt_num(accumulation_day_count_display, 0))
                    dv3.metric("Distributionstage", fmt_num(distribution_day_count_display, 0))
                    dv4.metric("Volumentrend", f"{fmt_num(volume_trend_display,0)}/100")

                    st.markdown(
                        """
                        <div class="section-head">
                            <div class="section-title">Katalysatoren & Event-Kontext</div>
                            <div class="section-meta-line">Bewertet Earnings-Nähe, Reaktionsqualität nach Events und den aktuellen Katalysator-Rückenwind.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Katalysator-Score", f"{fmt_num(catalyst_score_display,0)}/100", catalyst_text_display)
                    c2.metric("Event-Score", f"{fmt_num(earnings_event_score_display,0)}/100", event_phase_text(event_phase_label_display))
                    c3.metric("Revision/Momentum", f"{fmt_num(revision_momentum_score_display,0)}/100")
                    c4.metric("Event-Risiko", f"{fmt_num(event_risk_score_display,0)}/100")

                    e1, e2 = st.columns(2)
                    e1.metric("Post-Earnings 5d", fmt_num(earnings_reaction_5d_display, 1, "%"))
                    e2.metric("Post-Earnings 10d", fmt_num(earnings_reaction_10d_display, 1, "%"))

                    st.markdown(
                        """
                        <div class="section-head">
                            <div class="section-title">Institutionelle Qualität</div>
                            <div class="section-meta-line">Bewertet Cashflow-Stabilität, Margenqualität und die Robustheit des Unternehmens aus Investorensicht.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    iq1, iq2, iq3 = st.columns(3)
                    iq1.metric("Institutionelle Qualität", f"{fmt_num(institutional_quality_display,0)}/100", institutional_quality_text_display)
                    iq2.metric("Cashflow-Stabilität", f"{fmt_num(cashflow_stability_display,0)}/100")
                    iq3.metric("Margenstabilität", f"{fmt_num(margin_stability_display,0)}/100")
            st.markdown("**Kurzfazit**")
            st.write(short_thesis)

            st.markdown("**Kurzbeschreibung**")
            summary_short = company_summary[:900] + "..." if len(company_summary) > 900 else company_summary
            st.write(summary_short)

            st.markdown(f"**Chart & Performance · {ticker} ({ccy})**")
            chart_range = st.selectbox(
                "Zeitraum",
                ["3 Monate", "6 Monate", "1 Jahr", "3 Jahre"],
                index=2,
                key="chart_range"
            )

            chart_df = compute_chart_df(df, chart_range)
            fig = build_candlestick_chart(chart_df, ticker, ccy)
            st.plotly_chart(fig, use_container_width=True)

            perf_start = float(chart_df["Close"].iloc[0]) if not chart_df.empty else np.nan
            perf_end = float(chart_df["Close"].iloc[-1]) if not chart_df.empty else np.nan
            perf_pct = ((perf_end / perf_start) - 1) * 100 if pd.notna(perf_start) and perf_start != 0 else np.nan

            p4, p5, p6 = st.columns(3)
            p4.metric("Start", fmt_num(perf_start, 2, f" {ccy}"))
            p5.metric("Aktuell", fmt_num(perf_end, 2, f" {ccy}"))
            p6.metric("Performance", fmt_num(perf_pct, 1, "%"))

        with t1:
            st.subheader("Trading-Case")
            st.markdown('<div class="panel-caption">Technisches Setup, Momentum, Volumen und kurzfristige Struktur.</div>', unsafe_allow_html=True)
            st.markdown(f"**Benchmark:** {benchmark_label} (`{benchmark_symbol}`) | **Marktregime:** {market_info['ampel']} {market_regime_label(market_info['regime'])}")

            cols = st.columns(2)
            items = [
                ("S3 Trend", s3a, s3, s3t),
                ("S4 Momentum", s4a, s4, s4t),
                ("S5 Volumen", s5a, s5, s5t),
                ("S6 Volatilitaet", s6a, s6, s6t),
                ("52W-Lage", ampel(w52), w52, f"{dist52:.1f}% vom 52W-Hoch"),
                ("Relative Stärke", ampel(rs_score), rs_score, f"RS composite: {fmt_num(rs_composite,1,'%')}"),
            ]
            for i, (lab, ico, score, com) in enumerate(items):
                with cols[i % 2]:
                    st.markdown(
                        f'<div class="metric-card {card_class(score)}"><b>{ico} {lab}</b>'
                        f'<span style="float:right;font-size:1.3rem;font-weight:700">{score}</span>'
                        f'<br><small style="color:#aaa">{com}</small></div>',
                        unsafe_allow_html=True,
                    )

            tech_df = pd.DataFrame({
                "Indikator": [
                    "Kurs", "MA20", "MA50", "MA150", "MA200",
                    "RSI(14)", "MACD", "Signal", "MACD-Hist", "ADX", "ATR", "ATR in %",
                    "Stoch %K", "Stoch %D", "Williams %R", "ROC20", "ROC60",
                    "52W-Hoch", "52W-Tief", "Abstand zum 52W-Hoch",
                    "1M Aktie", "1M Benchmark", "1M Outperformance",
                    "3M Aktie", "3M Benchmark", "3M Outperformance",
                    "6M Aktie", "6M Benchmark", "6M Outperformance",
                    "RS Composite", "Marktregime"
                ],
                "Wert": [
                    f"{price:.2f}", f"{ma20:.2f}", f"{ma50:.2f}", f"{ma150:.2f}", f"{ma200:.2f}",
                    f"{rsi:.1f}", f"{macd_v:.3f}", f"{signal_v:.3f}", f"{macd_hist_current:.3f}", f"{adx:.1f}",
                    f"{atr:.3f}", f"{atr_pct:.1f}%", f"{stoch_k_v:.1f}", f"{stoch_d_v:.1f}", f"{willr_v:.1f}",
                    f"{roc20:.1f}%", f"{roc60:.1f}%", f"{high52:.2f}", f"{low52:.2f}", f"{dist52:.1f}%",
                    fmt_num(ret21, 1, "%"), fmt_num(bench_ret21, 1, "%"), fmt_num(rs_vs_benchmark_21, 1, "%"),
                    fmt_num(ret63, 1, "%"), fmt_num(bench_ret63, 1, "%"), fmt_num(rs_vs_benchmark_63, 1, "%"),
                    fmt_num(ret126, 1, "%"), fmt_num(bench_ret126, 1, "%"), fmt_num(rs_vs_benchmark_126, 1, "%"),
                    fmt_num(rs_composite, 1, "%"), market_regime_label(market_info["regime"])
                ],
            })
            st.dataframe(tech_df, hide_index=True, use_container_width=True)

        with t2:
            st.subheader("Kurzfristiges Signalbild")
            c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.15, 1.15])
            c1.metric("Kurzfrist Core", f"{short_term_score}/100", ampel(short_term_score))
            c2.metric("Kurzfrist Hilfsboard", f"{stb_score} Punkte", stb_signal)
            with c3:
                st.markdown(
                    """
                    <div class="wrap-metric-card">
                        <div class="wrap-metric-label">Core Fokus</div>
                        <div class="wrap-metric-value">Momentum und Volumen</div>
                        <div class="wrap-metric-sub">Die Kernsignale für kurzfristige Stärke.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    """
                    <div class="wrap-metric-card">
                        <div class="wrap-metric-label">Board Fokus</div>
                        <div class="wrap-metric-value">Einzelne Handelssignale</div>
                        <div class="wrap-metric-sub">Additive Trigger aus dem Hilfsboard.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.dataframe(
                pd.DataFrame({
                    "Kennzahl": ["Kurzfrist Core", "Kurzfrist Hilfsboard", "Board-Signal", "Board-Treiber"],
                    "Wert": [f"{short_term_score}/100", str(stb_score), stb_signal, stb_text],
                    "Kommentar": [
                        "S4 Momentum 45%, S5 Volumen 28%, S6 Volatilitaet 17%, RS 10%",
                        "Additive Kurzfrist-Punkte aus MA/RSI/Momentum/ADX/Stoch/Williams",
                        stb_empf,
                        stb_text
                    ],
                }),
                hide_index=True,
                use_container_width=True,
            )

        with t3:
            st.subheader("TradingBoard & Timing")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Kurzfrist-Timing", f"{tb_score_100}/100", tb_timing_text)
            c2.metric("Board-Signal", tb_signal, tb_empf)
            c3.metric("TradingBoard Stop-Loss", f"{price - (2.5 * atr):.2f} {ccy}")
            c4.metric("TradingBoard Kursziel 2", f"{tp2:.2f} {ccy}")

            st.dataframe(tb_df, hide_index=True, use_container_width=True)

            st.markdown("**Board-Details (Kurzfrist-Timing)**")
            st.text("\n".join(tb_details))

            st.markdown("**Board-Kontext (nicht im Score)**")
            st.text("\n".join(tb_context))

        with t4:
            st.subheader("Investment-Case")
            st.markdown('<div class="panel-caption">Fundamentale Qualität, Bewertung und mittelfristige Attraktivität.</div>', unsafe_allow_html=True)
            st.markdown(
                f"<div class='small-note'>Datenabdeckung Fundamentaldaten: {fund_cov*100:.0f}% | Geladene Felder: {fund_fields_loaded}/21</div>",
                unsafe_allow_html=True
            )

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Fundamental-Confidence", confidence_info["confidence"], confidence_info["confidence_icon"])
            f2.metric("Coverage", f"{confidence_info['coverage']*100:.0f}%")
            f3.metric("Geladene Felder", f"{confidence_info['loaded']}/{confidence_info['total']}")
            f4.metric("Abgeleitete Felder", str(confidence_info["derived_estimate"]))

            fund_df = pd.DataFrame({
                "Fundament-Block": [
                    "Qualitaet", "Wachstum", "Growth Quality", "Bewertung",
                    "Bilanz", "Sentiment", "Risiko"
                ],
                "Score": [
                    quality_score, growth_score, growth_quality, valuation_score,
                    balance_score, sentiment_score, risk_score
                ],
                "Kommentar": [
                    f"Gewinnmarge {fmt_num(profit_margin*100 if pd.notna(profit_margin) else np.nan,1,'%')} | Operative Marge {fmt_num(oper_margin*100 if pd.notna(oper_margin) else np.nan,1,'%')} | ROE {fmt_num(roe*100 if pd.notna(roe) else np.nan,1,'%')}",
                    f"Umsatzwachstum {fmt_num(revenue_growth*100 if pd.notna(revenue_growth) else np.nan,1,'%')} | EPS-Wachstum {fmt_num(earnings_growth*100 if pd.notna(earnings_growth) else np.nan,1,'%')}",
                    f"Wachstum + Cashflow + Margen | Stil: {stock_style}",
                    f"KGV {fmt_num(pe,1)} | PEG {fmt_num(peg,2)} | KUV {fmt_num(ps,2)} | KBV {fmt_num(pb,2)}",
                    f"Current Ratio {fmt_num(current_ratio,2)} | Quick Ratio {fmt_num(quick_ratio,2)} | D/E {fmt_num(debt_to_equity,1)}",
                    f"Analystenmeinung {rec_label} | Anzahl {fmt_num(analysts,0)} | Mean {fmt_num(rec_mean,2)}",
                    f"Beta {fmt_num(beta,2)} | Short-Quote {fmt_num(short_pct*100 if pd.notna(short_pct) else np.nan,1,'%')} | ATR% {fmt_num(atr_pct,1,'%')}",
                ],
            })
            st.dataframe(fund_df, hide_index=True, use_container_width=True)

            st.markdown("**Strukturierte Red Flags**")
            st.dataframe(red_flags_df, hide_index=True, use_container_width=True)

        with t5:
            st.subheader("Sicherheit & Checks")
            safeguard_df = pd.DataFrame({
                "Safeguard": [
                    "S0 Currency/Exchange",
                    "S0 Preis-Typ-Lock",
                    "S1 Earnings",
                    "S2 Regime",
                    "S3 Konfluenz-Cap",
                    "S4 Datenabdeckung",
                    "S5 Marktfilter",
                    "S6 Modus",
                    "S7 Red Flags",
                    "S8 Fundamental-Confidence"
                ],
                "Status": [
                    "🟢",
                    "🟢",
                    sg_earn,
                    reg_amp,
                    "🟢" if kb >= 3 else ("🟡" if kb == 2 else "🔴"),
                    "🟢" if fund_cov >= 0.55 else ("🟡" if fund_cov >= 0.35 else "🔴"),
                    market_info["ampel"],
                    "🟢",
                    "🟢" if red_flag_penalty_total == 0 else ("🟡" if red_flag_penalty_total <= 10 else "🔴"),
                    confidence_info["confidence_icon"],
                ],
                "Kommentar": [
                    f"{ccy} | {exch}",
                    "auto_adjust=True Yahoo Finance",
                    sg_earn_txt,
                    regime,
                    f"{kb}/4 Kernbloecke",
                    f"Fundamental-Coverage {fund_cov*100:.0f}%",
                    f"{benchmark_label} | {market_regime_label(market_info['regime'])}",
                    mode_label,
                    f"Penalty {red_flag_penalty_total}",
                    confidence_info["confidence"]
                ],
            })
            st.dataframe(safeguard_df, hide_index=True, use_container_width=True)

        with t6:
            st.subheader("Trade-Plan")
            st.markdown('<div class="panel-caption">Konkrete Ableitung von Entry, Stop, Zielen und Chance-Risiko-Verhältnis.</div>', unsafe_allow_html=True)
            st.markdown('<div class="muted-meta">Setup-Ziele werden nur gezeigt, wenn aus dem konkreten Muster ein belastbares Primär- oder Sekundärziel ableitbar ist.</div>', unsafe_allow_html=True)
            if not valid_trade_setup:
                st.error("Kein valides Trade-Setup: Score, Marktumfeld oder Konfluenz reichen aktuell nicht aus.")
                st.write(
                    f"Aktuell: Investment Score {investment}/100 | "
                    f"Setup Quality {setup_adj}/100 | "
                    f"Konfluenz {kb}/4 | "
                    f"Marktregime {market_regime_label(market_info['regime'])}"
                )
                if has_upcoming_earnings and pd.notna(days_earn) and days_earn < 7:
                    st.write("Zusatzhinweis: Earnings-Veto aktiv.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Einstiegskurs", f"{price:.2f} {ccy}")
                c2.metric("ATR-basierter Stop-Loss", f"{atr_stop:.2f} {ccy}", f"-{(price-atr_stop)/price*100:.1f}%" if atr_stop < price else "-")
                c3.metric("Aktueller Stop-Loss", f"{stop_used:.2f} {ccy}", f"-{stop_dist:.1f}%")

                c4, c5, c6 = st.columns(3)
                c4.metric("Kursziel 1 (1R)", f"{tp1:.2f} {ccy}", f"+{(tp1/price-1)*100:.1f}%")
                c5.metric("Kursziel 2 (Hauptziel)", f"{tp2:.2f} {ccy}", f"+{(tp2/price-1)*100:.1f}%")
                c6.metric("Kursziel 3", f"{tp3:.2f} {ccy}", f"+{(tp3/price-1)*100:.1f}%")

                c7, c8, c9 = st.columns(3)
                c7.metric(f"Chance-Risiko-Verhältnis {ampel_crv(crv)}", f"{crv:.1f}:1")
                c8.metric("Positionsgroesse", f"{pos_size} Stueck", f"Risiko {risk_eur:.0f} EUR ({risk_pct}%)")
                c9.metric("Zeitlicher Stop", time_stop, "wenn der Kurs nicht anschiebt")

                st.markdown("**Konkreter Einstiegsvorschlag**")
                e1, e2, e3 = st.columns(3)
                e1.metric("Entry-Zone", suggested_entry_zone)
                e2.metric("Entry-Herleitung", entry_source)
                e3.metric("Aktuelle Lage", entry_quality)

                st.markdown("**Case-Komponenten**")
                tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                tc1.metric("Investment-Case", f"{investment_case_score}/100", investment_case_text)
                tc2.metric("Einstiegs-Case", f"{trading_case_score}/100", trading_case_text)
                tc3.metric("CRV-Score", fmt_num(trade_crv_score, 0))
                tc4.metric("Entry-Score", fmt_num(trade_entry_score, 0))
                tc5.metric("Setup-Confidence", fmt_num(setup_confidence, 0))

                st.markdown("**Herleitung von Stop und Zielen**")
                st.write(f"• Stop: {stop_source}")
                st.write(f"• TP1: {tp1_source}")
                st.write(f"• TP2: {tp2_source}")
                st.write(f"• TP3: {tp3_source}")

                td1, td2 = st.columns(2)
                td1.metric("Primärziel aus Setup", ui_target_text(technical_target_1, ccy))
                td2.metric("Sekundärziel aus Setup", ui_target_text(technical_target_2, ccy, "kein zweites Setup-Ziel"))

        with t7:
            st.subheader("Position")
            st.markdown('<div class="panel-caption">Positionssicht mit Exit-Score, Exit-Aktion, Kontext und den wichtigsten Verkaufsgründen.</div>', unsafe_allow_html=True)

            px1, px2, px3, px4 = st.columns(4)
            px1.metric("Exit-Score", f"{exit_score_display}/100")
            px2.metric("Exit-Aktion", exit_action_display)
            px3.metric("Hauptgrund", exit_reason_top_display)
            px4.metric("P&L-Kontext", result.get("pnl_bucket", "-"))

            exs1, exs2, exs3, exs4, exs5 = st.columns(5)
            exs1.metric("Trendbruch", f"{result.get('trend_break_score', 0)}/100")
            exs2.metric("Momentum", f"{result.get('momentum_collapse_score', 0)}/100")
            exs3.metric("Rel. Schwäche", f"{result.get('relative_weakness_score', 0)}/100")
            exs4.metric("Distribution", f"{result.get('distribution_score', 0)}/100")
            exs5.metric("Trigger", f"{result.get('exit_trigger_score', 0)}/100")

            st.subheader("Position")
            if not position_mode:
                st.info("Dieser Bereich ist nur relevant, wenn ein Buy-in gesetzt ist und damit der Positionsmodus aktiv ist.")
            else:
                p1, p2, p3 = st.columns(3)
                p1.metric("Positions-Aktion", position_action)
                p2.metric("Nachkauf sinnvoll?", add_on_action)
                p3.metric("Teilgewinn prüfen?", partial_profit_action)

                p4, p5, p6 = st.columns(3)
                p4.metric("Stop-Anpassung", stop_action)
                p5.metric("Performance seit Einstieg", fmt_num(tb_perf, 1, "%"))
                p6.metric("Risiko-Hinweis", risk_note)

                st.markdown("**Einordnung**")
                st.write(
                    f"Die Positionsentscheidung kombiniert Investment-Case ({investment_case_score}/100), "
                    f"Einstiegs-Case ({trading_case_score}/100), Setup-Confidence ({fmt_num(setup_confidence,0)}/100), "
                    f"Marktumfeld ({market_regime_label(market_info['regime'])}) und die bisherige Performance seit Einstieg."
                )


        with t8:
            st.subheader("Watchlist & Trigger")
            if position_mode:
                st.info("Dieser Bereich ist vor allem für den Watchlist-Modus gedacht. Im Positionsmodus sind Trigger nur nachrangig relevant.")
            else:
                w1, w2, w3 = st.columns(3)
                w1.metric("Trigger-Status", display_trigger_status_label(trigger_status))
                w2.metric("Watchlist-Priorität", watchlist_priority)
                w3.metric("Nächster Trigger", next_trigger)

                st.markdown("**Warum steht die Aktie auf der Watchlist so weit oben?**")
                st.write(trigger_reason)

                w4, w5, w6 = st.columns(3)
                w4.metric("Setup-Typ", setup_type)
                w5.metric("Entry-Lage", entry_quality)
                w6.metric("Einstieg jetzt attraktiv?", f"{trading_case_score}/100", trading_case_text)

                st.markdown("**Praktische Watchlist-Logik**")
                st.write(
                    "Die Watchlist-Logik priorisiert Werte, die entweder bereits valide sind, "
                    "kurz vor einem Trigger stehen oder einen starken Investment-Case haben, "
                    "aber noch auf besseres Timing warten."
                )




    if is_pro_mode or is_expert_mode:
        # ---------- Horizon lamps ----------
        st.divider()
        with st.expander("Zeithorizonte anzeigen", expanded=False):
            st.subheader("5 Zeithorizont-Ampeln")
            st.markdown('<div class="panel-caption">Verdichtete Eignung je Anlagehorizont. Die Werte sind enger an Trading-Case, Investment-Case, Marktregime, Red Flags und bei Positionen auch an die Exit-Sicht gekoppelt.</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            for col, (lab, scv) in zip(cols, hmap.items()):
                tone, status_word, status_text = horizon_status_meta(scv)
                icon = horizon_icon(scv)
                col.markdown(
                    f"""
                    <div class="horizon-card {tone}">
                        <div class="horizon-top">
                            <div class="horizon-label">{lab}</div>
                            <div class="horizon-icon">{icon}</div>
                        </div>
                        <div class="horizon-value">{status_word}</div>
                        <div class="horizon-score">{scv}/100</div>
                        <div class="horizon-sub">{status_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ---------- Recommendation + Why block ----------
        st.divider()
        with st.expander("Erweiterte Entscheidungsansicht", expanded=False):
            st.subheader("Handlungsempfehlung")
            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.markdown(
                    f"""
                    <div class="reco-card context">
                        <div>
                            <div class="reco-top">
                                <div class="reco-label">Analysekontext</div>
                                <div class="reco-icon">🧭</div>
                            </div>
                            <div class="reco-value">{display_mode_label(mode_label)}</div>
                        </div>
                        <div class="reco-chip">Aktueller Rahmen</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                main_label = position_action if position_mode else display_emp_label(result.get("emp", "-"))
                main_chip = "Positions-Aktion" if position_mode else "Zentrale Aussage"
                st.markdown(
                    f"""
                    <div class="reco-card main">
                        <div>
                            <div class="reco-top">
                                <div class="reco-label">Haupteinschätzung</div>
                                <div class="reco-icon">🎯</div>
                            </div>
                            <div class="reco-value">{main_label}</div>
                        </div>
                        <div class="reco-chip">{main_chip}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div class="reco-card conviction">
                        <div>
                            <div class="reco-top">
                                <div class="reco-label">Überzeugungsgrad</div>
                                <div class="reco-icon">📌</div>
                            </div>
                            <div class="reco-value">{display_conv_label(result.get("conv", "-"))}</div>
                        </div>
                        <div class="reco-chip">Vertrauen ins Setup</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c4:
                st.markdown(
                    f"""
                    <div class="reco-card signal">
                        <div>
                            <div class="reco-top">
                                <div class="reco-label">Kurzfristsignal</div>
                                <div class="reco-icon">⚡</div>
                            </div>
                            <div class="reco-value">{display_stb_label(stb_signal)}</div>
                        </div>
                        <div class="reco-delta">Timing: {tb_timing_text} | Score: {stb_score}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c5:
                st.markdown(
                    f"""
                    <div class="reco-card market">
                        <div>
                            <div class="reco-top">
                                <div class="reco-label">Marktumfeld</div>
                                <div class="reco-icon">🌍</div>
                            </div>
                            <div class="reco-value">{market_regime_label(market_info["regime"])}</div>
                        </div>
                        <div class="reco-delta">{market_info["ampel"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption("Diese erweiterte Sicht zeigt die zusätzlichen Diagnose- und Einordnungsbausteine der Entscheidung.")
