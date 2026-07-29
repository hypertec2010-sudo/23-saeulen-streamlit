# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from modules.page_runtime import _activate_page_context

AREAS = [
    "📡 Live-Screener",
    "📐 Risiko-Rechner",
    "📌 Positionen / Exit",
    "📓 Trade-Journal",
    "🧾 Historie & Details",
]

st.set_page_config(page_title="Cockpit State CI")
_activate_page_context("Watchlisten", "📡 Live-Screener", "CI-Watchlisten")
selected = st.radio("Cockpit", AREAS, key="watchlist_cockpit_area_v2413")
st.write(f"Aktiver Bereich: {selected}")
st.button("Normaler Rerun", key="normal_rerun")
