# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, timedelta

import streamlit as st

from modules.live_refresh_policy import build_cache_key, evaluate_refresh

st.set_page_config(page_title="Refresh Policy CI")
interval = int(st.number_input("Intervall Sekunden", min_value=60, value=900, step=60))
elapsed = int(st.number_input("Sekunden seit Scan", min_value=0, value=300, step=1))
now = datetime(2026, 7, 27, 10, 0, 0)
key = build_cache_key("CI", ["AAPL"], "Charttechnik", "Swing")
decision = evaluate_refresh(
    now=now,
    cache={"key": key, "ts": (now - timedelta(seconds=elapsed)).isoformat()},
    expected_cache_key=key,
    interval_seconds=interval,
)
st.metric("Fällig", "Ja" if decision.due else "Nein")
st.write(f"Restsekunden: {decision.remaining_seconds}")
