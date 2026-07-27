# -*- coding: utf-8 -*-
"""Runtime bridge between native Streamlit pages and the stable legacy UI."""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LEGACY_APP = ROOT / "legacy_app.py"
VALID_WORKSPACES = {"Sofortanalyse", "Watchlisten", "Positionen", "Kandidaten-Radar"}
VALID_COCKPIT_AREAS = {
    "📡 Live-Screener",
    "📐 Risiko-Rechner",
    "📌 Positionen / Exit",
    "📓 Trade-Journal",
    "🧾 Historie & Details",
}


def _clear_legacy_workspace_query() -> None:
    """Prevent old Live-Monitor query params from overriding native navigation."""
    try:
        for key in ("workspace", "live_monitor", "refresh", "live_horizon"):
            if key in st.query_params:
                del st.query_params[key]
    except Exception:
        pass
    st.session_state["_v2411_live_query_restore_done"] = True


def _activate_page_context(
    workspace: str,
    cockpit_area: Optional[str],
    page_label: Optional[str],
) -> bool:
    """Activate a native page without overwriting in-page cockpit navigation.

    Returns True when the user has actually entered another native page. Widget
    reruns on the same page must preserve the cockpit radio selection.
    """
    requested_page = page_label or workspace
    previous_page = st.session_state.get("active_native_page_v282")
    page_changed = previous_page != requested_page

    st.session_state["workspace_mode"] = workspace

    if cockpit_area is not None:
        current_cockpit = st.session_state.get("watchlist_cockpit_area_v2413")
        invalid_cockpit = current_cockpit not in VALID_COCKPIT_AREAS
        if page_changed or invalid_cockpit:
            st.session_state["watchlist_cockpit_area_v2413"] = cockpit_area

    st.session_state["active_native_page_v282"] = requested_page
    return page_changed


def run_workspace_page(
    workspace: str,
    *,
    cockpit_area: Optional[str] = None,
    page_label: Optional[str] = None,
) -> None:
    if workspace not in VALID_WORKSPACES:
        raise ValueError(f"Unbekannter Workspace: {workspace}")
    if cockpit_area is not None and cockpit_area not in VALID_COCKPIT_AREAS:
        raise ValueError(f"Unbekannter Cockpit-Bereich: {cockpit_area}")
    if not LEGACY_APP.exists():
        st.error("legacy_app.py fehlt. Bitte den vollständigen v28.3.2-Paketinhalt deployen.")
        st.stop()

    page_changed = _activate_page_context(workspace, cockpit_area, page_label)
    # Query-Parameter nur beim echten nativen Seitenwechsel bereinigen. Eine
    # Bereinigung bei jedem Widget- oder Auto-Refresh-Rerun kann den laufenden
    # Fragment-Zeitplan des Live-Screeners unnoetig destabilisieren.
    if page_changed:
        _clear_legacy_workspace_query()

    os.environ["CAPITAL_HILL_MULTIPAGE"] = "1"
    runpy.run_path(str(LEGACY_APP), run_name="__capital_hill_legacy_v2832__")
