# -*- coding: utf-8 -*-
"""Runtime bridge between native Streamlit pages and the stable legacy UI."""

from __future__ import annotations

import os
import runpy
import time
from pathlib import Path
from typing import Optional

import streamlit as st

from modules.version_info import APP_VERSION

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


def _render_performance_sidebar(snapshot: dict | None) -> None:
    """Render the last completed performance snapshot before legacy code runs.

    Rendering before ``runpy.run_path`` keeps the diagnostic visible even when
    legacy code ends the current Streamlit run via ``st.stop`` or ``st.rerun``.
    The figures therefore intentionally describe the last *completed* page run.
    """
    try:
        with st.sidebar.expander("⏱ Performance", expanded=False):
            if not snapshot:
                st.caption("Messung wird nach dem ersten vollständig abgeschlossenen Seitenlauf angezeigt.")
                return

            elapsed_s = float(snapshot.get("elapsed_s", 0.0) or 0.0)
            storage_perf = snapshot.get("storage") or {}
            st.caption(f"Letzter vollständiger Seitenlauf: {elapsed_s:.2f} s")
            if storage_perf:
                reads = int(storage_perf.get("backend_loads", 0) or 0)
                hits = int(storage_perf.get("cache_hits", 0) or 0)
                backend_ms = float(storage_perf.get("backend_ms", 0.0) or 0.0)
                saves = int(storage_perf.get("save_calls", 0) or 0)
                st.caption(
                    f"Storage: {reads} Backend-Reads · {hits} Doppel-Reads vermieden · "
                    f"{backend_ms/1000.0:.2f} s Remote-Zeit · {saves} Writes"
                )
                slow = list(storage_perf.get("slow_namespaces") or [])[:4]
                if slow:
                    st.caption("Langsamste Storage-Bereiche des letzten Laufs:")
                    for row in slow:
                        st.caption(
                            f"• {row.get('namespace', '-')}: {float(row.get('ms', 0.0) or 0.0)/1000.0:.2f} s "
                            f"({int(row.get('backend_loads', 0) or 0)} Read)"
                        )
            if elapsed_s >= 1.5:
                st.caption(
                    "Hinweis: Ein Streamlit-Widget löst weiterhin einen Seiten-Rerun aus. "
                    "Die Messung zeigt, ob Storage oder der übrige Seitenaufbau dominiert."
                )
    except Exception:
        pass


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
        st.error(f"legacy_app.py fehlt. Bitte den vollständigen {APP_VERSION}-Paketinhalt deployen.")
        st.stop()

    page_changed = _activate_page_context(workspace, cockpit_area, page_label)
    # Query-Parameter nur beim echten nativen Seitenwechsel bereinigen. Eine
    # Bereinigung bei jedem Widget- oder Auto-Refresh-Rerun kann den laufenden
    # Fragment-Zeitplan des Live-Screeners unnoetig destabilisieren.
    if page_changed:
        _clear_legacy_workspace_query()

    os.environ["CAPITAL_HILL_MULTIPAGE"] = "1"

    # v30.21af: render the last completed measurement *before* legacy code.
    # Some legacy paths legitimately call st.rerun()/st.stop(), which means any
    # UI emitted after runpy.run_path may never be reached on that rerun.
    try:
        _render_performance_sidebar(st.session_state.get("_chsm_perf_last_run_v3021ae"))
    except Exception:
        _render_performance_sidebar(None)

    # v30.21ae: lightweight whole-page timing. The legacy bridge still runs the
    # complete page script on every Streamlit interaction, so this gives us an
    # objective baseline while the storage layer reports its own share.
    started = time.perf_counter()
    runpy.run_path(str(LEGACY_APP), run_name="__capital_hill_legacy_v2832__")
    elapsed_s = max(0.0, time.perf_counter() - started)

    storage_perf = {}
    try:
        manager = st.session_state.get("_chsm_storage_manager_v3021ae")
        if manager is not None and callable(getattr(manager, "performance_snapshot", None)):
            storage_perf = manager.performance_snapshot() or {}
    except Exception:
        storage_perf = {}

    snapshot = {
        "page": requested_page if (requested_page := (page_label or workspace)) else workspace,
        "workspace": workspace,
        "elapsed_s": round(elapsed_s, 3),
        "storage": storage_perf,
    }
    try:
        st.session_state["_chsm_perf_last_run_v3021ae"] = snapshot
        history = list(st.session_state.get("_chsm_perf_history_v3021ae") or [])
        history.append(snapshot)
        st.session_state["_chsm_perf_history_v3021ae"] = history[-12:]
    except Exception:
        pass
