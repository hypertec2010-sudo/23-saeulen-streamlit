# -*- coding: utf-8 -*-
"""Application shell for the v28.4.1 Streamlit multipage app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

APP_VERSION = "v28.4.1"
ROOT = Path(__file__).resolve().parents[1]


def configure_app() -> None:
    st.set_page_config(
        page_title=f"Capital-Hill-Score-Modell {APP_VERSION}",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _user_value(name: str, default: str = "") -> str:
    try:
        value = getattr(st.user, name, None)
        if value:
            return str(value)
    except Exception:
        pass
    try:
        return str(st.user.get(name, default) or default)
    except Exception:
        return default


def require_access() -> None:
    """Apply the existing Google OIDC gate before a page is executed."""
    try:
        logged_in = bool(st.user.is_logged_in)
    except Exception:
        logged_in = False

    if not logged_in:
        st.title("23 Säulen Analyse")
        st.info("Bitte mit Google anmelden, um die App zu nutzen.")
        st.button("Mit Google anmelden", on_click=st.login, type="primary", key="google_login_shell_v282")
        st.stop()

    try:
        allowed = {
            str(mail).strip().lower()
            for mail in st.secrets.get("access", {}).get("allowed_emails", [])
            if str(mail).strip()
        }
    except Exception:
        allowed = set()

    email = _user_value("email").strip().lower()
    if allowed and email not in allowed:
        st.error(f"Dieses Konto ist nicht freigeschaltet: {email or 'unbekannt'}")
        st.button("Abmelden", on_click=st.logout, key="google_logout_denied_shell_v282")
        st.stop()

    with st.sidebar:
        name = _user_value("name") or email or "angemeldet"
        st.caption(f"Angemeldet als: {name}")
        if email and email.lower() not in name.lower():
            st.caption(email)
        st.button("Abmelden", on_click=st.logout, key="google_logout_shell_v282", use_container_width=True)
        st.divider()


def _page(path: str, title: str, icon: str, url_path: str, *, default: bool = False):
    return st.Page(
        path,
        title=title,
        icon=icon,
        url_path=url_path,
        default=default,
    )


def render_navigation() -> None:
    """Render native navigation with a compatibility fallback."""
    if hasattr(st, "navigation") and hasattr(st, "Page"):
        navigation = st.navigation(
            {
                "Analyse": [
                    _page("pages/analysis.py", "Sofortanalyse", "🔎", "analyse", default=True),
                    _page("pages/radar.py", "Kandidaten-Radar", "🎯", "radar"),
                ],
                "Trading": [
                    _page("pages/watchlists.py", "Watchlisten", "📋", "watchlisten"),
                    _page("pages/positions.py", "Positionen / Exit", "🛡️", "positionen"),
                    _page("pages/trade_journal.py", "Trade-Journal", "📓", "trade-journal"),
                ],
            },
            position="sidebar",
        )
        navigation.run()
        return

    # Kompatibilität für ältere Streamlit-Versionen ohne st.navigation.
    choice = st.sidebar.radio(
        "Arbeitsbereich",
        ["Sofortanalyse", "Kandidaten-Radar", "Watchlisten", "Positionen / Exit", "Trade-Journal"],
        key="fallback_navigation_v282",
    )
    from modules.page_runtime import run_workspace_page

    mapping = {
        "Sofortanalyse": ("Sofortanalyse", None),
        "Kandidaten-Radar": ("Kandidaten-Radar", None),
        "Watchlisten": ("Watchlisten", "📡 Live-Screener"),
        "Positionen / Exit": ("Positionen", "📌 Positionen / Exit"),
        "Trade-Journal": ("Watchlisten", "📓 Trade-Journal"),
    }
    workspace, cockpit = mapping[choice]
    run_workspace_page(workspace, cockpit_area=cockpit, page_label=choice)
