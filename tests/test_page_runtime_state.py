from __future__ import annotations

from types import SimpleNamespace

import modules.page_runtime as runtime


def test_cockpit_selection_survives_same_page_reruns(monkeypatch) -> None:
    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(runtime, "st", fake_st)

    assert runtime._activate_page_context("Watchlisten", "📡 Live-Screener", "Watchlisten") is True
    assert fake_st.session_state["watchlist_cockpit_area_v2413"] == "📡 Live-Screener"

    fake_st.session_state["watchlist_cockpit_area_v2413"] = "📓 Trade-Journal"
    assert runtime._activate_page_context("Watchlisten", "📡 Live-Screener", "Watchlisten") is False
    assert fake_st.session_state["watchlist_cockpit_area_v2413"] == "📓 Trade-Journal"

    assert runtime._activate_page_context("Positionen", "📌 Positionen / Exit", "Positionen / Exit") is True
    assert fake_st.session_state["watchlist_cockpit_area_v2413"] == "📌 Positionen / Exit"


def test_invalid_current_cockpit_is_repaired(monkeypatch) -> None:
    fake_st = SimpleNamespace(
        session_state={
            "active_native_page_v282": "Watchlisten",
            "watchlist_cockpit_area_v2413": "unbekannt",
        }
    )
    monkeypatch.setattr(runtime, "st", fake_st)

    changed = runtime._activate_page_context("Watchlisten", "📡 Live-Screener", "Watchlisten")
    assert changed is False
    assert fake_st.session_state["watchlist_cockpit_area_v2413"] == "📡 Live-Screener"
