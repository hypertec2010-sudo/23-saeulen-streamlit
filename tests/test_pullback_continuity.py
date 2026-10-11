from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from modules import live_monitor
from modules.structural_trade_plan import (
    confirmed_support_stop, evaluate_pullback_continuity, plan_stop_reference,
    stop_evidence, update_pullback_plan,
)

NOW = datetime(2026, 10, 11, 15, 0)


@pytest.fixture
def anchor():
    return dict(
        price=110, entry_low=100, entry_high=105, invalidation=95,
        target=125, target_source="Widerstand", quality="confirmed",
        created_at=(NOW - timedelta(days=2)).strftime("%d.%m.%Y %H:%M:%S"),
    )


@pytest.mark.parametrize("price,text", [
    (111, "kein Rücksetzer"), (107, "Richtung Plan-Zone"),
    (102, "in Plan-Zone"), (98, "über Invalidierung"),
    (95, "invalidiert"), (94, "invalidiert"),
])
def test_original_entry_and_invalidation_define_pullback(anchor, price, text):
    assert text in evaluate_pullback_continuity(anchor, {"price": price}, now_at=NOW)


@pytest.mark.parametrize("state", ["observe", "plan", "wait_entry", "buy"])
def test_rescans_keep_original_anchor_and_creation_date(anchor, state):
    original = dict(anchor)
    candidate = dict(anchor, entry_low=102, entry_high=107, invalidation=97)
    kept = update_pullback_plan(anchor, {"price": 103, "decision_state": state}, candidate, now_at=NOW)
    assert kept == original
    assert anchor == original  # Never mutate the caller's plan.


@pytest.mark.parametrize("gate", ["hard_gate", "invalidated"])
def test_current_hard_gate_always_invalidates_history(anchor, gate):
    current = {"price": 102, gate: True, "decision_state": "buy"}
    assert "⛔" in evaluate_pullback_continuity(anchor, current, now_at=NOW)
    assert update_pullback_plan(anchor, current, anchor, now_at=NOW) == {}


def test_expiration_and_missing_date_do_not_preserve_unverified_anchor(anchor):
    current = {"price": 102, "decision_state": "observe"}
    for date in ["", "bad date", (NOW - timedelta(days=31)).strftime("%d.%m.%Y %H:%M:%S")]:
        old = dict(anchor, created_at=date)
        assert update_pullback_plan(old, current, anchor, now_at=NOW) == {}
    old = dict(anchor, created_at=(NOW - timedelta(days=31)).strftime("%d.%m.%Y %H:%M:%S"))
    fresh = update_pullback_plan(old, dict(current, decision_state="plan"), anchor, now_at=NOW)
    assert fresh["created_at"] == NOW.strftime("%d.%m.%Y %H:%M:%S")


def test_missing_fresh_price_is_not_a_current_validity_confirmation(anchor):
    status = evaluate_pullback_continuity(anchor, {"price": None}, now_at=NOW)
    assert "nicht aktuell prüfbar" in status
    kept = update_pullback_plan(anchor, {"price": None, "decision_state": "observe"}, {}, now_at=NOW)
    assert kept == anchor
    assert "buy" not in kept and "decision_state" not in kept


def test_invalid_new_plan_is_not_recorded(anchor):
    for invalid in [dict(anchor, invalidation=106), dict(anchor, entry_high=None), dict(anchor, price=None)]:
        assert update_pullback_plan({}, {"decision_state": "plan"}, invalid, now_at=NOW) == {}


def test_target_change_is_explicit_but_does_not_move_old_plan(anchor):
    current = {"price": 102, "target": 135, "decision_state": "observe"}
    assert "Zielbasis gewechselt" in evaluate_pullback_continuity(anchor, current, now_at=NOW)
    assert update_pullback_plan(anchor, current, dict(anchor, target=135), now_at=NOW)["target"] == 125


def test_stop_provenance_follows_selected_plan_reference():
    result = {"chart_invalidation_level": 95, "chart_invalidation_source": "Retest-Tief",
              "stop_used": 94, "stop_source": "ATR"}
    decision = {"crv_structure_source": "Kein mehrfach bestätigtes Swing-Tief", "crv_stop_quality": "proxy"}
    assert plan_stop_reference(result, decision) == {"stop": 95, "source": "Retest-Tief", "quality": "estimated"}
    result.pop("chart_invalidation_level")
    assert plan_stop_reference(result, decision) == {"stop": 94, "source": "ATR", "quality": "proxy"}


def test_confirmed_pivots_and_buffer_remain_diagnostic():
    lows = [100.0] * 40
    lows[8], lows[23] = 95, 95.2
    df = pd.DataFrame({"Low": lows})
    structure = confirmed_support_stop(df, current_price=110, entry_reference=103, atr=2)
    assert structure["quality"] == "confirmed_swing"
    assert structure["touches"] >= 2 and structure["stop"] < 95
    assert structure["support_buffer"] > 0
    used = stop_evidence(96, "ATR", structure)
    assert used["quality"] == "proxy" and used["used_stop"] == 96
    assert used["confirmed_structure_stop"] == structure["stop"]
    assert stop_evidence(96, "Bestätigtes Swing-Support", {})["quality"] != "confirmed"
    lows[23] = 100
    assert confirmed_support_stop(pd.DataFrame({"Low": lows}), current_price=110, entry_reference=103)["quality"] == "unverified"


def test_history_integration_keeps_old_plan_without_promoting_current_state(monkeypatch, anchor):
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    session = State()
    monkeypatch.setattr(live_monitor, "st", SimpleNamespace(session_state=session))
    monkeypatch.setattr(live_monitor, "get_current_berlin_time", lambda: NOW, raising=False)
    monkeypatch.setattr(live_monitor, "_v227_load_persistent_live_history", lambda: ({}, []))
    monkeypatch.setattr(live_monitor, "_v227_save_persistent_live_history", lambda *args: None)
    monkeypatch.setattr(live_monitor, "_v2416_log_event", lambda **kwargs: None, raising=False)

    def scan(state, price, gate=False):
        icon = {"plan": "🟡", "observe": "⚪", "buy": "🟢", "blocked": "🔴"}[state]
        row = {"Ticker": "META", "Kurs": price, "Ampel": icon, "Status": state,
               "Live-Score": "54/100", "__decision_state_v3021az": state,
               "__entry_hard_gate": gate, "__invalidated": False,
               "Plan-Entry-Unterkante": 100, "Plan-Entry-Oberkante": 105,
               "Plan-Invalidierung": 95, "Plan-Stopqualität": "confirmed", "Plan-Zielwert": 125}
        return live_monitor.apply_live_watchlist_status_history_v220(
            pd.DataFrame([row]), watchlist_name="Test", style_name="Ausgewogen | Swing")[0]

    scan("plan", 110)
    key = "Test::Ausgewogen | Swing::META"
    original = dict(session.live_watchlist_status_state_v220[key]["trade_plan"])
    frame = scan("observe", 102)
    assert "in Plan-Zone" in frame.iloc[0]["Plan-Status"]
    assert frame.iloc[0]["Status"] == "observe" and frame.iloc[0]["Ampel"] == "⚪"
    assert frame.iloc[0]["Trade-State"] == "Beobachten"
    assert session.live_watchlist_status_state_v220[key]["trade_plan"] == original
    scan("buy", 103)
    assert session.live_watchlist_status_state_v220[key]["trade_plan"] == original
    frame = scan("blocked", 103, gate=True)
    assert "⛔" in frame.iloc[0]["Plan-Status"]
    assert session.live_watchlist_status_state_v220[key]["trade_plan"] == {}
