"""Actual chart recognition and the boundary from evidence to monitoring."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from modules import trade_watch as w
from modules import watch_setups as setups
from modules.data_freshness import session_day
from modules.live_screener_snapshot import dataframe_to_payload, dataframe_from_payload

NOW = "2026-10-27T14:10:11Z"


def range_result():
    days = pd.bdate_range(end="2026-10-26", periods=100)
    days = pd.DatetimeIndex([d for d in days if session_day(d.date())])[-65:]
    close = np.linspace(87, 98, 65)
    frame = pd.DataFrame({"Open": close - .1, "High": close + .4, "Low": close - .5,
                          "Close": close, "Volume": 1000000.}, index=days)
    tail = frame.index[-20:]
    frame.loc[tail, ["Open", "High", "Low", "Close"]] = [99., 99.5, 98.5, 99.2]
    frame.loc[tail[[4, 11]], "High"] = 100.
    frame.loc[tail[[7, 14]], "Low"] = 97.5
    return dict(ticker="HALO", df=frame, price=99.2, ccy="USD", technical_target_1=110.,
                valid_trade_setup=True, setup_type="Trendfolge", days_earn=30,
                has_upcoming_earnings=True)


def pullback_result():
    r = range_result()
    frame = r["df"]
    frame.loc[frame.index[-25:], ["Open", "High", "Low", "Close"]] = [102., 103., 101., 102.5]
    frame.loc[frame.index[[50, 60]], "Low"] = 100.
    frame.loc[frame.index[56], "High"] = 107.
    r["price"] = 102.5
    return r


@pytest.mark.parametrize("factory,kind", [(range_result, "range_breakout"), (pullback_result, "support_pullback")])
def test_real_detector_derives_levels_from_completed_structure(factory, kind):
    r = factory(); before = r["df"].copy(deep=True)
    p = setups.detect(r, now=NOW)
    assert p["eligible"], p["reason"]
    assert p["kind"] == kind and len(p["checks"]) >= 4
    assert 0 < p["invalidation"] < r["price"] < p["entry"] <= p["max_entry"]
    assert p["mode"] == "close5" and p["stop_quality"] == "confirmed_setup_structure"
    assert setups.validate(p, now=NOW, ticker="HALO")[0]
    assert p["basis_date"] == "2026-10-26"
    pd.testing.assert_frame_equal(before, r["df"])


def test_green_and_setup_label_are_not_evidence():
    r = range_result()
    r["df"].loc[r["df"].index[-20:], "High"] = 99.5
    r["df"].loc[r["df"].index[-20:], "Low"] = 98.5
    r.update(ampel="🟢", setup_type="Breakout", stop_used=96., suggested_entry_zone=(100, 101))
    assert not setups.detect(r, now=NOW)["eligible"]


@pytest.mark.parametrize("change", ["single_top", "single_low", "wide", "weak_trend", "bad_volume", "bad_ohlc", "short", "stale", "weekly", "duplicate", "future", "hard_gate", "earnings", "price_broken", "price_chased"])
def test_structurally_incomplete_or_blocked_candidates_are_rejected(change):
    r = range_result(); frame = r["df"]; hard = False
    if change == "single_top": frame.loc[frame.index[-20 + 11], "High"] = 99.5
    if change == "single_low": frame.loc[frame.index[-20 + 14], "Low"] = 98.5
    if change == "wide": frame.loc[frame.index[[-13, -6]], "Low"] = 80.
    if change == "weak_trend": frame["Close"] = np.linspace(110, 90, len(frame)); frame["Open"] = frame.Close; frame["High"] = frame.Close + 1; frame["Low"] = frame.Close - 1
    if change == "bad_volume": frame.iloc[-1, frame.columns.get_loc("Volume")] = 0
    if change == "bad_ohlc": frame.iloc[-1, frame.columns.get_loc("High")] = 1
    if change == "short": r["df"] = frame.tail(40)
    if change == "stale": r["df"] = frame.iloc[:-1]
    if change == "weekly": frame.index = pd.date_range(end="2026-10-26", periods=65, freq="7D")
    if change == "duplicate": r["df"] = pd.concat([frame, frame.tail(1)])
    if change == "future": frame.index = frame.index + pd.Timedelta(days=10)
    if change == "hard_gate": hard = True
    if change == "earnings": r["days_earn"] = 2
    if change == "price_broken": r["price"] = 95
    if change == "price_chased": r["price"] = 110
    p = setups.detect(r, now=NOW, hard_gate=hard)
    assert not p["eligible"], (change, p)
    assert p["reason"]


def test_incomplete_current_daily_bar_never_creates_structure():
    r = range_result()
    before = setups.detect(r, now=NOW)
    today = r["df"].tail(1).copy(); today.index = pd.DatetimeIndex(["2026-10-27"])
    today.loc[:, "High"] = 150
    r["df"] = pd.concat([r["df"], today])
    after = setups.detect(r, now=NOW)
    assert after == before
    r["df"].loc[r["df"].index[-1], "Low"] = before["invalidation"] - .1
    assert not setups.detect(r, now=NOW)["eligible"]


def test_crv_and_old_market_veto_do_not_create_or_remove_a_chart_structure():
    r = range_result()
    first = setups.detect(r, now=NOW)
    r.update(crv_now=.01, technical_crv_now=.01, valid_trade_setup=False,
             market_info={"regime": "NEGATIV"})
    assert setups.detect(r, now=NOW) == first
    r["technical_target_1"] = 100.5
    second = setups.detect(r, now=NOW)
    assert second["eligible"] and second["crv"] < 1
    assert first["entry"] == second["entry"] and first["invalidation"] == second["invalidation"]


def test_integrity_binding_expiry_and_old_free_ideas_are_enforced():
    p = setups.detect(range_result(), now=NOW)
    source = {"watch_setup": p}
    args = dict(ticker="HALO", entry=p["entry"], invalidation=p["invalidation"], max_entry=p["max_entry"],
                mode=p["mode"], target=p["target"], source=source, expires_at=p["valid_until"], now=NOW)
    i = w.make_idea(**args)
    assert w.active(i)
    for changes in [{"entry": p["entry"] - .1}, {"mode": "contact"}, {"source": {}},
                    {"ticker": "HPE"}, {"expires_at": "2026-10-30T20:00Z"}]:
        with pytest.raises(ValueError): w.make_idea(**{**args, **changes})
    edited = deepcopy(p); edited["entry"] -= .1
    assert not setups.validate(edited)[0]
    assert not setups.validate(p, now="2026-10-28T14:00Z")[0]
    old = deepcopy(i); old["source"] = {"stop": 95.}
    assert not w.active(old)
    class Forbidden:
        def get_history_light(self, *args, **kwargs): raise AssertionError("Unverified ideas must not fetch quotes")
    assert w.poll([old], Forbidden(), now=NOW) == [old]


def test_snapshot_roundtrip_preserves_evidence_and_no_legacy_promotion():
    p = setups.detect(range_result(), now=NOW)
    frame = pd.DataFrame([{"Ticker": "HALO", "__watch_setup": p}])
    restored = dataframe_from_payload(json.loads(json.dumps(dataframe_to_payload(frame))))
    assert restored.iloc[0]["__watch_setup"] == p
    assert setups.validate(restored.iloc[0]["__watch_setup"], now=NOW)[0]


def test_xetra_uses_berlin_completed_day_and_provider_free_detection():
    r = range_result(); r["ticker"] = "SAP.DE"
    p = setups.detect(r, now="2026-10-27T09:00:00Z")
    assert p["eligible"] and p["basis_date"] == "2026-10-26"


def test_old_records_stay_readable_but_cannot_poll_or_count_as_a_setup(tmp_path):
    from modules.storage.manager import StorageManager
    from modules.storage.local_backend import LocalJsonBackend
    manager = StorageManager(user_id="old", local_backend=LocalJsonBackend(tmp_path))
    old = dict(id="old-free", ticker="HALO", state="waiting", entry=100., invalidation=95., max_entry=101.,
               mode="contact", source={}, events=[], expires_at="2026-10-30T20:00Z")
    w.save_ideas(manager, [old])
    loaded = w.load_ideas(manager)
    assert loaded[0]["setup_blocked"] and not w.active(loaded[0])
    assert manager.load_namespace(w.NAMESPACE)["ideas"][0] == old


def test_evidence_survives_actual_live_history_cleanup(monkeypatch):
    from types import SimpleNamespace
    from datetime import datetime
    from modules import live_monitor as lm
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    monkeypatch.setattr(lm, "st", SimpleNamespace(session_state=State()))
    monkeypatch.setattr(lm, "get_current_berlin_time", lambda: datetime(2026, 10, 27, 15), raising=False)
    monkeypatch.setattr(lm, "_v227_load_persistent_live_history", lambda: ({}, []))
    monkeypatch.setattr(lm, "_v227_save_persistent_live_history", lambda *args: None)
    monkeypatch.setattr(lm, "_v2416_log_event", lambda **kwargs: None, raising=False)
    package = setups.detect(range_result(), now=NOW)
    raw = pd.DataFrame([dict(Ticker="HALO", Kurs=99.2, Ampel="🟢", Status="Kauftrigger aktiv",
                            **{"Live-Score": "80/100", "__watch_setup": package, "__temporary": 99})])
    history, _ = lm.apply_live_watchlist_status_history_v220(raw, watchlist_name="Test")
    assert history.iloc[0]["__watch_setup"] == package and "__temporary" not in history


@pytest.mark.parametrize("evidence", ["missing", "rejected", "expired"])
def test_ui_has_no_capture_form_for_an_unverified_stock(evidence, monkeypatch):
    from streamlit.testing.v1 import AppTest
    real_utc = w.utc
    monkeypatch.setattr(w, "utc", lambda value=None: real_utc(value if value is not None else NOW))
    row = {"Ticker": "HALO", "Ampel": "🟢", "Status": "Kauftrigger aktiv"}
    if evidence != "missing":
        p = setups.detect(range_result(), now=NOW)
        if evidence == "rejected": p = setups.sealed({**p, "eligible": False, "reason": "Keine Struktur"})
        if evidence == "expired": p = setups.sealed({**p, "valid_until": "2026-10-27T13:00Z"})
        row["__watch_setup"] = p
    script = f'''
import streamlit as st
import pandas as pd
from modules.trade_watch_ui import render_capture
render_capture(st,pd.DataFrame([{row!r}]))
'''
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception and not app.number_input and not app.button
    assert any("Kein überwachbares Chartsetup" in i.value for i in app.info)


def test_single_analysis_capture_survives_the_next_rerender_clock(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from modules.storage.manager import StorageManager
    from modules.storage.local_backend import LocalJsonBackend
    real_utc = w.utc
    clock = {"now": NOW}
    frozen = lambda value=None: real_utc(value if value is not None else clock["now"])
    monkeypatch.setattr(w, "utc", frozen)
    monkeypatch.setattr(setups, "utc", frozen)
    monkeypatch.setenv("APP_STORAGE_BACKEND", "local")
    monkeypatch.setenv("APP_STORAGE_LOCAL_DIR", str(tmp_path))
    monkeypatch.setenv("APP_USER_ID", "single")
    script = '''
import streamlit as st
from modules.trade_watch_ui import capture_analysis
from test_watch_setups import range_result
capture_analysis(st, range_result(), key="single")
'''
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception and not app.number_input
    checkbox_key = app.checkbox[0].key
    app.checkbox[0].check()
    clock["now"] = "2026-10-27T14:11:11Z"
    app.button[0].click().run()
    assert not app.exception and not app.error and app.success
    assert app.checkbox[0].key == checkbox_key
    manager = StorageManager(user_id="single", local_backend=LocalJsonBackend(tmp_path))
    saved = w.load_ideas(manager)[0]
    assert saved["source"]["watch_setup"]["detected_at"] == frozen().isoformat()
