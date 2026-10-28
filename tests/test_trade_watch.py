"""Trade watcher: actual rule engine, provider boundary, storage and Streamlit UI."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from modules import trade_watch as w
from modules.storage.manager import StorageManager
from modules.storage.local_backend import LocalJsonBackend

ARM = "2026-10-27T14:00:00Z"
NOW = "2026-10-27T14:10:11Z"


def idea(**changes):
    args = dict(ticker="HALO", entry=100, invalidation=95, max_entry=102,
                target=110, mode="close5", expires_at="2026-10-30T20:00Z", now=ARM,
                source={"stop_quality": "proxy", "fingerprint": "original"})
    args.update(changes)
    return w.make_idea(**args)


def bars(start=ARM, periods=11, close=99.):
    idx = pd.date_range(start, periods=periods, freq="min")
    return pd.DataFrame({"Open": close, "High": close + .2, "Low": close - .2, "Close": close}, index=idx)


def breakout():
    frame = bars()
    frame.loc[frame.index[9:], ["Open", "High", "Low", "Close"]] = [100.5, 100.7, 100.3, 100.5]
    return frame


@pytest.mark.parametrize("changes", [
    {"invalidation": 100}, {"invalidation": 0}, {"entry": float("nan")},
    {"max_entry": 99}, {"target": 99}, {"mode": "unknown"},
    {"ticker": "SAP.F"}, {"expires_at": ARM}, {"expires_at": "2027-01-01T00:00Z"},
    {"expires_at": "2026-10-30 20:00"},
])
def test_invalid_ideas_are_rejected(changes):
    with pytest.raises(ValueError):
        idea(**changes)


def test_complete_closed_bar_confirms_once_and_preserves_frozen_analysis():
    original = idea()
    before = deepcopy(original)
    result = w.evaluate(original, breakout(), now=NOW)
    assert original == before
    assert result["state"] == "triggered" and result["triggered_at"] == "2026-10-27T14:10:00+00:00"
    assert result["source"] == original["source"] and result["invalidation"] == 95
    again = w.evaluate(result, breakout(), now="2026-10-27T14:10:30Z")
    assert len([e for e in again["events"] if e["kind"] == "trigger"]) == 1


@pytest.mark.parametrize("mode,n", [("close5", 5), ("close15", 15)])
def test_closed_bar_uses_full_interval_and_publication_buffer(mode, n):
    frame = bars(periods=n, close=100.5)
    end = pd.Timestamp(ARM) + pd.Timedelta(minutes=n)
    i = idea(mode=mode)
    early = w.evaluate(i, frame, now=end + pd.Timedelta(seconds=9))
    assert early["state"] != "triggered"
    final = w.evaluate(early, frame, now=end + pd.Timedelta(seconds=11))
    assert final["state"] == "triggered"


def test_same_minute_invalidation_beats_entry_and_gaps_do_not_improve_signal():
    frame = breakout()
    frame.loc[frame.index[9], "Low"] = 94
    result = w.evaluate(idea(), frame, now=NOW)
    assert result["state"] == "invalidated" and not result["triggered_at"]
    gap = w.evaluate(idea(), breakout().drop(breakout().index[3]), now=NOW)
    assert gap["gap"] and gap["state"] != "triggered"
    recovered = w.evaluate(gap, breakout(), now="2026-10-27T14:10:30Z")
    assert recovered["gap"] and recovered["state"] != "triggered"


def test_invalidation_after_confirmation_retains_both_events_in_order():
    frame = breakout()
    frame.loc[frame.index[10], "Low"] = 94
    result = w.evaluate(idea(), frame, now=NOW)
    assert result["state"] == "invalidated"
    assert [e["kind"] for e in result["events"]] == ["trigger", "invalidated"]


def test_pre_creation_and_partial_first_bar_cannot_trigger_or_invalidate():
    frame = breakout()
    frame.loc[frame.index[:1], "Low"] = 90
    result = w.evaluate(idea(now="2026-10-27T14:00:30Z"), frame, now=NOW)
    assert result["state"] == "triggered" and not result["gap"]
    old = bars(start="2026-10-27T13:50:00Z", periods=10, close=101)
    result = w.evaluate(idea(), old, now=NOW)
    assert result["state"] == "waiting" and result["price"] is None


@pytest.mark.parametrize("price,state", [(100.5, "triggered"), (103, "overextended"), (99.7, "near"), (98, "waiting")])
def test_contact_uses_current_sample_with_upper_entry_limit(price, state):
    result = w.evaluate(idea(mode="contact"), bars(close=price), now=NOW)
    assert result["state"] == state


def test_historical_breakouts_and_stale_contact_never_become_fresh_signals():
    stale = w.evaluate(idea(), breakout(), now="2026-10-27T14:20:00Z")
    assert stale["state"] != "triggered" and any(e["kind"] == "historical" for e in stale["events"])
    contact = w.evaluate(idea(mode="contact"), bars(close=101), now="2026-10-27T14:20:00Z")
    assert contact["state"] != "triggered" and "veraltet" in contact["health"]


def test_a_lone_current_sample_cannot_hide_a_gap_or_rewind_the_cursor():
    frame = breakout().tail(1)
    result = w.evaluate(idea(mode="contact"), frame, now=NOW)
    assert result["gap"] and result["state"] != "triggered"
    prior = w.evaluate(idea(mode="contact"), bars(), now=NOW)
    rewind = w.evaluate(prior, breakout().iloc[:-1], now="2026-10-27T14:10:30Z")
    assert rewind["state"] != "triggered" and rewind["last_bar_end"] == prior["last_bar_end"]
    assert "älteren Stand" in rewind["health"]


@pytest.mark.parametrize("kind", ["naive", "future", "duplicate", "negative", "bad_ohlc", "empty"])
def test_bad_market_data_never_changes_a_technical_state(kind):
    frame = breakout()
    if kind == "naive": frame.index = frame.index.tz_localize(None)
    if kind == "future": frame.index += pd.Timedelta(hours=2)
    if kind == "duplicate": frame = pd.concat([frame, frame.tail(1)])
    if kind == "negative": frame.iloc[0, 3] = -1
    if kind == "bad_ohlc": frame.iloc[0, 1] = 1
    if kind == "empty": frame = frame.iloc[:0]
    result = w.evaluate(idea(), frame, now=NOW)
    assert result["state"] == "waiting" and not result["triggered_at"]


def test_expiry_pause_and_archive_never_get_new_signals():
    i = idea(expires_at="2026-10-27T14:05Z")
    assert w.evaluate(i, breakout(), now=NOW)["state"] == "expired"
    for flag in ["paused", "archived"]:
        i = idea(); i[flag] = True
        assert w.evaluate(i, breakout(), now=NOW) == i


@pytest.mark.parametrize("time,open_", [
    ("2026-10-27T13:29Z", False), ("2026-10-27T13:30Z", True),
    ("2026-10-27T20:00Z", False), ("2026-11-02T14:29Z", False),
    ("2026-11-02T14:30Z", True), ("2026-11-26T17:00Z", False),
    ("2026-11-27T17:59Z", True), ("2026-11-27T18:00Z", False),
])
def test_us_dst_holiday_and_early_close(time, open_):
    assert w.in_session(time, "US") is open_


def test_weekend_gap_is_not_a_monitoring_gap():
    i = idea(now="2026-10-23T19:59Z", expires_at="2026-10-27T20:00Z")
    i["last_bar_end"] = "2026-10-23T20:00:00Z"
    result = w.evaluate(i, bars(start="2026-10-26T13:30Z", periods=2), now="2026-10-26T13:31:11Z")
    assert not result["gap"]


class Quotes:
    def __init__(self): self.calls = []
    def get_history_light(self, ticker, **kwargs):
        self.calls.append((ticker, kwargs))
        return breakout()


def test_poll_is_lightweight_grouped_throttled_and_scope_is_reset():
    from modules.provider_manager import _HISTORY_REFRESH_TOKEN
    provider = Quotes()
    i = idea(); second = idea(mode="contact")
    result = w.poll([i, second], provider, now=NOW)
    assert len(provider.calls) == 1
    assert provider.calls[0][1] == dict(period="5d", interval="1m", prepost=False, auto_adjust=False)
    assert all(x["state"] == "triggered" for x in result)
    w.poll(result, provider, now="2026-10-27T14:10:20Z")
    assert len(provider.calls) == 1 and not _HISTORY_REFRESH_TOKEN.get()
    w.poll(result, provider, now="2026-10-27T14:10:21Z", force=True)
    assert len(provider.calls) == 2
    w.poll([idea()], provider, now="2026-10-27T21:00Z", force=True)
    assert len(provider.calls) == 2


def test_rate_limit_stops_batch_and_failed_fetch_preserves_last_quote_time():
    from modules.provider_manager import MarketDataRateLimitError
    provider = Quotes()
    def fail(ticker, **kwargs):
        provider.calls.append(ticker)
        raise MarketDataRateLimitError("429")
    provider.get_history_light = fail
    prior = w.evaluate(idea(), bars(), now=NOW)
    result = w.poll([prior, idea(ticker="HPE")], provider, now="2026-10-27T14:11Z", force=True)
    assert provider.calls == ["HALO"]
    assert all("Abruffehler" in x["health"] for x in result)
    assert result[0]["last_fetch_at"] == prior["last_fetch_at"]


def manager_at(path, user="first"):
    return StorageManager(user_id=user, local_backend=LocalJsonBackend(path))


def test_storage_survives_new_session_is_user_scoped_and_rejects_duplicates(tmp_path):
    first = manager_at(tmp_path)
    w.add_idea(first, idea())
    assert w.load_ideas(manager_at(tmp_path))[0]["source"]["fingerprint"] == "original"
    assert w.load_ideas(manager_at(tmp_path, "another")) == []
    with pytest.raises(ValueError, match="bereits"):
        w.add_idea(first, idea())
    for n in range(9): w.add_idea(first, idea(ticker=f"STOCK{n}"))
    with pytest.raises(ValueError, match="Höchstens"):
        w.add_idea(first, idea(ticker="LAST"))


def test_corrupt_or_failed_storage_is_never_overwritten(tmp_path):
    manager = manager_at(tmp_path)
    manager.save_namespace(w.NAMESPACE, {"bad": "format"})
    with pytest.raises(RuntimeError, match="Speicherformat"):
        w.add_idea(manager, idea())
    assert manager.load_namespace(w.NAMESPACE) == {"bad": "format"}


def test_standalone_page_creates_and_reloads_idea_without_analysis_or_provider(tmp_path):
    from streamlit.testing.v1 import AppTest
    script = f'''
import streamlit as st
from modules.storage.manager import StorageManager
from modules.storage.local_backend import LocalJsonBackend
from modules.trade_watch_ui import render_monitor
m = StorageManager(user_id='ui', local_backend=LocalJsonBackend({str(tmp_path)!r}))
def forbidden():
    raise AssertionError('Opening/saving must not fetch quotes')
render_monitor(st, manager=m, provider_factory=forbidden)
'''
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception and not app.error
    app.text_input(key="watch_manual_manual_ticker").set_value("HALO")
    app.number_input(key="watch_manual_manual_entry").set_value(100.)
    app.number_input(key="watch_manual_manual_stop").set_value(95.)
    app.number_input(key="watch_manual_manual_ceiling").set_value(102.)
    app.checkbox(key="watch_manual_manual_reviewed").check()
    next(b for b in app.button if b.label == "Trade-Idee verfolgen").click().run()
    assert not app.exception and not app.error and app.success
    assert len(w.load_ideas(manager_at(tmp_path, "ui"))) == 1
    restored = AppTest.from_string(script).run(timeout=20)
    assert not restored.exception and restored.dataframe[0].value.iloc[0]["Ticker"] == "HALO"
    restored.button(key="watch_pause").click().run()
    assert not restored.exception and w.load_ideas(manager_at(tmp_path, "ui"))[0]["paused"]
    restored.button(key="watch_archive").click().run()
    assert not restored.exception and w.load_ideas(manager_at(tmp_path, "ui"))[0]["archived"]


def test_real_ui_poll_with_independent_fragment(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from modules import trade_watch_ui as ui
    provider = Quotes()
    # Only clock, persistence and provider boundary are replaced.
    real_utc = w.utc
    monkeypatch.setattr(w, "utc", lambda value=None: real_utc(value if value is not None else NOW))
    monkeypatch.setattr(ui, "_test_quotes", provider, raising=False)
    manager = manager_at(tmp_path, "ui")
    w.add_idea(manager, idea())
    script = f'''
import streamlit as st
from modules.storage.manager import StorageManager
from modules.storage.local_backend import LocalJsonBackend
from modules import trade_watch_ui as ui
m = StorageManager(user_id='ui', local_backend=LocalJsonBackend({str(tmp_path)!r}))
ui.render_monitor(st, manager=m, provider_factory=lambda: ui._test_quotes)
'''
    app = AppTest.from_string(script).run(timeout=20)
    app.button(key="watch_poll").click().run()
    assert not app.exception and not app.error
    assert len(provider.calls) == 1 and w.load_ideas(manager)[0]["state"] == "triggered"
    app.toggle(key="watch_auto").set_value(True).run()
    assert not app.exception
    assert len(provider.calls) == 1  # same instant, the 30s throttle still applies


def test_integration_has_native_page_and_capture_paths_without_analyzer_import():
    root = Path(__file__).resolve().parents[1]
    page = (root / "pages/trade_watch.py").read_text()
    legacy = (root / "legacy_app.py").read_text()
    assert "run_workspace_page" not in page
    assert "pages/trade_watch.py" in (root / "modules/app_shell.py").read_text()
    assert '_capture_trade_watch_bx(st, _summary_frame_v3021l' in legacy
    assert '_capture_single_watch_bx(st, _risk_module.build_analysis_risk_basis(result)' in legacy


def test_light_provider_bypasses_previous_15_minute_history_cache(monkeypatch):
    from modules.provider_manager import MarketDataProvider, _HISTORY_REFRESH_TOKEN
    class Primary:
        name = "fake"
        calls = 0
        def history(self, ticker, **kwargs):
            self.calls += 1
            return breakout() if self.calls > 1 else bars()
    primary = Primary()
    provider = MarketDataProvider(primary=primary)
    monkeypatch.setattr(provider, "_pace_request", lambda **kwargs: None)
    args = dict(period="5d", interval="1m", prepost=False, auto_adjust=False)
    provider.get_history_light("HALO", **args)
    result = w.poll([idea()], provider, now=NOW)
    assert primary.calls == 2 and result[0]["state"] == "triggered"
    assert not _HISTORY_REFRESH_TOKEN.get()


def test_capture_from_persisted_screener_basis_needs_no_analysis(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from modules.risk_calculator import RISK_BASIS_VERSION
    monkeypatch.setenv("APP_STORAGE_BACKEND", "local")
    monkeypatch.setenv("APP_STORAGE_LOCAL_DIR", str(tmp_path))
    monkeypatch.setenv("APP_USER_ID", "capture")
    basis = dict(version=RISK_BASIS_VERSION, ticker="HALO", price=99., zone_high=100., stop=95.,
                 target=110., fingerprint="persisted", currency="USD", stop_quality="proxy", data_date="26.10.2026")
    script = f'''
import streamlit as st
import pandas as pd
from modules.trade_watch_ui import render_capture
render_capture(st, pd.DataFrame([{{'Ticker':'HALO', 'Status':'Setup planen', '__risk_basis':{basis!r}}}]))
'''
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception and app.number_input(key="watch_screener_persisted_entry").value == 100.
    app.checkbox(key="watch_screener_persisted_reviewed").check()
    next(b for b in app.button if b.label == "Trade-Idee verfolgen").click().run()
    assert not app.exception and app.success
    captured = w.load_ideas(manager_at(tmp_path, "capture"))[0]
    assert captured["source"] == basis and captured["entry"] == 100.
