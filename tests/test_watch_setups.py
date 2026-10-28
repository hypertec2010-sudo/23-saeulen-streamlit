"""Full synthetic OHLCV recognition, reason counts, and read-only UI boundary."""
from copy import deepcopy
from datetime import datetime
import json

import numpy as np
import pandas as pd
import pytest

from modules import trade_watch as w
from modules import watch_setups as s
from modules.data_freshness import session_day
from modules.storage.manager import StorageManager
from modules.storage.local_backend import LocalJsonBackend

NOW = "2026-10-28T15:00:11Z"


def range_result(length=20, *, end="2026-10-27"):
    dates = pd.bdate_range(end=end, periods=170)
    dates = pd.DatetimeIndex([d for d in dates if session_day(d.date())])[-125:]
    close = np.linspace(80, 98, 125)
    df = pd.DataFrame(dict(Open=close-.1, High=close+.4, Low=close-.5, Close=close, Volume=1000000.), index=dates)
    base = df.index[-length-1:-1]
    df.loc[base, ["Open","High","Low","Close"]] = [99.,99.5,98.5,99.2]
    second = {20:11,40:25,60:45}[length]
    df.loc[base[[4,second]], "High"] = 100.
    df.loc[base[[7,second+3]], "Low"] = 97.5
    df.loc[df.index[-1], ["Open","High","Low","Close"]] = [99.2,99.6,98.8,99.2]
    return dict(ticker="HALO", df=df, price=99.2, ccy="USD", technical_target_1=110., days_earn=30, has_upcoming_earnings=True)


def higher_result(pending=False):
    r = range_result(); df = r["df"]
    df.loc[df.index[-30:], ["Open","High","Low","Close"]] = [102.,103.,101.5,102.5]
    df.loc[df.index[110], "Low"] = 97.
    df.loc[df.index[116], "High"] = 107.5
    df.loc[df.index[124 if pending else 120], "Low"] = 100.5
    if not pending:
        df.loc[df.index[122], "High"] = 103.2
    r["price"] = 102.5
    return r


def support_result():
    r = higher_result()
    r["df"].loc[r["df"].index[110], "Low"] = 100.6
    return r


def retest_result():
    r = range_result(); df = r["df"]
    base = df.index[96:116]
    df.loc[base, ["Open","High","Low","Close"]] = [99.,99.5,97.8,99.2]
    df.loc[base[[4,11]], "High"] = 100.
    values = [[100.5,104.,100.2,103.], [102.,103.,101.,102.], [101.8,102.,100.5,101.5],
              [100.5,101.2,99.9,100.5], [100.8,101.6,100.5,101.], [101.,102.,100.7,101.5],
              [101.,101.8,100.8,101.2], [101.2,101.8,101.,101.3], [101.2,101.9,101.,101.5]]
    df.loc[df.index[116:], ["Open","High","Low","Close"]] = values
    df.loc[df.index[116], "Volume"] = 2000000.
    r["price"] = 101.5
    return r


@pytest.mark.parametrize("length", [20,40,60])
def test_confirmed_consolidation_is_recognized_without_lowering_width_rules(length):
    r = range_result(length); before = r["df"].copy(deep=True)
    p = s.detect(r, now=NOW)
    assert p["eligible"], p
    assert p["kind"] == "range_breakout" and p["window"] == length
    assert p["evidence"]["width_pct"] == pytest.approx(2.5)
    assert 0 < p["invalidation"] < r["price"] < p["entry"] <= p["max_entry"]
    pd.testing.assert_frame_equal(before, r["df"])


@pytest.mark.parametrize("factory,kind", [(support_result,"support_pullback"), (higher_result,"higher_low_pullback"), (retest_result,"breakout_retest")])
def test_additional_patterns_require_real_pivot_and_breakout_evidence(factory, kind):
    p = s.detect(factory(), now=NOW)
    assert p["eligible"], p
    assert p["kind"] == kind and p["phase"] == "ready"
    assert s.validate(p, now=NOW)[0] and len(p["checks"]) >= 3


def test_higher_low_requires_a_lower_preceding_confirmed_pivot():
    r = higher_result()
    r["df"].loc[r["df"].index[110], "Low"] = 101.
    a = s.pullback_candidate(r["df"], 2., r["price"], higher=True)
    assert a["phase"] != "ready" and a["reason_code"] == "higher_low_missing"


def test_an_intervening_higher_pivot_cannot_be_skipped_to_invent_a_higher_low():
    r=higher_result()
    r["df"].loc[r["df"].index[114],"Low"]=101.
    a=s.pullback_candidate(r["df"],2.,r["price"],higher=True)
    assert a["phase"]!="ready" and a["reason_code"]=="higher_low_missing"


def test_pending_right_hand_confirmation_is_forming_and_never_eligible():
    r = higher_result(pending=True)
    p = s.detect(r, now=NOW)
    assert not p["eligible"] and p["phase"] == "forming"
    assert p["reason_code"] == "pivot_confirmation_pending"
    assert not s.validate(p, now=NOW)[0]


@pytest.mark.parametrize("price,phase", [(99.2,"ready"),(100.3,"triggered"),(102.,"missed"),(96.,"invalid")])
def test_fixed_range_levels_explain_all_development_stages(price, phase):
    r = range_result(); p = s.detect(r, now=NOW)
    r["price"] = price
    after = s.detect(r, now=NOW)
    assert after["phase"] == phase and after["eligible"] == (phase == "ready")
    assert after["entry"] == p["entry"] and after["invalidation"] == p["invalidation"]


def test_latest_completed_breakout_is_a_past_trigger_not_a_new_ready_setup():
    r=range_result();before=s.detect(r,now=NOW)
    r["df"].loc[r["df"].index[-1],["Open","High","Low","Close"]]=[100.2,100.5,100.1,100.3]
    r["price"]=100.3
    p=s.detect(r,now=NOW)
    assert p["phase"]=="triggered" and not p["eligible"]
    assert p["entry"]==before["entry"]


def test_consolidation_without_a_confirmed_lower_boundary_is_preparation_only():
    r = range_result(); base = r["df"].index[-21:-1]
    r["df"].loc[base[14], "Low"] = 98.5
    p = s.detect(r, now=NOW)
    assert not p["eligible"] and p["phase"] == "forming"
    assert p["reason_code"] == "range_pivots_incomplete"


def test_the_pullback_trigger_does_not_move_up_with_later_bars():
    r = higher_result(); p = s.detect(r, now=NOW)
    r["df"].loc[r["df"].index[-1], ["Open","High","Low","Close"]] = [103.5,105.,103.,104.]
    r["price"] = 104.
    after = s.detect(r, now=NOW)
    assert after["entry"] == p["entry"] and after["phase"] in {"triggered","missed"}


@pytest.mark.parametrize("change,code", [("missing","daily_data_missing"),("short","daily_short"),("stale","daily_stale"),
    ("volume","daily_invalid"),("ohlc","daily_invalid"),("duplicate","daily_duplicate"),("future","daily_future"),
    ("gap","daily_gap"),("trend","trend_missing"),("gate","entry_gate"),("earnings","earnings"),("held","held_position")])
def test_shared_blockers_have_specific_codes_instead_of_just_no_pattern(change,code):
    r = range_result(); df = r["df"]; gate = False
    if change == "missing": r.pop("df")
    if change == "short": r["df"] = df.tail(40)
    if change == "stale": r["df"] = df.iloc[:-1]
    if change == "volume": df.iloc[-1,df.columns.get_loc("Volume")] = 0
    if change == "ohlc": df.iloc[-1,df.columns.get_loc("High")] = 1
    if change == "duplicate": r["df"] = pd.concat([df,df.tail(1)])
    if change == "future": df.index = df.index + pd.Timedelta(days=10)
    if change == "gap": df.index = pd.date_range(end="2026-10-27",periods=len(df),freq="7D")
    if change == "trend": df["Close"] = np.linspace(110,90,len(df)); df["Open"] = df.Close; df["High"] = df.Close+1; df["Low"] = df.Close-1
    if change == "gate": gate = True
    if change == "earnings": r["days_earn"] = 2
    if change == "held": r["buy_in"] = 95.
    p = s.detect(r,now=NOW,hard_gate=gate)
    assert not p["eligible"] and p["reason_code"] == code
    _, counts, _ = s.diagnostic_tables(pd.DataFrame([{"Ticker":"HALO","__watch_setup":p}]),now=NOW)
    assert counts.iloc[0]["Code"] == code


@pytest.mark.parametrize("change", ["no_volume","failed_close","no_retest"])
def test_retest_cannot_be_promoted_without_its_evidence(change):
    r = retest_result(); df = r["df"]
    if change == "no_volume": df.loc[df.index[116],"Volume"] = 500000.
    if change == "failed_close": df.loc[df.index[117],["Open","High","Low","Close"]] = [101.,103.,97.,98.]
    if change == "no_retest": df.loc[df.index[119],"Low"] = 100.7
    candidate = s.retest_candidate(df,2.,r["price"])
    assert candidate["phase"] != "ready"


def test_incomplete_today_cannot_create_a_structure_but_can_invalidate_it():
    r = range_result(); before = s.detect(r,now=NOW)
    day = r["df"].tail(1).copy(); day.index = pd.DatetimeIndex(["2026-10-28"])
    r["df"] = pd.concat([r["df"],day])
    assert s.detect(r,now=NOW) == before
    r["df"].loc[day.index,"Low"] = 90.
    assert s.detect(r,now=NOW)["phase"] == "invalid"


def test_scores_crv_and_old_market_veto_do_not_invent_or_remove_a_pattern():
    r = range_result(); before = s.detect(r,now=NOW)
    r.update(Ampel="🟢", crv_now=.01, valid_trade_setup=False, market_info={"regime":"NEGATIV"})
    assert s.detect(r,now=NOW) == before
    r["technical_target_1"] = 100.5
    after = s.detect(r,now=NOW)
    assert after["eligible"] and after["crv"] < 1 and after["entry"] == before["entry"]


def test_green_label_without_pivots_is_not_a_ready_setup():
    r = range_result(); df = r["df"]
    df.loc[df.index[-21:],"High"] = 99.5
    df.loc[df.index[-21:],"Low"] = 98.5
    r.update(Ampel="🟢",setup_type="Breakout",suggested_entry_zone=(100,101),stop_used=95)
    assert not s.detect(r,now=NOW)["eligible"]


def test_old_unused_ohlcv_does_not_add_an_unrelated_rejection_gate():
    r=range_result();expected=s.detect(r,now=NOW)
    r["df"].iloc[0,r["df"].columns.get_loc("Volume")]=0.
    assert s.detect(r,now=NOW)==expected


def test_integrity_free_input_and_stage_checks_are_enforced_by_the_watcher():
    p = s.detect(range_result(),now=NOW)
    args = dict(ticker="HALO", entry=p["entry"], invalidation=p["invalidation"], max_entry=p["max_entry"],
                target=p["target"], mode=p["mode"], expires_at=p["valid_until"], source={"watch_setup":p}, now=NOW)
    i = w.make_idea(**args); assert w.active(i)
    for changes in ({"source":{}},{"entry":p["entry"]-.01},{"mode":"contact"},{"ticker":"HPE"},{"expires_at":"2026-10-30T20:00Z"}):
        with pytest.raises(ValueError): w.make_idea(**{**args,**changes})
    bad = deepcopy(p); bad["entry"] += .1
    assert not s.validate(bad)[0]
    forming = s.sealed({**p,"eligible":False,"phase":"forming"})
    with pytest.raises(ValueError): w.make_idea(**{**args,"source":{"watch_setup":forming}})
    old = deepcopy(i); old["source"] = {}
    class Forbidden:
        def get_history_light(self,*args,**kwargs): raise AssertionError("Unverified idea fetched quotes")
    assert not w.active(old) and w.poll([old],Forbidden(),now=NOW) == [old]


def test_existing_by_evidence_remains_usable_without_promoting_bx_free_ideas():
    p = s.detect(range_result(),now=NOW)
    p = {k:v for k,v in p.items() if k not in {"fingerprint","phase","phase_label","attempts"}}
    p["policy"] = s.PREVIOUS_POLICY
    p = s.sealed(p)
    assert s.validate(p,now=NOW)[0]
    assert not s.validate({"policy":"chart_watch_v30.21bx"})[0]


def test_hundred_row_diagnostics_count_each_stock_once_and_preserve_the_real_blocker():
    ready = s.detect(range_result(),now=NOW)
    blocked = s.sealed({**ready,"eligible":False,"phase":"blocked","reason_code":"trend_missing","reason":"Trend fehlt","attempts":[]})
    rows = [{"Ticker":f"T{i}","__watch_setup":blocked if i<80 else ready} for i in range(100)]
    stocks, counts, patterns = s.diagnostic_tables(pd.DataFrame(rows),now=NOW)
    assert len(stocks)==100 and counts.Aktien.sum()==100
    assert counts.loc[counts.Code=="trend_missing","Aktien"].iloc[0]==80
    assert len(patterns)>100  # multiple checks per remaining stock, not more stocks


def test_old_scan_and_expired_evidence_have_separate_counts():
    p = s.detect(range_result(),now=NOW)
    p = s.sealed({**p,"valid_until":"2026-10-28T14:00Z"})
    _,counts,_ = s.diagnostic_tables(pd.DataFrame([{"Ticker":"HALO","__watch_setup":p},{"Ticker":"HPE"}]),now=NOW)
    assert set(counts.Code)=={"setup_expired","old_scan"} and counts.Aktien.sum()==2


def test_snapshot_and_actual_history_cleanup_keep_the_evidence(monkeypatch):
    from types import SimpleNamespace
    from modules import live_monitor as lm
    from modules.live_screener_snapshot import dataframe_to_payload,dataframe_from_payload
    class State(dict):
        __getattr__=dict.__getitem__
        __setattr__=dict.__setitem__
    monkeypatch.setattr(lm,"st",SimpleNamespace(session_state=State()))
    monkeypatch.setattr(lm,"get_current_berlin_time",lambda:datetime(2026,10,28,16),raising=False)
    monkeypatch.setattr(lm,"_v227_load_persistent_live_history",lambda:({},[]))
    monkeypatch.setattr(lm,"_v227_save_persistent_live_history",lambda *a:None)
    monkeypatch.setattr(lm,"_v2416_log_event",lambda **kw:None,raising=False)
    p=s.detect(range_result(),now=NOW)
    raw=pd.DataFrame([dict(Ticker="HALO",Kurs=99.2,Ampel="🟢",Status="Kauftrigger aktiv",**{"Live-Score":"80/100","__watch_setup":p,"__temporary":1})])
    history,_=lm.apply_live_watchlist_status_history_v220(raw,watchlist_name="Test")
    assert history.iloc[0]["__watch_setup"]==p and "__temporary" not in history
    restored=dataframe_from_payload(json.loads(json.dumps(dataframe_to_payload(history))))
    assert restored.iloc[0]["__watch_setup"]==p


@pytest.mark.parametrize("phase",["forming","triggered","missed","invalid"])
def test_non_ready_ui_stages_are_visible_but_have_no_activation(phase,monkeypatch):
    from streamlit.testing.v1 import AppTest
    real=w.utc;monkeypatch.setattr(w,"utc",lambda value=None:real(value if value is not None else NOW))
    p=s.detect(range_result(),now=NOW)
    p=s.sealed({**p,"eligible":False,"phase":phase,"phase_label":s.PHASES[phase]})
    script=f'''
import streamlit as st
from modules.trade_watch_ui import capture_form
capture_form(st,{{'ticker':'HALO','watch_setup':{p!r}}})
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and not app.button and not app.number_input
    assert any("Noch keine aktive" in i.value for i in app.info)


def test_single_analysis_form_survives_clock_rerender(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    real=w.utc;clock={"now":NOW}
    frozen=lambda value=None:real(value if value is not None else clock["now"])
    monkeypatch.setattr(w,"utc",frozen);monkeypatch.setattr(s,"utc",frozen)
    monkeypatch.setenv("APP_STORAGE_BACKEND","local");monkeypatch.setenv("APP_STORAGE_LOCAL_DIR",str(tmp_path));monkeypatch.setenv("APP_USER_ID","single")
    script='''
import streamlit as st
from modules.trade_watch_ui import capture_analysis
from test_watch_setups import range_result
capture_analysis(st,range_result(),key='single')
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and app.checkbox
    key=app.checkbox[0].key;app.checkbox[0].check();clock["now"]="2026-10-28T15:01:11Z"
    app.button[0].click().run()
    assert not app.exception and not app.error and app.success and app.checkbox[0].key==key
    manager=StorageManager(user_id="single",local_backend=LocalJsonBackend(tmp_path))
    assert len(w.load_ideas(manager))==1
