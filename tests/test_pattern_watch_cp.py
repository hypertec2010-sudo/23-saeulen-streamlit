from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
import json
import pandas as pd
import pytest
from modules import pattern_watch as pw,chart_patterns as cp
from test_chart_patterns_cm import fixture,NOW

MON='2026-11-09T22:00:00Z'
LIVE='2026-11-09T15:02:00Z'


def idea(kind='double_top',phase='forming'):
    r=fixture(kind,phase);p=cp.build_chart_patterns(r,now=NOW)
    v=next(v for v in p['patterns'] if v['kind']==kind)
    return pw.create([],p,v,now=NOW)[0],r


def append_day(r,close=90,vol=3_000_000):
    f=r['df'].copy();f.loc[pd.Timestamp('2026-11-09')]=[close-.1,close+.3,close-.3,close,vol];return f


def minutes(close=91,stamp='2026-11-09T15:00:00Z'):
    return pd.DataFrame(dict(Open=[close],High=[close+.1],Low=[close-.1],Close=[close],Volume=[1000]),index=pd.DatetimeIndex([pd.Timestamp(stamp)]))


class Store:
    user_id='pattern-test'
    def __init__(self):self.value=None
    def load_namespace(self,key,default=None):return deepcopy(self.value)
    def save_namespace(self,key,data):self.value=deepcopy(data);return True
    def status(self):return {}


@pytest.mark.parametrize('kind',list(cp.KINDS))
def test_capture_all_six_fixed_rules_no_mutation(kind):
    item,r=idea(kind);p=cp.build_chart_patterns(r,now=NOW)
    assert item['original'] in p['patterns'] and item['state']=='forming'
    assert item['expires_at']>'2026-11-07' and len(item['events'])==1
    with pytest.raises(ValueError,match='bereits'):pw.create([item],p,item['original'],now=NOW)


def test_storage_roundtrip_user_scope_and_corrupt_payload():
    item,r=idea();s=Store();pw.save(s,[item]);assert pw.load(s)==[item]
    json.dumps(s.value,allow_nan=False)
    s.user_id='other'
    with pytest.raises(ValueError):pw.load(s)
    s.user_id='pattern-test';s.value['items']=[{}]
    with pytest.raises(ValueError):pw.load(s)


def test_stale_invalid_consumed_and_expired_source_cannot_be_captured():
    item,r=idea();p=item['source'];v=item['original'];p['stale']=True
    with pytest.raises(ValueError):pw.create([],p,v,now=NOW)
    p['stale']=False
    with pytest.raises(ValueError):pw.create([],p,v,now=MON)
    v=deepcopy(v);v['projection_touched']=True;p['patterns']=[v]
    with pytest.raises(ValueError):pw.create([],p,v,now=NOW)


def test_closed_day_confirms_same_formation_and_freezes_rules():
    item,r=idea();before=deepcopy(item)
    out=pw.evaluate(item,append_day(r),now=MON)
    assert out['state']=='confirmed' and out['current']['confirmed_at']=='2026-11-09'
    assert out['original']==item['original'] and item==before
    assert out['current']['invalidation']==item['original']['invalidation']
    assert out['current']['confirmation_buffer']==item['original']['confirmation_buffer']
    assert out['current_package']['chart'][-1]['date']=='2026-11-09'


@pytest.mark.parametrize('kind',list(cp.KINDS))
def test_replay_same_daily_basis_matches_original_pattern(kind):
    item,r=idea(kind)
    out=pw.evaluate(item,r['df'],now=NOW)
    assert out['current']['phase']==item['original']['phase']
    for field in ['projection','invalidation','boundary_now','confirmation_buffer']:
        assert out['current'][field]==pytest.approx(item['original'][field])


def test_missing_volume_never_turns_intraday_hint_into_confirmation():
    item,r=idea('double_bottom');f=append_day(r,110,vol=float('nan'))
    # Mirrored double bottom neckline is around 108; no volume means no daily confirmation.
    out=pw.evaluate(item,f,now=MON)
    assert out['state']!='confirmed' and out['current']['phase']=='forming'


@pytest.mark.parametrize('price,state',[(91,'intraday'),(92.1,'near'),(100,'forming')])
def test_intraday_hints_are_separate_from_day_confirmation(price,state):
    item,r=idea();out=pw.evaluate(item,r['df'],minutes(price),now=LIVE)
    assert out['state']==state and out['current']['phase']=='forming'
    assert out['price_source'].startswith('Abgeschlossene 1m')


def test_old_future_and_naive_quotes_not_used():
    item,r=idea()
    for stamp in ['2026-11-09T14:30:00Z','2026-11-09T15:05:00Z','2026-11-09T15:00:00']:
        out=pw.evaluate(item,r['df'],minutes(91,stamp),now=LIVE)
        assert out['state']!='intraday' and out['price_source']=='Abgeschlossener Tagesschluss'
        assert 'Kein frischer Intraday' in out['health']


def test_intraday_invalidation_does_not_claim_closed_day_break():
    item,r=idea();out=pw.evaluate(item,r['df'],minutes(110),now=LIVE)
    assert out['state']=='forming' and 'Invalidierung' in out['health']
    out=pw.evaluate(item,append_day(r,110),now=MON)
    assert out['state']=='invalidated'


def test_history_revision_and_missing_original_structure_require_new_evidence():
    item,r=idea();f=r['df'].copy();f.loc[f.index[-1],['Open','High','Low','Close']]*=.5
    out=pw.evaluate(item,f,now=NOW)
    assert out['state']=='superseded' and 'Kursbasis geändert' in out['health']


def test_stale_daily_data_cannot_show_current_confirmation():
    item,r=idea();out=pw.evaluate(item,r['df'],now=MON)
    assert out['data_blocked'] and out['price'] is None
    assert out['last_checked_at'] is None


def test_pause_archive_expiry():
    item,r=idea()
    for key in ['paused','archived']:
        v=dict(item,**{key:True});assert pw.evaluate(v,r['df'],now=MON)==v
    assert pw.evaluate(item,r['df'],now='2026-11-22T10:00:00Z')['state']=='expired'


def test_poll_groups_ticker_does_not_scan_other_values_and_records_changes():
    item,r=idea();second=deepcopy(item);second['id']='second'
    class Provider:
        def __init__(self):self.calls=[]
        def quarantine_state(self):return None
        def get_history_light(self,ticker,**kw):self.calls.append((ticker,kw));return minutes(91) if kw.get('interval') else r['df']
    provider=Provider();out=pw.poll([item,second],provider,now=LIVE,force=True)
    assert len(provider.calls)==2 and {t for t,k in provider.calls}=={'HALO'}
    assert all(i['state']=='intraday' for i in out)
    assert all(i['events'][-1]['kind']=='intraday' for i in out)


def test_quarantine_rate_limit_throttle_and_closed_session():
    item,r=idea()
    class Provider:
        def __init__(self,q=False):self.calls=[];self.q=q
        def quarantine_state(self):return SimpleNamespace(active=True) if self.q else None
        def get_history_light(self,ticker,**kw):self.calls.append(ticker);raise RuntimeError('429 Too Many Requests')
    p=Provider(True);out=pw.poll([item],p,now=LIVE,force=True);assert not p.calls and 'Cooldown' in out[0]['health']
    other=deepcopy(item);other.update(id='other',ticker='AAPL')
    p=Provider();out=pw.poll([item,other],p,now=LIVE,force=True)
    assert p.calls==['HALO'] and all(i.get('data_blocked') for i in out)
    p=Provider();out=pw.poll([item],p,now=NOW);assert not p.calls
    throttled=dict(item,last_attempt_at='2026-11-09T15:01:30Z')
    p=Provider();pw.poll([throttled],p,now=LIVE);assert not p.calls


def test_confirmed_patterns_still_report_current_overextension():
    item,r=idea('double_top','confirmed');out=pw.evaluate(item,r['df'],minutes(85),now=LIVE)
    assert out['state']=='confirmed' and out['price']==85
    assert out['beyond_threshold_pct']>0 and out['projection_room_pct'] is not None


def test_real_ui_saved_list_without_radar_and_capture(monkeypatch):
    from streamlit.testing.v1 import AppTest
    item,r=idea();s=Store();pw.save(s,[item])
    script='''import streamlit as st
from modules.pattern_watch_ui import render_watcher
render_watcher(st,manager=st.session_state.manager,navigate=lambda st,x:st.session_state.update(opened=x))
'''
    app=AppTest.from_string(script);app.session_state['manager']=s
    app.run(timeout=20)
    assert not app.exception and app.button(key='pattern_watch_refresh').disabled
    assert app.dataframe[0].value.iloc[0]['Ticker']=='HALO'
    app.button(key='pattern_watch_analyze').click().run()
    assert app.session_state['opened']==['HALO'] and not app.exception


def test_identity_ignores_rolling_window_index_offsets():
    item,r=idea();p=item['source'];v=item['original'];shifted=deepcopy(v)
    shifted['boundary']['anchor_index']+=10
    for point in shifted['points']:point['index']+=10
    assert pw.identity(p,v)==pw.identity(p,shifted)


def test_expiry_is_local_even_when_provider_quarantined():
    item,r=idea()
    class Provider:
        def quarantine_state(self):raise AssertionError('No provider needed for local expiry')
    out=pw.poll([item],Provider(),now='2026-11-22T10:00:00Z')
    assert out[0]['state']=='expired' and out[0]['events'][-1]['kind']=='expired'


def test_after_close_automatic_check_can_see_completed_daily_bar():
    item,r=idea()
    class Provider:
        def quarantine_state(self):return None
        def get_history_light(self,ticker,**kw):return append_day(r)
    # Daily confirmation has a 30-minute safety buffer; polling must continue beyond it.
    out=pw.poll([item],Provider(),now='2026-11-09T21:40:00Z')
    assert out[0]['state']=='confirmed'


def test_capture_button_saves_and_panel_pause_archive(monkeypatch):
    from streamlit.testing.v1 import AppTest
    original=pw.utc;monkeypatch.setattr(pw,'utc',lambda value=None:original(NOW if value is None else value))
    item,r=idea();s=Store()
    hit=dict(ticker='HALO',pattern=item['original'],evidence=item['source'])
    script=f'''import streamlit as st
from modules.pattern_watch_ui import render_capture,render_watcher
render_watcher(st,manager=st.session_state.manager)
render_capture(st,{{}},[{hit!r}],manager=st.session_state.manager,fresh=True)
'''
    app=AppTest.from_string(script);app.session_state['manager']=s;app.run(timeout=20)
    app.multiselect(key='pattern_watch_capture').set_value([0]).run()
    app.button(key='pattern_watch_add').click().run()
    assert not app.exception and len(app.session_state['manager'].value['items'])==1
    app.button(key='pattern_watch_pause').click().run()
    assert app.session_state['manager'].value['items'][0]['paused']
    app.button(key='pattern_watch_archive').click().run()
    assert app.session_state['manager'].value['items'][0]['archived'] and not app.exception
