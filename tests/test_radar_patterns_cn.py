from datetime import datetime, timezone
from types import SimpleNamespace
import json
import pytest
from modules import candidate_radar as radar, provider_manager
from modules.radar_patterns import pattern_hits
from test_chart_patterns_cm import fixture


def run(monkeypatch,entries,loader,**kwargs):
    monkeypatch.setattr(provider_manager,'get_market_data_provider',lambda:SimpleNamespace(quarantine_state=lambda:None))
    def forbidden(*a,**k):raise AssertionError('No full analysis, grade or gate allowed')
    return radar.run_two_stage_scan(universe='Eigene Liste',style='Chartmuster',entries=entries,
        history_loader=loader,analyze=forbidden,decide=forbidden,entry_package=forbidden,
        model_version='v30.21cn',clock=lambda:datetime(2026,11,7,10,tzinfo=timezone.utc),**kwargs)


def test_all_values_without_finalist_limit(monkeypatch):
    calls=[];f=fixture('double_bottom','confirmed')['df']
    def load(t):calls.append(t);return f
    symbols=['A'+chr(65+i) for i in range(22)]
    p=run(monkeypatch,symbols,load,full_analysis_limit=1)
    assert calls==symbols and len(p['rows'])==22 and radar.snapshot_valid(p)
    assert p['full_analysis_count']==0 and len({h['ticker'] for h in pattern_hits(p)})==22
    assert all(r['grade'] is None and r['crv'] is None for r in p['rows'])


@pytest.mark.parametrize('kind',['bull_flag','bear_flag','head_shoulders','inverse_head_shoulders','double_bottom','double_top'])
def test_all_six_discovered(monkeypatch,kind):
    p=run(monkeypatch,['HALO'],lambda t:fixture(kind,'confirmed')['df'])
    assert any(h['pattern']['kind']==kind and h['pattern']['phase']=='confirmed' for h in pattern_hits(p))
    assert json.loads(json.dumps(radar.json_safe(p)))['rows'][0]['chart_patterns']['available']


def test_filters_invalid_and_blocked(monkeypatch):
    p=run(monkeypatch,['HALO'],lambda t:fixture('double_top','invalid')['df'])
    assert not pattern_hits(p,kind='double_top')
    p=run(monkeypatch,['HALO'],lambda t:fixture('double_top','forming')['df'])
    assert pattern_hits(p,phase='forming',direction=-1,kind='double_top')
    assert not pattern_hits(p,phase='confirmed',kind='double_top')
    assert not pattern_hits(p,direction=1,kind='double_top')
    p=run(monkeypatch,['HALO'],lambda t:fixture('double_top')['df'].iloc[:5])
    assert radar.snapshot_valid(p) and not pattern_hits(p)
    assert not p['rows'][0]['chart_patterns']['available']


def test_error_continues_but_rate_limit_aborts(monkeypatch):
    def load(t):
        if t=='BAD':raise ValueError('bad data')
        return fixture('double_top')['df']
    p=run(monkeypatch,['BAD','HALO'],load)
    assert radar.snapshot_valid(p) and len(p['errors'])==1 and len(p['rows'])==1
    calls=[]
    def limited(t):calls.append(t);raise RuntimeError('429 Too Many Requests')
    p=run(monkeypatch,['HALO','AAPL'],limited)
    assert calls==['HALO'] and p['abort_reason']=='provider_rate_limit'
    assert not radar.snapshot_valid(p) and p['skipped']==1


def test_quarantine(monkeypatch):
    monkeypatch.setattr(provider_manager,'get_market_data_provider',lambda:SimpleNamespace(quarantine_state=lambda:SimpleNamespace(active=True,remaining_seconds=60)))
    def fail(*a,**k):raise AssertionError('quarantined')
    p=radar.run_two_stage_scan(universe='Eigene Liste',style='Chartmuster',entries=['HALO'],history_loader=fail,analyze=fail,decide=fail,entry_package=fail)
    assert p['abort_reason']=='provider_quarantine'


def test_real_ui_handoff_and_historical_scan(monkeypatch):
    from streamlit.testing.v1 import AppTest
    p=run(monkeypatch,['HALO'],lambda t:fixture('double_top','confirmed')['df'])
    app=AppTest.from_string('''import streamlit as st
from modules.radar_patterns import render_pattern_radar
render_pattern_radar(st,st.session_state.payload,fresh=st.session_state.fresh,limit=5,
    navigate=lambda st,chosen:st.session_state.update(opened=chosen))
''')
    app.session_state['payload']=p;app.session_state['fresh']=True
    app.run(timeout=20)
    assert not app.exception
    app.multiselect[0].set_value(['HALO']).run()
    app.button[0].click().run()
    assert app.session_state['opened']==['HALO']
    app.session_state['fresh']=False;app.run()
    assert app.multiselect[0].options==[] and app.button[0].disabled
    assert not app.exception


def test_snapshot_owner_roundtrip_and_complete_no_hits(monkeypatch):
    p=run(monkeypatch,['HALO'],lambda t:fixture('double_top')['df'].iloc[:5])
    class Store:
        user_id='test-pattern-owner'
        def save_namespace(self,key,value):self.value=json.loads(json.dumps(value));return True
        def load_namespace(self,key,default=None):return self.value
        def status(self):return {}
    store=Store()
    assert radar.save_snapshot(store,p)[0]
    saved=radar.load_snapshot(store,p['key'])
    assert saved['rows'][0]['chart_patterns']==p['rows'][0]['chart_patterns']
    assert not pattern_hits(saved)
    store.user_id='other-owner'
    assert radar.load_snapshot(store,p['key']) is None


def test_complete_radar_ui_routes_pattern_profile_without_general_filters(monkeypatch):
    from streamlit.testing.v1 import AppTest
    p=run(monkeypatch,['HALO'],lambda t:fixture('double_top','confirmed')['df'])
    p['owner']='pattern-ui-owner'
    script=f'''import streamlit as st
from datetime import datetime,timezone
from modules.candidate_radar_ui import render_candidate_radar
st.session_state.setdefault('radar_universe','Eigene Liste')
st.session_state.setdefault('radar_screening_style','Chartmuster')
st.session_state.setdefault('radar_custom_input','HALO')
class Storage:
    user_id='pattern-ui-owner'
    def load_namespace(self,*args,**kwargs):return {p!r}
    def status(self):return {{'last_backend':'cloud'}}
def forbidden(*a,**k):raise AssertionError('Display must not analyze or load holdings')
render_candidate_radar(st,storage=Storage(),scan=forbidden,catalog_loader=forbidden,
    watchlists_loader=forbidden,positions_loader=forbidden,queue_watchlist=forbidden,
    model_version='v30.21cn',clock=lambda:datetime(2026,11,7,10,tzinfo=timezone.utc))
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and not app.error
    assert app.selectbox(key='radar_patterns_phase').options==['Alle','Bestätigt','Entsteht']
    assert app.multiselect[0].options==['HALO']
    assert not any(s.key=='radar_v3021_trade_style' for s in app.selectbox)
