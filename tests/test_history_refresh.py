"""Manual analysis/provider cache bridge, using fake provider boundaries only."""
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
import pandas as pd
import pytest
from modules.provider_manager import MarketDataProvider,history_refresh_scope,MarketDataRateLimitError,ProviderCoolingDownError
from modules.cache_layer import make_cached_analyzer
from test_analysis_position_mode import offline_engines


class Quotes:
    name='fake'
    def __init__(self):self.calls=[];self.fail=False;self.day='2026-10-23'
    def history(self,symbol,**kwargs):
        assert not any('refresh' in k or 'token' in k or k.startswith('_') for k in kwargs)
        self.calls.append((symbol,dict(kwargs)))
        if self.fail:raise MarketDataRateLimitError('HTTP 429')
        return pd.DataFrame({'Close':[100.]*20},index=pd.bdate_range(end=self.day,periods=20))
    def download(self,*a,**kw):raise AssertionError('Unexpected fallback')


@pytest.fixture
def provider(monkeypatch):
    q=Quotes();p=MarketDataProvider(primary=q)
    monkeypatch.setattr(p,'_pace_request',lambda **kw:p.ensure_available())
    return p,q


def test_manual_token_refreshes_history_once_and_preserves_fetch_origin(provider):
    p,q=provider
    first=p.get_history('HALO',period='3y',interval='1d')
    before=deepcopy(first.attrs)
    q.day='2026-10-26'
    cached=p.get_history('HALO',period='3y',interval='1d')
    assert len(q.calls)==1 and cached.attrs['chsm_history']['from_cache']
    assert cached.attrs['chsm_history']['fetched_at']==first.attrs['chsm_history']['fetched_at']
    with history_refresh_scope('manual-A'):
        fresh=p.get_history('HALO',period='3y',interval='1d')
        same_run=p.get_history('HALO',period='3y',interval='1d')
    assert len(q.calls)==2
    assert fresh.attrs['chsm_history']['data_date']=='2026-10-26'
    assert not fresh.attrs['chsm_history']['from_cache']
    assert same_run.attrs['chsm_history']['from_cache']
    assert first.attrs==before
    with history_refresh_scope('manual-B'):p.get_history('HALO',period='3y',interval='1d')
    assert len(q.calls)==3


def test_manual_batch_reuses_shared_benchmark_and_scope_does_not_leak(provider):
    p,q=provider
    for ticker in ['HALO','HPE']:
        with history_refresh_scope('one-full-scan'):
            p.get_history(ticker,period='3y',interval='1d')
            p.get_history('SPY',period='1y',interval='1d')
    assert [x[0] for x in q.calls].count('SPY')==1
    normal=p.get_history('SPY',period='1y',interval='1d')
    assert not normal.attrs['chsm_history']['fresh_requested']
    with pytest.raises(RuntimeError):
        with history_refresh_scope('failed-scope'):raise RuntimeError('test')
    assert not p.get_history('SPY',period='1y',interval='1d').attrs['chsm_history']['fresh_requested']


def test_manual_refresh_keeps_rate_limit_quarantine_and_never_returns_old_data_as_fresh(provider):
    p,q=provider
    p.get_history('HALO',period='3y',interval='1d')
    q.fail=True
    with history_refresh_scope('manual-A'),pytest.raises(MarketDataRateLimitError):
        p.get_history('HALO',period='3y',interval='1d')
    calls=len(q.calls)
    with history_refresh_scope('manual-B'),pytest.raises(ProviderCoolingDownError):
        p.get_history('HALO',period='3y',interval='1d')
    assert len(q.calls)==calls


class FakeStreamlit:
    def cache_data(self,**options):
        def decorate(fn):
            store={}
            def cached(**kwargs):
                key=tuple(sorted((k,repr(v)) for k,v in kwargs.items() if not k.startswith('_')))
                if key not in store:store[key]=deepcopy(fn(**kwargs))
                return deepcopy(store[key])
            cached.clear=store.clear
            return cached
        return decorate


def test_actual_analyzer_wrapper_bypasses_both_caches_for_manual_and_reuses_auto(provider):
    p,q=provider
    calls=[]
    def analyze(**kwargs):
        calls.append(kwargs['ticker'])
        df=p.get_history(kwargs['ticker'],period='3y',interval='1d')
        return {'history_metadata':dict(df.attrs['chsm_history'])}
    cached=make_cached_analyzer(FakeStreamlit(),analyze)
    args=dict(ticker='HALO',horizon='Swing',depot=10000,risk_pct=1,override=0,buy_in_override=0,smart_money_default=True,strict_mode=True)
    a=cached(**args,market_bucket='auto-one')
    b=cached(**args,market_bucket='auto-one')
    assert not a['history_metadata']['analysis_cache_hit'] and b['history_metadata']['analysis_cache_hit']
    cached(**args,market_bucket='auto-two')
    assert len(q.calls)==1 and len(calls)==2
    q.day='2026-10-26'
    manual=cached(**args,market_bucket='manual-v287b-123')
    assert len(q.calls)==2 and manual['history_metadata']['data_date']=='2026-10-26'
    cached(**args,market_bucket='manual-v287b-123')
    assert len(q.calls)==2 # double-click protection remains
    cached(**args,market_bucket='selective-v3021d-456')
    assert len(q.calls)==3
    cached.clear()


@pytest.mark.parametrize('engine',['analyze_stock','_legacy_analyze_stock_impl'])
def test_actual_complete_engines_keep_stock_history_provenance(offline_engines,engine):
    scope,df=offline_engines
    meta={'fetched_at':'2026-10-26T14:00:00+00:00','from_cache':False,'data_date':'2026-10-26'}
    df.attrs['chsm_history']=dict(meta)
    r=scope[engine]('HALO','Swing (1-4 Wochen)',20000,1,0,0,False,False)
    assert r['history_metadata']==meta and df.attrs['chsm_history']==meta


def test_real_streamlit_cache_reports_analysis_cache_hit():
    from streamlit.testing.v1 import AppTest
    script='''
import streamlit as st
from modules.cache_layer import make_cached_analyzer

def analyze(**kwargs):
    st.session_state['executions']=st.session_state.get('executions',0)+1
    return {'history_metadata':{'fetched_at':'2026-10-26T14:00:00+00:00','from_cache':False}}
f=make_cached_analyzer(st,analyze)
if st.session_state.get('first',True):
    f.clear()
    st.session_state['first']=False
r=f(ticker='CACHE_TEST',horizon='Swing',depot=10000,risk_pct=1,override=0,buy_in_override=0,smart_money_default=True,strict_mode=True,market_bucket='manual-v287b-real-test')
st.session_state['last_hit']=r['history_metadata']['analysis_cache_hit']
st.write(st.session_state['executions'])
'''
    at=AppTest.from_string(script).run(timeout=20)
    assert not at.exception and not at.session_state['last_hit']
    at.run()
    assert not at.exception and at.session_state['last_hit'] and at.session_state['executions']==1
