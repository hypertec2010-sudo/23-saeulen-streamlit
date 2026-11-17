from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import time
import pandas as pd
import pytest
from modules.provider_manager import MarketDataProvider, MarketDataRateLimitError, ProviderCoolingDownError
from modules.provider_supplements import FIELDS


class Ticker:
    def __init__(self):self.calls=[];self.fail=False
    def get_info(self):
        self.calls.append('get_info')
        if self.fail:raise RuntimeError('429 Too Many Requests')
        return {'targetMeanPrice':110,'marketState':'REGULAR','regularMarketPrice':100}
    @property
    def info(self):raise AssertionError('Do not repeat get_info alias')
    @property
    def fast_info(self):raise AssertionError('Do not fetch fast_info unnecessarily')
    def __getattr__(self,name):
        if name in FIELDS:
            self.calls.append(name)
            if self.fail:raise RuntimeError('429 Too Many Requests')
            return {'Earnings Date':[]} if name=='calendar' else pd.DataFrame({'2025':[1.]},index=['Revenue'])
        raise AttributeError(name)
    def get_earnings_dates(self,limit=8):self.calls.append(('dates',limit));return pd.DataFrame()


def provider(monkeypatch):
    raw=Ticker();p=MarketDataProvider(primary=SimpleNamespace(name='fake',ticker=lambda symbol:raw))
    monkeypatch.setattr(p,'resolve_symbol',lambda symbol:symbol)
    paced=[]
    def pace(**kw):p.ensure_available();paced.append(1)
    monkeypatch.setattr(p,'_pace_request',pace)
    return p,raw,paced


def test_normal_info_one_call_cached_and_mutation_safe(monkeypatch):
    p,raw,paced=provider(monkeypatch)
    a,info=p.get_info_bundle('HALO');info['targetMeanPrice']=1
    b,again=p.get_info_bundle('HALO')
    assert raw.calls==['get_info'] and len(paced)==1
    assert again['targetMeanPrice']==110 and again['_chsm_info_fetched_at']==info['_chsm_info_fetched_at']
    assert b.get_info()['targetMeanPrice']==110 and raw.calls==['get_info']


@pytest.mark.parametrize('field',sorted(FIELDS))
def test_supplements_cached_paced_and_copied(monkeypatch,field):
    p,raw,paced=provider(monkeypatch);a,info=p.get_info_bundle('HALO')
    first=getattr(a,field)
    if isinstance(first,pd.DataFrame):first.iloc[0,0]=999
    b,_=p.get_info_bundle('HALO');second=getattr(b,field)
    assert raw.calls.count(field)==1 and len(paced)==2
    if isinstance(second,pd.DataFrame):assert second.iloc[0,0]==1


def test_supplement_ttl_not_refetched_after_info_ttl(monkeypatch):
    p,raw,paced=provider(monkeypatch);a,_=p.get_info_bundle('HALO');a.income_stmt
    key=('HALO','info');stamp,value=p._supplement_cache.values[key]
    p._supplement_cache.values[key]=(stamp-901,value)
    b,_=p.get_info_bundle('HALO');b.income_stmt
    assert raw.calls.count('get_info')==2 and raw.calls.count('income_stmt')==1
    key=('HALO','income_stmt');stamp,value=p._supplement_cache.values[key]
    p._supplement_cache.values[key]=(stamp-3601,value);b.income_stmt
    assert raw.calls.count('income_stmt')==2


def test_earnings_parameters_are_part_of_cache_key(monkeypatch):
    p,raw,paced=provider(monkeypatch);a,_=p.get_info_bundle('HALO')
    a.get_earnings_dates(limit=8);a.get_earnings_dates(limit=8);a.get_earnings_dates(limit=4)
    assert raw.calls.count(('dates',8))==1 and raw.calls.count(('dates',4))==1


def test_rate_limit_on_supplement_stops_following_aliases_and_properties(monkeypatch):
    p,raw,paced=provider(monkeypatch);a,_=p.get_info_bundle('HALO');raw.fail=True
    with pytest.raises(MarketDataRateLimitError):a.income_stmt
    with pytest.raises(ProviderCoolingDownError):a.balance_sheet
    with pytest.raises(ProviderCoolingDownError):p.get_info_bundle('AAPL')
    assert raw.calls==['get_info','income_stmt'] and p.quarantine_state().active


def test_info_rate_limit_no_fallback_or_failed_cache(monkeypatch):
    p,raw,paced=provider(monkeypatch);raw.fail=True
    with pytest.raises(MarketDataRateLimitError):p.get_info_bundle('HALO')
    assert raw.calls==['get_info'] and not p._supplement_cache.values


def legacy_function(name):
    path=Path(__file__).parents[1]/'legacy_app.py';tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.decorator_list=[]
    return ast.Module(body=[fn],type_ignores=[])


def test_actual_extended_quote_uses_no_raw_http_or_redundant_histories(monkeypatch):
    import numpy as np
    calls=[]
    class P:
        def ensure_available(self):pass
        def get_info_bundle(self,t):return None,{'marketState':'REGULAR','regularMarketPrice':100,'_chsm_info_fetched_at':time.time()}
        def get_history_light(self,*a,**k):calls.append(k);raise AssertionError('No history needed')
    env=dict(np=np,pd=pd,time=time,_market_provider_v2845a=P())
    exec(compile(legacy_function('load_extended_market_quote'),'<legacy>','exec'),env)
    out=env['load_extended_market_quote']('HALO')
    assert out['display_price']==100 and not calls
    p=env['_market_provider_v2845a']
    p.get_info_bundle=lambda t:(None,{'_chsm_info_fetched_at':time.time()-901,'regularMarketPrice':999})
    def history(*a,**k):
        calls.append(k)
        return pd.DataFrame({'Close':[101.]},index=pd.DatetimeIndex(['2026-11-17T15:00:00Z']))
    p.get_history_light=history
    out=env['load_extended_market_quote']('HALO')
    assert out['display_price']==101 and len(calls)==1
    assert calls[0]['prepost'] and calls[0]['interval']=='2m'


def test_load_data_does_not_publish_partial_analysis_after_swallowed_429(monkeypatch):
    p,raw,paced=provider(monkeypatch)
    p.get_history=lambda *a,**k:pd.DataFrame({'Close':[100]})
    def derive(t,info):
        raw.fail=True
        try:t.income_stmt
        except Exception:pass
        return info
    env=dict(_market_provider_v2845a=p,derive_fundamentals_from_statements=derive,
        extract_analyst_data=lambda t,i:i,extract_earnings_data=lambda t,i:i)
    exec(compile(legacy_function('load_data'),'<legacy>','exec'),env)
    with pytest.raises(ProviderCoolingDownError):env['load_data']('HALO')
