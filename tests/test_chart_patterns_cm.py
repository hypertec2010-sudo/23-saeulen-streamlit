from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from modules import chart_patterns as cp
from modules.chart_pattern_ui import pattern_rows,pattern_figure,render_chart_patterns,render_screener_patterns

NOW='2026-11-07T10:00:00Z'


def mirror(r):
    out=deepcopy(r);f=out['df'];old=f.copy()
    for col,other in [('Open','Open'),('High','Low'),('Low','High'),('Close','Close')]:f[col]=200-old[other]
    out['price']=float(f.Close.iloc[-1])
    return out


def fixture(kind,phase='forming'):
    inverse=kind in {'inverse_head_shoulders','double_bottom','bear_flag'}
    if 'flag' in kind:
        c=np.r_[np.linspace(80,100,109),[104.,108.,112.,116.,120.], [119.,118.5,118.,117.5,117.],117.5]
        f=pd.DataFrame(dict(Open=c-.1,High=c+.3,Low=c-.3,Close=c,Volume=1_000_000.),index=pd.bdate_range(end='2026-11-06',periods=120))
        for i,close in enumerate([104.,108.,112.,116.,120.],109):
            f.loc[f.index[i],['Open','High','Low','Close','Volume']]=[close-3,close+1,close-3.5,close,2_000_000.]
        for i in range(114,119):
            f.loc[f.index[i],['Open','High','Low','Close','Volume']]=[119-(i-114)*.5,120-(i-114)*.5,118-(i-114)*.5,119-(i-114)*.5,800_000.]
        if phase=='confirmed':f.loc[f.index[-1],['Open','High','Low','Close','Volume']]=[118.,122.,117.,121.,3_000_000.]
        elif phase=='invalid':f.loc[f.index[-1],['Open','High','Low','Close']]=[115.,116.,109.,110.]
    else:
        hs='head_shoulders' in kind
        knots=[0,45,65,75,85,95,105,110,119] if hs else [0,45,70,85,100,110,119]
        vals=[80,90,100,90,110,90,100,92,92] if hs else [80,90,105,92,105,99,99]
        c=np.interp(np.arange(120),knots,vals)
        f=pd.DataFrame(dict(Open=c-.1,High=c+.3,Low=c-.3,Close=c,Volume=1_000_000.),index=pd.bdate_range(end='2026-11-06',periods=120))
        if phase!='forming':
            close=(88. if hs else 90.) if phase=='confirmed' else (105. if hs else 110.)
            f.loc[f.index[-1],['Open','High','Low','Close','Volume']]=[close-.1,close+.3,close-.3,close,3_000_000.]
    r=dict(ticker='HALO',ccy='USD',price=float(f.Close.iloc[-1]),df=f)
    return mirror(r) if inverse else r


@pytest.mark.parametrize('kind',list(cp.KINDS))
@pytest.mark.parametrize('phase',['forming','confirmed','invalid'])
def test_all_six_pattern_geometries_have_their_own_three_states(kind,phase):
    r=fixture(kind,phase);p=cp.build_chart_patterns(r,now=NOW)
    item=next(x for x in p['patterns'] if x['kind']==kind)
    assert item['phase']==phase,item
    assert item['points'] and item['structure_known_at']
    assert item['invalidation']>0 and item['boundary_now']>0
    assert item['direction']==cp.KINDS[kind][1]
    if phase=='confirmed':assert item['confirmed_at']>=item['structure_known_at']
    json.dumps(p,allow_nan=False)


def test_missing_head_is_not_a_head_shoulders_pattern():
    r=fixture('head_shoulders');f=r['df']
    f.loc[f.index[76:95],['Open','High','Low','Close']]=[94.9,95.3,94.7,95.]
    p=cp.build_chart_patterns(r,now=NOW)
    item=next(x for x in p['patterns'] if x['kind']=='head_shoulders')
    assert item['phase']=='none'


def test_running_bar_cannot_confirm_or_invalidate_daily_pattern():
    r=fixture('double_bottom');r['df'].index=pd.bdate_range(end='2026-11-05',periods=120)
    before=cp.build_chart_patterns(r,now='2026-11-06T16:00Z')
    extra=pd.DataFrame(dict(Open=[100.],High=[150.],Low=[50.],Close=[140.],Volume=[9_000_000.]),index=pd.to_datetime(['2026-11-06']))
    r['df']=pd.concat([r['df'],extra])
    assert cp.build_chart_patterns(r,now='2026-11-06T16:00Z')==before


def test_unfinished_right_shoulder_is_not_claimed_as_confirmed_pivot():
    r=fixture('head_shoulders');r['df']=r['df'].iloc[:108]
    # Right shoulder at index 105 has only two later bars, not three.
    p=cp.build_chart_patterns(r,now=NOW)
    item=next(x for x in p['patterns'] if x['kind']=='head_shoulders')
    assert item['phase']=='none'


@pytest.mark.parametrize('kind',['bull_flag','bear_flag','double_bottom','inverse_head_shoulders'])
def test_bullish_reversals_and_both_flags_need_breakout_volume(kind):
    r=fixture(kind,'confirmed');r['df']['Volume']=float('nan')
    p=cp.build_chart_patterns(r,now=NOW)
    item=next(x for x in p['patterns'] if x['kind']==kind)
    assert item['phase']=='forming' and item['volume_ratio'] is None
    assert 'volumen' in item['reason'].lower()
    json.dumps(p,allow_nan=False)


def test_bearish_top_can_be_price_confirmed_with_explicit_missing_volume_note():
    r=fixture('double_top','confirmed');r['df']=r['df'].drop(columns='Volume')
    item=next(x for x in cp.build_chart_patterns(r,now=NOW)['patterns'] if x['kind']=='double_top')
    assert item['phase']=='confirmed' and 'kursbasiert' in item['volume_note']


def test_same_ohlcv_is_not_blocked_by_general_analysis_or_watcher_gates(monkeypatch):
    from modules import watch_setups
    monkeypatch.setattr(watch_setups,'detect',lambda *a,**k:(_ for _ in ()).throw(AssertionError('watcher must not run')))
    r=fixture('double_bottom','confirmed');p=cp.build_chart_patterns(r,now=NOW)
    r.update(buy_in=100.,days_earn=1,has_upcoming_earnings=True,technical_target_1=1.,grade='F',trading_score=0)
    assert cp.build_chart_patterns(r,now=NOW)==p


def test_shape_and_projection_do_not_use_general_analysis_target():
    r=fixture('head_shoulders','confirmed');before=r['df'].copy(deep=True)
    a=cp.build_chart_patterns(r,now=NOW)
    r['technical_target_1']=9999.
    assert cp.build_chart_patterns(r,now=NOW)==a and r['df'].equals(before)


def test_confirmed_break_returning_above_neckline_is_invalidated():
    r=fixture('double_top','confirmed');p=cp.build_chart_patterns(r,now=NOW)
    f=r['df'];extra=f.tail(1).copy();extra.index=pd.to_datetime(['2026-11-09'])
    extra.loc[:,['Open','High','Low','Close']]=[98.,99.,97.,98.]
    r['df']=pd.concat([f,extra])
    item=next(x for x in cp.build_chart_patterns(r,now='2026-11-09T22:00Z')['patterns'] if x['kind']=='double_top')
    assert item['phase']=='invalid' and 'Fehlausbruch' in item['reason']


def test_volume_baseline_excludes_signal_and_future_bars():
    d={'Volume':[100.]*20+[120.,9_000_000.]}
    assert cp._volume_ratio(d,20)==1.2
    d['Volume'][-1]=1.
    assert cp._volume_ratio(d,20)==1.2


def test_missing_data_return_six_unavailable_results_not_six_negative_patterns():
    p=cp.build_chart_patterns({'ticker':'HALO'},now=NOW)
    assert not p['available'] and len(p['patterns'])==6
    assert all(x['phase']=='blocked' for x in p['patterns'])


def test_no_pattern_on_flat_series_and_old_data_remain_historical():
    r=fixture('double_top');r['df'].loc[:,['Open','High','Low','Close']]=[100.,101.,99.,100.]
    p=cp.build_chart_patterns(r,now=NOW)
    assert all(x['phase']=='none' for x in p['patterns'])
    assert cp.patterns_brief(p)=='Kein aktueller Musterbeleg'
    r['df']=r['df'].iloc[:-1];p=cp.build_chart_patterns(r,now=NOW)
    assert p['stale'] and 'historisch' in cp.patterns_brief(p)


def test_brief_and_figure_include_evidence_but_do_not_modify_inputs():
    r=fixture('head_shoulders','confirmed');p=cp.build_chart_patterns(r,now=NOW);before=deepcopy(p)
    top=next(x for x in p['patterns'] if x['kind']=='head_shoulders')
    assert 'Schulter' in cp.patterns_brief(p) and len(pattern_rows(p))==6
    fig=pattern_figure(p,top)
    assert len(fig.data)==3 and len(fig.layout.shapes)==2
    assert list(fig.data[1].x)==[x['date'] for x in top['points']]
    assert p==before


def test_real_streamlit_views_render_all_six_patterns():
    from streamlit.testing.v1 import AppTest
    p=cp.build_chart_patterns(fixture('double_bottom','confirmed'),now=NOW)
    script=f'''
import streamlit as st
import pandas as pd
from modules.chart_pattern_ui import render_chart_patterns,render_screener_patterns
p={p!r}
render_chart_patterns(st,p)
render_screener_patterns(st,pd.DataFrame([{{'Ticker':'HALO','Ampel':'🔴','__chart_patterns':p}}]))
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and len(app.dataframe)==2
    assert len(app.dataframe[0].value)==6 and len(app.dataframe[1].value)==6


def test_pattern_evidence_survives_real_history_cleanup_and_json_snapshot(monkeypatch):
    from modules import live_monitor as lm
    from modules.live_screener_snapshot import dataframe_to_payload,dataframe_from_payload
    class State(dict):
        __getattr__=dict.__getitem__
        __setattr__=dict.__setitem__
    monkeypatch.setattr(lm,'st',SimpleNamespace(session_state=State()))
    monkeypatch.setattr(lm,'get_current_berlin_time',lambda:datetime(2026,11,7,10),raising=False)
    monkeypatch.setattr(lm,'_v227_load_persistent_live_history',lambda:({},[]))
    monkeypatch.setattr(lm,'_v227_save_persistent_live_history',lambda *a:None)
    monkeypatch.setattr(lm,'_v2416_log_event',lambda **kw:None,raising=False)
    p=cp.build_chart_patterns(fixture('double_top','confirmed'),now=NOW)
    raw=pd.DataFrame([{'Ticker':'HALO','Kurs':90.,'Ampel':'🔴','Status':'Abwarten','Live-Score':'20/100',
                      '__chart_patterns':p,'__temporary':1}])
    enriched,_=lm.apply_live_watchlist_status_history_v220(raw,watchlist_name='Test')
    restored=dataframe_from_payload(json.loads(json.dumps(dataframe_to_payload(enriched))))
    assert restored.loc[0,'__chart_patterns']==p and '__temporary' not in enriched
    assert restored.loc[0,'Chartmuster-Kurz']==cp.patterns_brief(p)
    assert 'Chartmuster-Kurz' not in raw


def test_direct_overview_contains_pattern_column_and_escaped_mobile_text():
    text=(Path(__file__).resolve().parents[1]/'legacy_app.py').read_text()
    assert '"Support-Kurz", "Chartmuster-Kurz", "Volatilität"' in text
    assert '_patterns_brief_cm = html.escape(' in text and '{_patterns_brief_cm}</div>' in text


@pytest.mark.parametrize('kind',list(cp.KINDS))
def test_confirmed_evidence_cannot_predate_its_available_structure(kind):
    p=cp.build_chart_patterns(fixture(kind,'confirmed'),now=NOW)
    item=next(x for x in p['patterns'] if x['kind']==kind)
    assert item['confirmed_at']>=item['structure_known_at']
    if 'flag' not in kind:
        assert item['known_index']==item['points'][-1]['index']+3


def test_live_quote_is_context_not_daily_confirmation():
    r=fixture('double_bottom');a=cp.build_chart_patterns(r,now=NOW)
    r['price']=9999.;b=cp.build_chart_patterns(r,now=NOW)
    assert a==b


def test_flag_breakout_is_not_absorbed_into_a_wider_new_flag():
    r=fixture('bull_flag','confirmed')
    a=next(x for x in cp.build_chart_patterns(r,now=NOW)['patterns'] if x['kind']=='bull_flag')
    extra=r['df'].tail(1).copy();extra.index=pd.to_datetime(['2026-11-09'])
    extra.loc[:,['Open','High','Low','Close']]=[121.,122.,120.,121.5]
    r['df']=pd.concat([r['df'],extra])
    b=next(x for x in cp.build_chart_patterns(r,now='2026-11-09T22:00Z')['patterns'] if x['kind']=='bull_flag')
    assert a['phase']==b['phase']=='confirmed'
    assert a['points']==b['points'] and a['boundary']==b['boundary']
    assert a['invalidation']==b['invalidation'] and a['projection']==b['projection']


def test_converging_pennant_is_not_claimed_as_parallel_flag():
    r=fixture('bull_flag')
    for i in range(114,119):r['df'].loc[r['df'].index[i],'Low']=115+(i-114)*.5
    item=next(x for x in cp.build_chart_patterns(r,now=NOW)['patterns'] if x['kind']=='bull_flag')
    assert item['phase']=='none'


def test_same_day_pre_confirmation_wick_is_not_a_later_projection_hit():
    dates=pd.bdate_range(end='2026-11-06',periods=30)
    d={'Close':[99.]*30,'High':[100.]*30,'Low':[98.]*30,
       'ATR':[1.]*30,'Volume':[100.]*30,'date':[x.date().isoformat() for x in dates]}
    d['Close'][-1]=102.;d['High'][-1]=120.
    item=cp._evaluate('double_bottom',d,end=26,known=29,anchor=26,value=100.,slope=0.,
                      height=10.,invalidation=90.,points=[],volume_required=False)
    assert item['phase']=='confirmed' and item['projection']==110.
    assert not item['projection_touched']


def test_confirmation_volume_threshold_is_exact_and_not_a_probability():
    r=fixture('double_bottom','confirmed')
    prior=float(r['df'].Volume.iloc[-21:-1].mean())
    r['df'].loc[r['df'].index[-1],'Volume']=prior*1.19999
    a=next(x for x in cp.build_chart_patterns(r,now=NOW)['patterns'] if x['kind']=='double_bottom')
    assert a['phase']=='forming'
    r['df'].loc[r['df'].index[-1],'Volume']=prior*1.2
    b=next(x for x in cp.build_chart_patterns(r,now=NOW)['patterns'] if x['kind']=='double_bottom')
    assert b['phase']=='confirmed' and b['volume_ratio']==1.2


def test_missing_or_corrupt_snapshot_is_not_a_negative_pattern_finding():
    assert cp.patterns_brief(None)=='Neu scannen'
    assert cp.patterns_brief({'policy':cp.POLICY,'available':True,'patterns':[None]})=='Beleg unvollständig · neu scannen'


def test_projection_touch_after_invalidation_is_not_credited_to_the_pattern():
    dates=pd.bdate_range(end='2026-11-06',periods=30)
    d={'Close':[99.]*30,'High':[100.]*30,'Low':[98.]*30,
       'ATR':[1.]*30,'Volume':[100.]*30,'date':[x.date().isoformat() for x in dates]}
    d['Close'][25]=102.;d['High'][25]=103.
    d['Close'][26]=89.;d['Low'][26]=88.
    d['High'][29]=120.
    item=cp._evaluate('double_bottom',d,end=22,known=25,anchor=22,value=100.,slope=0.,
                      height=10.,invalidation=90.,points=[],volume_required=False)
    assert item['phase']=='invalid' and item['invalidated_index']==26
    assert not item['projection_touched']
