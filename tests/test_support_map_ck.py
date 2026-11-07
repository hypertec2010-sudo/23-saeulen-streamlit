from copy import deepcopy
import json

import pandas as pd
import pytest

from modules.support_map import (build_support_map, _clusters, selected_zones,
                                 support_rows, render_support_map, render_screener_supports)

NOW='2026-11-07T10:00:00Z'


def result(*,n=150,end='2026-11-06',points=None):
    f=pd.DataFrame({'Open':110.,'High':111.,'Low':109.,'Close':110.,'Volume':1_000_000.},
                   index=pd.bdate_range(end=end,periods=n))
    for i,v in (points if points is not None else [(20,100.),(45,100.5),(70,99.8)]):
        f.iloc[i,f.columns.get_loc('Low')]=v
    return {'ticker':'HALO','price':108.,'ccy':'USD','df':f}


def test_multitest_zone_has_dates_separation_and_bounded_width():
    r=result();before=r['df'].copy(deep=True);p=build_support_map(r,now=NOW)
    assert p['available'] and not p['stale'] and p['bars']==150
    assert len(p['zones'])==1
    z=p['zones'][0]
    assert z['touches']==3 and z['span_bars']==50 and z['pronounced']
    assert z['low']<99.8<z['high'] and z['low']<100.5<z['high']
    assert z['state']=='below_price' and z['distance_pct']<0
    assert z['last_test']==r['df'].index[70].date().isoformat()
    assert z['confirmed_at']==r['df'].index[73].date().isoformat()
    assert r['df'].equals(before)
    json.dumps(p,allow_nan=False)


def test_cluster_cannot_chain_distant_levels():
    groups=_clusters([(1,100.),(8,101.),(15,102.),(30,103.)])
    assert all(max(v for _,v in g)/min(v for _,v in g)<=1.015 for g in groups)
    assert len(groups)==2


def test_nearby_pivots_in_same_episode_are_not_multiple_tests():
    p=build_support_map(result(points=[(20,100.),(25,100.2)]),now=NOW)
    assert p['available'] and not p['zones'] and len(p['references'])==1


@pytest.mark.parametrize('points,pronounced',[
    ([(20,100.),(26,100.1),(32,100.2)],False),
    ([(20,100.),(30,100.1),(40,100.2)],True),
    ([(20,100.),(70,100.1)],False),
])
def test_pronounced_label_requires_three_tests_and_time_span(points,pronounced):
    z=build_support_map(result(points=points),now=NOW)['zones'][0]
    assert z['pronounced'] is pronounced


@pytest.mark.parametrize('quote,last_close,state',[
    (108.,110.,'below_price'),(100.,110.,'in_zone'),
    (95.,110.,'intraday_below'),(95.,95.,'broken'),
    (108.,95.,'intraday_reclaim'),
])
def test_quote_and_completed_close_have_distinct_break_meanings(quote,last_close,state):
    r=result();r['price']=quote
    r['df'].loc[r['df'].index[-1],['Open','High','Low','Close']]=[last_close,last_close+1,last_close-1,last_close]
    z=build_support_map(r,now=NOW)['zones'][0]
    assert z['state']==state
    assert z['distance_pct']==0 if state=='in_zone' else z['distance_pct']!=0


def test_reclaim_after_break_needs_a_new_retest():
    r=result();f=r['df']
    f.loc[f.index[120],['Open','High','Low','Close']]=[95.,96.,94.,95.]
    assert build_support_map(r,now=NOW)['zones'][0]['state']=='reclaimed'
    f.loc[f.index[130],'Low']=100.2
    assert build_support_map(r,now=NOW)['zones'][0]['state']=='below_price'


def test_old_zones_are_marked_and_chart_uses_at_most_250_bars():
    p=build_support_map(result(n=300),now=NOW)
    assert p['zones'][0]['older'] and len(p['chart'])==250
    assert 'älter' in support_rows(p)[0]['Historischer Beleg']


def test_future_and_running_bars_do_not_confirm_a_support():
    r=result(end='2026-11-05')
    before=build_support_map(r,now='2026-11-06T16:00Z')
    r['df']=pd.concat([r['df'],pd.DataFrame({'Open':[110.],'High':[112.],'Low':[1.],'Close':[108.],'Volume':[1.]},index=pd.to_datetime(['2026-11-06']))])
    after=build_support_map(r,now='2026-11-06T16:00Z')
    assert after==before
    r['df'].index=list(r['df'].index[:-1])+[pd.Timestamp('2026-11-09')]
    assert not build_support_map(r,now=NOW)['available']


@pytest.mark.parametrize('change', ['short','nan','duplicate','unsorted','no_ohlc','gap','weekly','foreign'])
def test_invalid_data_are_explained_without_fabricated_zones(change):
    r=result();f=r['df']
    if change=='short':r['df']=f.tail(40)
    elif change=='nan':f.iloc[-10,f.columns.get_loc('Low')]=float('nan')
    elif change=='duplicate':r['df']=pd.concat([f,f.tail(1)])
    elif change=='unsorted':r['df']=f.iloc[::-1]
    elif change=='no_ohlc':r['df']=f[['Close']]
    elif change=='gap':f.index=list(f.index[:70]-pd.Timedelta(days=30))+list(f.index[70:])
    elif change=='weekly':f.index=pd.date_range(end='2026-11-06',periods=len(f),freq='7D')
    elif change=='foreign':r['ticker']='SHEL.L'
    p=build_support_map(r,now=NOW)
    assert not p['available'] and p['reason'] and not p['zones']


def test_stale_data_remain_explicitly_historical_not_fresh():
    p=build_support_map(result(end='2026-11-05'),now=NOW)
    assert p['available'] and p['stale'] and p['expected_date']=='2026-11-06'
    assert p['basis_date']=='2026-11-05'


def test_missing_quote_uses_named_completed_close_reference():
    r=result();r.pop('price');p=build_support_map(r,now=NOW)
    assert p['price']==110. and p['price_source']=='Letzter abgeschlossener Tagesschluss'


def test_single_low_never_becomes_a_strong_zone():
    p=build_support_map(result(points=[(40,80.)]),now=NOW)
    assert p['available'] and not p['zones'] and len(p['references'])==1
    assert p['references'][0]['level']==80.


def test_screener_and_detail_share_same_snapshot_and_do_not_modify_it():
    r=result();p=build_support_map(r,now=NOW);before=deepcopy(p)
    class Fake:
        def __init__(self):self.frames=[];self.messages=[];self.fig=None
        def expander(self,*args,**kwargs):return self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def dataframe(self,f,**kwargs):self.frames.append(f.copy())
        def caption(self,s):self.messages.append(s)
        def markdown(self,s):self.messages.append(s)
        def info(self,s):self.messages.append(s)
        def warning(self,s):self.messages.append(s)
        def table(self,f):self.frames.append(f.copy())
        def plotly_chart(self,fig,**kwargs):self.fig=fig
    ui=Fake();render_support_map(ui,p)
    assert ui.fig is not None and len(ui.fig.layout.shapes)>=2
    rows=ui.frames[0]
    screener=Fake();render_screener_supports(screener,pd.DataFrame([{'Ticker':'HALO','Status':'Abwarten / rot','__support_map':p}]))
    assert screener.frames[0].iloc[0]['Trendstatus']=='Abwarten / rot'
    assert rows.iloc[0]['Zone'] in screener.frames[0].iloc[0]['Nächste Tiefzone']
    assert before==p


def test_real_streamlit_detail_and_screener_render_without_exception():
    from streamlit.testing.v1 import AppTest
    p=build_support_map(result(),now=NOW)
    script=f'''
import streamlit as st
import pandas as pd
from modules.support_map import render_support_map, render_screener_supports
p={p!r}
render_support_map(st,p)
render_screener_supports(st,pd.DataFrame([{{'Ticker':'HALO','Status':'Rot','__support_map':p}}]))
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and len(app.dataframe)==2
    assert app.dataframe[1].value.iloc[0]['Trendstatus']=='Rot'
    assert any('Historische Unterstützungszonen' in m.value for m in app.markdown)


def test_missing_history_is_not_reported_as_proof_of_no_support():
    from streamlit.testing.v1 import AppTest
    p=build_support_map({'ticker':'HALO'},now=NOW)
    script=f'''
import streamlit as st
import pandas as pd
from modules.support_map import render_screener_supports
render_screener_supports(st,pd.DataFrame([{{'Ticker':'HALO','Status':'Rot','__support_map':{p!r}}}]))
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]['Nächste Tiefzone']=='Nicht berechenbar'
