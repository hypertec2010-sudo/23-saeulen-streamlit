from copy import deepcopy
from datetime import datetime, timezone

import pandas as pd
import pytest

from modules import candidate_radar as r
from modules import radar_trade_style as h

NOW=datetime(2026,10,28,15,tzinfo=timezone.utc)


def result():
    idx=pd.bdate_range(end='2026-10-27',periods=25)
    frame=pd.DataFrame(dict(High=101.,Low=99.,Close=100.),index=idx)
    return dict(ticker='HALO',df=frame,setup_type='Konsolidierung',valid_trade_setup=True,
        price=100.,confidence_info={'coverage':1.},trigger_status='aktiv',horizon_label='Langfristig (6-24 Monate)')


def row(**changes):
    return dict(ticker='HALO',name='Halo',status=r.READY,valid_setup=True,price=100.,stop=98.,target=104.,
        target_provenance={'label':'Technisches Ziel'},quote_date='2026-10-27',
        trade_style_evidence=h.daily_evidence(result(),asof=NOW)) | changes


@pytest.mark.parametrize('target,stop,style',[(104,97,'Kurzfristiger Trade'),(104.0001,98,'Swing'),
    (104,96.999,'Swing'),(112,94,'Swing'),(112.0001,94,'Zeithorizont offen'),(108,93.99,'Zeithorizont offen')])
def test_explicit_atr_boundaries_are_not_rounded_or_treated_as_intraday(target,stop,style):
    x=row(target=target,stop=stop);before=deepcopy(x)
    d=h.describe(x)
    assert d['style']==style and d['target_atr']==pytest.approx((target-100)/2)
    assert x==before and 'Intraday' not in d['style']


@pytest.mark.parametrize('change',[
    {'trade_style_evidence':None},{'trade_style_evidence':{'policy':h.POLICY,'basis':'5m','atr':2}},
    {'target_provenance':{'label':'Analystenziel'}},{'target_provenance':{'label':'Technisches Ziel + rechnerische Anhebung'}},
    {'target_provenance':{'label':'R-Planungsziel'}},{'target':99},{'stop':101},
    {'quote_date':'2026-11-09'}, {'price':float('nan')},
])
def test_insufficient_or_different_plan_data_leaves_horizon_open(change):
    assert h.describe(row(**change))['style']=='Zeithorizont offen'


@pytest.mark.parametrize('status,readiness',[(r.READY,'Frische Einstiegsprüfung nötig'),
    (r.NEAR,'Bestätigung abwarten'),(r.BLOCKED,'Gesperrt'),(r.HISTORY,'Historisch · neu prüfen')])
def test_style_never_overrides_current_entry_gates_or_history(status,readiness):
    d=h.describe(row(status=status))
    assert d['style']=='Kurzfristiger Trade' and d['readiness']==readiness


def test_no_setup_is_observation_even_with_a_grade_and_near_target():
    d=h.describe(row(valid_setup=False,grade='A'))
    assert d['style']=='Beobachtung'


@pytest.mark.parametrize('source',['Technisches Setup-Ziel','CHSM-Widerstandszone (3 Berührungen)',
    'Breakout-Projektion aus 20T-Range (4.0% Range-Höhe)'])
def test_existing_production_technical_target_labels_are_supported(source):
    assert h.describe(row(target_provenance={'label':source}))['style']=='Kurzfristiger Trade'


@pytest.mark.parametrize('kind',['intraday','weekly','future','invalid','short'])
def test_daily_volatility_requires_actual_usable_daily_history(kind):
    x=result();f=x['df']
    if kind=='intraday':f.index=pd.date_range('2026-10-27T12:00Z',periods=len(f),freq='5min')
    if kind=='weekly':f.index=pd.date_range(end='2026-10-27',periods=len(f),freq='7D')
    if kind=='future':f.index=f.index+pd.Timedelta(days=10)
    if kind=='invalid':f['High']=1.
    if kind=='short':x['df']=f.tail(10)
    assert h.daily_evidence(x,asof=NOW)['atr'] is None


def test_running_daily_bar_and_selected_analysis_horizon_do_not_change_classification():
    x=result();before=h.daily_evidence(x,asof=NOW)
    running=x['df'].tail(1).copy();running.index=pd.DatetimeIndex(['2026-10-28']);running['High']=500.
    x['df']=pd.concat([x['df'],running]);x['horizon_label']='Intraday'
    assert h.daily_evidence(x,asof=NOW)==before
    assert before['basis_date']=='2026-10-27' and before['atr']==2.


def test_european_daily_midnight_uses_the_same_date_basis_as_radar_quotes():
    x=result();x['df'].index=x['df'].index.tz_localize('Europe/Berlin')
    evidence=h.daily_evidence(x,asof=NOW)
    assert evidence['basis_date']=='2026-10-27'
    assert h.describe(row(trade_style_evidence=evidence,quote_date=r.quote_date(x)))['style']=='Kurzfristiger Trade'


def test_candidate_pipeline_preserves_existing_decisions_and_freezes_small_evidence():
    x=result();before=x['df'].copy(deep=True)
    decide=lambda *_:dict(grade='A',score=88,top_chance_rank=80,bucket='Jetzt prüfbar')
    package=lambda *_:dict(crv=2.,price=100.,stop=98.,tp1=104.,entry_zone='99–100',entry_distance_pct=0.,target_source='Technisches Ziel')
    candidate=r.build_candidate(x,ticker='HALO',style='Charttechnik',decide=decide,entry_package=package,analyzed_at=NOW)
    assert candidate['grade']=='A' and candidate['score']==88 and candidate['crv']==2.
    assert candidate['trade_style_evidence']['atr']==2.
    assert h.describe(candidate)['style']=='Kurzfristiger Trade'
    assert 'df' not in candidate and len(str(candidate['trade_style_evidence']))<500
    pd.testing.assert_frame_equal(x['df'],before)


def test_real_radar_card_shows_style_readiness_and_collapsed_reason():
    from streamlit.testing.v1 import AppTest
    x=row(grade='A',why='Test',next_step='Frisch prüfen',crv=2.,sector='Software',asset_type='Aktie',coverage=1.)
    script=f'''
import streamlit as st
from modules.candidate_radar_ui import render_result_card
render_result_card(st,{x!r})
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    assert any('Kurzfristiger Trade' in x.value for x in app.markdown)
    assert any('Frische Einstiegsprüfung nötig' in x.value for x in app.markdown)
    assert any(e.label=='Begründung zum Handelsstil' and not e.proto.expanded for e in app.expander)


def radar_payload():
    rows=[]
    for ticker,goal in [('HALO',104.),('HPE',108.)]:
        x=result();x['ticker']=ticker
        rows.append(r.build_candidate(x,ticker=ticker,style='Charttechnik',
            decide=lambda *_:dict(grade='A',score=88,top_chance_rank=80,bucket='Jetzt prüfbar'),
            entry_package=lambda *_,goal=goal:dict(crv=(goal-100)/2,price=100.,stop=98.,tp1=goal,
                entry_zone='99–100',entry_distance_pct=0.,target_source='Technisches Ziel'),analyzed_at=NOW))
    return dict(schema_version=r.SCHEMA_VERSION,radar_version=r.RADAR_VERSION,completed=True,
        owner='radar-test',key=r.request_key('Eigene Liste','Charttechnik',['HALO','HPE']),
        started_at=NOW.isoformat(),completed_at=NOW.isoformat(),scan_id='fixture',source='manual',
        requested=2,processed=2,symbols=['HALO','HPE'],rows=rows,errors=[],resolution=[])


def test_snapshot_reload_and_history_do_not_create_fresh_entry_readiness():
    payload=radar_payload();before=deepcopy(payload)
    assert r.snapshot_valid(payload)
    rows=r.view_rows(payload,now='2026-11-01T15:00Z')
    assert all(h.describe(x)['readiness']=='Historisch · neu prüfen' for x in rows)
    assert payload==before
    payload['rows'][0].pop('trade_style_evidence')
    assert r.snapshot_valid(payload)
    assert h.describe(r.view_rows(payload,now=NOW)[0])['style']=='Zeithorizont offen'


def test_radar_page_filters_trade_style_without_scan_or_membership_changes():
    from streamlit.testing.v1 import AppTest
    payload=radar_payload()
    script=f'''
import streamlit as st
from datetime import datetime,timezone
from modules.candidate_radar_ui import render_candidate_radar
st.session_state.setdefault('radar_universe','Eigene Liste')
st.session_state.setdefault('radar_screening_style','Charttechnik')
st.session_state.setdefault('radar_custom_input','HALO,HPE')
class Storage:
    user_id='radar-test'
    def load_namespace(self,*args,**kwargs):return {payload!r}
    def status(self):return {{'last_backend':'cloud','degraded':False}}
def forbidden(*args,**kwargs):raise AssertionError('Display must not trigger scan/queue')
render_candidate_radar(st,storage=Storage(),scan=forbidden,
    catalog_loader=lambda:[{{'Watchlist_Name':'Test','Watchlist_Type':'Watchlist'}}],
    watchlists_loader=lambda:[],positions_loader=lambda:{{}},queue_watchlist=forbidden,
    clock=lambda:datetime(2026,10,28,15,tzinfo=timezone.utc))
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    assert 'Swing' in app.selectbox(key='radar_v3021_trade_style').options
    app.selectbox(key='radar_v3021_trade_style').select('Swing').run()
    assert not app.exception and not app.error
    assert app.multiselect[0].options==['HPE']
    assert any('Handelsstil' in str(x.value.columns) for x in app.dataframe)
    assert any('Handelsstil: **Swing**' in x.value for x in app.markdown)
