import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from modules import rotation_radar as radar
from modules.environment_context import (build_environment_context, attach_environment_columns,
    stock_daily, expected_daily_date, resolve_groups)

NOW='2026-10-17T15:00:00Z'
META={'snapshot_id':'verified-test','snapshot_source':'v30.3d verified storage',
      'legacy_fallback':False,'data_through':'2026-10-16','generated_at':'2026-10-16T21:00:00Z'}


def row(symbol,**overrides):
    data={'Ticker':symbol,'Kursdatum':'2026-10-16','Benchmarkdatum':'2026-10-16',
          'Vorheriger Handelstag':'2026-10-15','Kontext-Datenbasis':'aligned',
          'Perf 1T %':.5,'Perf 5T %':2.,'RS 5T %':1.,'Rotation Δ1T':1.,
          'Rotation Δ5T':2.,'Phase':'🟢 Leading','Leadership':80.,'Rotation':70.}
    return {**data,**overrides}


def evaluate(symbol='MU',rows=None,meta=None,**result):
    return build_environment_context({'ticker':symbol,**result},
        pd.DataFrame(rows or [row('SPY'),row('XLK'),row('SMH')]), META if meta is None else meta,now=NOW)


@pytest.mark.parametrize('symbol',['MU','MRVL','NVDA','AMD'])
def test_semiconductor_mapping_uses_same_radar_smh(symbol):
    p=evaluate(symbol)
    assert p['groups']['industry']=='SMH' and p['severity']==0
    assert [level['symbol'] for level in p['levels']]==['SPY','XLK','SMH']


@pytest.mark.parametrize('ticker,sector,industry,expected',[
    ('ANET','Technology','Communication Equipment','XLK'),
    ('BE','Industrials','Electrical Equipment & Parts','XLI')])
def test_anet_be_are_not_forced_into_semis_or_ai(ticker,sector,industry,expected):
    p=evaluate(ticker,sector=sector,industry=industry,rows=[row('SPY'),row(expected)])
    assert p['groups']['industry'] is None and p['groups']['sector']==expected
    assert 'Nur Sektorproxy' in p['reason']


@pytest.mark.parametrize('symbol,role,value,severity',[
    ('SPY','Perf 1T %',-2.,2),('SPY','Perf 1T %',-.75,1),
    ('SMH','Perf 1T %',-3.,2),('SMH','Perf 1T %',-1.,1),
    ('SMH','Perf 5T %',-6.,2),('SMH','Rotation Δ1T',-8.,1),
    ('SMH','Rotation Δ5T',-10.,1),('SMH','Phase','🔴 Rotating Out',2)])
def test_warnings_separate_absolute_selloff_and_rotation(symbol,role,value,severity):
    rows=[row(s,**({role:value} if s==symbol else {})) for s in ['SPY','XLK','SMH']]
    p=evaluate(rows=rows)
    assert p['severity']==severity
    assert symbol in p['reason']


def test_absolute_selloff_not_hidden_by_positive_relative_strength():
    p=evaluate(rows=[row('SPY',**{'Perf 1T %':-3.5}),row('XLK'),
                     row('SMH',**{'Perf 1T %':-3.,'RS 5T %':8.})])
    assert p['severity']==2


@pytest.mark.parametrize('change',[
    {'legacy_fallback':True},{'snapshot_source':'legacy'},{'snapshot_id':None},
    {'data_through':'2026-10-15'},{'generated_at':'2026-10-16T15:00:00Z'},
    {'generated_at':None},{'generated_at':'2026-10-18T21:00:00Z'}])
def test_unverified_old_partial_day_or_future_snapshot_cannot_show_green(change):
    p=evaluate(meta={**META,**change})
    assert p['severity'] is None and p['status']=='⚪ Unbekannt'


@pytest.mark.parametrize('change',[
    {'Kursdatum':None},{'Kontext-Datenbasis':'missing'},
    {'Benchmarkdatum':'2026-10-15'},{'Perf 1T %':None},{'Rotation Δ1T':None}])
def test_missing_group_evidence_prevents_overall_green(change):
    p=evaluate(rows=[row('SPY'),row('XLK'),row('SMH',**change)])
    assert p['severity'] is None


def test_known_warning_is_visible_even_when_other_context_missing():
    p=evaluate(rows=[row('SPY',**{'Perf 1T %':-2.5})])
    assert p['severity']==2 and p['partial'] and 'Teilbasis' in p['reason']


def test_relative_stock_return_requires_identical_two_dates():
    df=pd.DataFrame({'Close':[100.,97.,90.]},index=pd.to_datetime(['2026-10-15','2026-10-16','2026-10-19']))
    rows=[row('SPY'),row('XLK'),row('SMH',**{'Perf 1T %':-4.})]
    p=evaluate(rows=rows,df=df)
    assert p['stock_rs_1d']==pytest.approx(1.)
    df.index=pd.to_datetime(['2026-10-14','2026-10-16','2026-10-19'])
    assert evaluate(rows=rows,df=df)['stock_rs_1d'] is None


def test_intraday_stock_bars_are_not_daily_proof():
    df=pd.DataFrame({'Close':[100,99]},index=pd.to_datetime(['2026-10-16 09:00','2026-10-16 10:00']))
    assert stock_daily({'df':df},now=NOW)=={}


def test_close_clock_is_weekend_aware_and_conservative():
    assert str(expected_daily_date(NOW))=='2026-10-16'
    assert str(expected_daily_date('2026-10-19T18:00:00Z'))=='2026-10-16'
    assert str(expected_daily_date('2026-10-19T21:00:00Z'))=='2026-10-19'


def test_one_snapshot_both_views_and_original_technical_fields_unchanged():
    frame=pd.DataFrame([{'Ticker':'MU','Ampel':'🟢','Live-Score':'85/100','Setup-Stärke':'82/100',
                         'Freigabe-Check':'Keine offene Freigabebedingung','Status':'Kaufen',
                         '__env_stock':float('nan')},
                        {'Ticker':'MRVL','Ampel':'🟡','Live-Score':'60/100','Status':'Planen'}])
    before=frame.copy(deep=True);rr=pd.DataFrame([row('SPY'),row('XLK'),row('SMH',**{'Perf 1T %':-5.})])
    radar_before=rr.copy(deep=True);meta=deepcopy(META)
    out=attach_environment_columns(frame,rr,META,now=NOW)
    pd.testing.assert_frame_equal(frame,before)
    pd.testing.assert_frame_equal(out[before.columns],before)
    pd.testing.assert_frame_equal(rr,radar_before)
    assert META==meta and (out['Umfeldrisiko']=='🔴 Hoch').all()
    single=evaluate(rows=rr.to_dict('records'))
    assert out.iloc[0]['Umfeld-Grund']==single['reason']
    assert (out['Umfeld-Snapshot']==single['snapshot_id']).all()


def test_old_saved_rows_and_empty_frames_are_supported():
    out=attach_environment_columns(pd.DataFrame([{'Ticker':'ANET','Ampel':'🟢'}]),pd.DataFrame(),{},now=NOW)
    assert out.iloc[0]['Ampel']=='🟢' and out.iloc[0]['Umfeldrisiko']=='⚪ Unbekannt'
    assert attach_environment_columns(pd.DataFrame(),pd.DataFrame(),{},now=NOW).empty


def test_new_radar_metrics_require_aligned_days_and_preserve_existing_scores():
    idx=pd.bdate_range('2025-09-01',periods=280)
    prices=pd.DataFrame({'SMH':np.linspace(100,150,280),'SPY':np.linspace(100,120,280)},index=idx)
    spec=next(s for s in radar.UNIVERSE if s.symbol=='SMH')
    base=radar._metrics_for(spec,prices)
    assert base['Kontext-Datenbasis']=='aligned' and base['Perf 1T %']>0
    assert base['Kursdatum']==idx[-1].date().isoformat()
    # New metrics add evidence, never change the original metric formulas.
    original=radar._context_daily_metrics
    try:
        radar._context_daily_metrics=lambda *a:{}
        old=radar._metrics_for(spec,prices)
    finally:radar._context_daily_metrics=original
    assert all(base[k]==v for k,v in old.items())
    prices.loc[idx[-2],'SPY']=np.nan
    assert radar._metrics_for(spec,prices)['Kontext-Datenbasis']=='missing'


def test_actual_streamlit_risk_rubric_and_both_screener_layouts():
    from streamlit.testing.v1 import AppTest
    script="""
import streamlit as st
from modules.environment_context import build_environment_context,render_environment_context
import pandas as pd
render_environment_context(st,build_environment_context({'ticker':'ANET'},pd.DataFrame(),{}))
"""
    at=AppTest.from_string(script).run(timeout=20)
    assert not at.exception and len(at.get('table'))==1
    assert any('Unbekannt' in e.value for e in at.markdown)
    root=Path(__file__).resolve().parents[1]
    source=(root/'legacy_app.py').read_text();tree=ast.parse(source)
    main=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='main_cols' for t in n.targets)
              and isinstance(n.value,ast.List)
              and any(isinstance(e,ast.Constant) and e.value=='Invalidierung-Detail' for e in n.value.elts))
    assert any(isinstance(e,ast.Constant) and e.value=='Umfeldrisiko' for e in main.value.elts)
    compact=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='_compact_cols_v3021l' for t in n.targets)
              and isinstance(n.value,ast.List))
    assert any(isinstance(e,ast.Constant) and e.value=='Umfeldrisiko' for e in compact.value.elts)
    assert '{_env_badge_v3021bp}' in source and 'render_environment_context(st, _environment_single_v3021bp)' in source


def test_actual_snapshot_fingerprint_protects_new_risk_evidence():
    import hashlib,json
    from datetime import datetime,date
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'legacy_app.py').read_text())
    names={'_v303d_rotation_frame_fingerprint','_v303d_json_safe'}
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'pd':pd,'np':np,'hashlib':hashlib,'json':json,'datetime':datetime,'date':date}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'actual_snapshot_fingerprint','exec'),scope)
    frame=pd.DataFrame([row('SMH')]);fingerprint=scope['_v303d_rotation_frame_fingerprint']
    first=fingerprint(frame)
    frame.loc[0,'Perf 1T %']=-5
    assert fingerprint(frame)!=first
    second=fingerprint(frame)
    frame.loc[0,'Kursdatum']='2026-10-15'
    assert fingerprint(frame)!=second


def test_cached_forming_stock_bar_not_promoted_after_close():
    frame=pd.DataFrame({'Close':[100,101,99]},index=pd.to_datetime(['2026-10-14','2026-10-15','2026-10-16']))
    p=stock_daily({'df':frame},now='2026-10-16T21:00:00Z')
    assert p['date']=='2026-10-15'


def test_technical_decision_has_no_dependency_on_environment_context():
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'modules/live_monitor.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_decision_state_v3021az')
    source=ast.unparse(node).casefold()
    assert 'environment' not in source and 'umfeld' not in source
