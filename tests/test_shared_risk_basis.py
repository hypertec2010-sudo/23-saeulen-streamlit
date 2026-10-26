"""Recent trade structure, long-term reference and one persistent risk basis."""
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
import ast
import json
from pathlib import Path
import pandas as pd
import pytest
from modules.stop_selection import select_productive_stop
from modules.structural_trade_plan import TRADE_STOP_POLICY
from modules.technical_crv import build_technical_crv_package,technical_crv_eligibility
from modules.risk_calculator import build_analysis_risk_basis,risk_basis_from_row,atr_stop_scenario,_v230_calculate_position_size
from modules.stop_diagnostics import build_stop_diagnostic_rows
from modules.live_screener_snapshot import dataframe_to_payload,dataframe_from_payload


def halo(recent=False,wide=False):
    lows=[125.]*80
    lows[8],lows[23]=105.50,105.6
    if recent:lows[58],lows[70]=(105.50,105.6) if wide else (123.,123.1)
    df=pd.DataFrame({'Low':lows})
    plan=select_productive_stop(df,price=125.63,entry_zone=(125,126),atr=3.61,
        setup_type='Breakout',fallback_stop=121.23,fallback_source='Praxis-Mindestabstand')
    return dict(ticker='HALO',price=125.63,df=df,atr=3.61,ts='26.10.2026',currency='USD',
        suggested_entry_zone=(125,126),setup_type='Breakout',valid_trade_setup=True,
        stop_used=plan['stop'],stop_source=plan['source'],stop_selection=plan,
        technical_target_1=145.55,chart_invalidation_level=122.96)


def test_halo_distant_reference_does_not_silently_become_the_trade_stop():
    r=halo();p=build_technical_crv_package(r)
    assert r['stop_used']==119.13 and r['stop_selection']['selected']=='atr_proxy'
    assert p['technical_crv_stop_quality']=='proxy' and not technical_crv_eligibility(p)['now']
    assert p['technical_crv_now']==pytest.approx((145.55-125.63)/(125.63-119.13))
    d=p['technical_crv_stop_diagnostics']
    assert d['overarching_structure']['support']==105.5
    assert d['overarching_structure']['stop']==pytest.approx(104.5975)
    assert d['structure']['quality']!='confirmed_swing'
    rows=build_stop_diagnostic_rows(d,ccy='USD')
    assert any(x['Baustein']=='Übergeordnete Unterstützung' and x['Kurs']=='105.50 USD' for x in rows)
    assert next(x for x in rows if x['Baustein']=='Tatsächlich verwendeter Stop')['Kurs']=='119.13 USD'
    # The model-only level 122.96 is not promoted to confirmed evidence.
    assert build_analysis_risk_basis(r)['chart_invalidation_stop'] is None


def test_recent_retested_structure_is_preferred_without_borrowing_model_confirmation():
    r=halo(recent=True);p=build_technical_crv_package(r)
    assert r['stop_used']==122.10
    assert r['stop_selection']['selected']=='confirmed_structure'
    assert p['technical_crv_stop_quality']=='confirmed'
    assert p['technical_crv_stop_diagnostics']['structure']['support']==123


def test_recent_but_distant_reference_is_retained_without_forcing_a_tighter_structure():
    r=halo(recent=True,wide=True);p=build_technical_crv_package(r)
    assert r['stop_used']==119.13
    assert r['stop_selection']['structure']['reason_code']=='stop_distance'
    assert p['technical_crv_stop_quality']=='proxy'
    assert p['technical_crv_stop_diagnostics']['overarching_structure']['quality']=='confirmed_swing'


def test_position_mode_keeps_existing_stop():
    r=halo()
    p=select_productive_stop(r['df'],price=125.63,entry_zone=(125,126),atr=3.61,
        setup_type='Breakout',fallback_stop=104.60,fallback_source='Existing stop',position_mode=True)
    assert p['stop']==104.6 and p['selected']=='existing' and 'policy' not in p


@pytest.mark.parametrize('recent',[False,True])
def test_snapshot_calculation_and_optional_atr_use_the_exact_central_basis(recent):
    r=halo(recent=recent);p=build_technical_crv_package(r)
    b=build_analysis_risk_basis(r,generated_at='2026-10-26T12:00:00+01:00')
    row={'Ticker':'HALO','Kurs':127.,'__risk_basis':b}
    original=deepcopy(b)
    restored=dataframe_from_payload(json.loads(json.dumps(dataframe_to_payload(pd.DataFrame([row])))))
    read=risk_basis_from_row(restored.iloc[0].to_dict())
    assert read==original and read['price']==125.63 # different live quote never replaces basis
    calc=_v230_calculate_position_size(read['entry_default'],read['stop'],read['target'],50000,.5)
    assert read['stop']==p['technical_crv_stop_now'] and calc['crv']==p['technical_crv_now']
    scenario=atr_stop_scenario(read)
    assert scenario['stop']<=read['stop'] and read==original
    changed=build_analysis_risk_basis(r,generated_at='2026-10-26T12:30:00+01:00')
    assert changed['fingerprint']!=read['fingerprint']
    assert risk_basis_from_row({'Ticker':'OTHER','__risk_basis':read}) is None
    assert risk_basis_from_row({'Ticker':'HALO'}) is None


def test_actual_risk_wrapper_has_no_independent_stop_selection_or_analysis():
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'legacy_app.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_v230_extract_position_inputs')
    scope={'get_current_berlin_time':lambda:datetime(2026,10,26,12)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_wrapper','exec'),scope)
    r=halo();b=scope['_v230_extract_position_inputs'](r)
    assert b['stop']==r['stop_used'] and b['target']==145.55


def test_risk_basis_survives_actual_history_cleanup(monkeypatch):
    from modules import live_monitor as lm
    class State(dict):
        __getattr__=dict.__getitem__
        __setattr__=dict.__setitem__
    monkeypatch.setattr(lm,'st',SimpleNamespace(session_state=State()))
    monkeypatch.setattr(lm,'get_current_berlin_time',lambda:datetime(2026,10,26,12),raising=False)
    monkeypatch.setattr(lm,'_v227_load_persistent_live_history',lambda:({},[]))
    monkeypatch.setattr(lm,'_v227_save_persistent_live_history',lambda *a:None)
    monkeypatch.setattr(lm,'_v2416_log_event',lambda **kw:None,raising=False)
    b=build_analysis_risk_basis(halo(),generated_at='2026-10-26T12:00:00')
    raw=pd.DataFrame([dict(Ticker='HALO',Kurs=125.63,Ampel='🟢',Status='Kauftrigger aktiv',
        **{'Live-Score':'80/100','__risk_basis':b,'__temporary':99,'__decision_state_v3021az':'buy'})])
    hist,_=lm.apply_live_watchlist_status_history_v220(raw,watchlist_name='Test')
    assert hist.iloc[0]['__risk_basis']==b and '__temporary' not in hist
    assert risk_basis_from_row(hist.iloc[0].to_dict())==b


def test_real_calculator_modes_keep_standard_immutable_and_reset_on_new_scan():
    from streamlit.testing.v1 import AppTest
    script='''
import streamlit as st
from test_shared_risk_basis import halo
from modules.risk_calculator import build_analysis_risk_basis,render_snapshot_calculator
r=halo(recent=True)
b=build_analysis_risk_basis(r,generated_at=st.session_state.get('scan_at','first'))
st.session_state['original_basis']=b
render_snapshot_calculator(st,{'Ticker':'HALO','Kurs':125.63,'__risk_basis':b},currency='USD')
'''
    at=AppTest.from_string(script).run(timeout=20)
    assert not at.exception
    def input(label):return next(n for n in at.number_input if n.label==label)
    assert input('Stop').value==122.10 and input('Stop').disabled
    assert input('Geplanter Entry').value==125.63
    at.selectbox[0].select('ATR-Vergleich').run()
    assert not at.exception and input('Stop').value<122.10
    assert at.session_state['original_basis']['stop']==122.10
    at.selectbox[0].select('Manuelles Szenario').run()
    input('Stop').set_value(110.).run()
    assert not at.exception and input('Stop').value==110
    assert at.session_state['original_basis']['stop']==122.10
    at.session_state['scan_at']='second'
    at.run()
    assert not at.exception and at.selectbox[0].value=='Analysewerte (Standard)'
    assert input('Stop').value==122.10
    at.selectbox[0].select('Manuelles Szenario').run()
    input('Stop').set_value(111.).run()
    at.button[0].click().run()
    assert not at.exception and at.selectbox[0].value=='Analysewerte (Standard)'
    assert input('Stop').value==122.10


def test_old_snapshot_requests_rescan_instead_of_a_second_analysis():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_string('''
import streamlit as st
from modules.risk_calculator import render_snapshot_calculator
render_snapshot_calculator(st,{'Ticker':'HALO','Kurs':125.63})
''').run(timeout=20)
    assert not at.exception and not at.number_input
    assert 'neu scannen' in at.info[0].value
    # The actual calculator page has no independent analysis/cache fallback.
    tree=ast.parse((Path(__file__).resolve().parents[1]/'legacy_app.py').read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                and ast.unparse(n.test)=="cockpit_area == '📐 Risiko-Rechner'")
    calls=[n.func for statement in branch.body for n in ast.walk(statement) if isinstance(n,ast.Call)]
    assert not any(isinstance(f,ast.Name) and f.id=='analyze_stock' for f in calls)
    assert any(isinstance(f,ast.Attribute) and f.attr=='render_snapshot_calculator' for f in calls)
