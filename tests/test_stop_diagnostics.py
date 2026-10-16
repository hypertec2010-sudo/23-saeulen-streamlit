from copy import deepcopy
from pathlib import Path
import pandas as pd
import pytest
from modules.structural_trade_plan import confirmed_support_stop, stop_evidence
from modules.technical_crv import build_technical_crv_package, technical_crv_eligibility
from modules.stop_diagnostics import build_stop_diagnostic_rows
from test_crv_consistency import radar_rr


def lows(*values, baseline=100):
    data=[float(baseline)]*40
    for index,value in values:data[index]=value
    return pd.DataFrame({'Low':data})


@pytest.mark.parametrize('df,code',[
    (None,'inputs_missing'),(pd.DataFrame({'Close':[100]*40}),'lows_missing'),
    (pd.DataFrame({'Low':[100]*12}),'history_short'),(lows(),'no_pivots'),
    (lows((8,99.4),(23,99.5)),'no_downside_pivot'),
    (lows((8,95)),'no_retest'),(lows((8,95),(23,98)),'no_retest'),
    (lows((8,60),(23,60.2)),'stop_distance')])
def test_rejection_stages_are_specific_without_granting_stop(df,code):
    p=confirmed_support_stop(df,current_price=100,entry_reference=100)
    assert p['reason_code']==code and p['reason']
    assert p['quality']=='unverified' and p['stop'] is None and p['support'] is None
    if code=='no_retest':assert p['candidate_support'] is not None
    if code=='stop_distance':assert p['candidate_support']==60 and p['candidate_stop']<60


def fixture(source='Swing-Support',stop=None):
    df=lows((8,95),(23,95.2))
    ref=confirmed_support_stop(df,current_price=100,entry_reference=100,atr=2)
    return dict(price=100,df=df,atr=2,stop_used=ref['stop'] if stop is None else stop,
                stop_source=source,suggested_entry_zone=(100,100),technical_target_1=110)


def test_confirmed_reference_preserves_exact_stop_and_does_not_mutate_history():
    df=lows((8,95),(23,95.2));before=df.copy(deep=True)
    p=confirmed_support_stop(df,current_price=100,entry_reference=100,atr=2)
    assert p['quality']=='confirmed_swing' and p['support']==95 and p['stop']==94.43
    assert p['touches']==2 and p['reason_code']=='confirmed'
    pd.testing.assert_frame_equal(df,before)


@pytest.mark.parametrize('source,stop,quality,reason',[
    ('Swing-Support',None,'confirmed','passt zur bestätigten'),
    ('Praxis-Mindestabstand',None,'proxy','verwendet wird ein Ersatzstop'),
    ('Swing-Support',97,'estimated','0,2-%-Toleranz')])
def test_actual_package_radar_and_rows_distinguish_reference_from_used_stop(radar_rr,source,stop,quality,reason):
    r=fixture(source,stop);before=deepcopy(r)
    pkg=build_technical_crv_package(r);diag=pkg['technical_crv_stop_diagnostics']
    assert diag['used']['quality']==quality and reason in diag['used']['reason']
    rows=build_stop_diagnostic_rows(diag,ccy='USD')
    assert rows[0]['Kurs']=='95.00 USD' and rows[1]['Kurs']=='94.43 USD'
    assert rows[2]['Kurs']==f"{r['stop_used']:.2f} USD"
    assert ('verwendeter Stop ist bestätigt' if quality=='confirmed' else 'nicht automatisch') in rows[1]['Einordnung']
    assert technical_crv_eligibility(pkg)['now']==(quality=='confirmed')
    rr=radar_rr(r)
    assert rr['stop_diagnostics']==diag
    assert rr['crv_now_eligible']==(quality=='confirmed')
    if quality!='confirmed':
        assert 'Verwendeter Stop nicht als Strukturstop bestätigt' in rr['crv_evidence_reason']
        assert reason in rr['crv_evidence_reason']
    pd.testing.assert_frame_equal(r['df'],before['df'])
    assert r['stop_used']==before['stop_used']


def test_single_pivot_and_rejected_distance_are_not_displayed_as_confirmed():
    for df,word in [(lows((8,95)),'Mehrfachbestätigung fehlt'),
                    (lows((8,60),(23,60.2)),'Stop-Prüfung nicht bestanden')]:
        p=build_technical_crv_package(dict(price=100,df=df,stop_used=90,stop_source='ATR',technical_target_1=110))
        rows=build_stop_diagnostic_rows(p['technical_crv_stop_diagnostics'])
        assert word in rows[0]['Einordnung']
        assert not technical_crv_eligibility(p)['now']


def test_missing_stop_is_separate_from_missing_structure():
    r=fixture();r.pop('stop_used')
    diag=build_technical_crv_package(r)['technical_crv_stop_diagnostics']
    assert diag['structure']['quality']=='confirmed_swing'
    assert diag['used']['quality']=='missing' and 'Verwendeter Stop fehlt' in diag['used']['reason']


def test_real_streamlit_table_renders_all_three_stages():
    from streamlit.testing.v1 import AppTest
    script='''
import streamlit as st
import pandas as pd
from modules.technical_crv import build_technical_crv_package
from modules.stop_diagnostics import render_stop_diagnostics
data=[100.]*40;data[8]=95;data[23]=95.2
pkg=build_technical_crv_package(dict(price=100,df=pd.DataFrame({'Low':data}),
    stop_used=96,stop_source='ATR',technical_target_1=110))
render_stop_diagnostics(st,pkg['technical_crv_stop_diagnostics'],ccy='USD')
'''
    at=AppTest.from_string(script).run(timeout=20)
    assert not at.exception
    assert len(at.get('table'))==1
    assert any('Ersatzstop' in e.value for e in at.caption)
    source=(Path(__file__).resolve().parents[1]/'legacy_app.py').read_text()
    assert 'render_stop_diagnostics(st, _bd_diag_v3021bd.get("crv_stop_diagnostics"), ccy=ccy)' in source
