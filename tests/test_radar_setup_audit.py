from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from modules import candidate_radar as r
from modules import radar_setup_audit as a
from modules import watch_setups as s
from test_watch_setups import range_result, NOW
from test_radar_trade_style import radar_payload


def test_preview_is_not_a_full_context_or_crv_release():
    result=range_result();before=result['df'].copy(deep=True)
    preview=a.chart_audit(result,now=NOW,full_context=False)
    assert preview['phase']=='ready' and not preview['full_context']
    assert 'rr_allowed' not in preview and 'entry' not in preview and 'fingerprint' not in preview
    pd.testing.assert_frame_equal(result['df'],before)


def test_full_context_applies_known_earnings_and_frozen_setup_crv():
    result=range_result()
    full=a.chart_audit(result,now=NOW,full_context=True)
    package=s.detect(result,now=NOW)
    from modules.watch_quality import assess
    assert full['rr_cap']==assess(package)['rr_cap'] and full['rr_allowed']
    result['technical_target_1']=100.5
    low=a.chart_audit(result,now=NOW,full_context=True)
    assert low['phase']=='ready' and not low['rr_allowed']
    result['days_earn']=1
    assert a.chart_audit(result,now=NOW,full_context=True)['reason_code']=='earnings'


def test_diagnostic_exception_never_discards_existing_candidate(monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('private provider text must not leak')
    monkeypatch.setattr(s,'detect',fail)
    out=a.chart_audit(range_result(),now=NOW,full_context=True)
    assert out['reason_code']=='audit_error' and 'private' not in str(out)


def test_full_vorscan_is_preserved_and_selection_and_provider_counts_are_unchanged(monkeypatch):
    from modules import provider_manager
    monkeypatch.setattr(provider_manager,'get_market_data_provider',lambda:SimpleNamespace(quarantine_state=lambda:None))
    symbols=[f'STOCK{i:02}' for i in range(35)]
    history_calls=[];analysis_calls=[]
    result=range_result()
    def history(ticker):history_calls.append(ticker);return result['df']
    def analyze(**kwargs):
        analysis_calls.append(kwargs['ticker'])
        return {**result,'ticker':kwargs['ticker'],'confidence_info':{'coverage':1.},'valid_trade_setup':True}
    score=r.build_light_candidate(result['df'],ticker='X',style='Charttechnik')['light_score']
    payload=r.run_two_stage_scan(universe='Eigene Liste',style='Charttechnik',entries=symbols,
        analyze=analyze,decide=lambda *_:dict(grade='A',score=80,bucket='Jetzt prüfbar'),
        entry_package=lambda *_:dict(price=99.2,stop=97.,tp1=110.,crv=4.,entry_zone='99–100',entry_distance_pct=0.),
        history_loader=history,full_analysis_limit=15,clock=lambda:pd.Timestamp(NOW))
    assert len(history_calls)==35 and analysis_calls==symbols[:15]
    assert payload['preselected_symbols']==symbols[:15] and len(payload['preselection'])==35
    assert all(x['light_score']==score for x in payload['preselection'])
    assert r.snapshot_valid(payload)
    selection,full=a.tables(payload)
    assert len(selection)==35 and len(full)==15
    outside=selection.loc[~selection['Vollanalyse ausgewählt']]
    assert len(outside)==20 and (outside['OHLCV-Musterphase']=='Bestätigungsbereit').all()
    assert all('chart_audit' in x for x in payload['rows'])


def test_old_scan_is_unknown_and_never_implies_no_pattern_or_good_crv():
    payload=radar_payload()
    for row in payload['rows']:row.pop('chart_audit',None)
    selection,full=a.tables(payload)
    assert selection.empty and set(full['Diagnose'])=={'Chartdiagnose fehlt'}


def test_full_categories_are_disjoint_and_do_not_mix_radar_crv_with_setup_crv():
    p=radar_payload();rows=[]
    for n in range(4):
        row=deepcopy(p['rows'][0]);row['ticker']=str(n)
        row['chart_audit']=dict(policy=a.POLICY,phase='ready',rr_allowed=True,rr_cap=2.)
        row['crv']=.19
        rows.append(row)
    rows[0]['gates']=['Earnings']
    rows[1]['chart_audit']['phase']='forming'
    rows[2]['chart_audit']['rr_allowed']=False
    p['rows']=rows
    _,full=a.tables(p)
    assert len(set(full['Diagnose']))==4
    assert full.iloc[3]['Diagnose']=='Chartsetup und Setup-CRV im Scan prüfenswert'
    assert 'Unter' in full.iloc[3]['Radar-CRV-Einordnung']
    assert full.iloc[3]['Zielabstand %']==pytest.approx(4.)
    assert full.iloc[3]['Stopabstand %']==pytest.approx(2.)


def test_card_warns_low_radar_crv_without_changing_status_or_grade():
    from streamlit.testing.v1 import AppTest
    row=radar_payload()['rows'][0];row['crv']=.19;before=deepcopy(row)
    script=f'''
import streamlit as st
from modules.candidate_radar_ui import render_result_card
render_result_card(st,{row!r})
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and any('Radar-CRV 0,19' in x.value for x in app.warning)
    assert row==before and any('Qualitätsnote **A**' in x.value for x in app.markdown)


def test_diagnostic_panel_shows_unselected_candidates_without_data_fetch():
    from streamlit.testing.v1 import AppTest
    p=radar_payload();p['light_processed']=2;p['preselected_symbols']=['HALO']
    p['preselection']=[dict(ticker=t,light_score=80,chart_preview=dict(policy=a.POLICY,phase='ready',reason='Test')) for t in ['HALO','HPE']]
    script=f'''
import streamlit as st
from modules.radar_setup_audit import render_audit
render_audit(st,{p!r})
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and len(app.dataframe)==2
    assert any('1 mit OHLCV-Musterphase' in x.value for x in app.markdown)
    assert any('Vollanalyse außerhalb' in x.value and 'HPE' in x.value for x in app.markdown)
