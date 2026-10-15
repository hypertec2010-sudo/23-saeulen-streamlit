from copy import deepcopy
from datetime import datetime
import numpy as np
import pytest
from modules.decision_diagnostics import build_invalidation_diagnostics, explain_components
from modules.live_monitor import build_decision_state_v3021az
from test_setup_alert_crv_evidence import setup_alerts


def test_exact_used_quote_and_level_with_separate_conflicting_numeric_reference():
    r={'price':100,'live_price':98.5,'wave_structure_pkg':{'wave_invalidation_price':97}}
    d={'wave_invalidation':'Unter 99 USD'}
    before=deepcopy((r,d))
    p=build_invalidation_diagnostics(r,d,price=98.5,used_level=99)
    assert p['price_source']=='live_price' and p['level']==99 and p['breached']
    assert p['distance_pct']==pytest.approx((98.5/99-1)*100)
    assert p['numeric_model_level']==97 and p['mismatch']
    assert 'Modell' in p['source'] and 'nicht allein' in p['detail']
    assert (r,d)==before


@pytest.mark.parametrize('price,level',[(None,100),(100,None),(float('nan'),100),(100,0),(100,-1)])
def test_missing_diagnostic_inputs_are_not_invented(price,level):
    p=build_invalidation_diagnostics({}, {},price=price,used_level=level)
    assert not p['breached'] and p['distance_pct'] is None


@pytest.mark.parametrize('price,breached',[(99,True),(100,True),(101,False)])
def test_diagnostic_boundary_matches_existing_alert(price,breached):
    assert build_invalidation_diagnostics({}, {},price=price,used_level=100)['breached']==breached


def test_defaults_are_neither_drivers_nor_measured_brakes():
    p=explain_components({'Timing':26,'Konfluenz':42,'Chart':50,'Trend':50,'CRV':48},
        {'Timing':True,'Konfluenz':True},setup_strength=40,action_score=0,invalidated=True)
    assert 'Keine ausgeprägten' in p['drivers']
    assert 'Timing 26' in p['brakes'] and 'Konfluenz 42' in p['brakes']
    assert 'CRV' not in p['brakes'] and 'Chart' not in p['brakes']
    assert 'CRV' in p['neutral'] and '−45' in p['explanation']


@pytest.mark.parametrize('extra', [{},{'entry_hard_gate':True},{'invalidated':True}])
def test_detail_does_not_change_central_permission(extra):
    args=dict(valid_trade_setup=True,signal_packages_complete=True,timing_score=85,conf_score=71,
              trigger_status='Aktiv',crv_now_eligible=False,crv_entry_eligible=False)
    args.update(extra)
    a=build_decision_state_v3021az(**args)
    b=build_decision_state_v3021az(**args,invalidation_detail='Modelllevel 101, Kurs 100')
    assert (a['state'],a['buy_now'],a['setup_ready'])==(b['state'],b['buy_now'],b['setup_ready'])
    if extra.get('invalidated'):assert 'Modelllevel 101' in b['release_check']


@pytest.mark.parametrize('ticker,timing,conf,raw', [('MU',25.9,41.8,'Passiv'),('MRVL',36.1,53,'Beobachten')])
@pytest.mark.parametrize('horizon', ['Swing / 1-4 Wochen','Kurzfrist / Trading'])
def test_real_monitor_retains_block_and_old_score_but_preserves_strength(monkeypatch,setup_alerts,ticker,timing,conf,raw,horizon):
    from modules import live_monitor as lm
    # Use real alert parsing and generation, including exact reference capture.
    scope=setup_alerts.__globals__
    for name in ['_v210_alert_price','_v210_alert_num']:
        monkeypatch.setattr(lm,name,scope[name],raising=False)
    monkeypatch.setattr(lm,'build_setup_alerts_v210',setup_alerts,raising=False)
    monkeypatch.setattr(lm,'get_current_berlin_time',lambda:datetime(2026,10,15,10),raising=False)
    monkeypatch.setattr(lm,'_v214_monitor_final_release_check',lambda *a:(False,[]),raising=False)
    monkeypatch.setattr(lm,'shorten_text',lambda v,n:str(v)[:n],raising=False)
    monkeypatch.setattr(lm,'_v232_is_current_baseline_source',lambda v:False,raising=False)
    r=dict(ticker=ticker,price=100,ma20=100,valid_trade_setup=False,trigger_status=raw,
           signal_package_complete=True,timing_action_confidence_pkg={'score':timing},
           trigger_confluence_pkg={'score':conf},wave_structure_pkg={'wave_invalidation_price':101})
    d=dict(bucket='Starke Watchlist',grade='B',score=80,crv=2,crv_entry=2,
           crv_now_eligible=False,crv_entry_eligible=False,entry_zone='99 - 101',
           wave_invalidation='101',crv_evidence_reason='Strukturstop fehlt')
    row=lm._v212_monitor_status_from_decision(r,d,live_horizon=horizon)
    assert row['Ampel']=='🔴' and row['__decision_state_v3021az']=='blocked'
    assert row['Live-Score']=='0/100'
    strength=int(row['Setup-Stärke'].split('/')[0]);assert 35<strength<50
    assert row['Invalidierung-Level']==101 and row['Invalidierung-Kursbasis']==100
    assert '101.00' in row['Freigabe-Check'] and '100.00' in row['Freigabe-Check']
    assert 'Modell' in row['Invalidierung-Quelle']
    assert 'CRV' not in row['Score-Bremsen']
    assert 'Chart 50' not in row['Score-Treiber']
    assert '−45' in row['Warum dieser Score?']
    d['wave_invalidation']='99'
    other=lm._v212_monitor_status_from_decision(r,d,live_horizon=horizon)
    assert other['Setup-Stärke']==row['Setup-Stärke']
    assert other['Ampel']=='⚪'


def test_actual_summary_names_the_invalidated_model_reference():
    import ast,re
    from pathlib import Path
    tree=ast.parse((Path(__file__).resolve().parents[1]/'legacy_app.py').read_text())
    names={'_v243_clean_cell','_v304a_harvest_num','_v305c_live_takeaway'}
    funcs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {f.name for f in funcs}==names
    scope={'re':re};exec(compile(ast.Module(body=funcs,type_ignores=[]),'actual_summary','exec'),scope)
    detail='Wave-/Setup-Modelllevel 101.00; Vergleichskurs 100.00; Abstand -0.99%'
    row={'Ampel':'🔴','Status':'Blockiert','Aktive Einstiegsgates':'-',
         'Invalidierung-Detail':detail,'Freigabe-Check':'Invalidierung gebrochen',
         'Harvest-Score':68}
    text,action,tone=scope['_v305c_live_takeaway'](row)
    assert detail in text and 'Harvest 68' in text
    main=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='main_cols' for t in n.targets)
        and isinstance(n.value,ast.List)
        and any(isinstance(e,ast.Constant) and e.value=='Invalidierung-Detail' for e in n.value.elts))
    assert any(isinstance(e,ast.Constant) and e.value=='Freigabe-Check' for e in main.value.elts)
    assert tone=='warning' and 'erneuter technischer Freigabe' in action


def test_actual_compact_table_selects_strength_instead_of_zero_action_score():
    import ast,pandas as pd
    from pathlib import Path
    tree=ast.parse((Path(__file__).resolve().parents[1]/'legacy_app.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='_compact_cols_v3021l' for t in n.targets)
        and isinstance(n.value,ast.List))
    scope={'_desktop_live_display_v304a':pd.DataFrame([{'Setup-Stärke':'40/100','Live-Score':'0/100','Ampel':'🔴'}])}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_compact_columns','exec'),scope)
    assert 'Setup-Stärke' in scope['_compact_cols_v3021l']
    assert 'Live-Score' not in scope['_compact_cols_v3021l']
