"""Evidence eligibility and central release integration, independent of providers."""
import math
import pandas as pd
import pytest
from modules.technical_crv import build_technical_crv_package, technical_crv_eligibility, build_crv_basis_comparison
from modules.structural_trade_plan import confirmed_support_stop
from modules.live_monitor import build_decision_state_v3021az
from test_crv_consistency import radar_rr
from test_setup_alert_crv_evidence import setup_alerts


def result():
    lows = [100.0] * 40
    lows[8], lows[23] = 95.0, 95.2
    df = pd.DataFrame({"Low": lows})
    ref = confirmed_support_stop(df, current_price=100, entry_reference=100, atr=2)
    return dict(price=100, stop_used=ref['stop'], stop_source=ref['source'], df=df, atr=2,
                suggested_entry_zone=(100, 100), technical_target_1=110)


def decide(pkg, **extra):
    quality = technical_crv_eligibility(pkg)
    return build_decision_state_v3021az(valid_trade_setup=True, signal_packages_complete=True,
        timing_score=85, conf_score=71, trigger_status='Aktiv', crv_now=pkg['technical_crv_now'],
        crv_entry=pkg['technical_crv_entry'], crv_now_eligible=quality['now'],
        crv_entry_eligible=quality['entry'], crv_evidence_reason=quality['reason'], **extra)


def test_confirmed_actual_stop_and_technical_target_allow_release(radar_rr):
    r = result()
    pkg = build_technical_crv_package(r)
    assert technical_crv_eligibility(pkg)['now']
    assert technical_crv_eligibility(pkg)['entry']
    assert decide(pkg)['state'] == 'buy'
    rr = radar_rr(r)
    assert rr['crv_now_eligible'] and rr['crv_entry_eligible']
    assert build_crv_basis_comparison(r, pkg)[0]['operative']


@pytest.mark.parametrize('source', ['Praxis-Mindestabstand', 'ATR-Stop (1,8x)', 'Fallback-Stop', 'MA20-Naeherung'])
def test_proxy_never_qualifies_even_with_coincident_structural_reference(source, radar_rr):
    r = result(); r['stop_source'] = source
    pkg = build_technical_crv_package(r)
    assert math.isfinite(pkg['technical_crv_now']) and pkg['technical_crv_now'] > 1.5
    assert not technical_crv_eligibility(pkg)['now']
    assert not technical_crv_eligibility(pkg)['entry']
    ds = decide(pkg)
    assert ds['state'] == 'buy'
    assert not ds['crv_evaluated']
    assert 'Verwendeter Stop nicht als Strukturstop bestätigt' in ds['release_check']
    assert 'unter 1,50' not in ds['release_check']
    assert not radar_rr(r)['crv_now_eligible']
    assert not build_crv_basis_comparison(r,pkg)[0]['operative']


def test_structural_label_without_chart_evidence_never_qualifies():
    r=result(); r.pop('df')
    assert not technical_crv_eligibility(build_technical_crv_package(r))['now']


def test_different_used_stop_cannot_borrow_confirmed_reference():
    r=result();r['stop_used']=97
    assert not technical_crv_eligibility(build_technical_crv_package(r))['now']


def test_genuine_structure_does_not_have_a_percentage_floor():
    r=result();r['df'].loc[8,'Low']=98;r['df'].loc[23,'Low']=98.1
    ref=confirmed_support_stop(r['df'],current_price=100,entry_reference=100,atr=1)
    assert ref['stop'] > 96.5
    r.update(stop_used=ref['stop'],stop_source=ref['source'],atr=1)
    assert technical_crv_eligibility(build_technical_crv_package(r))['entry']


@pytest.mark.parametrize('source', ['2R-Fallback', 'Analystenziel', 'TP2', 'synthetisches Ziel'])
def test_nontechnical_target_never_qualifies(source):
    r=result();r.pop('technical_target_1');r.update(tp2=110,tp2_source=source)
    pkg=build_technical_crv_package(r)
    assert not technical_crv_eligibility(pkg)['now']
    assert decide(pkg)['state']=='buy'
    assert not decide(pkg)['crv_evaluated']
    assert 'technisches Ziel fehlt' in decide(pkg)['release_check']


def test_missing_target_is_separate_from_low_confirmed_crv():
    r=result();r.pop('technical_target_1')
    ds=decide(build_technical_crv_package(r))
    assert 'technisches Ziel fehlt' in ds['release_check']
    r['technical_target_1']=103
    ds=decide(build_technical_crv_package(r))
    assert 'Enger Zielraum' in ds['crv_context']
    assert not any('CRV' in x for x in ds['release_missing'])
    assert 'nicht bewertbar' not in ds['release_check']


def test_no_implicit_eligibility_from_numbers_at_decision_boundary():
    ds=build_decision_state_v3021az(valid_trade_setup=True,signal_packages_complete=True,
        timing_score=90,conf_score=90,trigger_status='Aktiv',crv_now=4,crv_entry=4)
    assert ds['state']=='buy'
    assert not ds['crv_evaluated']


@pytest.mark.parametrize('extra',[{'entry_hard_gate':True},{'invalidated':True}])
def test_structural_crv_cannot_override_hard_gates(extra):
    assert decide(build_technical_crv_package(result()), **extra)['state']=='blocked'


@pytest.mark.parametrize('trigger,icon', [('Aktiv','🟢'), ('Beobachten','🟡')])
def test_actual_monitor_row_keeps_proxy_as_scenario(monkeypatch, radar_rr, setup_alerts, trigger, icon):
    from datetime import datetime
    from modules import live_monitor as lm
    monkeypatch.setattr(lm, '_v210_alert_price', lambda r: r.get('price'), raising=False)
    monkeypatch.setattr(lm, '_v210_alert_num', lambda v, default=None: float(v) if v is not None else default, raising=False)
    monkeypatch.setattr(lm, 'build_setup_alerts_v210', setup_alerts, raising=False)
    monkeypatch.setattr(lm, 'get_current_berlin_time', lambda: datetime(2026,10,11,20), raising=False)
    monkeypatch.setattr(lm, '_v214_monitor_final_release_check', lambda *a: (False, []), raising=False)
    monkeypatch.setattr(lm, 'shorten_text', lambda v, n: str(v)[:n], raising=False)
    monkeypatch.setattr(lm, '_v232_is_current_baseline_source', lambda v: False, raising=False)
    r=result();r.update(ticker='META',valid_trade_setup=True,trigger_status=trigger,ma20=99,
        signal_package_complete=True,
        timing_action_confidence_pkg={'score':85},trigger_confluence_pkg={'score':71})
    r['stop_source']='Praxis-Mindestabstand'
    rr=radar_rr(r)
    d=dict(rr,bucket='Nahe am Trigger',grade='B',score=80,crv_target_source=rr['target_source'],
           crv_target_value=rr['tp1'],crv_stop_quality=rr['stop_quality'],crv_entry_plan_quality=rr['entry_plan_quality'],
           crv_stop_diagnostics=rr['stop_diagnostics'])
    row=lm._v212_monitor_status_from_decision(r,d)
    assert row['Ampel']==icon
    assert 'verwendet wird ein Ersatzstop' in row['Stop-Diagnose']
    assert 'CRV nicht bewertet' in row['Freigabe-Check']
    assert '(Szenario)' in row['CRV jetzt']
    assert 'Verwendeter Stop nicht als Strukturstop bestätigt' in row['Freigabe-Check']
    assert 'unter 1,50' not in row['Freigabe-Check']


@pytest.mark.parametrize('scenario', [0.1, 0.93, 2.5, 20, None])
@pytest.mark.parametrize('trigger,state', [('Beobachten', 'plan'), ('Aktiv', 'buy')])
def test_unassessed_scenario_neither_blocks_nor_confirms_crv(scenario, trigger, state):
    ds=build_decision_state_v3021az(valid_trade_setup=True,signal_packages_complete=True,
        timing_score=85,conf_score=71,trigger_status=trigger,crv_now=scenario,crv_entry=scenario,
        crv_now_eligible=False,crv_entry_eligible=False,crv_evidence_reason='Strukturstop fehlt')
    assert ds['state']==state
    assert ds['crv_now'] is None and ds['crv_entry'] is None
    assert not ds['crv_now_ok'] and not ds['crv_entry_ok']
    assert not ds['crv_evaluated']
    assert not any('CRV' in reason for reason in ds['release_missing'])
    assert 'CRV nicht bewertet' in ds['release_check']


@pytest.mark.parametrize('extra,state', [
    ({'timing_score':59}, 'observe'), ({'conf_score':59}, 'observe'),
    ({'timing_score':69}, 'plan'), ({'conf_score':64}, 'plan'),
    ({'valid_trade_setup':False}, 'observe'), ({'signal_packages_complete':False}, 'observe'),
    ({'trigger_status':'inaktiv'}, 'plan'), ({'entry_distance_pct':5.1}, 'wait_entry'),
    ({'ma20_stretch_pct':12.1}, 'wait_entry'), ({'entry_hard_gate':True}, 'blocked'),
    ({'invalidated':True}, 'blocked'),
])
def test_neutral_crv_does_not_override_other_requirements(extra, state):
    args=dict(valid_trade_setup=True,signal_packages_complete=True,timing_score=85,conf_score=71,
              trigger_status='Aktiv',crv_now=5,crv_entry=5,crv_now_eligible=False,crv_entry_eligible=False)
    args.update(extra)
    assert build_decision_state_v3021az(**args)['state']==state


def test_assessed_now_low_crv_cannot_borrow_unassessed_entry_scenario():
    ds=build_decision_state_v3021az(valid_trade_setup=True,signal_packages_complete=True,
        timing_score=85,conf_score=71,trigger_status='Aktiv',crv_now=0.9,crv_entry=10,
        crv_now_eligible=True,crv_entry_eligible=False)
    assert ds['state']=='buy' and ds['crv_evaluated']
    assert 'Enger Zielraum' in ds['crv_context']
    assert not any('CRV' in x for x in ds['release_missing'])


def test_assessed_entry_only_does_not_override_technical_location():
    ds=build_decision_state_v3021az(valid_trade_setup=True,signal_packages_complete=True,
        timing_score=85,conf_score=71,trigger_status='Aktiv',crv_now=5,crv_entry=2,
        crv_now_eligible=False,crv_entry_eligible=True)
    assert ds['state']=='buy' and ds['buy_now']
    assert ds['crv_assessment'].startswith('Jetzt offen')


def test_assessed_but_missing_numeric_value_is_disclosed_without_block():
    ds=build_decision_state_v3021az(valid_trade_setup=True,signal_packages_complete=True,
        timing_score=85,conf_score=71,trigger_status='Aktiv',crv_now=None,
        crv_now_eligible=True,crv_entry_eligible=False)
    assert ds['state']=='buy' and ds['crv_evaluated']
