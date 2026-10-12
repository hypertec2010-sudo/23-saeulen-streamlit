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
    assert ds['state'] == 'observe'
    assert 'Strukturstop fehlt' in ds['release_check']
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
    assert decide(pkg)['state']=='observe'
    assert 'technisches Ziel fehlt' in decide(pkg)['release_check']


def test_missing_target_is_separate_from_low_confirmed_crv():
    r=result();r.pop('technical_target_1')
    ds=decide(build_technical_crv_package(r))
    assert 'technisches Ziel fehlt' in ds['release_check']
    r['technical_target_1']=103
    ds=decide(build_technical_crv_package(r))
    assert 'unter 1,50' in ds['release_check']
    assert 'nicht bewertbar' not in ds['release_check']


def test_no_implicit_eligibility_from_numbers_at_decision_boundary():
    ds=build_decision_state_v3021az(valid_trade_setup=True,signal_packages_complete=True,
        timing_score=90,conf_score=90,trigger_status='Aktiv',crv_now=4,crv_entry=4)
    assert ds['state']=='observe'


@pytest.mark.parametrize('extra',[{'entry_hard_gate':True},{'invalidated':True}])
def test_structural_crv_cannot_override_hard_gates(extra):
    assert decide(build_technical_crv_package(result()), **extra)['state']=='blocked'


def test_actual_monitor_row_keeps_proxy_as_scenario(monkeypatch, radar_rr, setup_alerts):
    from datetime import datetime
    from modules import live_monitor as lm
    monkeypatch.setattr(lm, '_v210_alert_price', lambda r: r.get('price'), raising=False)
    monkeypatch.setattr(lm, '_v210_alert_num', lambda v, default=None: float(v) if v is not None else default, raising=False)
    monkeypatch.setattr(lm, 'build_setup_alerts_v210', setup_alerts, raising=False)
    monkeypatch.setattr(lm, 'get_current_berlin_time', lambda: datetime(2026,10,11,20), raising=False)
    monkeypatch.setattr(lm, '_v214_monitor_final_release_check', lambda *a: (False, []), raising=False)
    monkeypatch.setattr(lm, 'shorten_text', lambda v, n: str(v)[:n], raising=False)
    monkeypatch.setattr(lm, '_v232_is_current_baseline_source', lambda v: False, raising=False)
    r=result();r.update(ticker='META',valid_trade_setup=True,trigger_status='Aktiv',ma20=99,
        signal_package_complete=True,
        timing_action_confidence_pkg={'score':85},trigger_confluence_pkg={'score':71})
    r['stop_source']='Praxis-Mindestabstand'
    rr=radar_rr(r)
    d=dict(rr,bucket='Nahe am Trigger',grade='B',score=80,crv_target_source=rr['target_source'],
           crv_target_value=rr['tp1'],crv_stop_quality=rr['stop_quality'],crv_entry_plan_quality=rr['entry_plan_quality'])
    row=lm._v212_monitor_status_from_decision(r,d)
    assert row['Ampel']=='⚪'
    assert '(Szenario)' in row['CRV jetzt']
    assert 'Strukturstop fehlt' in row['Freigabe-Check']
    assert 'unter 1,50' not in row['Freigabe-Check']
