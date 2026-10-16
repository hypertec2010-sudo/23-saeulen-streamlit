import ast
from pathlib import Path
import pandas as pd
import pytest
from modules.stop_selection import select_productive_stop
from modules.technical_crv import build_technical_crv_package, technical_crv_eligibility
from modules.action_risk_context import build_action_risk_context
from modules.live_monitor import build_decision_state_v3021az
from modules.chart_overlays import build_trade_setup_overlay_v193
from test_crv_consistency import radar_rr

ROOT=Path(__file__).resolve().parents[1]


def unh_history():
    # Reconstructed support/ATR fixture, not an original UNH market snapshot.
    data=[410.]*80;data[8]=395.37;data[23]=395.6
    return pd.DataFrame({'Low':data})


def choose(**overrides):
    args=dict(price=409.65,entry_zone='407.60 - 411.70 USD',atr=11.96,
              setup_type='Breakout',fallback_stop=390.60,fallback_source='ATR (1,6x)')
    args.update(overrides)
    return select_productive_stop(unh_history(),**args)


def test_unh_selects_technical_stop_and_preserves_previous_atr_as_diagnostic():
    p=choose()
    assert p['stop']==392.38 and p['selected']=='confirmed_structure'
    assert p['previous_stop']==390.6 and p['previous_source']=='ATR (1,6x)'
    assert 'Swing-Support' in p['source']


@pytest.mark.parametrize('overrides',[
    {'position_mode':True},{'setup_type':'Kein sauberes Setup'},
    {'setup_type':'unbekannt'},{'entry_zone':'-'},{'price':float('nan')},
    {'entry_zone':'390 - 411.70'},{'entry_zone':(392.38,411.70)}])
def test_no_eligible_whole_zone_or_position_preserves_previous_stop(overrides):
    p=choose(**overrides)
    assert p['selected']=='existing' and p['stop']==390.6 and p['source']=='ATR (1,6x)'
    assert p['reason']


def test_no_structure_keeps_existing_stop_without_borrowing_confirmation():
    p=select_productive_stop(pd.DataFrame({'Low':[410.]*80}),price=409.65,
        entry_zone='407.60 - 411.70',atr=11.96,setup_type='Breakout',
        fallback_stop=390.6,fallback_source='ATR (1,6x)')
    assert p['stop']==390.6 and p['selected']=='existing'


def test_confirmed_stop_does_not_get_widened_to_practice_floor():
    data=[100.]*40;data[8]=98;data[23]=98.1
    p=select_productive_stop(pd.DataFrame({'Low':data}),price=100,
        entry_zone='99 - 101',atr=1,setup_type='Pullback an MA20',
        fallback_stop=96.5,fallback_source='Praxis-Mindestabstand')
    assert p['selected']=='confirmed_structure' and p['stop']>96.5


def test_cent_rounding_cannot_select_zero_stop():
    data=[.01]*40;data[8]=.005;data[23]=.00501
    p=select_productive_stop(pd.DataFrame({'Low':data}),price=.006,
        entry_zone=(.0055,.006),atr=.0001,setup_type='Rebound',
        fallback_stop=.004,fallback_source='ATR')
    assert p['selected']=='existing' and p['stop']==.004


def test_cent_rounding_cannot_fake_a_confirmed_used_stop():
    data=[.012]*40;data[8]=.01012;data[23]=.01013
    p=select_productive_stop(pd.DataFrame({'Low':data}),price=.012,
        entry_zone=(.0115,.012),atr=.0001,setup_type='Rebound',
        fallback_stop=.009,fallback_source='ATR')
    assert p['selected']=='existing' and p['stop']==.009
    assert 'Toleranz' in p['reason']


def engine_risk_block(path):
    tree=ast.parse((ROOT/path).read_text())
    # Run actual statements from each engine rather than a test recreation.
    block=next(n.body for n in ast.walk(tree) if isinstance(n,ast.If)
               and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_stop_selection_v3021bo'
                   for t in x.targets) for x in n.body))
    start=next(i for i,n in enumerate(block) if isinstance(n,ast.ImportFrom) and n.module=='modules.stop_selection')
    end=next(i for i,n in enumerate(block) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='tp1_source' for t in n.targets))
    assert start<end
    sizing=[n for n in block if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id in {'risk_eur','pos_size'} for t in n.targets)]
    assert len(sizing)==2
    return compile(ast.Module(body=block[start:end+1]+sizing,type_ignores=[]),'actual_engine_risk','exec')


@pytest.mark.parametrize('engine',['analysis_core.py','modules/legacy_analysis_core.py'])
@pytest.mark.parametrize('target,state',[(430,'observe'),(450,'buy')])
def test_both_actual_engines_size_chart_radar_and_permission_share_chosen_stop(engine,target,state,radar_rr):
    scope=dict(df=unh_history(),price=409.65,suggested_entry_zone='407.60 - 411.70 USD',
               atr=11.96,setup_type='Breakout',stop_used=390.6,stop_source='ATR (1,6x)',
               buy_in_override=0,depot=20000,risk_pct=1)
    exec(engine_risk_block(engine),scope)
    assert scope['stop_used']==392.38 and scope['risk_per_share']==pytest.approx(17.27)
    assert scope['tp1']==426.92 and scope['pos_size']==11
    r=dict(price=scope['price'],df=scope['df'],atr=scope['atr'],setup_type='Breakout',
           suggested_entry_zone=scope['suggested_entry_zone'],stop_used=scope['stop_used'],
           stop_source=scope['stop_source'],stop_selection=scope['_stop_selection_v3021bo'],
           technical_target_1=target,tp1=scope['tp1'])
    pkg=build_technical_crv_package(r);r.update(pkg)
    rr=radar_rr(r);overlay=build_trade_setup_overlay_v193(r)
    assert pkg['technical_crv_stop_now']==overlay['stop']==392.38
    assert rr['crv_now_eligible'] and technical_crv_eligibility(pkg)['now']
    assert overlay['crv_basis'][0]['stop']==392.38 and overlay['crv_basis'][0]['operative']
    assert rr['crv']==pytest.approx((target-409.65)/17.27)
    args=dict(valid_trade_setup=True,signal_packages_complete=True,timing_score=85,
              conf_score=75,trigger_status='Aktiv',crv_now=rr['crv'],crv_entry=rr['crv_entry'],
              crv_now_eligible=rr['crv_now_eligible'],crv_entry_eligible=rr['crv_entry_eligible'])
    assert build_decision_state_v3021az(**args)['state']==state
    assert build_decision_state_v3021az(**args,invalidated=True)['state']=='blocked'
    assert build_decision_state_v3021az(**args,entry_hard_gate=True)['state']=='blocked'
    risk=build_action_risk_context(r,entry_zone=r['suggested_entry_zone'],ccy='USD')
    assert 'Stop bei 392.38 USD' in risk['text'] and 'Keine bestätigte' not in risk['text']


@pytest.mark.parametrize('engine',['analysis_core.py','modules/legacy_analysis_core.py'])
def test_actual_engine_position_mode_preserves_existing_stop_and_size(engine):
    scope=dict(df=unh_history(),price=409.65,suggested_entry_zone='407.60 - 411.70 USD',
               atr=11.96,setup_type='Breakout',stop_used=390.6,stop_source='ATR (1,6x)',
               buy_in_override=400,depot=20000,risk_pct=1)
    exec(engine_risk_block(engine),scope)
    assert scope['stop_used']==390.6 and scope['pos_size']==10


def test_existing_reference_without_selected_stop_is_no_longer_called_missing():
    p=choose(position_mode=True)
    r=dict(price=409.65,df=unh_history(),atr=11.96,stop_used=p['stop'],stop_source=p['source'])
    r.update(build_technical_crv_package(r))
    text=build_action_risk_context(r,entry_zone='407.60 - 411.70')['text']
    assert 'Bestätigte Support-Referenz vorhanden' in text and 'Keine bestätigte' not in text
