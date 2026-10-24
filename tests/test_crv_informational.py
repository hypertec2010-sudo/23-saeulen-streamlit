"""CRV may change its own evidence/display, never technical readiness or rank."""
import ast
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from modules.live_monitor import build_decision_state_v3021az
from test_crv_consistency import radar_rr
from test_setup_alert_crv_evidence import setup_alerts
from test_crv_evidence import result

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('extra,expected',[
    ({},'buy'),({'trigger_status':'Beobachten'},'plan'),
    ({'conf_score':59.9},'observe'),({'timing_score':69.9},'plan'),
    ({'valid_trade_setup':False},'observe'),({'entry_distance_pct':5.1},'wait_entry'),
    ({'entry_hard_gate':True},'blocked'),({'invalidated':True},'blocked'),
])
def test_crv_evidence_changes_only_context_for_every_technical_state(extra,expected):
    args=dict(valid_trade_setup=True,signal_packages_complete=True,timing_score=85,
              conf_score=71,trigger_status='Aktiv',entry_distance_pct=0)
    args.update(extra)
    for n,e in [(0.04,0.1),(0.95,0.9),(1.49,3),(3,.2),(None,None),(float('nan'),2)]:
        for en,ee in [(True,True),(False,False),(True,False),(False,True)]:
            d=build_decision_state_v3021az(**args,crv_now=n,crv_entry=e,
                crv_now_eligible=en,crv_entry_eligible=ee)
            assert d['state']==expected
            assert not any('CRV' in x for x in d['release_missing'])
            assert 'CRV verändert weder' in d['crv_context']
            if not en:assert d['crv_now'] is None
            if not ee:assert d['crv_entry'] is None


@pytest.mark.parametrize('file',['analysis_core.py','modules/legacy_analysis_core.py'])
def test_actual_core_score_and_raw_trigger_do_not_depend_on_crv(file):
    tree=ast.parse((ROOT/file).read_text())
    assignments={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in {'tradeability_score','trading_case_score'} and isinstance(node.value,ast.Call):
                    # The first full calculation, not the final clamp/round.
                    if target.id not in assignments and not any(isinstance(x,ast.Name) and x.id==target.id for x in ast.walk(node.value)):
                        assignments[target.id]=node
    trigger=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and ast.unparse(n.test)=='not position_mode'
        and isinstance(n.body[0],ast.If) and ast.unparse(n.body[0].test).startswith('has_upcoming_earnings')
        and any(isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='trigger_status' for t in a.targets) for a in ast.walk(n)))
    outputs=[]
    for score in [15,95,float('nan')]:
        scope=dict(round=round,clamp=lambda x:max(0,min(100,x)),crv_score=score,
            stop_score=80,timing_trade_score=85,market_trade_score=85,entry_score=90,
            setup_adj=85,entry_location_score=90,position_mode=False,
            has_upcoming_earnings=False,days_earn=np.nan,pd=pd,
            market_info={'regime':'POSITIV'},valid_trade_setup=True,entry_quality='gut',
            setup_confidence=85,setup_type='Trendfolge',investment_case_score=85)
        exec(compile(ast.Module(body=[assignments['tradeability_score'],assignments['trading_case_score'],trigger],type_ignores=[]),file,'exec'),scope)
        outputs.append((scope['tradeability_score'],scope['trading_case_score'],scope['trigger_status']))
    assert outputs[0]==outputs[1]==outputs[2] and outputs[0][2]=='Aktiv'
    # Existing position management retains its historical weighting.
    scope.update(position_mode=True,crv_score=15)
    exec(compile(ast.Module(body=[assignments['tradeability_score']],type_ignores=[]),file,'exec'),scope)
    assert scope['tradeability_score']==round(15*.34+80*.18+85*.22+85*.10+90*.16)


def test_actual_radar_bucket_grade_score_and_rank_ignore_crv():
    tree=ast.parse((ROOT/'legacy_app.py').read_text())
    names={'build_professional_radar_decision_v18','_radar_v183_top_chance_rank'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={
        'radar_candidate_type':lambda r:'Leader','radar_risk_bucket':lambda r:'ruhig',
        'radar_setup_maturity':lambda r:'prüfbar','display_trigger_status_label':str,
        '_radar_v18_text':lambda r,k,d='':r.get(k,d),
        '_radar_safe_num':lambda v,d=50:float(v) if v is not None else d,
        '_radar_nested_num':lambda r,p,k,d:r.get(p,{}).get(k,d),
        '_radar_v18_clip':lambda v:max(0,min(100,float(v))),
        'radar_chart_impulse_pack':lambda r:{'score':85},
        '_radar_v18_entry_position':lambda r:('In Entry-Zone',90),
        '_radar_v183_style_fit':lambda *a:{'score':85},
        '_radar_v192_wave_impact':lambda r:{'score':80,'impact':0},
        '_radar_v200_mtf_impact':lambda r:{'score':80,'impact':0},
    }
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_radar','exec'),scope)
    r={k:85 for k in ['investment_case_score','trading_case_score','tradeability_score',
        'setup_confidence','leadership_score','trend_quality_score','base_quality_score',
        'volume_quality_score','catalyst_score','short_term_score','tb_score_100']}
    r.update(valid_trade_setup=True,trigger_status='Aktiv',market_info={'regime':'POSITIV'},
             accumulation_score=70,distribution_pressure_score=40)
    outputs=[]
    for crv,score in [(0.04,20),(0.95,40),(3,95),(None,45)]:
        rr={'crv':crv,'crv_entry':crv,'crv_now_eligible':True,'crv_entry_eligible':True,
            'rr_score':score,'entry_position':'In Entry-Zone','entry_score':90,'entry_distance_pct':0}
        scope['build_radar_entry_rr_package_v182']=lambda r:rr
        d=scope['build_professional_radar_decision_v18'](r)
        outputs.append(tuple(d[k] for k in ['score','grade','bucket','penalty','top_chance_rank']))
    assert all(o==outputs[0] for o in outputs)


def test_actual_live_row_retains_score_and_technical_state_when_target_changes(monkeypatch,radar_rr,setup_alerts):
    from modules import live_monitor as lm
    monkeypatch.setattr(lm,'_v210_alert_price',lambda r:r.get('price'),raising=False)
    monkeypatch.setattr(lm,'_v210_alert_num',lambda v,default=None:float(v) if v is not None else default,raising=False)
    monkeypatch.setattr(lm,'build_setup_alerts_v210',setup_alerts,raising=False)
    monkeypatch.setattr(lm,'get_current_berlin_time',lambda:datetime(2026,10,24,12),raising=False)
    monkeypatch.setattr(lm,'_v214_monitor_final_release_check',lambda *a:(True,[]),raising=False)
    monkeypatch.setattr(lm,'shorten_text',lambda v,n:str(v)[:n],raising=False)
    monkeypatch.setattr(lm,'_v232_is_current_baseline_source',lambda v:False,raising=False)
    outputs=[]
    for target in [100.2,103,110,130]:
        r=result();r.update(ticker='TEST',technical_target_1=target,valid_trade_setup=True,
            trigger_status='Aktiv',ma20=99,signal_package_complete=True,
            market_info={'regime':'POSITIV'},rs_vs_benchmark_63=10,atr_pct=3,
            charttechnik_setup_pkg={'score':70},
            timing_action_confidence_pkg={'score':85},trigger_confluence_pkg={'score':71})
        rr=radar_rr(r)
        d=dict(rr,bucket='Jetzt prüfbar',grade='B',score=80,crv_target_source=rr['target_source'],
           crv_target_value=rr['tp1'],crv_stop_quality=rr['stop_quality'],
           crv_entry_plan_quality=rr['entry_plan_quality'],crv_stop_diagnostics=rr['stop_diagnostics'])
        before=deepcopy(d)
        row=lm._v212_monitor_status_from_decision(r,d)
        outputs.append(tuple(row[k] for k in ['Ampel','Status','Priorität','Setup-Stärke','Handlungs-Score','Engine-Score','Guarded Engine-Score','Engine-Empfehlung']))
        assert 'CRV-Einordnung' in row and 'CRV verändert weder' in row['CRV-Hinweis']
        assert 'CRV' not in row['Score-Bremsen'] and 'CRV' not in row['Score-Treiber']
        assert d==before
    assert all(o==outputs[0] for o in outputs) and outputs[0][0]=='🟢'
