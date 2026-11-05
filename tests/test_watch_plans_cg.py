"""Pattern-bound targets, the supplied TXN geometry, and legacy isolation."""
from copy import deepcopy

import pandas as pd
import pytest

from modules import watch_setups as s
from modules import watch_quality as q
from modules import trade_watch as w
from modules.watch_plans import bind_plan,candidate_rank,PLAN_POLICY
from test_watch_setups import range_result,higher_result,retest_result,qualifying_result,NOW
from test_trade_watch import idea,breakout,NOW as ENGINE_NOW


def highs(peak=105.):
    f=pd.DataFrame({'High':[98.]*15},index=pd.bdate_range(end='2026-10-27',periods=15))
    f.iloc[6,0]=peak
    return f


def test_supplied_txn_levels_have_already_exceeded_their_pullback_reference():
    support=294.70001220703125;depth=5.408242315644137
    previous_high=support/(1-depth/100)
    confirmation=317.20001220703125
    entry=318.3179;stop=291.9052;cap=321.5011
    candidate=dict(kind='higher_low_pullback',label=s.KINDS['higher_low_pullback'],phase='ready',
        entry=entry,invalidation=stop,max_entry=cap,reason_code='ready',reason='Structure ready',evidence={})
    out=bind_plan(candidate,highs(),objective=previous_high,source='Vorhoch',basis_date='2026-10-30',observed_high=confirmation)
    assert previous_high==pytest.approx(311.54935633019795)
    assert out['phase']=='unsuitable' and out['reason_code']=='target_not_above_entry'
    assert out['target']<entry and not q.assess(out)['allowed']
    assert (out['entry'],out['invalidation'],out['max_entry'])==(entry,stop,cap)
    assert candidate['phase']=='ready'


def test_actual_pullback_detector_rejects_a_bounce_past_the_previous_high():
    r=higher_result();df=r['df']
    df.loc[df.index[122],'High']=109.
    a=s.pullback_candidate(df,3.,102.5,higher=True)
    assert a['phase']=='unsuitable' and a['reason_code']=='target_not_above_entry'
    assert a['evidence']['pre_pullback_high']==107.5
    assert a['target']==107.5


@pytest.mark.parametrize('factory',[range_result,qualifying_result,retest_result])
def test_general_analysis_target_never_controls_pattern_target(factory):
    r=factory();a=s.detect(r,now=NOW)
    r['technical_target_1']=a['entry']+.01
    b=s.detect(r,now=NOW)
    r['technical_target_1']=9999.
    c=s.detect(r,now=NOW)
    assert a==b==c
    assert a['plan_policy']==PLAN_POLICY and a['target_source'] and a['target_basis_date']


def test_range_projection_does_not_manufacture_one_point_five_r():
    p=s.detect(range_result(),now=NOW)
    e=p['evidence']
    assert p['target']==pytest.approx(e['resistance']+(e['resistance']-e['support']))
    assert q.assess(p)['rr_cap']<1.5 and not q.admission(p)[0]
    assert p['phase']=='ready' and 'Tradeplan ungeeignet' in p['phase_label']


def test_range_uses_a_real_prior_overhead_resistance_independent_of_general_target():
    r=range_result();df=r['df']
    df.loc[df.index[-40],'High']=110.
    r['technical_target_1']=100.2
    p=s.detect(r,now=NOW)
    assert p['kind']=='range_breakout' and p['target']==110. and q.admission(p)[0]
    assert 'Pivot-Widerstand' in p['target_source']
    assert p['evidence']['range_projection']==102.5


def test_range_never_skips_a_nearer_confirmed_level_to_gain_rr():
    r=range_result();df=r['df']
    df.loc[df.index[-40],'High']=110.
    df.loc[df.index[-30],'High']=101.
    p=s.detect(r,now=NOW)
    assert p['target']==101. and not q.admission(p)[0]


def test_qualifying_pullback_keeps_its_prior_high_stop_and_unrounded_rr():
    p=s.detect(qualifying_result(),now=NOW)
    assert p['kind']=='higher_low_pullback' and p['target']==107.5
    assert q.admission(p)[0] and q.assess(p)['rr_cap']>1.5
    assert p['phase_label']=='Übernahmebereit · Bestätigung abwarten'
    assert p['invalidation']==pytest.approx(99.897)


def test_nearby_confirmed_resistance_caps_target_and_is_not_skipped():
    candidate=dict(phase='ready',entry=100.,max_entry=101.,invalidation=95.)
    a=bind_plan(candidate,highs(105.),objective=110.,source='Vorhoch',basis_date='2026-10-20')
    assert a['target']==105. and a['target_reference']==110.
    assert not q.assess(a)['allowed'] and a['target_obstacle']==105.
    b=bind_plan(candidate,highs(105.),objective=110.,source='Vorhoch',basis_date='2026-10-20',observed_high=106.)
    assert b['phase']=='missed'


def test_invalidation_has_priority_over_spent_target():
    candidate=dict(phase='invalid',entry=100.,max_entry=101.,invalidation=95.,reason_code='broken',reason='Invalid')
    a=bind_plan(candidate,highs(),objective=99.,source='Vorhoch',basis_date='2026-10-20',observed_high=110.)
    assert a['phase']=='invalid' and a['reason_code']=='broken'


def test_candidate_selection_prefers_a_qualifying_ready_plan_over_first_bad_plan(monkeypatch):
    def range_variant(data,n,atr,price):
        return dict(kind='range_breakout',label=s.KINDS['range_breakout'],phase='ready',reason_code='ready',
            reason='Test',entry=100.,max_entry=101.,invalidation=95. if n==20 else 98.,
            target=102. if n==20 else 110.,window=n,plan_policy=PLAN_POLICY,evidence={},checks=[])
    monkeypatch.setattr(s,'range_candidate',range_variant)
    monkeypatch.setattr(s,'pullback_candidate',lambda *a,**kw:s.attempt('support_pullback','none','None'))
    monkeypatch.setattr(s,'retest_candidate',lambda *a,**kw:s.attempt('breakout_retest','none','None'))
    p=s.detect(range_result(),now=NOW)
    assert p['window']==40 and p['target']==110. and q.admission(p)[0]
    assert not p['attempts'][0]['plan_allowed'] and p['attempts'][1]['plan_allowed']


def test_old_rules_still_monitor_but_cannot_be_newly_armed():
    legacy=idea();pkg=legacy['source']['watch_setup'];pkg.pop('plan_policy')
    legacy['source']['watch_setup']=s.sealed(pkg)
    old=deepcopy(legacy)
    assert w.active(legacy) and w.evaluate(legacy,breakout(),now=ENGINE_NOW)['state']=='triggered'
    assert legacy==old and not q.admission(legacy)[0]
    with pytest.raises(ValueError,match='Alter Beleg'):
        w.make_idea(ticker=legacy['ticker'],entry=legacy['entry'],invalidation=legacy['invalidation'],
            max_entry=legacy['max_entry'],target=legacy['target'],mode=legacy['mode'],expires_at=legacy['expires_at'],
            source=legacy['source'],now=legacy['created_at'])


@pytest.mark.parametrize('factory,expected,has_form',[(range_result,'Tradeplan ungeeignet',False),
    (qualifying_result,'Übernahmebereit',True)])
def test_real_form_distinguishes_structure_and_trade_plan(factory,expected,has_form,monkeypatch):
    from streamlit.testing.v1 import AppTest
    real=w.utc;monkeypatch.setattr(w,'utc',lambda value=None:real(value if value is not None else NOW))
    p=s.detect(factory(),now=NOW)
    script=f'''
import streamlit as st
from modules.trade_watch_ui import capture_form
capture_form(st,{{'ticker':'HALO','watch_setup':{p!r}}})
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and bool(app.checkbox) is has_form
    assert any(expected in x.value for x in app.markdown)


def test_frozen_pattern_objective_does_not_chase_later_confirmation_bars():
    r=qualifying_result();a=s.detect(r,now=NOW)
    r['df'].loc[r['df'].index[-1],'High']=104.
    b=s.detect(r,now=NOW)
    assert b['target']==a['target'] and b['entry']==a['entry']
    assert b['invalidation']==a['invalidation']
