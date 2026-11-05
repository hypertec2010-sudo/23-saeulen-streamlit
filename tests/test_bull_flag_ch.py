from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from modules import watch_setups as s
from modules import watch_quality as q
from modules import trade_watch as w
from modules.watch_bull_flag import bull_flag_candidate,entry_volume_check,ENTRY_VOLUME_RULE
from test_watch_setups import range_result,NOW
from test_trade_watch import idea,breakout,NOW as ENGINE_NOW

ARM='2026-10-28T15:30:00Z'
END='2026-10-28T15:35:00Z'
CHECK='2026-10-28T15:35:11Z'


def flag_result():
    r=range_result();f=r['df'];n=len(f)
    close=np.linspace(70,100,n)
    f.loc[:,['Open','High','Low','Close']]=np.column_stack([close-.2,close+1,close-1,close])
    f.loc[:, 'Volume']=1_000_000.
    f.loc[f.index[94:114],['Open','High','Low','Close']]=[100.,101.,99.,100.]
    for j,c in enumerate([104.,108.,112.,116.,120.],114):
        f.loc[f.index[j],['Open','High','Low','Close','Volume']]=[c-3,c+1,c-3.5,c,2_000_000.]
    flags=[[119.,120.,118.,119.],[118.8,119.2,117.8,118.5],[118.4,119.,117.6,118.3],
           [118.2,119.,117.8,118.4],[118.3,119.,117.9,118.4]]
    f.loc[f.index[119:124],['Open','High','Low','Close']]=flags
    f.loc[f.index[119:124],'Volume']=800_000.
    f.loc[f.index[124],['Open','High','Low','Close','Volume']]=[118.4,119.,118.,118.6,800_000.]
    r['price']=118.6;r['technical_target_1']=119.
    return r


def flag_idea():
    p=s.detect(flag_result(),now=ARM)
    assert p['kind']=='bull_flag' and q.admission(p)[0],p
    return w.make_idea(ticker='HALO',entry=p['entry'],invalidation=p['invalidation'],max_entry=p['max_entry'],
        target=p['target'],mode=p['mode'],expires_at=p['valid_until'],source={'watch_setup':p},now=ARM)


def volume_minutes(i,ratio=1.3):
    idx=pd.date_range('2026-10-28T13:30Z',periods=126,freq='min')
    close=np.where(idx<pd.Timestamp(ARM),i['entry']-.5,i['entry']+.1)
    vol=np.full(len(idx),100.)
    vol[(idx>=pd.Timestamp(ARM))&(idx<pd.Timestamp(END))]=ratio*100.
    return pd.DataFrame(dict(Open=close,High=close+.2,Low=close-.2,Close=close,Volume=vol),index=idx)


def test_flag_has_own_impulse_target_and_pattern_stop_without_mutating_source():
    r=flag_result();before=r['df'].copy(deep=True)
    p=s.detect(r,now=NOW)
    assert p['kind']=='bull_flag' and p['phase']=='ready' and q.admission(p)[0]
    assert p['target']==142. and p['target_source'].startswith('Einfache Flaggenmast')
    assert p['evidence']['pole_height']==21. and p['evidence']['flag_days']==5
    assert p['invalidation']<117.6 and p['entry']>121. and p['max_entry']<p['target']
    assert p['entry_volume_rule']==ENTRY_VOLUME_RULE and p['stop_quality']=='flag_boundary'
    assert 'Volume' in p['chart'][0]
    r['technical_target_1']=9999.
    assert s.detect(r,now=NOW)==p
    pd.testing.assert_frame_equal(r['df'],before)


@pytest.mark.parametrize('problem',['no_pole','pole_volume','flag_volume','flag_width','too_deep'])
def test_weak_or_non_contracting_structures_do_not_become_ready_flags(problem):
    r=flag_result();f=r['df']
    if problem=='no_pole':
        f.loc[f.index[114:119],['Open','High','Low','Close']]=[118.,120.,117.,119.]
    if problem=='pole_volume':f.loc[f.index[114:119],'Volume']=1_000_000.
    if problem=='flag_volume':f.loc[f.index[119:124],'Volume']=2_000_000.
    if problem=='flag_width':f.loc[f.index[119:124],'Low']=110.
    if problem=='too_deep':f.loc[f.index[121],'Low']=100.
    a=bull_flag_candidate(f,3.,118.6)
    assert a['phase']!='ready'


def test_latest_daily_bar_cannot_move_flag_levels_and_crossing_is_not_fresh_entry():
    r=flag_result();p=s.detect(r,now=NOW)
    r['df'].loc[r['df'].index[-1],['Open','High','Low','Close']]=[121.5,122.,121.,121.7]
    r['price']=121.7
    b=s.detect(r,now=NOW)
    assert b['phase']=='triggered' and b['entry']==p['entry'] and b['target']==p['target']
    assert b['invalidation']==p['invalidation'] and not b['eligible']


def test_real_overhead_resistance_caps_flag_projection_and_blocks_poor_rr():
    r=flag_result();r['df'].loc[r['df'].index[-40],'High']=123.
    p=s.detect(r,now=NOW)
    assert p['kind']=='bull_flag' and p['target']==123. and not q.admission(p)[0]
    assert p['target_reference']==max(p['evidence']['flag_high'],p['evidence']['pole_high'])+p['evidence']['pole_height']


@pytest.mark.parametrize('ratio,confirmed',[(1.3,True),(1.29999,False),(2.,True)])
def test_full_closed_price_group_needs_its_own_volume_confirmation(ratio,confirmed):
    i=flag_idea();f=volume_minutes(i,ratio)
    original=deepcopy(i)
    out=w.evaluate(i,f,now=CHECK)
    assert (out['state']=='triggered') is confirmed
    assert out['last_entry_volume']['confirmed'] is confirmed and i==original
    if not confirmed:
        assert out['state']=='volume_wait' and 'Volumenbestätigung fehlt' in out['health']
        again=w.evaluate(out,f,now=CHECK)
        assert again['state']=='volume_wait' and again['events']==out['events']
    else:
        assert sum(e['kind']=='trigger' for e in out['events'])==1
        assert w.evaluate(out,f,now=CHECK)['events']==out['events']


@pytest.mark.parametrize('problem',['missing','nan','zero','short','current_in_baseline'])
def test_missing_or_unusable_volume_does_not_create_a_trigger(problem):
    i=flag_idea();f=volume_minutes(i,1.)
    if problem=='missing':f=f.drop(columns='Volume')
    if problem=='nan':f.loc[pd.Timestamp(ARM),'Volume']=float('nan')
    if problem=='zero':f.loc[pd.Timestamp(ARM),'Volume']=0.
    if problem=='short':f=f.loc[f.index>=pd.Timestamp('2026-10-28T14:00Z')]
    if problem=='current_in_baseline':
        # 20 prior blocks average 1000, the current block 500: future data cannot lower it.
        f.loc[f.index<pd.Timestamp(ARM),'Volume']=200.
        f.loc[f.index>=pd.Timestamp(END),'Volume']=1.
    out=w.evaluate(i,f,now=CHECK)
    assert out['state']!='triggered' and not out['last_entry_volume']['confirmed']


def test_volume_baseline_uses_pre_arm_data_but_never_future_signal_groups():
    i=flag_idea();f=volume_minutes(i,1.3)
    f.loc[f.index>=pd.Timestamp(END),'Volume']=999999.
    ok,detail=entry_volume_check(f,pd.Timestamp(END),ENTRY_VOLUME_RULE)
    assert ok and '1.30' in detail


def test_volume_failure_does_not_override_invalidation_or_closed_bar_buffer():
    i=flag_idea();f=volume_minutes(i,1.3)
    assert w.evaluate(i,f,now='2026-10-28T15:35:09Z')['state']!='triggered'
    f.loc[pd.Timestamp(ARM),'Low']=i['invalidation']-.1
    f=f.drop(columns='Volume')
    out=w.evaluate(i,f,now=CHECK)
    assert out['state']=='invalidated' and not out['triggered_at']


def test_old_patterns_still_trigger_without_intraday_volume_rule():
    i=idea();out=w.evaluate(i,breakout(),now=ENGINE_NOW)
    assert out['state']=='triggered' and 'last_entry_volume' not in out


def test_incomplete_flag_volume_proof_cannot_be_armed():
    p=s.detect(flag_result(),now=NOW)
    p.pop('entry_volume_rule');p=s.sealed(p)
    assert not s.validate(p,now=NOW)[0]


def test_latest_daily_low_break_invalidates_the_frozen_flag_even_after_recovery():
    r=flag_result();r['df'].loc[r['df'].index[-1],'Low']=115.
    a=bull_flag_candidate(r['df'],3.,118.6)
    assert a['phase']=='invalid' and a['reason_code']=='flag_broken'


def test_range_fallback_explains_model_constraint_not_just_market_crv():
    p=s.detect(range_result(),now=NOW);report=q.assess(p)
    assert not report['allowed'] and 'Modellbedingte Sperre' in report['reason']
    assert p['model_rr_ceiling']==1. and q.MIN_ENTRY_RR==1.5
