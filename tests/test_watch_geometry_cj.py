"""Separate impossible initial reward geometry from an actually spent target."""
from copy import deepcopy

import pandas as pd
import pytest

from modules import watch_setups as s, watch_quality as q
from modules.watch_plans import bind_plan, candidate_rank
from test_watch_plans_cg import highs
from test_watch_setups import higher_result, NOW


def ma_plan():
    return dict(kind='higher_low_pullback',label=s.KINDS['higher_low_pullback'],
        phase='ready',entry=662.6072,max_entry=668.9429,invalidation=601.37,
        reason_code='ready',reason='Test',evidence={})


def test_supplied_ma_levels_are_unsuitable_not_a_missed_entry():
    original=ma_plan();before=deepcopy(original)
    p=bind_plan(original,highs(),objective=633.29,source='Vorhoch',
                basis_date='2026-10-22',observed_high=670.)
    assert p['phase']=='unsuitable' and p['reason_code']=='target_not_above_entry'
    assert p['target']==633.29 and original==before
    report=q.assess(p)
    assert not report['allowed'] and round(report['rr_entry'],2)==-.48
    assert round(report['rr_cap'],2)==-.53


@pytest.mark.parametrize('target,code',[(662.6072,'target_not_above_entry'),
    (665.,'target_inside_entry_band'),(668.9429,'target_inside_entry_band')])
def test_entry_band_boundaries_cannot_be_called_spent(target,code):
    p=bind_plan(ma_plan(),highs(),objective=target,source='Test',basis_date='2026-10-22',observed_high=700.)
    assert p['phase']=='unsuitable' and p['reason_code']==code
    assert not q.assess(p)['allowed']


def test_valid_target_can_still_be_consumed():
    p=bind_plan(ma_plan(),highs(),objective=800.,source='Test',basis_date='2026-10-22',observed_high=800.)
    assert p['phase']=='missed' and p['reason_code']=='pattern_target_consumed'


def test_real_invalidation_remains_primary():
    c=ma_plan();c.update(phase='invalid',reason_code='broken')
    p=bind_plan(c,highs(),objective=633.29,source='Test',basis_date='2026-10-22',observed_high=700.)
    assert p['phase']=='invalid' and p['reason_code']=='broken'


def test_live_quote_cannot_rename_impossible_pullback_as_missed():
    r=higher_result();f=r['df'];f.loc[f.index[122],'High']=109.
    r['price']=111.
    p=s.detect(r,now=NOW)
    matches=[a for a in p['attempts'] if a['kind']=='higher_low_pullback']
    assert matches and matches[0]['phase']=='unsuitable'
    assert matches[0]['reason_code']=='target_not_above_entry'
    stocks,_,patterns=s.diagnostic_tables(pd.DataFrame([{'Ticker':'HALO','__watch_setup':p}]),now=NOW)
    row=patterns.loc[patterns.Muster==s.KINDS['higher_low_pullback']].iloc[0]
    assert row.Phase=='Tradeplan ungeeignet' and row.Code=='target_not_above_entry'


def test_independently_qualified_breakout_beats_unsuitable_pullback():
    bad=ma_plan();bad.update(phase='unsuitable',target=633.29)
    good=dict(kind='range_breakout',phase='ready',entry=100.,max_entry=101.,
              invalidation=98.,target=110.)
    assert max([bad,good],key=candidate_rank) is good
