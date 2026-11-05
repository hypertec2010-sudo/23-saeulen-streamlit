"""Analysis vetoes must not replace the watcher's own technical checks."""
from copy import deepcopy

import numpy as np
import pytest

from modules import watch_setups as s, watch_quality as q
from test_watch_setups import range_result, qualifying_result, NOW
from test_bull_flag_ch import flag_result


@pytest.mark.parametrize('factory', [range_result, qualifying_result, flag_result])
def test_analysis_gate_does_not_change_any_chart_evidence_or_admission(factory):
    result = factory()
    before = result['df'].copy(deep=True)
    without_gate = s.detect(result, now=NOW, hard_gate=False)
    with_gate = s.detect(result, now=NOW, hard_gate=True)
    assert with_gate == without_gate
    assert with_gate['attempts'] and with_gate['reason_code'] != 'entry_gate'
    assert q.admission(with_gate) == q.admission(without_gate)
    assert result['df'].equals(before)


@pytest.mark.parametrize('condition,code', [
    ('earnings','earnings'), ('trend','trend_missing'),
    ('missing','daily_data_missing'), ('held','held_position'),
])
def test_own_protections_remain_when_general_gate_is_present(condition,code):
    r=qualifying_result()
    if condition=='earnings': r['days_earn']=2
    elif condition=='missing': r.pop('df')
    elif condition=='held': r['buy_in']=95.
    elif condition=='trend':
        f=r['df'];f['Close']=np.linspace(110,90,len(f))
        f['Open']=f.Close;f['High']=f.Close+1;f['Low']=f.Close-1
    p=s.detect(r,now=NOW,hard_gate=True)
    assert p['phase']=='blocked' and p['reason_code']==code
    assert not p['eligible']


def test_removing_general_gate_does_not_grant_low_rr_range_admission():
    p=s.detect(range_result(),now=NOW,hard_gate=True)
    assert p['phase']=='ready' and p['attempts']
    assert not q.admission(p)[0]


def test_own_entry_band_still_blocks_chasing():
    r=qualifying_result()
    p=s.detect(r,now=NOW)
    assert q.admission(p)[0]
    r['price']=p['max_entry']+0.1
    updated=s.detect(r,now=NOW,hard_gate=True)
    matching=[a for a in updated['attempts'] if a['kind']==p['kind']]
    assert matching and all(a['phase']!='ready' for a in matching)
    assert not updated['plan_allowed']
