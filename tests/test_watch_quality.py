"""Admission uses frozen levels; historical monitoring remains independent."""
from copy import deepcopy

import pandas as pd
import pytest

from modules import trade_watch as w
from modules import watch_quality as q
from modules import watch_setups as s
from test_watch_setups import range_result, qualifying_result, NOW
from test_trade_watch import idea, breakout, NOW as ENGINE_NOW


def rules(**changes):
    return dict(entry=100., invalidation=95., max_entry=101., target=110., **{}) | changes


@pytest.mark.parametrize('target,allowed', [(110., True), (109.999, False), (101., False),
    (100.5, False), (None, False), (float('nan'), False), (float('inf'), False), (0, False)])
def test_rr_gate_checks_the_worst_permitted_entry_without_rounding(target, allowed):
    levels = rules(target=target)
    before = deepcopy(levels)
    result = q.assess(levels)
    assert result['allowed'] is allowed
    for k in levels:
        assert levels[k] == before[k] or pd.isna(levels[k]) and pd.isna(before[k])
    if target == 110.:
        assert result['rr_entry'] == 2. and result['rr_cap'] == 1.5


@pytest.mark.parametrize('change', [{'entry':0}, {'invalidation':100}, {'max_entry':99},
    {'entry':None}, {'max_entry':float('inf')}])
def test_invalid_levels_fail_closed(change):
    assert not q.assess(rules(**change))['allowed']


def test_analysis_crv_and_analysis_levels_cannot_override_frozen_rules():
    levels = rules(target=105., crv=99, source={'stop':99, 'target':1000, 'crv':99})
    assert not q.assess(levels)['allowed']
    assert q.assess(levels)['rr_cap'] == pytest.approx(4/6)


def test_quality_explains_actual_structure_without_a_profit_grade():
    p = s.detect(range_result(), now=NOW)
    before = deepcopy(p)
    rows = q.quality_rows(p)
    assert any('3/3' in r['Befund'] for r in rows)
    assert any('Nicht Bestandteil' in r['Befund'] for r in rows)
    assert p == before
    p['entry'] += 1
    assert q.quality_rows(p)[0]['Befund'] == 'Nicht nachgewiesen'


def test_low_rr_does_not_erase_a_valid_detected_chart_pattern():
    r = range_result(); r['technical_target_1'] = 100.5
    p = s.detect(r, now=NOW)
    assert p['eligible'] and s.validate(p, now=NOW)[0]
    assert not q.assess(p)['allowed']
    stocks, _, _ = s.diagnostic_tables(pd.DataFrame([{'Ticker':'HALO','__watch_setup':p}]), now=NOW)
    assert stocks.iloc[0]['Phase'] == 'Struktur bestätigt'
    assert 'Entry' in stocks.iloc[0]['Übernahmeprüfung']


def test_new_creation_requires_rr_but_existing_idea_still_monitors():
    original = idea()
    legacy = deepcopy(original)
    p = legacy['source']['watch_setup']
    p['target'] = legacy['target'] = 105.
    legacy['source']['watch_setup'] = s.sealed(p)
    assert w.active(legacy)
    before = deepcopy(legacy)
    result = w.evaluate(legacy, breakout(), now=ENGINE_NOW)
    assert result['state'] == 'triggered' and legacy == before
    with pytest.raises(ValueError, match='CRV'):
        idea(target=105.)
    # Source's optional target remains absent rather than being fabricated.
    with pytest.raises(ValueError, match='Ziel'):
        idea(target=None)


def test_direct_add_cannot_bypass_admission_or_write_storage(monkeypatch):
    legacy = idea()
    legacy['target'] = 105.
    p = legacy['source']['watch_setup']; p['target'] = 105.
    legacy['source']['watch_setup'] = s.sealed(p)
    real = w.utc
    monkeypatch.setattr(w, 'utc', lambda value=None: real(value if value is not None else ENGINE_NOW))
    class Forbidden:
        def load_result(self, *a, **kw): raise AssertionError('must reject before storage')
        def save_result(self, *a, **kw): raise AssertionError('must not write')
    with pytest.raises(ValueError, match='CRV'):
        w.add_idea(Forbidden(), legacy)


def test_rule_card_uses_top_level_values_and_plain_german():
    levels = rules(ticker='TEAM', price=99., currency='USD', mode='close5', state='waiting',
                   source={'stop':1., 'target':999.})
    rows = {r['Regel']:r['Wert'] for r in q.rule_rows(levels)}
    assert rows['Invalidierung'] == '95 USD'
    assert rows['Ziel (Referenz)'] == '110 USD'
    assert '5-Minuten' in rows['Einstiegsbestätigung']
    assert 'close5' not in str(rows)


@pytest.mark.parametrize('target,has_form', [(110.,True),(100.5,True),(None,True)])
def test_capture_ui_explains_admission_without_inputs_or_provider(target, has_form, monkeypatch):
    from streamlit.testing.v1 import AppTest
    real = w.utc
    monkeypatch.setattr(w,'utc',lambda value=None: real(value if value is not None else NOW))
    r=qualifying_result();r['technical_target_1']=target
    p=s.detect(r, now=NOW)
    script=f'''
import streamlit as st
from modules.trade_watch_ui import capture_form
capture_form(st,{{'ticker':'HALO','watch_setup':{p!r}}})
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and not app.number_input
    assert bool(app.checkbox) is has_form
    assert any('oberen Entry-Rand' in str(t.value) for t in app.table)
    assert any(e.label=='Technische Details zum Chartbeleg' and not e.proto.expanded for e in app.expander)
