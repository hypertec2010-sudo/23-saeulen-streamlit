import ast
from copy import deepcopy
from pathlib import Path
import pytest
from modules.action_risk_context import build_action_risk_context


def test_unh_entry_band_is_warning_not_stop_or_invalidation():
    r={'technical_crv_stop_quality':'proxy','technical_crv_stop_now':395}
    before=deepcopy(r)
    p=build_action_risk_context(r,entry_zone='409.14 - 413.26 USD',ccy='USD')
    assert 'Unter 409.14 USD' in p['text']
    assert 'Unter 413.26' not in p['text'] and '395' not in p['text']
    assert 'innerhalb' in p['text'] and 'keine automatische Invalidierung' in p['text']
    assert 'Keine bestätigte' in p['text'] and 'Struktur-Risiko' in p['label']
    assert r==before


def test_confirmed_actual_stop_stays_separate_from_entry():
    p=build_action_risk_context({'technical_crv_stop_quality':'confirmed','technical_crv_stop_now':395.5},
                               entry_zone='409.14 - 413.26',ccy='USD')
    assert 'Unter 409.14 USD' in p['text']
    assert 'Stop bei 395.50 USD' in p['text']


@pytest.mark.parametrize('quality',['missing','estimated','proxy',None])
def test_numerical_stop_and_alternative_reference_cannot_fake_confirmation(quality):
    p=build_action_risk_context({'technical_crv_stop_quality':quality,'technical_crv_stop_now':395,
                                'technical_crv_structure_stop':390},entry_zone='409.14 - 413.26')
    assert 'Keine bestätigte' in p['text'] and '390' not in p['text']


@pytest.mark.parametrize('value',[None,float('nan'),-1,0])
def test_invalid_confirmed_stop_is_not_displayed(value):
    p=build_action_risk_context({'technical_crv_stop_quality':'confirmed','technical_crv_stop_now':value})
    assert 'Keine bestätigte' in p['text']


def test_existing_model_block_remains_explicit_without_promoting_it_to_stop():
    p=build_action_risk_context({},entry_zone='409.14 - 413.26',invalidated=True,
                               invalidation_diagnostics={'level':420})
    assert 'Bestehende Sperre' in p['text'] and '420.00' in p['text']
    assert 'bestätigt keinen Strukturstop' in p['text']


def test_actual_render_path_preserves_position_mode_and_uses_full_escaped_risk():
    source=(Path(__file__).resolve().parents[1]/'legacy_app.py').read_text()
    tree=ast.parse(source)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
              and any(isinstance(x,ast.Name) and x.id=='_risk_context_v3021bm' for x in ast.walk(n)))
    # Select the small new if block, not its enclosing render function/condition.
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.If)
           and isinstance(n.test,ast.UnaryOp) and isinstance(n.test.op,ast.Not)
           and isinstance(n.test.operand,ast.Name) and n.test.operand.id=='position_mode'
           and any(isinstance(x,ast.Name) and x.id=='_risk_context_v3021bm' for x in ast.walk(n))]
    assert len(nodes)==1
    for positioned in [True,False]:
        scope={'position_mode':positioned,'result':{},'suggested_entry_zone':'409.14 - 413.26',
               'ccy':'USD','_invalidated_v3021az':False,'_invalid_diag_single_v3021bk':{},
               '_invalid_txt':'Originaler Positionsplan','_invalid_label':'Exit prüfen'}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_risk_render','exec'),scope)
        if positioned:assert scope['_invalid_txt']=='Originaler Positionsplan'
        else:assert 'Unter 409.14 USD' in scope['_invalid_txt'] and 'Keine bestätigte' in scope['_invalid_txt']
    assert 'html.escape(_invalid_txt)' in source
    assert 'shorten_text(_invalid_txt, 190)' not in source
    assert '_v17_13_2_invalid_zone_for_action' not in source
    assert 'exec_redflag = _invalid_txt if not position_mode' in source
