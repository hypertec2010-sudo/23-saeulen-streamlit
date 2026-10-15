import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from modules.decision_diagnostics import build_wave_reference_evidence, build_invalidation_diagnostics
from test_setup_alert_crv_evidence import setup_alerts


def history():
    return pd.DataFrame({'Close':[101.,99.,95.]},index=pd.to_datetime(['2026-10-13','2026-10-14','2026-10-15']))


def test_only_completed_daily_close_and_exact_reference_date():
    df=history();before=df.copy(deep=True)
    e=build_wave_reference_evidence(df,swing={'idx':0},as_of='2026-10-15')
    assert e=={'source':'swing_low','reference_date':'2026-10-13',
               'completed_close':99.,'completed_close_date':'2026-10-14'}
    pd.testing.assert_frame_equal(df,before)


@pytest.mark.parametrize('index', [pd.RangeIndex(3),pd.to_datetime(['2026-10-14 09:00','2026-10-14 10:00','2026-10-15 00:00']),
                                  pd.to_datetime(['2026-10-14','2026-10-13','2026-10-15'])])
def test_non_daily_or_unsorted_data_cannot_confirm_close(index):
    df=history();df.index=index
    assert build_wave_reference_evidence(df,swing={'idx':0},as_of='2026-10-15')['completed_close'] is None


@pytest.mark.parametrize('close,expected',[(99,True),(100,True),(101,False)])
def test_quote_breach_separate_from_daily_close(close,expected):
    r={'price':98,'wave_structure_pkg':{'wave_invalidation_price':100,
       'wave_invalidation_evidence':{'source':'swing_low','reference_date':'2026-10-12',
                                    'completed_close':close,'completed_close_date':'2026-10-14'}}}
    p=build_invalidation_diagnostics(r,{},price=98,used_level=100)
    assert p['breached'] and p['completed_close_breached'] is expected
    assert '2026-10-14' in p['detail'] and 'Swing-Tief' in p['detail']
    if not expected:assert 'heutiger Schlussbruch nicht bestätigt' in p['detail']


@pytest.mark.parametrize('numeric,reference_date',[(110,'2026-10-12'),(100,'2026-10-15'),(100,None)])
def test_conflicting_or_newer_or_undated_reference_cannot_prove_past_break(numeric,reference_date):
    r={'wave_structure_pkg':{'wave_invalidation_price':numeric,
        'wave_invalidation_evidence':{'source':'ma20_fallback','reference_date':reference_date,
                                     'completed_close':90,'completed_close_date':'2026-10-14'}}}
    p=build_invalidation_diagnostics(r,{},price=98,used_level=100)
    assert p['completed_close_breached'] is None
    assert 'MA20-Ersatzreferenz' in p['detail']


@pytest.mark.parametrize('fallback',[False,True])
def test_actual_wave_builder_attaches_provenance_without_changing_alert(fallback,setup_alerts):
    tree=ast.parse((Path(__file__).resolve().parents[1]/'legacy_app.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_wave_structure_context_v190')
    swings=[{'idx':10+i*10,'type':'H' if fallback or i%2 else 'L','price':100.+i} for i in range(4)]
    scope={'pd':pd,'np':np,'build_wave_structure_context_v1532':lambda *a:{},
           '_wave_v190_swing_points':lambda *a,**kw:swings,
           '_wave_v190_fmt_num':lambda v:f'{v:.2f}' if v is not None else 'n/a',
           '_wave_v191_build_readable_fields':lambda b:{}}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_wave','exec'),scope)
    df=pd.DataFrame({'Close':np.full(80,100.),'High':np.full(80,105.),'Low':np.full(80,95.)},
                    index=pd.date_range('2026-01-01',periods=80))
    pkg=scope[node.name](df,{})
    e=pkg['wave_invalidation_evidence']
    assert e['source']==('ma20_fallback' if fallback else 'swing_low')
    assert e['reference_date']==df.index[-1 if fallback else 30].date().isoformat()
    assert ('MA20-Ersatzreferenz' in pkg['wave_invalidation']) is fallback
    # The actual old text parser must still see the PRICE, never the "20" of MA20.
    old={'wave_invalidation':str(pkg['wave_invalidation_price'])}
    new={'wave_invalidation':pkg['wave_invalidation']}
    for price in [99,100,101,102,103]:
        a=setup_alerts({'price':price},decision=old)
        b=setup_alerts({'price':price},decision=new)
        assert [r['Alert-Typ'] for r in a]==[r['Alert-Typ'] for r in b]


def test_legacy_records_do_not_invent_reference_provenance():
    p=build_invalidation_diagnostics({}, {},price=90,used_level=100)
    assert p['reference_kind'] is None and p['completed_close_breached'] is None
    assert 'Herkunft nicht belegt' in p['detail']
