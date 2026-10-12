"""Exercise the actual alert callback, including the CRV-eligibility branch."""
import ast
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from modules.technical_crv import parse_price_zone


@pytest.fixture
def setup_alerts():
    names = {'_v210_alert_num', '_v210_alert_price', '_v210_alert_level',
             '_v210_alert_add', 'build_setup_alerts_v210', '_radar_v182_parse_zone', 'fmt_num'}
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'legacy_app.py').read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in funcs} == names
    scope = {'np': np, 'pd': pd, 're': re, '_parse_crv_price_zone_v3021be': parse_price_zone}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'actual_setup_alerts', 'exec'), scope)
    return scope['build_setup_alerts_v210']


@pytest.mark.parametrize('eligible', [True, False, None])
@pytest.mark.parametrize('bucket', ['Jetzt prüfbar', 'Nahe am Trigger', 'Starke Watchlist', 'Später beobachten'])
def test_real_alert_branch_requires_technical_evidence(setup_alerts, eligible, bucket):
    d = dict(bucket=bucket, grade='B', score=80, crv=2.0,
             entry_zone='99 - 101', gate_reasons='Keine harten Gates')
    if eligible is not None:
        d['crv_now_eligible'] = eligible
    rows = setup_alerts({'ticker': 'AMD', 'price': 100}, decision=d)
    types = {r['Alert-Typ'] for r in rows}
    assert ('CRV attraktiv' in types) == (eligible is True and bucket in {'Jetzt prüfbar', 'Nahe am Trigger'})


def test_actual_alerts_keep_invalidation_and_hard_gates(setup_alerts):
    d = dict(bucket='Jetzt prüfbar', grade='A', score=90, crv=3,
             crv_now_eligible=True, entry_zone='99 - 101',
             wave_invalidation='101', gate_reasons='Distribution dominiert')
    rows = setup_alerts({'ticker': 'MRVL', 'price': 100}, decision=d)
    assert [r['Alert-Typ'] for r in rows] == ['Invalidierung gebrochen']
