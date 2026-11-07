from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import ast
import json

import pandas as pd
import pytest

from modules.support_map import build_support_map,support_brief,attach_support_briefs
from test_support_map_ck import result,NOW


@pytest.mark.parametrize('price,close,expected',[
    (108.,110.,'Support '), (100.,110.,'In Zone '),
    (95.,110.,'Unter Zone '), (95.,95.,'Zone '),
    (108.,95.,'Über Zone '),
])
def test_compact_statement_distinguishes_current_support_and_break(price,close,expected):
    r=result();r['price']=price
    r['df'].loc[r['df'].index[-1],['Open','High','Low','Close']]=[close,close+1,close-1,close]
    p=build_support_map(r,now=NOW);before=deepcopy(p)
    text=support_brief(p)
    assert text.startswith(expected) and 'USD' in text
    assert before==p and not any(x in text for x in ['Kaufen','Kaufsignal','Boden bestätigt'])
    if price==95. and close==95.:assert 'gebrochen' in text
    if price==95. and close==110.:assert 'Schluss offen' in text


def test_no_zone_missing_scan_and_missing_history_are_distinct():
    p=build_support_map(result(points=[]),now=NOW)
    assert support_brief(p)=='Keine bestätigte Tiefzone darunter'
    assert support_brief(None)=='Neu scannen'
    assert support_brief(float('nan'))=='Neu scannen'
    assert support_brief(build_support_map({'ticker':'HALO'},now=NOW))=='Nicht berechenbar'


def test_old_basis_and_reclaim_are_not_hidden():
    r=result(end='2026-11-05');f=r['df']
    f.loc[f.index[120],['Open','High','Low','Close']]=[95.,96.,94.,95.]
    p=build_support_map(r,now=NOW)
    assert 'Retest offen' in support_brief(p) and 'Stand 2026-11-05' in support_brief(p)


def test_nearest_zone_is_displayed_not_the_most_distant_high_touch_count():
    p=build_support_map(result(),now=NOW)
    low=deepcopy(p['zones'][0]);low.update(low=70.,high=71.,touches=9,distance_pct=(71/108-1)*100)
    p['zones'].insert(0,low)
    text=support_brief(p)
    assert '70,00' not in text and '100,' in text


def test_numeric_strings_are_safe_and_corrupt_records_are_explicit():
    p=build_support_map(result(),now=NOW)
    expected=support_brief(p)
    for z in p['zones']:
        for k in ['low','high','distance_pct']:z[k]=str(z[k])
    assert support_brief(p)==expected
    p['zones'][0]['low']='bad'
    assert support_brief(p)=='Daten unvollständig · neu scannen'


def test_briefs_do_not_change_any_other_column_or_input():
    p=build_support_map(result(),now=NOW)
    f=pd.DataFrame([{'Ticker':'HALO','Ampel':'🔴','Status':'Abwarten','Live-Score':'20/100','__support_map':p}])
    before=deepcopy(f.to_dict('records'))
    out=attach_support_briefs(f)
    assert 'Support-Kurz' not in f and out.loc[0,'Support-Kurz']==support_brief(p)
    assert before==f.to_dict('records')
    pd.testing.assert_frame_equal(out.drop(columns=['Support-Kurz']),f)


def test_real_history_cleanup_and_json_snapshot_keep_support_evidence(monkeypatch):
    from modules import live_monitor as lm
    from modules.live_screener_snapshot import dataframe_to_payload,dataframe_from_payload
    class State(dict):
        __getattr__=dict.__getitem__
        __setattr__=dict.__setitem__
    monkeypatch.setattr(lm,'st',SimpleNamespace(session_state=State()))
    monkeypatch.setattr(lm,'get_current_berlin_time',lambda:datetime(2026,11,7,10),raising=False)
    monkeypatch.setattr(lm,'_v227_load_persistent_live_history',lambda:({},[]))
    monkeypatch.setattr(lm,'_v227_save_persistent_live_history',lambda *a:None)
    monkeypatch.setattr(lm,'_v2416_log_event',lambda **kw:None,raising=False)
    p=build_support_map(result(),now=NOW)
    raw=pd.DataFrame([{'Ticker':'HALO','Kurs':108.,'Ampel':'🔴','Status':'Abwarten','Live-Score':'20/100',
                      '__support_map':p,'__temporary':1}])
    enriched,_=lm.apply_live_watchlist_status_history_v220(raw,watchlist_name='Test')
    assert enriched.loc[0,'__support_map']==p and '__temporary' not in enriched
    assert enriched.loc[0,'Support-Kurz']==support_brief(p)
    restored=dataframe_from_payload(json.loads(json.dumps(dataframe_to_payload(enriched))))
    assert restored.loc[0,'__support_map']==p
    assert restored.loc[0,'Support-Kurz']==support_brief(p)
    assert raw.loc[0,'Ampel']=='🔴' and 'Support-Kurz' not in raw


def test_desktop_compact_and_mobile_markup_expose_the_same_brief():
    text=(Path(__file__).resolve().parents[1]/'legacy_app.py').read_text()
    tree=ast.parse(text)
    compact=[node.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='_compact_cols_v3021l' for t in node.targets)
             and isinstance(node.value,ast.List)]
    assert compact
    for node in compact:
        constants=[x.value for x in node.elts if isinstance(x,ast.Constant)]
        assert constants.index('Support-Kurz')==constants.index('Kurs')+1
    assert '_support_brief_cl = html.escape(' in text
    assert '{_support_brief_cl}</div>' in text


def test_old_snapshot_gets_rescan_hint_instead_of_a_fabricated_level():
    out=attach_support_briefs(pd.DataFrame([{'Ticker':'HALO','Ampel':'🔴'}]))
    assert out.loc[0,'Support-Kurz']=='Neu scannen'
