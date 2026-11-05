from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules import candidate_radar as r
from modules import watch_setups as s
from modules import trade_watch as w
from modules.radar_setup_audit import chart_audit
from modules.radar_watch_handoff import watch_basis, render_handoff
from modules.watch_results import trade_results,summarize,render_results
from modules.chart_ma_table import moving_average_rows,render_moving_averages
from modules.storage.manager import StorageManager
from modules.storage.local_backend import LocalJsonBackend
from test_watch_setups import range_result,NOW
from test_watch_position import opened,CHECK,minutes
from modules.watch_position import close_position


def snapshot():
    x=range_result();x.update(valid_trade_setup=True,confidence_info={'coverage':1.},trigger_status='aktiv')
    row=r.build_candidate(x,ticker='HALO',style='Charttechnik',
        decide=lambda *_:dict(grade='A',score=88,top_chance_rank=80,bucket='Jetzt prüfbar'),
        entry_package=lambda *_:dict(crv=2.,price=99.2,stop=95.,tp1=110.,entry_zone='100–101',entry_distance_pct=0.),
        analyzed_at=NOW)
    return dict(schema_version=r.SCHEMA_VERSION,radar_version=r.RADAR_VERSION,completed=True,
        owner='radar-test',key=r.request_key('Eigene Liste','Charttechnik',['HALO']),
        started_at=NOW,completed_at=NOW,scan_id='cf-fixture',source='manual',requested=1,processed=1,
        symbols=['HALO'],rows=[row],errors=[],resolution=[])


def test_radar_proof_survives_json_and_transfers_exact_levels_without_rescan():
    snap=json.loads(json.dumps(snapshot()));before=deepcopy(snap)
    basis,reason=watch_basis(snap,'HALO',now=NOW,holdings_known=True)
    assert basis is not None,reason
    pkg=basis['watch_setup'];assert s.evidence_valid(pkg)
    i=w.make_idea(ticker='HALO',entry=pkg['entry'],invalidation=pkg['invalidation'],max_entry=pkg['max_entry'],
        target=pkg['target'],mode=pkg['mode'],expires_at=pkg['valid_until'],currency=pkg['currency'],source=basis,now=NOW)
    assert w.active(i) and i['source']['radar']['scan_id']==snap['scan_id'] and i['state']=='waiting'
    assert all(i[k]==pkg[k] for k in ('entry','invalidation','max_entry','target'))
    assert snap==before
    basis['watch_setup']['entry']+=1
    assert snap==before


@pytest.mark.parametrize('case',['old','expired_proof','held','unknown_holdings','gate','changed','missing',
    'preview','low_rr','stale_quote','other_owner','other_ticker'])
def test_radar_handoff_rejects_unfit_evidence(case):
    snap=snapshot();args=dict(now=NOW,holdings_known=True)
    row=snap['rows'][0];audit=row['chart_audit']
    if case=='old':args['now']='2026-10-30T15:00Z'
    if case=='expired_proof':args['now']='2026-10-28T21:00Z'
    if case=='held':args['held']={'halo'}
    if case=='unknown_holdings':args['holdings_known']=False
    if case=='gate':row['gates']=['Earnings']
    if case=='changed':audit['watch_setup']['entry']+=.1
    if case=='missing':audit.pop('watch_setup')
    if case=='preview':audit['full_context']=False
    if case=='low_rr':
        audit['watch_setup']=s.sealed({**audit['watch_setup'],'target':100.5})
    if case=='stale_quote':row['quote_date']='2026-10-01'
    if case=='other_owner':args['owner']='someone-else'
    if case=='other_ticker':audit['watch_setup']=s.sealed({**audit['watch_setup'],'ticker':'HPE'})
    basis,reason=watch_basis(snap,'HALO',**args)
    assert basis is None and reason


def test_light_audit_never_carries_armable_proof():
    preview=chart_audit(range_result(),now=NOW,full_context=False)
    assert 'watch_setup' not in preview


def test_real_radar_handoff_form_persists_once_and_keeps_rules(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    real=w.utc;monkeypatch.setattr(w,'utc',lambda value=None:real(value if value is not None else NOW))
    snap=snapshot()
    script=f'''
import streamlit as st
from modules.radar_watch_handoff import render_handoff
from modules.storage.manager import StorageManager
from modules.storage.local_backend import LocalJsonBackend
m=StorageManager(user_id='radar-test',local_backend=LocalJsonBackend({str(tmp_path)!r}))
st.session_state.setdefault('radar_watch_candidate','HPE')
render_handoff(st,{snap!r},['HALO'],storage=m,holdings_known=True,held=set(),now={NOW!r})
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and not app.number_input
    app.checkbox[0].check()
    next(b for b in app.button if b.label=='Bestätigungsbereites Setup verfolgen').click().run()
    assert not app.exception and app.success
    m=StorageManager(user_id='radar-test',local_backend=LocalJsonBackend(tmp_path))
    saved=w.load_ideas(m)
    assert len(saved)==1 and saved[0]['source']['watch_setup']==snap['rows'][0]['chart_audit']['watch_setup']
    assert not saved[0].get('position')
    next(b for b in app.button if b.label=='Bestätigungsbereites Setup verfolgen').click().run()
    assert app.error and len(w.load_ideas(m))==1


def finished(exit_price=111.5,**position_changes):
    i=close_position(w.evaluate(opened(),minutes(),now=CHECK),exit_price,now=CHECK)
    i['position'].update(position_changes)
    return i


def test_actual_closed_results_ignore_fake_cached_return_and_include_archives():
    ideas=[finished(111.5,return_pct=999),finished(95.),finished(100.5),opened()]
    ideas[0]['archived']=True
    before=deepcopy(ideas)
    f,excluded=trade_results(ideas,now=NOW);summary=summarize(f)
    assert excluded==0 and len(f)==3 and ideas==before
    assert summary['Gewinne']==1 and summary['Verluste']==1 and summary['Neutral']==1
    assert summary['Trefferquote %']==pytest.approx(100/3)
    assert f['Ergebnis R'].tolist()==pytest.approx([2.,-1.,0.])
    assert summary['Ø Ergebnis R']==pytest.approx(1/3)
    assert f.iloc[0]['Kursrendite %']==pytest.approx((111.5/100.5-1)*100)


def test_missing_proof_and_data_gaps_are_not_clean_or_invented_r():
    a=finished(gap=True);b=finished();b['source']={}
    c=finished(ambiguous_at=CHECK)
    f,ex=trade_results([a,b,c],now=NOW)
    assert ex==0 and len(f)==3 and not f['Verlauf für Mustervergleich'].any()
    assert pd.isna(f.iloc[1]['Ergebnis R'])
    assert summarize(f)['R verfügbar']==2 and summarize(f)['Mit unvollständigem Verlauf']==3
    assert summarize(f.loc[f['Verlauf für Mustervergleich']])['Trades']==0


def test_no_position_poll_or_late_manual_close_is_not_marked_as_observed():
    without_poll=close_position(opened(),105.,now=CHECK)
    late=finished(closed_at='2026-10-27T16:00Z')
    f,ex=trade_results([without_poll,late],now=NOW)
    assert ex==0 and len(f)==2 and not f['Verlauf für Mustervergleich'].any()


@pytest.mark.parametrize('change',[{'entry_price':0},{'exit_price':float('inf')},
    {'opened_at':None},{'closed_at':''},{'closed_at':'2027-01-01T12:00Z'},
    {'closed_at':'2026-10-26T12:00Z'}])
def test_invalid_closures_do_not_create_result(change):
    idea=finished();idea['position'].update(change)
    f,excluded=trade_results([idea],now=NOW)
    assert f.empty and excluded==1


def test_duplicate_closure_is_not_counted_twice_and_empty_sample_is_not_zero_winrate():
    i=finished();f,ex=trade_results([i,deepcopy(i)],now=NOW)
    assert len(f)==1 and ex==1
    assert summarize(f.iloc[:0])['Trefferquote %'] is None


def test_real_results_ui_filters_gaps_without_storage_writes():
    from streamlit.testing.v1 import AppTest
    ideas=[finished(),finished(gap=True)]
    script=f'''
import streamlit as st
from modules.watch_results import render_results
render_results(st,{ideas!r},now={NOW!r})
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and app.table[0].value.iloc[0]['Trades']==2
    app.checkbox(key='watch_results_clean').check().run()
    assert not app.exception and app.table[0].value.iloc[0]['Trades']==1


def daily(n=250):
    return pd.DataFrame({'Close':np.arange(1,n+1,dtype=float)},index=pd.bdate_range(end='2026-11-03',periods=n))


def test_ma200_uses_full_history_without_mutation_or_short_window_fallback():
    df=daily();before=df.copy(deep=True)
    rows=moving_average_rows({'df':df,'price':250.,'ma200':999.})
    assert rows[2]['Indikator']=='MA200 · 200 Handelstage' and rows[2]['Wert']=='150,50'
    assert rows[2]['Kursabstand']=='66,11 %'
    assert moving_average_rows({'df':df.tail(199),'price':250.})[2]['Wert']=='Nicht verfügbar'
    pd.testing.assert_frame_equal(df,before)


def test_hourly_or_bad_history_cannot_be_labeled_ma200_daily():
    df=daily();df.index=pd.date_range('2026-10-01',periods=len(df),freq='h')
    assert moving_average_rows({'df':df,'price':250})[2]['Wert']=='Nicht verfügbar'
    df=daily();df.iloc[-1,0]=float('nan')
    assert moving_average_rows({'df':df,'price':250})[2]['Wert']=='Nicht verfügbar'


def test_actual_chart_ma_table_renders_ma200_and_legacy_calls_it():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('''
import streamlit as st
from test_watch_extensions_cf import daily
from modules.chart_ma_table import render_moving_averages
render_moving_averages(st,{'df':daily(),'price':250.})
''').run(timeout=20)
    assert not app.exception and 'MA200' in str(app.table[0].value)
    code=(Path(__file__).resolve().parents[1]/'legacy_app.py').read_text()
    assert 'render_moving_averages(st, result)' in code
