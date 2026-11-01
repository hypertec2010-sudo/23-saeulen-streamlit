"""Actual watcher/provider/storage/UI paths for manually confirmed positions."""
from copy import deepcopy

import pandas as pd
import pytest

from modules import trade_watch as w
from modules import watch_position as p
from modules import watch_setups as s
from modules.storage.manager import StorageManager
from modules.storage.local_backend import LocalJsonBackend
from test_trade_watch import idea, breakout, NOW

OPEN = NOW  # 27 Oct, 14:10:11 UTC; first eligible minute is 14:11.
CHECK = '2026-10-27T14:13:11Z'


def triggered():
    return w.evaluate(idea(), breakout(), now=NOW)


def opened():
    return p.open_position(triggered(), 100.5, now=OPEN)


def minutes(start='2026-10-27T14:11:00Z', periods=3):
    idx=pd.date_range(start, periods=periods, freq='min')
    return pd.DataFrame(dict(Open=101.,High=102.,Low=100.,Close=101.),index=idx)


def test_manual_open_is_explicit_and_preserves_levels_source_and_original():
    i=triggered();before=deepcopy(i)
    out=p.open_position(i,100.5,now=OPEN)
    assert i==before and out['position']['status']=='open'
    for k in ('entry','invalidation','max_entry','target','source','triggered_at'):
        assert out[k]==before[k]
    assert out['position']['entry_price']==100.5
    assert p.status_label(i) is None
    assert w.active(out)
    with pytest.raises(ValueError):p.open_position(out,100.5,now=OPEN)


@pytest.mark.parametrize('change', [dict(state='waiting'),dict(paused=True),dict(archived=True),
    dict(gap=True),dict(quote_at='2026-10-27T14:00:00Z'),dict(last_attempt_at=''),
    dict(health='Abruffehler: Provider'),dict(source={})])
def test_unverified_or_old_trigger_cannot_be_registered_as_current_execution(change):
    i=triggered();i.update(change)
    with pytest.raises(ValueError):p.open_position(i,100.,now=OPEN)


@pytest.mark.parametrize('price',[None,0,-1,float('nan'),float('inf')])
def test_invalid_manual_fills_rejected(price):
    with pytest.raises(ValueError):p.open_position(triggered(),price,now=OPEN)
    with pytest.raises(ValueError):p.close_position(opened(),price,now=CHECK)


@pytest.mark.parametrize('level,event', [('High','position_target'),('Low','position_stop')])
def test_closed_post_execution_minute_records_contact_once(level,event):
    i=opened();frame=minutes()
    frame.loc[frame.index[1],level]=115. if level=='High' else 94.
    original=deepcopy(i)
    out=w.evaluate(i,frame,now=CHECK)
    assert i==original and p.is_open(out)
    assert len([e for e in out['events'] if e['kind']==event])==1
    again=w.evaluate(out,frame,now=CHECK)
    assert len([e for e in again['events'] if e['kind']==event])==1
    assert out['state']=='triggered' and out['source']==i['source']
    assert out['position']['last_bar_end']=='2026-10-27T14:13:00+00:00'


def test_contact_before_open_or_in_partial_confirmation_minute_does_not_count():
    frame=minutes(start='2026-10-27T14:09:00Z',periods=5)
    frame.loc[frame.index[:2],['High','Low']]=[120.,90.]
    out=w.evaluate(opened(),frame,now=CHECK)
    assert not out['position']['target_at'] and not out['position']['stop_at']


def test_running_minute_and_publication_buffer_cannot_create_exit_notice():
    frame=minutes(periods=2);frame.loc[frame.index[-1],'High']=115.
    i=opened()
    early=w.evaluate(i,frame,now='2026-10-27T14:13:09Z')
    assert not early['position']['target_at']
    final=w.evaluate(early,frame,now='2026-10-27T14:13:10Z')
    assert final['position']['target_at']


def test_same_minute_is_ambiguous_and_distinct_minutes_preserve_order():
    frame=minutes();frame.loc[frame.index[0],['High','Low']]=[115.,94.]
    out=w.evaluate(opened(),frame,now=CHECK)
    pos=out['position']
    assert pos['ambiguous_at']==pos['stop_at']==pos['target_at']
    assert 'unklar' in p.status_label(out)
    frame=minutes();frame.loc[frame.index[0],'High']=115.;frame.loc[frame.index[1],'Low']=94.
    out=w.evaluate(opened(),frame,now=CHECK)
    assert not out['position']['ambiguous_at']
    assert out['position']['target_at']<out['position']['stop_at']
    assert 'Invalidierung' in p.status_label(out)


def test_data_gap_keeps_known_contact_and_does_not_invent_missing_contacts():
    frame=minutes().drop(pd.Timestamp('2026-10-27T14:11Z'))
    frame.loc[frame.index[0],'High']=115.
    out=w.evaluate(opened(),frame,now=CHECK)
    assert out['position']['gap'] and out['position']['target_at']
    assert not out['position']['stop_at']
    assert 'Datenlücke' in out['health']
    again=w.evaluate(out,minutes(),now=CHECK)
    assert again['position']['gap'] and not again['position']['stop_at']


def test_delayed_data_is_historical_and_stale_provider_never_rolls_back():
    frame=minutes();frame.loc[frame.index[0],'High']=115.
    out=w.evaluate(opened(),frame,now='2026-10-27T14:20:11Z')
    assert out['position']['gap']
    event=next(e for e in out['events'] if e['kind']=='position_target')
    assert 'verspätet' in event['text'] and event['at']=='2026-10-27T14:12:00+00:00'
    old=w.evaluate(out,frame.iloc[:1],now='2026-10-27T14:21:11Z')
    assert old['position']['last_bar_end']==out['position']['last_bar_end']
    assert old['quote_at']==out['quote_at'] and old['events']==out['events']


def test_position_outlives_entry_expiry_but_closed_position_never_fetches():
    i=opened();package=i['source']['watch_setup'];i['expires_at']='2026-10-27T14:11Z'
    package['valid_until']=i['expires_at'];i['source']['watch_setup']=s.sealed(package)
    class Provider:
        calls=0
        def get_history_light(self,*args,**kwargs):
            self.calls+=1
            return minutes()
    provider=Provider()
    out=w.poll([i],provider,now=CHECK,force=True)[0]
    assert provider.calls==1 and p.is_open(out) and w.active(out)
    closed=p.close_position(out,105.,now=CHECK)
    assert not w.active(closed)
    before=deepcopy(closed)
    assert w.poll([closed],None,now=CHECK,force=True)==[before]
    assert closed['position']['return_pct']==pytest.approx((105/100.5-1)*100)
    from modules.trade_watch_ui import table_rows
    from modules.trade_watch_overview import data_notes
    assert data_notes(table_rows([closed],now='2026-10-28T15:00Z'))==([],[])
    with pytest.raises(ValueError):p.close_position(closed,105.,now=CHECK)


@pytest.mark.parametrize('kind',['error','bad_data','future','bad_proof'])
def test_failed_evidence_never_records_exit(kind):
    i=opened();frame=minutes();frame.loc[frame.index[0],'High']=115.
    kwargs={}
    if kind=='error':kwargs['fetch_error']='unavailable'
    if kind=='bad_data':frame['Low']=float('nan')
    if kind=='future':frame.index=frame.index+pd.Timedelta(days=1)
    if kind=='bad_proof':i['invalidation']=96.
    out=w.evaluate(i,frame,now=CHECK,**kwargs)
    assert not out['position']['target_at'] and not out['position']['stop_at']


def test_storage_keeps_position_events_and_levels_across_sessions(tmp_path):
    m=StorageManager(user_id='pos',local_backend=LocalJsonBackend(tmp_path))
    i=opened();frame=minutes();frame.loc[frame.index[0],'High']=115.
    i=w.evaluate(i,frame,now=CHECK);w.save_ideas(m,[i])
    again=StorageManager(user_id='pos',local_backend=LocalJsonBackend(tmp_path))
    assert w.load_ideas(again)==[i]


def test_ui_open_target_notice_close_and_overview_are_persisted(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from modules.trade_watch_overview import MAIN_COLUMNS
    real=w.utc;clock={'now':OPEN}
    monkeypatch.setattr(w,'utc',lambda value=None:real(value if value is not None else clock['now']))
    m=StorageManager(user_id='pos',local_backend=LocalJsonBackend(tmp_path))
    i=triggered();w.save_ideas(m,[i]);prefix='position_'+i['id']
    script=f'''
import streamlit as st
from modules.storage.manager import StorageManager
from modules.storage.local_backend import LocalJsonBackend
from modules.trade_watch_ui import render_monitor
from modules.trade_watch_overview import render_compact_monitor
m=StorageManager(user_id='pos',local_backend=LocalJsonBackend({str(tmp_path)!r}))
def forbidden():raise AssertionError('No automatic provider call')
render_compact_monitor(st,renderer=lambda view:render_monitor(view,manager=m,provider_factory=forbidden))
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    app.number_input(key=prefix+'_fill').set_value(100.5)
    next(b for b in app.button if b.label=='Position als eröffnet bestätigen').click().run()
    assert app.error and not w.load_ideas(m)[0].get('position')
    app.checkbox(key=prefix+'_executed').check()
    next(b for b in app.button if b.label=='Position als eröffnet bestätigen').click().run()
    assert not app.exception and p.is_open(w.load_ideas(m)[0])
    assert app.button(key='watch_archive').disabled and app.button(key='watch_pause').disabled
    clock['now']=CHECK
    frame=minutes();frame.loc[frame.index[0],'High']=115.
    stored=w.evaluate(w.load_ideas(m)[0],frame,now=CHECK);w.save_ideas(m,[stored])
    app.run()
    assert not app.exception and any('Ziel erreicht' in x.value for x in app.success)
    assert list(app.dataframe[0].value.columns)==MAIN_COLUMNS
    assert 'Ziel erreicht' in app.dataframe[0].value.iloc[0]['Status']
    app.number_input(key=prefix+'_exit').set_value(115.)
    app.checkbox(key=prefix+'_closed').check()
    next(b for b in app.button if b.label=='Position als geschlossen bestätigen').click().run()
    assert not app.exception and w.load_ideas(m)[0]['position']['status']=='closed'
    assert app.button(key='watch_poll').disabled
    assert not app.button(key='watch_archive').disabled


def test_provider_window_is_shared_by_two_positions_and_does_not_call_analysis():
    calls=[]
    class Provider:
        def get_history_light(self,ticker,**kwargs):
            calls.append((ticker,kwargs))
            return minutes()
    out=w.poll([opened(),opened()],Provider(),now=CHECK,force=True)
    assert len(calls)==1 and calls[0][1]==dict(period='5d',interval='1m',prepost=False,auto_adjust=False)
    assert all(i['position']['last_bar_end'] for i in out)


def test_ui_poll_fetches_position_even_after_entry_expiry(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    real=w.utc
    monkeypatch.setattr(w,'utc',lambda value=None:real(value if value is not None else CHECK))
    m=StorageManager(user_id='expired-entry',local_backend=LocalJsonBackend(tmp_path))
    i=opened();i['expires_at']='2026-10-27T14:11Z';w.save_ideas(m,[i])
    script=f'''
import streamlit as st
from modules.storage.manager import StorageManager
from modules.storage.local_backend import LocalJsonBackend
from modules.trade_watch_ui import render_monitor
from test_watch_position import minutes
class Provider:
    def get_history_light(self,*args,**kwargs):
        frame=minutes();frame.loc[frame.index[0],'High']=115.
        return frame
m=StorageManager(user_id='expired-entry',local_backend=LocalJsonBackend({str(tmp_path)!r}))
render_monitor(st,manager=m,provider_factory=Provider)
'''
    app=AppTest.from_string(script).run(timeout=20)
    app.button(key='watch_poll').click().run()
    assert not app.exception and not app.error
    assert w.load_ideas(m)[0]['position']['target_at']
    assert any('Ziel erreicht' in x.value for x in app.success)
