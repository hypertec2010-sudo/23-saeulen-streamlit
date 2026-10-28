import pandas as pd
import pytest
from modules.data_freshness import classify_date,freshness_table,format_stamp


@pytest.mark.parametrize('now,day,fetched,status,severity',[
 ('2026-10-26T14:29:00+01:00','2026-10-23','2026-10-26T12:00Z','Letzte abgeschlossene','none'),
 ('2026-10-26T14:31:00+01:00','2026-10-23','2026-10-26T13:31Z','heutige Tageskerze noch nicht geliefert','info'),
 ('2026-10-26T14:31:00+01:00','2026-10-26','2026-10-26T13:31Z','Laufende','none'),
 ('2026-10-26T21:31:00+01:00','2026-10-23','2026-10-26T20:31Z','Abgeschlossene Tageskerze fehlt','warning'),
 ('2026-10-26T21:31:00+01:00','2026-10-26','2026-10-26T14:00Z','nicht durch frischen Abruf','warning'),
 ('2026-10-26T21:31:00+01:00','2026-10-26','2026-10-26T20:31Z','Letzte abgeschlossene','none'),
 ('2026-11-02T15:29:00+01:00','2026-10-30','2026-11-02T13:00Z','Letzte abgeschlossene','none'),
 ('2026-11-02T15:31:00+01:00','2026-10-30','2026-11-02T14:31Z','heutige Tageskerze noch nicht geliefert','info'),
 ('2026-10-24T12:00:00+02:00','2026-10-23','2026-10-23T21:00Z','Letzte abgeschlossene','none'),
 ('2026-11-26T18:00:00+01:00','2026-11-25','2026-11-26T17:00Z','Letzte abgeschlossene','none'),
 ('2026-11-27T19:31:00+01:00','2026-11-27','2026-11-27T18:31Z','Letzte abgeschlossene','none'),
 ('2027-12-31T16:00:00+01:00','2027-12-30','2027-12-31T15:00Z','heutige Tageskerze noch nicht geliefert','info'),
])
def test_us_session_weekend_dst_holiday_and_close_evidence(now,day,fetched,status,severity):
    c=classify_date('HALO',day,now=now,fetched_at=fetched)
    assert status in c['status'] and c['severity']==severity


def test_other_exchanges_and_unknown_dates_do_not_get_false_us_freshness():
    assert classify_date('SAP.DE','2026-10-23',now='2026-10-26T18:00Z')['severity']=='none'
    assert 'unbekannt' in classify_date('HALO','bad')['status']
    assert 'nicht hinterlegt' in classify_date('HALO','2029-01-02',now='2029-01-02T18:00Z')['status']
    assert 'Abrufzeit nicht belegt' in classify_date('HALO','2026-10-23',now='2026-10-26T12:00Z')['status']


def test_times_display_in_berlin_and_old_day_month_dates_remain_readable():
    assert '14:31:00 CET' in format_stamp('2026-10-26T13:31:00Z')
    table=freshness_table(pd.DataFrame([{'Ticker':'HALO','__diag_setup_data_date':'23.10.2026'}]),now='2026-10-26T12:00Z')
    assert table.iloc[0]['Kerzendatum']=='23.10.2026'
    assert table.iloc[0]['Letzter Provider-Abruf']=='Unbekannt'


def test_actual_summary_does_not_call_42_us_friday_bars_stale_before_open():
    from streamlit.testing.v1 import AppTest
    script='''
import streamlit as st
import pandas as pd
from modules.data_freshness import render_freshness_summary
rows=[{'Ticker':f'US{i}','Kursdaten-Kerze':'2026-10-23','Kursdaten-Abruf':'2026-10-26T12:00Z','Kursdaten-Quelle':'Provider-Abruf'} for i in range(42)]
rows += [{'Ticker':t,'Kursdaten-Kerze':'2026-10-26'} for t in ['SAP.DE','YSN.DE','ENR.DE']]
render_freshness_summary(st,pd.DataFrame(rows),now=st.session_state.get('now','2026-10-26T14:00:00+01:00'))
'''
    at=AppTest.from_string(script).run(timeout=20)
    assert not at.exception and not at.warning
    at.session_state['now']='2026-10-26T15:00:00+01:00'
    at.run()
    assert not at.exception and not at.warning
    assert '42 US-Werte noch ohne heutige Tageskerze' in at.info[0].value
