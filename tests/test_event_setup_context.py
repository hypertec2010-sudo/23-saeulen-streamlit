from datetime import timedelta
import pandas as pd
import pytest
from modules.event_setup_context import build_event_setup_context, render_event_setup_context


def bars(following=5):
    index=pd.bdate_range('2026-08-03',periods=21+following)
    df=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Volume':100.},index=index)
    df.iloc[20]=[110,112,109,111,200]
    if following:df.iloc[21:]=[110,112,109,110,50]
    return df


def evaluate(df):
    return build_event_setup_context(df,as_of=df.index[-1].date()+timedelta(days=1))


def test_event_anchor_and_volume_use_only_prior_twenty_sessions():
    p=evaluate(bars());e=p['event']
    assert e['relative_volume']==2 and e['gap_pct']==pytest.approx(10)
    assert e['hvc']==111 and e['event_low']==109
    assert 'nicht verifiziert' in p['summary']


@pytest.mark.parametrize('volume,open_price',[(199,110),(200,109.9)])
def test_below_scan_threshold_is_not_an_event(volume,open_price):
    df=bars();df.iloc[20]=[open_price,112,109,111,volume]
    assert evaluate(df)['event'] is None


def test_current_day_is_excluded_even_if_it_has_huge_gap_and_volume():
    df=bars(0);df.iloc[20]=[130,150,125,145,10000]
    assert build_event_setup_context(df,as_of=df.index[-1].date())['event'] is None


def test_hvc_reclaim_is_distinct_from_already_above():
    df=bars();df.iloc[-1]=[111,114,110,113,70]
    assert 'zurückerobert' in evaluate(df)['event']['hvc_status']
    df.iloc[-2]=[111,114,110,113,70]
    assert 'keine neue Rückeroberung' in evaluate(df)['event']['hvc_status']


def test_flag_break_uses_prior_band_not_breakout_candle():
    df=bars();df.iloc[-1]=[111,114,110,113,70];e=evaluate(df)['event']
    assert e['flag_high']==112 and e['flag_low']==109
    assert 'über Konsolidierungshoch' in e['flag_status']


@pytest.mark.parametrize('case', ['wide','gap_fill','event_gap_fill','missing_following'])
def test_unconfirmed_consolidation_is_not_flag(case):
    df=bars()
    if case=='wide':df.iloc[21:]=[111,121,105,113,190];expected='noch nicht eng'
    elif case=='gap_fill':df.iloc[-1]=[111,114,99,113,70];expected='Gap vollständig'
    elif case=='event_gap_fill':df.iloc[20,2]=99;expected='Gap vollständig'
    else:df.iloc[22,4]=float('nan');expected='unvollständig'
    assert expected in evaluate(df)['event']['flag_status']


def test_event_without_following_session_waits():
    e=evaluate(bars(0))['event']
    assert 'spätere Reaktion abwarten' in e['hvc_status'] and 'Noch keine' in e['flag_status']


def test_old_event_outside_twenty_day_window_is_not_shown():
    assert evaluate(bars(25))['event'] is None


@pytest.mark.parametrize('bad', ['missing_column','intraday','missing_latest','nondatetime'])
def test_unusable_daily_data_are_explicitly_unavailable(bad):
    df=bars()
    if bad=='missing_column':df=df.drop(columns='Volume')
    elif bad=='intraday':df.index=pd.date_range('2026-08-03 09:30',periods=len(df),freq='5min')
    elif bad=='missing_latest':df.iloc[-1,3]=float('nan')
    else:df.index=range(len(df))
    assert not build_event_setup_context(df,as_of='2026-12-01')['available']


def test_no_input_mutation_or_operative_decision_fields():
    df=bars();before=df.copy(deep=True);p=evaluate(df)
    pd.testing.assert_frame_equal(df,before)
    for obj in [p,p['event']]:assert not {'score','crv','stop_used','decision_state','buy_now'} & set(obj)


def test_rubric_renders_inside_existing_expander_with_streamlit():
    from streamlit.testing.v1 import AppTest
    script='''import streamlit as st
from modules.event_setup_context import build_event_setup_context, render_event_setup_context
import pandas as pd
idx=pd.bdate_range('2026-08-03', periods=26)
df=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Volume':100.}, index=idx)
df.iloc[20]=[110,112,109,111,200]
df.iloc[21:]=[110,112,109,110,50]
with st.expander('Technische Chartdetails'):
    render_event_setup_context(st,build_event_setup_context(df,as_of='2026-12-01'),ccy='USD')
'''
    at=AppTest.from_string(script).run(timeout=20)
    assert not at.exception
    assert any('High Volume Close' in e.value for e in at.markdown)
    assert any('keine wirkung' in e.value.lower() for e in at.caption)
