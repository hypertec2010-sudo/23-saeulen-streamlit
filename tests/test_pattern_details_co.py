from copy import deepcopy
import pytest
from modules.chart_patterns import build_chart_patterns
from modules.chart_pattern_ui import pattern_detail, pattern_figure
from test_chart_patterns_cm import fixture,NOW


@pytest.mark.parametrize('kind',['bull_flag','bear_flag','head_shoulders','inverse_head_shoulders','double_bottom','double_top'])
def test_directional_confirmation_level_from_saved_evidence(kind):
    package=build_chart_patterns(fixture(kind,'forming'),now=NOW)
    pattern=next(p for p in package['patterns'] if p['kind']==kind)
    before=deepcopy(package)
    d=pattern_detail(package,pattern)
    assert d['threshold']==pytest.approx(pattern['boundary_now']+pattern['direction']*pattern['confirmation_buffer'])
    assert d['distance_pct']==pytest.approx((d['threshold']-d['close'])*pattern['direction']/d['close']*100)
    assert 'abwarten' in d['action'] and package==before


def test_stale_target_touched_and_late_break_are_explained():
    p=build_chart_patterns(fixture('double_top','confirmed'),now=NOW)
    v=next(x for x in p['patterns'] if x['kind']=='double_top')
    p['stale']=True;v['projection_touched']=True;v['break_before_known']=True
    action=pattern_detail(p,v)['action']
    assert 'neu scannen' in action and 'bereits berührt' in action and 'Pivot' in action


def test_invalid_and_missing_evidence_no_entry_manufactured():
    p=build_chart_patterns(fixture('double_top','invalid'),now=NOW)
    v=next(x for x in p['patterns'] if x['kind']=='double_top')
    assert 'verwerfen' in pattern_detail(p,v)['action']
    d=pattern_detail({},dict(phase='none',direction=-1))
    assert d['threshold'] is None and d['distance_pct'] is None


def test_focus_larger_labels_and_toggle_levels_do_not_change_evidence():
    p=build_chart_patterns(fixture('head_shoulders','confirmed'),now=NOW)
    v=next(x for x in p['patterns'] if x['kind']=='head_shoulders');before=deepcopy(p)
    focus=pattern_figure(p,v);full=pattern_figure(p,v,window='Gesamter Beleg')
    assert len(focus.data[0].x)<len(full.data[0].x)
    assert focus.layout.height==680 and focus.layout.xaxis.type=='category'
    assert focus.layout.yaxis.fixedrange is False and 'text' in focus.data[1].mode
    assert len(focus.layout.shapes)==2
    simple=pattern_figure(p,v,window='60',show_levels=False,show_labels=False)
    assert len(simple.data[0].x)==60 and not simple.layout.shapes
    assert simple.data[1].mode=='lines+markers' and p==before


def test_ui_controls_and_all_hit_details():
    from streamlit.testing.v1 import AppTest
    p=build_chart_patterns(fixture('double_top','confirmed'),now=NOW)
    payload=dict(rows=[dict(ticker=t,chart_patterns=dict(p,ticker=t)) for t in ['HALO','AAPL']],processed=2,requested=2,errors=[])
    script=f'''import streamlit as st
from modules.radar_patterns import render_pattern_radar
render_pattern_radar(st,{payload!r},fresh=True,limit=1,navigate=lambda st,x:None)
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    app.selectbox(key='radar_patterns_detail_ticker').select('HALO').run()
    assert app.selectbox(key='radar_pattern_HALO_window').value=='Musterfokus'
    app.selectbox(key='radar_pattern_HALO_window').select('60').run()
    app.checkbox(key='radar_pattern_HALO_levels').uncheck().run()
    assert not app.exception and any('Bestätigungsschwelle' in str(x.value) for x in app.table)
    assert len(app.get('download_button'))==2


def test_crossed_threshold_with_missing_volume_is_not_described_as_unbroken():
    r=fixture('double_bottom','confirmed');r['df']['Volume']=float('nan')
    p=build_chart_patterns(r,now=NOW)
    v=next(x for x in p['patterns'] if x['kind']=='double_bottom')
    d=pattern_detail(p,v)
    assert v['phase']=='forming' and d['distance_pct']<0
    assert 'bereits jenseits' in d['action'] and 'Bestätigung fehlt' in d['action']
