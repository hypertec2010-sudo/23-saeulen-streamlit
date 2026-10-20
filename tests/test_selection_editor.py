"""Exercise real data_editor edit events across successive script reruns."""
import ast
import json
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize('area,column', [('manage','Auswahl'),('screener','🔄')])
def test_first_click_multiple_rows_deselect_and_unrelated_rerun(area,column):
    # Use the actual preparation and data_editor call from each application area.
    source=Path(__file__).resolve().parents[1].joinpath('legacy_app.py').read_text()
    tree=ast.parse(source)
    target='_edited_selection_v3021aw' if area=='manage' else '_edited_live_display_v3021e'
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id==target for t in n.targets))
    call=ast.unparse(node)
    script=f'''
import streamlit as st
import pandas as pd
from modules.selection_editor import prepare_selection_editor
selected_watchlist_name='A'
_selection_key_v3021aw='selection'
_selection_state_key_v3021e='selection'
_partial_widget_suffix_v3021e='test'
revision=st.session_state.get('revision',0)
selected=st.session_state.get('selection',[])
rows=['MU','MRVL','ANET']
if st.session_state.get('reverse',False):rows.reverse()
frame=pd.DataFrame({{'{column}':[t in selected for t in rows],'Ticker':rows}})
frame, key=prepare_selection_editor(st.session_state,frame,selection_column='{column}',
    state_key='selection',widget_key='editor',revision=revision,
    reset='selection' not in st.session_state)
st.session_state['active_key']=key
_selection_df_v3021aw=frame
_desktop_rescan_editor_v3021e=frame
_selection_editor_key_v3021bs=key
_live_editor_key_v3021bs=key
{call}
result={target}
st.session_state['selection']=result.loc[result['{column}'],'Ticker'].tolist()
st.write(','.join(st.session_state['selection']))
'''
    at=AppTest.from_string(script).run(timeout=20)
    assert not at.exception
    browser_edits={}
    def browser_run(changes=None):
        # AppTest has no data_editor interaction API. Send the same JSON widget
        # event as the browser, retaining it for unrelated subsequent reruns.
        widget_id=at.dataframe[0].proto.id
        if changes is not None:browser_edits[widget_id]=changes
        ws=WidgetStates()
        ws.widgets.add(id=widget_id,string_value=json.dumps({
            'edited_rows':browser_edits.get(widget_id,{}),'added_rows':[],'deleted_rows':[]}))
        at._run(ws,timeout=20)
    def edit(changes,expected):
        key=at.session_state['active_key']
        browser_run(changes)
        assert not at.exception
        assert at.session_state['selection']==expected
        return key
    first=edit({0:{column:True}},['MU'])
    browser_run()  # Old implementation reverted the checkbox on this run.
    assert at.session_state['selection']==['MU'] and at.session_state['active_key']==first
    edit({0:{column:True},1:{column:True}},['MU','MRVL'])
    edit({0:{column:False},1:{column:True}},['MRVL'])
    at.session_state['reverse']=True
    browser_run()
    assert at.session_state['selection']==['MRVL']
    assert at.session_state['active_key']!=first
    edit({0:{column:True}},['ANET','MRVL'])
    # Programmatic green/clear commands use a fresh revision.
    at.session_state['selection']=['MU']
    at.session_state['revision']=1
    browser_run()
    assert at.session_state['selection']==['MU']
    at.session_state['selection']=[]
    at.session_state['revision']=2
    browser_run()
    assert at.session_state['selection']==[]
    edit({0:{column:True}},['ANET'])
    # Successful manage move/removal deletes the selection state.
    del at.session_state['selection']
    browser_run()
    assert at.session_state['selection']==[]
