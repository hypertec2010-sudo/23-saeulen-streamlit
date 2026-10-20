import ast
from copy import deepcopy
from pathlib import Path
import pandas as pd
import pytest
from modules.watchlist_move import plan_move,execute_move,copy_start_metadata
from modules.storage.watchlist_repository import WatchlistRepository


def item(name,ticker='',**extra):
    return dict(Watchlist_Name=name,Watchlist_Type='Watchlist',Ticker=ticker,
                Added_At='2026-09-01',Alert_Mode='Standard',Check_Frequency='4x täglich',**extra)


class Storage:
    def __init__(self,rows):
        self.rows=deepcopy(rows);self.writes=0;self.fail=False
    def load_namespace(self,*a,**kw):return {'rows':deepcopy(self.rows)}
    def save_namespace(self,namespace,payload):
        self.writes+=1
        if self.fail:return False
        self.rows=deepcopy(payload['rows']);return True


def test_repository_one_write_preserves_dates_metadata_and_empty_source():
    rows=[item('A','MU',Note='keep me'),item('A','MRVL'),item('B')]
    rows[2].update(Check_Frequency='Nur manuell',Alert_Mode='Leise')
    store=Storage(rows);repo=WatchlistRepository(store)
    baseline={'A::MU':{'Ticker':'MU','Watchlist_Name':'A','Startkurs':90,
                       'Added_At':'2026-09-01','Startkurs_Quelle':'Historisch'}}
    before=deepcopy(baseline)
    def save(data):baseline.clear();baseline.update(deepcopy(data));return True
    ok,msg=repo.move_entries_to_watchlist(' a ','B',['mu','MRVL','MU'],
        load_baselines=lambda:deepcopy(baseline),save_baselines=save)
    assert ok and store.writes==1
    src=[r for r in store.rows if r['Watchlist_Name']=='A']
    assert len(src)==1 and src[0]['Ticker']==''
    moved=[r for r in store.rows if r['Watchlist_Name']=='B' and r['Ticker']]
    assert {r['Ticker'] for r in moved}=={'MU','MRVL'}
    assert all(r['Added_At']=='2026-09-01' and r['Check_Frequency']=='Nur manuell' for r in moved)
    assert next(r for r in moved if r['Ticker']=='MU')['Note']=='keep me'
    assert baseline['A::MU']==before['A::MU'] and baseline['B::MU']['Startkurs']==90
    assert baseline['B::MU']['Added_At']=='2026-09-01' and baseline['B::MU']['Watchlist_Name']=='B'
    assert all(not pd.isna(value) for r in store.rows for value in r.values())


def test_existing_target_is_not_overwritten_or_duplicated():
    rows=[item('A','MU',Note='source'),item('B','MU',Note='target')]
    original=deepcopy(rows);p=plan_move(pd.DataFrame(rows),'A','B',['MU'])
    target=p['frame'].query("Watchlist_Name=='B'").to_dict('records')
    assert target==[original[1]] and p['duplicates']==['MU'] and rows==original
    baseline={'A::MU':{'Startkurs':90},'B::MU':{'Startkurs':100}}
    assert copy_start_metadata(baseline,p)==baseline


@pytest.mark.parametrize('source,target,tickers',[
    ('A','a',['MU']),('A','B',[]),('Missing','B',['MU']),('A','Missing',['MU']),
    ('A','B',['MU','UNKNOWN'])])
def test_invalid_or_stale_selection_never_writes(source,target,tickers):
    store=Storage([item('A','MU'),item('B')]);repo=WatchlistRepository(store)
    ok,msg=repo.move_entries_to_watchlist(source,target,tickers,
        load_baselines=lambda:{},save_baselines=lambda d:True)
    assert not ok and store.writes==0


@pytest.mark.parametrize('position_list',['A','B'])
def test_positions_are_not_implicitly_moved(position_list):
    rows=[item('A','MU'),item('B')]
    next(r for r in rows if r['Watchlist_Name']==position_list)['Watchlist_Type']='Positions-Watchlist'
    with pytest.raises(ValueError,match='Depotpositionen'):plan_move(pd.DataFrame(rows),'A','B',['MU'])


def test_baseline_write_failure_prevents_membership_change():
    store=Storage([item('A','MU'),item('B')]);repo=WatchlistRepository(store)
    ok,msg=repo.move_entries_to_watchlist('A','B',['MU'],
        load_baselines=lambda:{'A::MU':{'Startkurs':90}},save_baselines=lambda d:False)
    assert not ok and store.writes==0 and 'Startkursdaten' in msg


def test_unconfirmed_membership_write_keeps_source_baseline_and_membership():
    original=[item('A','MU'),item('B')];store=Storage(original);store.fail=True
    baseline={'A::MU':{'Startkurs':90}};repo=WatchlistRepository(store)
    def save(data):baseline.update(deepcopy(data));return True
    ok,msg=repo.move_entries_to_watchlist('A','B',['MU'],
        load_baselines=lambda:deepcopy(baseline),save_baselines=save)
    assert not ok and store.rows==original
    assert baseline['A::MU']['Startkurs']==90 and baseline['B::MU']['Startkurs']==90
    assert 'nicht bestätigt' in msg


def test_existing_archived_target_baseline_kept_and_repeated_move_does_not_write():
    store=Storage([item('A','MU'),item('B')]);repo=WatchlistRepository(store)
    metadata={'A::MU':{'Startkurs':90},'B::MU':{'Startkurs':80}}
    def forbidden(*a):raise AssertionError('should not overwrite target metadata')
    ok,_=repo.move_entries_to_watchlist('A','B',['MU'],load_baselines=lambda:metadata,save_baselines=forbidden)
    assert ok and store.writes==1
    ok,_=repo.move_entries_to_watchlist('A','B',['MU'],load_baselines=lambda:metadata,save_baselines=forbidden)
    assert not ok and store.writes==1


@pytest.mark.parametrize('fail',[False,True])
def test_actual_google_sheets_path_uses_one_update_without_clear(fail):
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'logging_utils.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='move_entries_to_watchlist')
    frame=pd.DataFrame([item('A','MU',Note='preserved'),item('B')]).fillna('')
    original=[list(frame.columns)]+frame.astype(str).values.tolist()
    class Worksheet:
        writes=0
        values=deepcopy(original)
        def get_all_values(self):return deepcopy(self.values)
        def update(self,**kwargs):
            self.writes+=1
            assert kwargs['range_name']=='A1' and kwargs['value_input_option']=='RAW'
            if fail:raise RuntimeError('unavailable')
            self.values=deepcopy(kwargs['values'])
    ws=Worksheet()
    scope={'pd':pd,'open_log_spreadsheet':lambda:(object(),None),
           'get_or_create_worksheet':lambda *a,**kw:ws}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_sheets_move','exec'),scope)
    ok,msg=scope['move_entries_to_watchlist']('A','B',['MU'],
        load_baselines=lambda:{},save_baselines=lambda d:True)
    assert ok is not fail and ws.writes==1
    if fail:assert ws.values==original
    else:
        moved=pd.DataFrame(ws.values[1:],columns=ws.values[0])
        assert moved.query("Watchlist_Name=='B' and Ticker=='MU'").iloc[0]['Note']=='preserved'


def test_actual_streamlit_controls_offer_target_and_execute_selected_batch():
    from streamlit.testing.v1 import AppTest
    script='''
import streamlit as st
import pandas as pd
from modules.watchlist_move import render_move_controls,execute_move
if 'rows' not in st.session_state:
    st.session_state.rows=[{'Watchlist_Name':name,'Watchlist_Type':'Watchlist','Ticker':ticker}
                          for name,ticker in [('A','MU'),('A','MRVL'),('B','')]]
def save(frame):
    st.session_state.rows=frame.to_dict('records');return True
def move(source,target,tickers):
    return execute_move(source,target,tickers,
        load_rows=lambda:(pd.DataFrame(st.session_state.rows),None),save_rows=save,
        load_baselines=lambda:{},save_baselines=lambda d:True)
render_move_controls(st,frame=pd.DataFrame(st.session_state.rows),source='A',
    selected=['MU','MRVL'],move=move,on_success=st.success)
'''
    at=AppTest.from_string(script).run(timeout=20)
    assert not at.exception and at.selectbox[0].options==['B']
    at.button[0].click().run()
    assert not at.exception and len(at.success)==1
    result=pd.DataFrame(at.session_state.rows)
    assert set(result.query("Watchlist_Name=='B'")['Ticker'])=={'','MU','MRVL'}


def test_manage_and_screener_controls_can_render_together_without_duplicate_widgets():
    from streamlit.testing.v1 import AppTest
    script='''
import streamlit as st
import pandas as pd
from modules.watchlist_move import render_move_controls
frame=pd.DataFrame([{'Watchlist_Name':n,'Watchlist_Type':'Watchlist','Ticker':t}
                    for n,t in [('A','MU'),('B',''),('Depot','MU')]])
frame.loc[frame['Watchlist_Name']=='Depot','Watchlist_Type']='Positions-Watchlist'
for prefix in ['watchlist_move','screener_move']:
    render_move_controls(st,frame=frame,source='A',selected=['MU'],
        move=lambda *a:(True,'ok'),on_success=st.success,key_prefix=prefix)
'''
    at=AppTest.from_string(script).run(timeout=20)
    assert not at.exception and len(at.selectbox)==2
    assert all(box.options==['B'] for box in at.selectbox)


def test_actual_success_callback_clears_selection_and_reruns_after_widget_creation():
    from streamlit.testing.v1 import AppTest
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'legacy_app.py').read_text())
    callback=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_move_success_v3021br')
    script='''
import streamlit as st
import pandas as pd
selected_watchlist_name='A'
_selection_key_v3021aw='watchlist_remove_selection_v3021aw::A'
def trigger_ui_refresh():st.rerun()
notice=st.session_state.pop('watchlist_move_notice_v3021br',None)
if notice:st.success(notice)
st.data_editor(pd.DataFrame({'Ticker':['MU'],'Auswahl':[True]}),key='watchlist_remove_editor_v3021aw::A')
'''+ast.unparse(callback)+'''
if st.button('Verschieben'):_move_success_v3021br('Erfolgreich verschoben')
'''
    at=AppTest.from_string(script).run(timeout=20)
    at.button[0].click().run()
    assert not at.exception and at.success[0].value=='Erfolgreich verschoben'
