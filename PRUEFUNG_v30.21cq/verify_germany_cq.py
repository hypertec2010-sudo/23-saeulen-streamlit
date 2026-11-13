from types import SimpleNamespace
import pandas as pd
from modules import candidate_radar as r,provider_manager
from modules.radar_universe import get_universe_map,normalize_entries
from test_chart_patterns_cm import fixture,NOW

name='Deutschland · Aktienauswahl'
symbols=get_universe_map()[name][0]
assert len(symbols)==65 and len(set(symbols))==65
assert all(t.endswith('.DE') for t in symbols)
assert 'SAP.DE' in symbols and 'SAP' not in symbols
assert normalize_entries(symbols)==(symbols,[])
provider_manager.get_market_data_provider=lambda:SimpleNamespace(quarantine_state=lambda:None)
frame=fixture('double_top')['df']
for style in r.STYLES:
    calls=[];full=[]
    def history(ticker):calls.append(ticker);return frame
    def analyze(**kw):
        full.append(kw['ticker'])
        return dict(ticker=kw['ticker'],df=frame,price=99.2,confidence_info={'coverage':1.},valid_trade_setup=True)
    p=r.run_two_stage_scan(universe=name,style=style,entries=symbols,history_loader=history,
        analyze=analyze,decide=lambda *_:dict(grade='A',score=80,bucket='Jetzt prüfbar'),
        entry_package=lambda *_:dict(price=99.2,stop=97.,tp1=110.,crv=4.,entry_zone='99–100',entry_distance_pct=0.),
        full_analysis_limit=2,clock=lambda:pd.Timestamp(NOW))
    assert calls==symbols and r.snapshot_valid(p)
    assert len(full)==(0 if style=='Chartmuster' else 2)
    print(style+': 65 deutsche Ticker verarbeitet, Snapshot gültig, Vollanalysen '+str(len(full)))
from streamlit.testing.v1 import AppTest
app=AppTest.from_string('''import streamlit as st
from modules.candidate_radar_ui import render_candidate_radar
class Store:
    user_id='germany-verify'
    def load_namespace(self,*a,**k):return None
    def status(self):return {}
def forbidden(*a,**k):raise AssertionError('No scan by selecting a list')
render_candidate_radar(st,storage=Store(),scan=forbidden,catalog_loader=lambda:[],
    watchlists_loader=lambda:[],positions_loader=lambda:{},queue_watchlist=forbidden)
''').run(timeout=30)
assert not app.exception
app.selectbox(key='radar_v3021_universe').select(name).run()
assert not app.exception and app.selectbox(key='radar_v3021_universe').value==name
print('Echte Streamlit-Auswahl erfolgreich; Auswahl allein startet keinen Scan.')
