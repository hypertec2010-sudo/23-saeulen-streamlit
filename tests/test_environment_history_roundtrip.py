from datetime import datetime
from types import SimpleNamespace
import json
import pandas as pd
import pytest
from modules import live_monitor as monitor
from modules.live_screener_snapshot import dataframe_to_payload, dataframe_from_payload
from modules.environment_context import attach_environment_columns
from test_environment_context import META, row


@pytest.mark.parametrize('ticker,sector,industry,group,state,icon',[
    ('HALO','Healthcare','Biotechnology','XBI','plan','🟡'),
    ('ACN','Technology','Information Technology Services','XLK','plan','🟡'),
    ('DHR','Healthcare','Diagnostics & Research','XLV','observe','⚪'),
    ('LLY','Healthcare','Drug Manufacturers - General','XLV','observe','⚪')])
def test_environment_survives_real_history_json_snapshot_and_render(monkeypatch,ticker,sector,industry,group,state,icon):
    class State(dict):
        __getattr__=dict.__getitem__
        __setattr__=dict.__setitem__
    monkeypatch.setattr(monitor,'st',SimpleNamespace(session_state=State()))
    monkeypatch.setattr(monitor,'get_current_berlin_time',lambda:datetime(2026,10,19,12),raising=False)
    monkeypatch.setattr(monitor,'_v227_load_persistent_live_history',lambda:({},[]))
    monkeypatch.setattr(monitor,'_v227_save_persistent_live_history',lambda *a:None)
    monkeypatch.setattr(monitor,'_v2416_log_event',lambda **kw:None,raising=False)
    stock={'date':'2026-10-16','prior_date':'2026-10-15','return_1d':1.}
    raw=pd.DataFrame([{'Ticker':ticker,'Kurs':100.,'Ampel':icon,'Status':state,
         'Live-Score':'71/100','Setup-Stärke':'78/100','Freigabe-Check':'Originaler Check',
         '__decision_state_v3021az':state,'__entry_hard_gate':False,'__invalidated':False,
         '__env_sector':sector,'__env_industry':industry,'__env_stock':stock,
         '__temporary_sort':123,'__env_debug':'must not persist'}])
    history,_=monitor.apply_live_watchlist_status_history_v220(raw,watchlist_name='Regression')
    assert history.iloc[0]['__env_sector']==sector
    assert history.iloc[0]['__env_industry']==industry
    assert history.iloc[0]['__env_stock']==stock
    assert '__temporary_sort' not in history and '__env_debug' not in history
    payload=json.loads(json.dumps(dataframe_to_payload(history)))
    restored=dataframe_from_payload(payload)
    radar=pd.DataFrame([row('SPY',**{'Perf 1T %':-1.1,'Perf 5T %':-2.}),
                        row('XLV'),row('XLK'),row('XBI')])
    rendered=attach_environment_columns(restored,radar,META,now='2026-10-19T10:07:00Z')
    assert rendered.iloc[0]['Umfeldrisiko']=='🟡 Erhöht'
    assert 'Teilbasis' not in rendered.iloc[0]['Umfeld-Grund']
    for key in ['Ampel','Status','Live-Score','Setup-Stärke','Freigabe-Check']:
        assert rendered.iloc[0][key]==raw.iloc[0][key]
    # The specific industry (HALO) or broad sector proxy (others) is retained
    # through the same path that previously discarded this metadata.
    from modules.environment_context import resolve_groups
    groups=resolve_groups({'ticker':ticker,'sector':restored.iloc[0]['__env_sector'],
                           'industry':restored.iloc[0]['__env_industry']})
    assert (groups['industry'] or groups['sector'])==group
