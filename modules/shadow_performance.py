from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

STORE = Path('shadow_performance_v287.json')
HORIZONS = (1,3,5,10,20)

def _first(row, names, default=None):
    for n in names:
        if n in row and pd.notna(row.get(n)) and str(row.get(n)).strip() not in ('','-','nan','None'):
            return row.get(n)
    return default

def _num(v):
    try:
        s=str(v).replace('%','').replace(',','.').strip()
        return float(s)
    except Exception: return None

def _load():
    try:
        x=json.loads(STORE.read_text(encoding='utf-8'))
        return x if isinstance(x,list) else []
    except Exception: return []

def _save(rows):
    try: STORE.write_text(json.dumps(rows,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    except Exception: pass

def _direction(live, shadow):
    rank={'🔴':0,'⚪':1,'🟡':2,'🟢':3}
    a,b=rank.get(str(live),1),rank.get(str(shadow),1)
    return 'Aufwertung' if b>a else ('Abwertung' if b<a else 'Unverändert')

def sync_events(shadow_df):
    rows=_load(); seen={r.get('id') for r in rows}
    if not isinstance(shadow_df,pd.DataFrame) or shadow_df.empty: return pd.DataFrame(rows)
    for _,r in shadow_df.iterrows():
        d=r.to_dict()
        ticker=str(_first(d,['Ticker','ticker','Symbol'],'')).upper().strip()
        ts=str(_first(d,['Zeit','Zeitpunkt','Timestamp','timestamp','Datum','date','ts'],''))
        live=str(_first(d,['Live-Ampel','Live Ampel','Ampel','live_ampel'],'-'))
        shadow=str(_first(d,['Shadow-Ampel','Shadow Ampel','Engine-Ampel','shadow_ampel'],'-'))
        price=_num(_first(d,['Kurs','Preis','Price','price','Event-Kurs','event_price']))
        if not ticker or live=='-' or shadow=='-' or live==shadow: continue
        if not ts: ts=datetime.now().isoformat(timespec='seconds')
        eid=f'{ticker}|{ts}|{live}|{shadow}'
        if eid in seen: continue
        rows.append({'id':eid,'ticker':ticker,'event_ts':ts,'live':live,'shadow':shadow,'richtung':_direction(live,shadow),'event_price':price,**{f'r{h}':None for h in HORIZONS}})
        seen.add(eid)
    _save(rows); return pd.DataFrame(rows)

def _event_date(v):
    try: return pd.to_datetime(v,errors='coerce').tz_localize(None)
    except Exception: return pd.NaT

def refresh_forward_returns(events, provider):
    rows=events.to_dict('records') if isinstance(events,pd.DataFrame) else list(events or [])
    by={}
    for r in rows: by.setdefault(str(r.get('ticker','')),[]).append(r)
    for ticker,items in by.items():
        if not ticker: continue
        try:
            hist=provider.get_history(ticker,period='6mo',auto_adjust=True)
            if hist is None or len(hist)<2: continue
            h=hist.copy(); h.index=pd.to_datetime(h.index).tz_localize(None)
            close=h['Close'].dropna()
            if close.empty: continue
            for r in items:
                dt=_event_date(r.get('event_ts'))
                if pd.isna(dt): continue
                pos=close.index.searchsorted(dt.normalize(),side='left')
                if pos>=len(close): continue
                base=r.get('event_price')
                if base is None or float(base)<=0: base=float(close.iloc[pos])
                for n in HORIZONS:
                    j=pos+n
                    if j<len(close): r[f'r{n}']=round((float(close.iloc[j])/float(base)-1)*100,3)
        except Exception:
            continue
    _save(rows); return pd.DataFrame(rows)

def build_dashboard(events):
    if not isinstance(events,pd.DataFrame) or events.empty: return pd.DataFrame(),pd.DataFrame()
    df=events.copy()
    out=[]
    for direction,g in df.groupby('richtung'):
        row={'Shadow-Richtung':direction,'Events':len(g)}
        for h in HORIZONS:
            vals=pd.to_numeric(g.get(f'r{h}'),errors='coerce').dropna()
            row[f'{h}T Ø']=('n/a' if vals.empty else f'{vals.mean():+.2f}%')
            row[f'{h}T Treffer']=('n/a' if vals.empty else f'{((vals>0).mean()*100):.0f}%')
        out.append(row)
    detail=df.rename(columns={'ticker':'Ticker','event_ts':'Event','live':'Live','shadow':'Shadow','richtung':'Richtung','event_price':'Event-Kurs',**{f'r{h}':f'{h}T %' for h in HORIZONS}})
    cols=['Ticker','Event','Live','Shadow','Richtung','Event-Kurs']+[f'{h}T %' for h in HORIZONS]
    return pd.DataFrame(out),detail[[c for c in cols if c in detail.columns]]
