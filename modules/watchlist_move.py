"""Move normal watchlist entries with one membership write, preserving metadata."""
from copy import deepcopy
import pandas as pd

FIELDS=['Watchlist_Name','Watchlist_Type','Ticker','Added_At','Alert_Mode','Check_Frequency']


def text(value):
    return '' if value is None or (not isinstance(value,(dict,list)) and pd.isna(value)) else str(value).strip()


def plan_move(frame, source, target, tickers):
    source=text(source);target=text(target)
    selected=list(dict.fromkeys(text(t).upper().replace(' ','') for t in tickers or [] if text(t)))
    if not source or not target or source.casefold()==target.casefold() or not selected:
        raise ValueError('Bitte unterschiedliche Quell-/Ziel-Watchlisten und mindestens einen Wert auswählen.')
    if not isinstance(frame,pd.DataFrame) or any(c not in frame for c in ['Watchlist_Name','Watchlist_Type','Ticker']):
        raise ValueError('Watchlist-Daten sind unvollständig; nichts verschoben.')
    rows=frame.fillna('').to_dict('records')
    group=lambda name:[r for r in rows if text(r.get('Watchlist_Name')).casefold()==name.casefold()]
    src=group(source);dst=group(target)
    if not src or not dst:raise ValueError('Quell- oder Ziel-Watchlist existiert nicht mehr.')
    if any(text(r.get('Watchlist_Type'))!='Watchlist' for r in src+dst):
        raise ValueError('Diese Funktion verschiebt normale Watchlisten-Einträge; Depotpositionen bitte in der Positionsverwaltung bearbeiten.')
    source=text(src[0]['Watchlist_Name']);target=text(dst[0]['Watchlist_Name'])
    norm=lambda r:text(r.get('Ticker')).upper().replace(' ','')
    available={norm(r) for r in src}
    absent=[t for t in selected if t not in available]
    if absent:raise ValueError('Nicht mehr in der Quelle vorhanden: '+', '.join(absent)+'. Bitte Auswahl neu laden.')
    existing={norm(r) for r in dst}
    duplicate=[t for t in selected if t in existing]
    result=[dict(r) for r in rows if not (text(r.get('Watchlist_Name')).casefold()==source.casefold() and norm(r) in selected)]
    for ticker in selected:
        if ticker in existing:continue
        item=dict(next(r for r in src if norm(r)==ticker))
        item.update(Watchlist_Name=target,Watchlist_Type='Watchlist',Ticker=ticker,
                    Alert_Mode=dst[0].get('Alert_Mode') or 'Standard',
                    Check_Frequency=dst[0].get('Check_Frequency') or '4x täglich')
        result.append(item);existing.add(ticker)
    if not any(text(r.get('Watchlist_Name')).casefold()==source.casefold() for r in result):
        empty={key:src[0].get(key,'') for key in FIELDS};empty['Ticker']=''
        result.append(empty)
    return {'frame':pd.DataFrame(result,columns=frame.columns).fillna(''),'source':source,'target':target,
            'tickers':selected,'duplicates':duplicate,
            'added':[t for t in selected if t not in duplicate]}


def copy_start_metadata(store, plan):
    copied=deepcopy(store or {})
    for ticker in plan['added']:
        source_key=next((k for k in copied if str(k).casefold()==f"{plan['source']}::{ticker}".casefold()),None)
        target_key=next((k for k in copied if str(k).casefold()==f"{plan['target']}::{ticker}".casefold()),None)
        if source_key is None or target_key is not None or not isinstance(copied[source_key],dict):continue
        item=deepcopy(copied[source_key]);item.update(Watchlist_Name=plan['target'],Ticker=ticker)
        copied[f"{plan['target']}::{ticker}"]=item
    return copied


def execute_move(source,target,tickers,*,load_rows,save_rows,load_baselines,save_baselines):
    try:
        frame,error=load_rows()
        if error:return False,str(error)
        plan=plan_move(frame,source,target,tickers)
        original=load_baselines()
        if not isinstance(original,dict):return False,'Startkursdaten konnten nicht vollständig gelesen werden; nichts verschoben.'
        metadata=copy_start_metadata(original,plan)
        # Copy before membership change, retaining source history. A failed or
        # ambiguous membership write cannot lose the original baseline.
        if metadata!=original and not save_baselines(metadata):
            return False,'Startkursdaten konnten nicht gespeichert werden; Watchlist-Einträge nicht verschoben.'
        if not save_rows(plan['frame']):
            return False,'Verschieben nicht bestätigt. Bitte Listen neu laden; Startkursdaten bleiben erhalten.'
        message=f"{len(plan['tickers'])} Wert(e) von '{plan['source']}' nach '{plan['target']}' verschoben."
        if plan['duplicates']:message+=f" {len(plan['duplicates'])} bereits im Ziel vorhanden; Zieleinträge unverändert behalten."
        return True,message
    except Exception as exc:return False,f'Verschieben nicht bestätigt: {exc}'


def render_move_controls(st, *, frame, source, selected, move, on_success, key_prefix='watchlist_move'):
    if not isinstance(frame,pd.DataFrame) or frame.empty:return
    own=frame.loc[frame['Watchlist_Name'].map(lambda x:text(x).casefold()==text(source).casefold())]
    if own.empty or any(text(v)!='Watchlist' for v in own['Watchlist_Type']):return
    targets=sorted({text(r['Watchlist_Name']) for r in frame.to_dict('records')
                    if text(r.get('Watchlist_Type'))=='Watchlist' and text(r.get('Watchlist_Name')).casefold()!=text(source).casefold()},key=str.casefold)
    st.markdown('**Ausgewählte Werte verschieben**')
    if not targets:
        st.caption('Lege eine weitere normale Watchlist als Ziel an.')
        return
    target=st.selectbox('Ziel-Watchlist',targets,key=f'{key_prefix}_target::{source}')
    valid=False
    if selected:
        try:
            plan=plan_move(frame,source,target,selected);valid=True
            st.caption(f"{len(selected)} ausgewählt · {len(plan['duplicates'])} bereits im Ziel. Aufnahmedatum und vorhandener Startkurs bleiben erhalten; bestehende Zieleinträge werden nicht überschrieben.")
        except ValueError as exc:st.warning(str(exc))
    else:st.caption('Häkchen oben vor den zu verschiebenden Werten setzen.')
    if st.button(f'Nach {target} verschieben ({len(selected)})',disabled=not valid,
                 key=f'{key_prefix}_apply::{source}',use_container_width=True):
        ok,message=move(source,target,list(selected))
        if ok:on_success(message)
        else:st.error(message)
