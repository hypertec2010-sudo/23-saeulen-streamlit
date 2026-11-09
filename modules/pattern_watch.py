"""Saved formation monitoring. Frozen evidence; no radar, score or order path."""
from copy import deepcopy
from hashlib import sha256
import json
import math

import pandas as pd
from modules import chart_patterns as cp
from modules.trade_watch import utc, number, market_for, clean_minutes, in_session, session_bounds
from modules.support_map import build_support_map
from modules.watch_setups import completed_day

NAMESPACE='pattern_watch_v1'
TERMINAL={'invalidated','expired','superseded'}
LABELS={'forming':'Entsteht','near':'Nahe an der Schwelle','intraday':'Schwelle intraday überschritten',
        'confirmed':'Tagesmuster bestätigt','invalidated':'Muster verworfen',
        'expired':'Beobachtung abgelaufen','superseded':'Beleg überholt / Datenbasis geändert'}


def identity(package,pattern):
    proof={k:pattern.get(k) for k in ['kind','invalidation','confirmation_buffer','structure_known_at']}
    proof['points']=[{k:p.get(k) for k in ['date','price','role']} for p in pattern.get('points',[])]
    proof['boundary']={k:(pattern.get('boundary') or {}).get(k) for k in ['anchor_date','value','slope']}
    return sha256(json.dumps([package.get('ticker'),package.get('policy'),proof],sort_keys=True).encode()).hexdigest()[:24]


def load(manager):
    owner=str(getattr(manager,'user_id','') or '')
    if not owner:raise ValueError('Nutzerzuordnung fehlt.')
    payload=manager.load_namespace(NAMESPACE,default=None)
    if payload is None:return []
    if not isinstance(payload,dict) or payload.get('owner')!=owner or payload.get('schema')!=1 or not isinstance(payload.get('items'),list):
        raise ValueError('Gespeicherte Musterliste nicht eindeutig lesbar; nicht überschreiben.')
    required={'id','ticker','name','state','source','original','created_at','expires_at','market'}
    if any(not isinstance(i,dict) or not required<=i.keys() or not isinstance(i.get('source'),dict) or not isinstance(i.get('original'),dict) or i['source'].get('policy')!=cp.POLICY for i in payload['items']):
        raise ValueError('Musterliste enthält unvollständige Einträge; nicht überschreiben.')
    return deepcopy(payload['items'])


def save(manager,items):
    owner=str(getattr(manager,'user_id','') or '')
    if not owner:raise ValueError('Nutzerzuordnung fehlt.')
    payload=dict(schema=1,owner=owner,items=deepcopy(items))
    json.dumps(payload,allow_nan=False)
    if not manager.save_namespace(NAMESPACE,payload):raise RuntimeError('Musterliste nicht gespeichert; vorheriger Stand bleibt maßgeblich.')


def active(item):return not item.get('archived') and not item.get('paused') and item.get('state') not in TERMINAL


def event(item,kind,now,text):
    item.setdefault('events',[]).append(dict(at=utc(now).isoformat(),kind=kind,text=text))
    item['events']=item['events'][-100:]


def create(items,package,pattern,*,now=None,days=14):
    current=utc(now)
    if package.get('policy')!=cp.POLICY or not package.get('available') or package.get('stale'):
        raise ValueError('Frischer, vollständiger Musterbeleg erforderlich.')
    if pattern not in package.get('patterns',[]) or pattern.get('phase') not in {'forming','confirmed'} or pattern.get('projection_touched'):
        raise ValueError('Nur entstehende oder bestätigte, noch nicht verbrauchte Muster vormerken.')
    ticker=package.get('ticker');market=market_for(ticker)
    if package.get('basis_date')!=completed_day(current,market).isoformat():
        raise ValueError('Tagesbasis nicht mehr aktuell; zuerst neu scannen.')
    ident=identity(package,pattern)
    if any(i.get('id')==ident and not i.get('archived') for i in items):raise ValueError('Diese Formation wird bereits beobachtet.')
    if len({i['ticker'] for i in items if active(i)}|{ticker})>30 or sum(active(i) for i in items)>=100:
        raise ValueError('Höchstens 30 aktive Aktien und 100 aktive Muster.')
    if days not in {7,14,30}:raise ValueError('Ungültige Beobachtungsdauer.')
    if any(number(pattern.get(k)) is None for k in ['boundary_now','confirmation_buffer','invalidation','projection']):
        raise ValueError('Gespeicherte Musterregeln unvollständig.')
    out=deepcopy(items)
    # An archived instance is retained; a deliberate new observation gets its own ID.
    if any(i.get('id')==ident for i in items):ident=ident+'-'+current.strftime('%Y%m%d%H%M%S%f')
    item=dict(id=ident,ticker=ticker,market=market,kind=pattern['kind'],name=pattern['name'],
        created_at=current.isoformat(),expires_at=(current+pd.Timedelta(days=days)).isoformat(),
        state=pattern['phase'],paused=False,archived=False,source=deepcopy(package),
        original=deepcopy(pattern),current=deepcopy(pattern),basis_date=package['basis_date'],
        last_attempt_at=None,last_checked_at=None,quote_at=None,price=None,health='Noch nicht aktualisiert',
        events=[])
    event(item,'created',current,'Formation mit festen Strukturpunkten und Regeln vorgemerkt. Keine Handelsfreigabe.')
    out.append(item);return out


def _data(frame,context):
    cutoff=pd.Timestamp(context['basis_date']).date()
    f=frame.loc[[d<=cutoff for d in frame.index.date]].tail(520).copy()
    data={k:pd.to_numeric(f[k],errors='coerce').astype(float).tolist() for k in ['Open','High','Low','Close']}
    data['Volume']=pd.to_numeric(f.Volume,errors='coerce').astype(float).tolist() if 'Volume' in f else [float('nan')]*len(f)
    data['date']=[i.date().isoformat() for i in f.index]
    tr=[max(data['High'][i]-data['Low'][i],abs(data['High'][i]-data['Close'][i-1]),abs(data['Low'][i]-data['Close'][i-1])) if i else data['High'][i]-data['Low'][i] for i in range(len(f))]
    data['ATR']=[sum(tr[max(0,i-13):i+1])/len(tr[max(0,i-13):i+1]) for i in range(len(f))]
    return data


def evaluate(item,frame,minutes=None,*,now=None):
    out=deepcopy(item);current=utc(now)
    if not active(out):return out
    if current>=utc(out['expires_at']):
        out.update(state='expired',health='Beobachtungsdauer abgelaufen. Für neue Regeln neuen Beleg vormerken.');return out
    context=build_support_map(dict(ticker=out['ticker'],df=frame),now=current)
    if not context.get('available') or context.get('stale'):
        out.update(health='Tagesprüfung nicht aktuell: '+context.get('reason','Daten fehlen'),data_blocked=True,price=None,quote_at=None)
        return out
    data=_data(frame,context);indices={d:i for i,d in enumerate(data['date'])}
    original=out['original'];source=out['source'];line=original['boundary']
    required=[line['anchor_date'],original['structure_known_at']]+[p['date'] for p in original['points']]
    if any(d not in indices for d in required):
        out.update(state='superseded',health='Ursprüngliche Struktur nicht mehr vollständig in der Historie.');return out
    # Never silently rebase frozen rules after a split, adjustment or OHLC revision.
    for b in source.get('chart') or []:
        if b['date'] not in indices:continue
        i=indices[b['date']]
        if any(not math.isclose(data[k][i],b[k.lower()],rel_tol=1e-6,abs_tol=1e-6) for k in ['Open','High','Low','Close']):
            out.update(state='superseded',health='Historische Kursbasis geändert. Neue Formation bewusst prüfen.');return out
    anchor=indices[line['anchor_date']];known=indices[original['structure_known_at']]
    end_date=source['chart'][-1]['date'] if not original.get('points') else max(p['date'] for p in original['points'])
    end=indices[end_date]
    original_target_index=original.get('known_index',original['end_index'])
    if original.get('confirmed_at'):
        old_dates={b['date']:b['index'] for b in source.get('chart') or []}
        original_target_index=old_dates.get(original['confirmed_at'],original_target_index)
    old_line=line['value']+line['slope']*(original_target_index-line['anchor_index'])
    height=abs(original['projection']-old_line)
    evidence=original.get('evidence') or {}
    base_ok=True
    if 'flag' in original['kind']:
        base_ok=(number(evidence.get('impulse_volume_ratio')) is not None and evidence['impulse_volume_ratio']>=1.2 and
                 number(evidence.get('flag_volume_ratio')) is not None and evidence['flag_volume_ratio']<=.8)
    points=[dict(p,index=indices[p['date']]) for p in original['points']]
    evaluated=cp._evaluate(original['kind'],data,end=end,known=known,anchor=anchor,value=line['value'],
        slope=line['slope'],height=height,invalidation=original['invalidation'],points=points,
        volume_required=bool(original.get('volume_required')),volume_base=base_ok,extra=evidence,
        expiry=min(evidence.get('flag_days',10),10) if 'flag' in original['kind'] else None,
        fixed_buffer=original['confirmation_buffer'])
    out.update(current=evaluated,basis_date=context['basis_date'],last_checked_at=current.isoformat(),
        data_blocked=False,price=data['Close'][-1],quote_at=context['basis_date'],price_source='Abgeschlossener Tagesschluss',
        health='Tagesregeln mit unveränderter Struktur geprüft.')
    out['state']='invalidated' if evaluated['phase']=='invalid' else evaluated['phase']
    threshold=evaluated['boundary_now']+evaluated['direction']*original['confirmation_buffer']
    out['threshold']=threshold;out['distance_pct']=(threshold-out['price'])*evaluated['direction']/out['price']*100
    # Quotes are early hints only; no intraday close can confirm a daily pattern.
    if out['state'] in {'forming','confirmed'} and minutes is not None and in_session(current,out['market']):
        try:
            intraday=clean_minutes(minutes,current,out['market'])
            finished=intraday.loc[intraday.index+pd.Timedelta(seconds=70)<=current]
            if finished.empty:raise ValueError('Keine abgeschlossene Minutenkerze.')
            stamp=finished.index[-1];age=(current-(stamp+pd.Timedelta(minutes=1))).total_seconds()
            if age>300:raise ValueError('Minutenkurs älter als fünf Minuten.')
            # Today's sloped line advances exactly one session beyond the last completed daily bar.
            threshold+=evaluated['boundary']['slope']
            price=float(finished.Close.iloc[-1]);direction=evaluated['direction']
            out.update(price=price,quote_at=stamp.isoformat(),price_source='Abgeschlossene 1m-Kerze · vorläufig',threshold=threshold)
            out['distance_pct']=(threshold-price)*direction/price*100
            if direction*(price-original['invalidation'])<=0:
                out['health']='Intraday an/jenseits der Muster-Invalidierung: Tagesprüfung abwarten; kein bestätigter Tagesbruch.'
            elif out['state']=='forming' and out['distance_pct']<=0:out['state']='intraday'
            elif out['state']=='forming' and out['distance_pct']<=1:out['state']='near'
        except ValueError as exc:out['health']+=' Kein frischer Intraday-Hinweis: '+str(exc)
    elif out['state']=='forming' and 0<out['distance_pct']<=1:out['state']='near'
    target=evaluated.get('projection')
    out['projection_room_pct']=(target-out['price'])*evaluated['direction']/out['price']*100 if target else None
    out['beyond_threshold_pct']=max(0,-out['distance_pct'])
    out['intraday_projection_touched']=bool(target and out.get('price_source','').startswith('Abgeschlossene 1m') and (out['price']-target)*evaluated['direction']>=0)
    out['current_package']=dict(policy=cp.POLICY,ticker=out['ticker'],available=True,stale=False,
        basis_date=context['basis_date'],currency=source.get('currency',''),bars=len(data['Close']),patterns=[evaluated],
        chart=[dict(index=i,date=data['date'][i],open=data['Open'][i],high=data['High'][i],low=data['Low'][i],close=data['Close'][i]) for i in range(max(0,len(data['Close'])-180),len(data['Close']))])
    return out


def poll(items,provider,*,now=None,force=False):
    from modules.provider_manager import history_refresh_scope
    from modules.candidate_radar import safe_scan_error
    current=utc(now);out=deepcopy(items);responses={};limited=False
    for item in out:
        if active(item) and current>=utc(item['expires_at']):
            item.update(state='expired',health='Beobachtungsdauer abgelaufen.');event(item,'expired',current,item['health'])
    if not any(active(i) for i in out):return out
    quarantine=provider.quarantine_state()
    if quarantine is not None and getattr(quarantine,'active',False):
        for item in out:
            if active(item):item['health']='Provider-Cooldown aktiv; keine Abfrage.'
        return out
    for item in out:
        if not active(item):continue
        before=item['state']
        if current>=utc(item['expires_at']):
            item.update(state='expired',health='Beobachtungsdauer abgelaufen.');event(item,'expired',current,item['health']);continue
        local=current.tz_convert('America/New_York' if item['market']=='US' else 'Europe/Berlin')
        bounds=session_bounds(local.date(),item['market'])
        if not force and not (bounds and bounds[0]<=current<=bounds[1]+pd.Timedelta(minutes=60)):
            item['health']='Handelsfenster geschlossen. Manuelle Tagesprüfung bleibt möglich.';continue
        if not force and item.get('last_attempt_at') and (current-utc(item['last_attempt_at'])).total_seconds()<60:continue
        ticker=item['ticker'];item['last_attempt_at']=current.isoformat()
        if ticker not in responses:
            if limited:responses[ticker]=(None,None,'Provider-Limit: weitere Abfragen ausgesetzt.')
            else:
                try:
                    daily=provider.get_history_light(ticker,period='3y',auto_adjust=True)
                    minutes=None;quote_error=''
                    if in_session(current,item['market']):
                        try:
                            with history_refresh_scope('pattern-watch-'+current.isoformat()):
                                minutes=provider.get_history_light(ticker,period='1d',interval='1m',prepost=False,auto_adjust=True)
                        except Exception as exc:
                            quote_error='Intraday nicht prüfbar: '+safe_scan_error(exc)['detail']
                            limited=safe_scan_error(exc)['category']=='Provider-Limit'
                    responses[ticker]=(daily,minutes,quote_error)
                except Exception as exc:
                    info=safe_scan_error(exc);limited=info['category']=='Provider-Limit'
                    responses[ticker]=(None,None,info['detail'])
        daily,minutes,error=responses[ticker]
        if daily is None:item.update(health='Keine aktuelle Prüfung: '+error,data_blocked=True,price=None,quote_at=None);continue
        try:
            updated=evaluate(item,daily,minutes,now=current)
            if error:updated['health']+=' '+error
            if updated['state']!=before:event(updated,updated['state'],current,LABELS.get(updated['state'],updated['state']))
            item.clear();item.update(updated)
        except Exception as exc:item.update(health='Prüfung fehlgeschlagen: '+type(exc).__name__,data_blocked=True,price=None,quote_at=None)
    return out
