"""Read-only historical support map; no provider, score, stop or entry effects.

Numerical thresholds describe initial display heuristics, not bounce odds.
Only strict low pivots with three completed bars on either side count. Clusters
have bounded total price width; adjacent pivot episodes count only once.
"""
import pandas as pd

from modules.trade_watch import number, utc, market_for
from modules.watch_setups import completed_day

POLICY = 'support_map_v30.21ck'


def _pivots(frame):
    values=frame.Low.tolist()
    return [(i,float(values[i])) for i in range(3,len(values)-3)
            if values[i]<min(values[i-3:i]+values[i+1:i+4])]


def _clusters(points):
    groups=[]
    for item in sorted(points,key=lambda x:(x[1],x[0])):
        if not groups or item[1]/groups[-1][0][1]>1.015:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def build_support_map(result, *, now=None):
    r=result or {};current=utc(now)
    base=dict(policy=POLICY,available=False,zones=[],references=[],bars=0,
              basis_date='',expected_date='',stale=False,chart=[],reason='')
    def unavailable(reason):
        return {**base,'reason':reason}
    try:
        ticker=str(r.get('ticker') or r.get('Ticker') or '').upper().strip()
        market=market_for(ticker)
        expected=completed_day(current,market)
        base['expected_date']=expected.isoformat()
        source=r.get('df')
        required=['Open','High','Low','Close']
        if (not isinstance(source,pd.DataFrame) or isinstance(source.columns,pd.MultiIndex)
                or not isinstance(source.index,pd.DatetimeIndex) or source.index.hasnans
                or not source.index.is_monotonic_increasing
                or not all(k in source for k in required)):
            return unavailable('Eindeutige, geordnete Tages-OHLC-Historie fehlt.')
        dates=list(source.index.date)
        if len(set(dates))!=len(dates):
            return unavailable('Mehrere Tageskerzen je Datum; keine eindeutige Basis.')
        today=current.tz_convert('America/New_York' if market=='US' else 'Europe/Berlin').date()
        if any(d>today for d in dates):
            return unavailable('Kursdaten enthalten zukünftige Tagesdaten.')
        frame=source.loc[[d<=expected for d in dates],required].tail(520).copy()
        if len(frame)<60:
            return unavailable('Mindestens 60 abgeschlossene Tageskerzen erforderlich.')
        frame=frame.apply(pd.to_numeric,errors='coerce')
        valid=frame.apply(lambda c:c.map(lambda x:number(x) is not None and x>0)).all(axis=1)
        valid &= frame.High>=frame[['Open','Close','Low']].max(axis=1)
        valid &= frame.Low<=frame[['Open','Close','High']].min(axis=1)
        if not valid.all():
            return unavailable('Ungültige OHLC-Werte in der verwendeten Historie.')
        gaps=[(b-a).days for a,b in zip(frame.index.date[:-1],frame.index.date[1:])]
        if max(gaps)>10 or sum(g<=4 for g in gaps)/len(gaps)<.7:
            return unavailable('Tageshistorie lückenhaft oder keine tägliche Kursreihe.')
        quote=number(r.get('price'))
        price=quote if quote is not None and quote>0 else float(frame.Close.iloc[-1])
        prev=frame.Close.shift()
        tr=pd.concat([frame.High-frame.Low,(frame.High-prev).abs(),(frame.Low-prev).abs()],axis=1).max(axis=1)
        atr=float(tr.tail(14).mean())
        base.update(available=True,ticker=ticker,bars=len(frame),basis_date=frame.index[-1].date().isoformat(),
                    stale=frame.index[-1].date()!=expected,price=price,
                    price_source='Analysekurs · Zeitstempel nicht separat geprüft' if quote is not None and quote>0 else 'Letzter abgeschlossener Tagesschluss',
                    currency=str(r.get('ccy') or r.get('currency') or (r.get('info') or {}).get('currency') or ''),
                    atr=atr,reason='Historische Tiefzonen; keine Bestätigung einer Bodenbildung.')
        points=_pivots(frame)
        zones=[];singles=[]
        for group in _clusters(points):
            tests=[]
            for i,v in sorted(group):
                if not tests or i-tests[-1][0]>=6:
                    tests.append((i,v))
            if len(tests)<2:
                if tests:singles.append(tests[0])
                continue
            low=min(v for _,v in tests);high=max(v for _,v in tests);mid=(low+high)/2
            padding=min(mid*.005,max(mid*.002,atr*.1))
            lower=max(low-padding,1e-9);upper=high+padding
            last=tests[-1][0];span=tests[-1][0]-tests[0][0]
            last_close=float(frame.Close.iloc[-1])
            if price<lower:
                state='broken' if last_close<lower else 'intraday_below'
            elif price<=upper:
                state='in_zone'
            elif last_close<lower:
                state='intraday_reclaim'
            elif (frame.Close.iloc[last+4:]<lower).any():
                state='reclaimed'
            else:
                state='below_price'
            distance=0. if lower<=price<=upper else (lower/price-1)*100 if price<lower else (upper/price-1)*100
            zones.append(dict(low=lower,high=upper,mid=mid,touches=len(tests),span_bars=span,
                first_test=frame.index[tests[0][0]].date().isoformat(),
                last_test=frame.index[last].date().isoformat(),
                confirmed_at=frame.index[last+3].date().isoformat(),
                age_bars=len(frame)-1-last,older=len(frame)-1-last>126,
                pronounced=len(tests)>=3 and span>=20,
                tests=[frame.index[i].date().isoformat() for i,_ in tests],
                state=state,distance_pct=distance))
        base['zones']=sorted(zones,key=lambda z:abs(z['distance_pct']))
        base['references']=[dict(level=v,date=frame.index[i].date().isoformat(),
                                   age_bars=len(frame)-1-i,distance_pct=(v/price-1)*100)
                            for i,v in sorted(singles,key=lambda x:abs(x[1]-price)) if v<price][:2]
        base['chart']=[dict(date=i.date().isoformat(),close=float(row.Close))
                       for i,row in frame.tail(250).iterrows()]
        return base
    except (ValueError,TypeError,KeyError,IndexError,OverflowError) as exc:
        return unavailable('Unterstützungsübersicht nicht berechenbar: '+str(exc)[:140])


STATES={'below_price':'Unterhalb des Kurses · Reaktion offen',
        'in_zone':'Kurs in Zone · Reaktion offen',
        'broken':'Gebrochen per Tagesschluss · möglicher Widerstand',
        'intraday_below':'Aktuell unterschritten · Schlussbestätigung fehlt',
        'intraday_reclaim':'Aktuell über Zone · Schlussbestätigung fehlt',
        'reclaimed':'Zurückerobert · neuer Tief-Retest fehlt'}


def support_brief(package):
    """Compact wording from the same snapshot; never infer a buy signal."""
    p=package
    if not isinstance(p,dict) or p.get('policy')!=POLICY:
        return 'Neu scannen'
    if not p.get('available'):
        return 'Nicht berechenbar'
    price=number(p.get('price'))
    if price is None or price<=0 or not isinstance(p.get('zones'),list):
        return 'Daten unvollständig · neu scannen'
    zones=[]
    for z in p['zones']:
        if not isinstance(z,dict):return 'Daten unvollständig · neu scannen'
        low,high,dist=(number(z.get(k)) for k in ('low','high','distance_pct'))
        if low is None or high is None or dist is None or not 0<low<=high or z.get('state') not in STATES:
            return 'Daten unvollständig · neu scannen'
        zones.append({**z,'low':low,'high':high,'distance_pct':dist})
    below=[z for z in zones if z['low']<=price]
    z=min(below,key=lambda z:abs(z['distance_pct'])) if below else min(zones,key=lambda z:z['distance_pct']) if zones else None
    if z is None:
        text='Keine bestätigte Tiefzone darunter'
    else:
        fmt=lambda v:(f'{v:.4f}' if abs(v)<1 else f'{v:.2f}').replace('.',',')
        band=fmt(z['low'])+'–'+fmt(z['high'])
        currency=str(p.get('currency') or '').strip()
        band+=(' '+currency) if currency else ''
        distance=f"{abs(z['distance_pct']):.1f}".replace('.',',')+' %'
        state=z['state']
        if state=='in_zone':text='In Zone '+band+' · Reaktion offen'
        elif state=='broken':text='Zone '+band+' gebrochen'
        elif state=='intraday_below':text='Unter Zone '+band+' · Schluss offen'
        elif state=='intraday_reclaim':text='Über Zone '+band+' · Schluss offen'
        elif state=='reclaimed':text='Zone '+band+' · '+distance+' darunter · Retest offen'
        else:text='Support '+band+' · '+distance+' darunter'
        if z.get('older'):text+=' · älter'
    if p.get('stale'):text+=' · Stand '+str(p.get('basis_date') or 'unbekannt')
    return text


def attach_support_briefs(frame):
    if not isinstance(frame,pd.DataFrame):return frame
    out=frame.copy()
    out['Support-Kurz']=[support_brief(r.get('__support_map')) for r in out.to_dict('records')]
    return out


def selected_zones(package):
    zones=(package or {}).get('zones') or []
    below=sorted([z for z in zones if z['high']>=package['price']>=z['low'] or z['high']<package['price']],
                 key=lambda z:abs(z['distance_pct']))
    selected=below[:3]
    major=sorted([z for z in below if z['pronounced']],key=lambda z:(-z['touches'],z['age_bars']))
    if major and major[0] not in selected:selected.append(major[0])
    above=[z for z in zones if z['low']>package['price']]
    if above:selected.append(min(above,key=lambda z:z['distance_pct']))
    return selected


def support_rows(package):
    rows=[]
    for z in selected_zones(package):
        evidence=f"{z['touches']} getrennte Tests · {z['span_bars']} Handelstage"
        if z['pronounced']:evidence+=' · ausgeprägt'
        if z['older']:evidence+=' · älter als 126 Handelstage'
        rows.append({'Zone':f"{z['low']:.2f}–{z['high']:.2f}",
                     'Abstand zur Zone':f"{z['distance_pct']:+.2f} %",
                     'Historischer Beleg':evidence,'Status':STATES[z['state']],
                     'Letzter Tief-Test':z['last_test']})
    return rows


def render_support_map(st, package):
    p=package or {}
    st.markdown('**Historische Unterstützungszonen**')
    if not p.get('available'):
        st.caption(p.get('reason') or 'Noch keine Unterstützungsübersicht vorhanden; neu analysieren.')
        return
    if p.get('stale'):
        st.warning(f"Historische Tagesbasis {p['basis_date']}; erwartet {p['expected_date']}. Status nicht als frisch bestätigt behandeln.")
    rows=support_rows(p)
    if rows:
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    below=[z for z in p['zones'] if z['low']<=p['price']]
    if not below:
        st.info('Unter dem Referenzkurs ist in dieser Historie keine mehrfach getestete Tiefzone bestätigt.')
    if not rows and not p['references']:
        st.caption('Keine ausreichend belegte Tiefzone oder einzelnes bestätigtes Tief gefunden.')
    if p['references']:
        with st.expander('Einzelne Tiefs · keine mehrfach bestätigte Unterstützung',expanded=False):
            st.table(pd.DataFrame([{'Einzeltief':f"{x['level']:.2f}",'Datum':x['date'],
                                    'Abstand':f"{x['distance_pct']:+.2f} %"} for x in p['references']]))
    st.caption(f"Kursreferenz {p['price']:.2f} {p['currency']} · {p['price_source']} · {p['bars']} Tageskerzen bis {p['basis_date']} nach Börsenkalender; Provider-Finalität ungeprüft. Abstand zur nächsten Zonenkante; innerhalb der Zone 0 %. Ausgeprägt = mindestens 3 Tests über mindestens 20 Handelstage, keine Abprallwahrscheinlichkeit.")
    st.caption('Eine nahe Unterstützung bestätigt noch keinen Boden. Trendampel, Analyse-Stop und Trade-Wächter-Regeln bleiben unverändert.')
    if rows:
        with st.expander('Unterstützungszonen im Kursverlauf',expanded=False):
            import plotly.graph_objects as go
            fig=go.Figure(go.Scatter(x=[x['date'] for x in p['chart']],y=[x['close'] for x in p['chart']],mode='lines',name='Tagesschluss'))
            for i,z in enumerate(selected_zones(p),1):
                broken=z['state'] in {'broken','intraday_below'}
                fig.add_hrect(y0=z['low'],y1=z['high'],fillcolor='orange' if broken else 'steelblue',opacity=.16,line_width=1,
                              annotation_text=f"{'Bruch' if broken else 'Zone'} {i} · {z['touches']} Tests",annotation_position='top left')
            fig.add_hline(y=p['price'],line_dash='dot',annotation_text='Kursreferenz')
            fig.update_layout(height=350,margin=dict(l=20,r=20,t=30,b=25),yaxis_title=p['currency'],showlegend=False)
            st.plotly_chart(fig,use_container_width=True,key='support_map_'+p['ticker'])


def render_screener_supports(st, frame):
    if not isinstance(frame,pd.DataFrame) or frame.empty:return
    if '__support_map' not in frame:
        return
    rows=[]
    for r in frame.to_dict('records'):
        p=r.get('__support_map')
        if not isinstance(p,dict) or p.get('policy')!=POLICY:
            rows.append({'Ticker':r.get('Ticker'),'Nächste Tiefzone':'Neu scannen'})
            continue
        if not p.get('available'):
            rows.append({'Ticker':r.get('Ticker'),'Trendstatus':r.get('Status'),
                         'Nächste Tiefzone':'Nicht berechenbar','Zonenstatus':'Nicht bewertbar',
                         'Beleg':p.get('reason') or 'Tageshistorie fehlt'})
            continue
        eligible=[z for z in p.get('zones',[]) if z['low']<=p.get('price',0)]
        z=min(eligible,key=lambda z:abs(z['distance_pct'])) if eligible else None
        rows.append({'Ticker':r.get('Ticker'),'Trendstatus':r.get('Status'),
                     'Nächste Tiefzone':f"{z['low']:.2f}–{z['high']:.2f} {p.get('currency','')}" if z else 'Keine mehrfach getestete Zone darunter',
                     'Abstand %':round(z['distance_pct'],2) if z else None,
                     'Beleg':f"{z['touches']} Tests"+(' · ausgeprägt' if z['pronounced'] else '') if z else p.get('reason'),
                     'Zonenstatus':STATES[z['state']] if z else 'Nicht belegt',
                     'Tagesbasis':p.get('basis_date'),
                     'Datenhinweis':'Historie veraltet' if p.get('stale') else 'Zeitstempel der Kursreferenz nicht separat geprüft'})
    with st.expander('🧱 Unterstützungszonen · auch bei roten Werten',expanded=False):
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption('Historische Tiefzonen unabhängig von der Trendampel; Nähe ist kein Kaufsignal. Für frische Zonen die gewünschten Werte neu scannen.')
