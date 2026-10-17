"""Display-only market/industry risk from one verified Rotation Radar snapshot.

Thresholds are transparent initial heuristics, not a backtested loss forecast.
No providers, score adjustments, entry gates, order management or persistence.
"""
from datetime import timedelta
import math
import pandas as pd

SECTORS={'technology':'XLK','technologie':'XLK','financial services':'XLF','financials':'XLF',
         'finanzen':'XLF','industrials':'XLI','industrie':'XLI','healthcare':'XLV',
         'gesundheit':'XLV','energy':'XLE','energie':'XLE','basic materials':'XLB',
         'materials':'XLB','consumer cyclical':'XLY','consumer defensive':'XLP',
         'utilities':'XLU','real estate':'XLRE','communication services':'XLC'}
INDUSTRIES=[(('semiconductor','halbleiter'),'SMH'),(('cybersecurity','cyber security'),'CIBR'),
            (('software',),'IGV'),(('biotech',),'XBI'),(('aerospace','defense'),'ITA'),
            (('solar',),'TAN')]
# Known Radar members, not holdings weights or an inferred "AI" industry.
SEMIS={'MU','MRVL','NVDA','AMD','AVGO','TSM','ASML','AMAT','LRCX','KLAC','QCOM','ARM'}


def number(value):
    try:
        n=float(value)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError):return None


def expected_daily_date(now=None):
    ts=pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
    if ts.tzinfo is None:ts=ts.tz_localize('UTC')
    ny=ts.tz_convert('America/New_York')
    day=ny.date()
    if ny.hour<16 or (ny.hour==16 and ny.minute<30):day-=timedelta(days=1)
    while day.weekday()>4:day-=timedelta(days=1)
    return day


def stock_daily(result, *, now=None):
    df=(result or {}).get('df')
    if not isinstance(df,pd.DataFrame) or not isinstance(df.index,pd.DatetimeIndex) or 'Close' not in df:
        return {}
    if df.index.hasnans or pd.Index(df.index.date).duplicated().any() or not df.index.is_monotonic_increasing:return {}
    cutoff=expected_daily_date(now)
    current=pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
    if current.tzinfo is None:current=current.tz_localize('UTC')
    # An analysis cache may contain a still-forming current-day bar. Never
    # promote that bar to confirmed stock-vs-group evidence after the close.
    cutoff=min(cutoff,current.tz_convert('America/New_York').date()-timedelta(days=1))
    ser=pd.to_numeric(df.loc[[d.date()<=cutoff for d in df.index],'Close'],errors='coerce')
    if len(ser)<2:return {}
    last=number(ser.iloc[-1]);prior=number(ser.iloc[-2])
    if last is None or prior is None or min(last,prior)<=0:return {}
    return {'date':ser.index[-1].date().isoformat(),'prior_date':ser.index[-2].date().isoformat(),
            'return_1d':(last/prior-1)*100}


def resolve_groups(result):
    r=result or {};info=r.get('info') or {}
    symbol=str(r.get('ticker') or r.get('Ticker') or '').upper().strip()
    sector=str(r.get('sector') or r.get('sector_label') or info.get('sector') or '').casefold()
    industry=str(r.get('industry') or r.get('industry_label') or info.get('industry') or '').casefold()
    sector_symbol=SECTORS.get(sector)
    industry_symbol='SMH' if symbol in SEMIS else None
    if not industry_symbol:
        industry_symbol=next((group for words,group in INDUSTRIES if any(word in industry for word in words)),None)
    if industry_symbol=='SMH' and not sector_symbol:sector_symbol='XLK'
    # US-listed ADRs use the US market proxy; German listings use the Radar's
    # US-listed Germany ETF. Other local exchanges remain explicitly unmapped.
    market='EWG' if symbol.endswith('.DE') else 'SPY' if symbol and '.' not in symbol else None
    return {'market':market,'sector':sector_symbol,'industry':industry_symbol,
            'sector_proxy':not bool(industry_symbol),'symbol':symbol}


def _level(frame, symbol, role, meta, now):
    out={'role':role,'symbol':symbol or '—','severity':None,'status':'⚪ Unbekannt',
         'reason':'Keine passende Radar-Gruppe zugeordnet.'}
    if not symbol:return out
    hits=frame.loc[frame['Ticker'].astype(str).str.upper()==symbol] if isinstance(frame,pd.DataFrame) and 'Ticker' in frame else pd.DataFrame()
    if len(hits)!=1:return {**out,'reason':f'{symbol}: Radar-Zeile fehlt oder ist mehrdeutig.'}
    row=hits.iloc[0].to_dict();out['row']=row
    date=str(row.get('Kursdatum') or '')
    try:dated=pd.Timestamp(date).date()
    except (ValueError,TypeError):return {**out,'reason':f'{symbol}: Kursdatum fehlt; Radar aktualisieren.'}
    if (meta.get('legacy_fallback') or meta.get('snapshot_source')!='v30.3d verified storage'
            or not meta.get('snapshot_id') or date!=str(meta.get('data_through') or '')):
        return {**out,'reason':f'{symbol}: Kein verifizierter, einheitlicher Radar-Datenstand.'}
    if dated!=expected_daily_date(now):
        return {**out,'reason':f'{symbol}: Tagesstand {date} ist für diesen Zeitpunkt nicht aktuell/bestätigt.'}
    try:
        generated=pd.Timestamp(meta.get('generated_at'))
        if pd.isna(generated):raise ValueError('missing timestamp')
        if generated.tzinfo is None:generated=generated.tz_localize('UTC')
        current=pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
        if current.tzinfo is None:current=current.tz_localize('UTC')
        if generated>current:return {**out,'reason':f'{symbol}: Snapshot-Zeitstempel liegt in der Zukunft.'}
        cutoff=pd.Timestamp(str(dated)+' 16:30',tz='America/New_York')
        if generated<cutoff:return {**out,'reason':f'{symbol}: Snapshot entstand vor Bestätigung dieses Tagesschlusses.'}
    except (ValueError,TypeError):return {**out,'reason':f'{symbol}: Snapshot-Zeitstempel fehlt.'}
    if row.get('Kontext-Datenbasis')!='aligned' or str(row.get('Benchmarkdatum'))!=date:
        return {**out,'reason':f'{symbol}: Kurs-/Benchmark-Tage nicht gemeinsam belegt.'}
    r1=number(row.get('Perf 1T %'));r5=number(row.get('Perf 5T %'))
    d1=number(row.get('Rotation Δ1T'));d5=number(row.get('Rotation Δ5T'))
    phase=str(row.get('Phase') or '')
    if any(v is None for v in [r1,r5,d1,d5]) or not phase:
        return {**out,'reason':f'{symbol}: Kurzfristige Radar-Metriken fehlen; Radar aktualisieren.'}
    market=role=='Gesamtmarkt'
    red1,red5=(-2.,-4.) if market else (-3.,-6.)
    yellow1,yellow5=(-.75,-1.5) if market else (-1.,-2.)
    severity=0;reason='Kein Warnsignal in den geprüften Tageskriterien.'
    if r1<=red1 or r5<=red5:
        severity=2;reason=f'Deutlicher Rückgang: 1T {r1:+.1f} %, 5T {r5:+.1f} %.'
    elif 'Rotating Out' in phase:
        severity=2;reason='Radar-Phase Rotating Out; längerfristige Schwäche.'
    elif r1<=yellow1 or r5<=yellow5:
        severity=1;reason=f'Kurzfristiger Druck: 1T {r1:+.1f} %, 5T {r5:+.1f} %.'
    elif d1<=-8 or d5<=-10:
        severity=1;reason=f'Rotation kühlt ab: Δ1T {d1:+.1f}, Δ5T {d5:+.1f} Punkte.'
    elif 'Cooling' in phase or 'Mature' in phase:
        severity=1;reason='Radar-Phase '+phase+'; Stärke/Rotation nicht durchgehend konstruktiv.'
    elif not market and number(row.get('RS 5T %')) is not None and number(row.get('RS 5T %'))<=-2:
        severity=1;reason='Branche/Sektor bleibt über 5 Tage hinter dem Markt zurück.'
    return {**out,'severity':severity,'status':['🟢 Unauffällig','🟡 Erhöht','🔴 Hoch'][severity],
            'reason':reason,'date':date}


def build_environment_context(result, frame, meta, *, now=None):
    meta=meta or {};groups=resolve_groups(result)
    levels=[_level(frame,groups['market'],'Gesamtmarkt',meta,now),
            _level(frame,groups['sector'],'Sektor',meta,now)]
    if groups['industry']:
        levels.append(_level(frame,groups['industry'],'Branche/Thema',meta,now))
    known=[level for level in levels if level['severity'] is not None]
    missing=[level for level in levels if level['severity'] is None]
    maximum=max((level['severity'] for level in known),default=0)
    # A real warning remains visible on a partial basis. Missing coverage can
    # never produce green and is shown alongside any known yellow/red warning.
    severity=maximum if maximum>0 or not missing else None
    status='⚪ Unbekannt' if severity is None else ['🟢 Unauffällig','🟡 Erhöht','🔴 Hoch'][severity]
    if severity is None:
        reason=missing[0]['reason']
        if len(missing)>1:reason+=' Weitere Ebenen ebenfalls unbekannt.'
    elif severity>0:
        top=next(level for level in known if level['severity']==maximum)
        reason=f"{top['role']} {top['symbol']}: {top['reason']}"
    else:reason='Markt und zugeordnete Radar-Gruppen ohne Warnsignal in den geprüften Tageskriterien.'
    if missing and severity is not None:reason+=' Teilbasis: weiterer Kontext unbekannt.'
    if groups['sector_proxy']:reason+=' Nur Sektorproxy; Branche nicht separat erfasst.'
    stock=(result or {}).get('__env_stock')
    if not isinstance(stock,dict):stock=stock_daily(result,now=now)
    group=next((level for level in reversed(levels) if level['severity'] is not None),None)
    rs=None
    if group and group['role']!='Gesamtmarkt':
        row=group['row']
        if stock.get('date')==group.get('date') and stock.get('prior_date')==row.get('Vorheriger Handelstag'):
            ret=number(stock.get('return_1d'));benchmark=number(row.get('Perf 1T %'))
            if ret is not None and benchmark is not None:rs=ret-benchmark
    return {'status':status,'severity':severity,'reason':reason,'levels':levels,
            'data_through':meta.get('data_through') or 'unbekannt',
            'snapshot_id':meta.get('snapshot_id') or 'unbekannt','groups':groups,
            'stock_rs_1d':rs,'partial':bool(missing)}


def attach_environment_columns(frame, radar, meta, *, now=None):
    out=frame.copy()
    packages=[]
    for row in out.to_dict('records'):
        result={'ticker':row.get('Ticker'),'sector':row.get('__env_sector'),
                'industry':row.get('__env_industry'),'__env_stock':row.get('__env_stock')}
        packages.append(build_environment_context(result,radar,meta,now=now))
    out['Umfeldrisiko']=[p['status'] for p in packages]
    out['Umfeld-Grund']=[p['reason'] for p in packages]
    out['Umfeld-Datenstand']=[p['data_through'] for p in packages]
    out['Umfeld-Snapshot']=[p['snapshot_id'] for p in packages]
    return out


def render_environment_context(st, package):
    st.markdown('**Markt-/Branchenrisiko · '+package['status']+'**')
    st.write(package['reason'])
    rows=[]
    for level in package['levels']:
        row=level.get('row') or {}
        rows.append({'Ebene':level['role'],'Radar-Gruppe':level['symbol'],'Risiko':level['status'],
                     '1T %':number(row.get('Perf 1T %')),'5T %':number(row.get('Perf 5T %')),
                     'Radar-Phase':str(row.get('Phase') or '—'),'Grund':level['reason']})
    with st.expander("Umfeld im Detail",expanded=False):
        st.table(rows)
        if package['stock_rs_1d'] is not None:
            st.caption(f"Aktie relativ zur zugeordneten Gruppe (gleicher Tageszeitraum): {package['stock_rs_1d']:+.2f} Prozentpunkte. Relative Stärke schützt nicht vor absoluten Verlusten.")
        else:st.caption('Aktie relativ zur Branche: gemeinsamer Tageszeitraum nicht belegt.')
    st.caption(f"Tagesdaten bis {package['data_through']} · Radar-Snapshot {package['snapshot_id']}. Separate Risikoheuristik; technischer Score und Freigabe unverändert. Kein Intraday-Frühwarnsystem.")
