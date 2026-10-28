"""Display-only US daily-bar freshness and provider/cache provenance.

NYSE published 2026–2027 regular-session holiday/early-close calendar:
https://www.nyse.com/markets/hours-calendars (checked 2026-10-26).
Other exchanges, exceptional closures and unsupported calendar years are not
silently assigned a US session. No scores, refresh loops or provider requests.
"""
from datetime import date,timedelta
import pandas as pd

HOLIDAYS={
 2026:'01-01 01-19 02-16 04-03 05-25 06-19 07-03 09-07 11-26 12-25',
 2027:'01-01 01-18 02-15 03-26 05-31 06-18 07-05 09-06 11-25 12-24',
}
EARLY_CLOSE={'2026-11-27','2026-12-24','2027-11-26'}


def stamp(value):
    try:
        p=pd.Timestamp(value)
        if pd.isna(p):return None
        return p.tz_localize('UTC') if p.tzinfo is None else p.tz_convert('UTC')
    except (ValueError,TypeError):return None


def bar_date(value):
    raw=str(value or '').strip()
    if not raw or raw in {'n/a','nan','None','NaT'}:return None
    try:
        return pd.to_datetime(raw,format='%d.%m.%Y').date() if '.' in raw and len(raw)==10 else pd.Timestamp(raw).date()
    except (ValueError,TypeError):return None


def is_us(symbol):
    s=str(symbol or '').upper()
    return bool(s and ('.' not in s or s in {'BRK.B','BRK.A'}) and not s.startswith('^') and '=' not in s)


def session_day(day):
    if day.year not in HOLIDAYS:return None
    return day.weekday()<5 and day.strftime('%m-%d') not in HOLIDAYS[day.year].split()


def previous_session(day):
    for _ in range(12):
        day-=timedelta(days=1)
        eligible=session_day(day)
        if eligible is None:return None
        if eligible:return day
    return None


def close_at(day):
    hour=13 if day.isoformat() in EARLY_CLOSE else 16
    return pd.Timestamp(f'{day.isoformat()} {hour}:00',tz='America/New_York')


def classify_date(symbol,value,*,now=None,fetched_at=None):
    current=stamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
    current=current if current is not None else pd.Timestamp.now(tz='UTC')
    day=bar_date(value)
    out={'date':day,'status':'Kerzendatum unbekannt','severity':'info','missing_current':False}
    if day is None:return out
    if not is_us(symbol):return {**out,'status':'Börsensitzung nicht geprüft','severity':'none'}
    ny=current.tz_convert('America/New_York');today=ny.date()
    eligible=session_day(today)
    if eligible is None:return {**out,'status':'US-Kalender für dieses Jahr nicht hinterlegt','severity':'info'}
    close=close_at(today)
    confirmed_after=pd.Timestamp(close.to_pydatetime()+timedelta(minutes=30))
    opened=pd.Timestamp(f'{today.isoformat()} 09:30',tz='America/New_York')
    completed=today if eligible and ny>=confirmed_after else previous_session(today)
    if completed is None:return {**out,'status':'Vorherige US-Sitzung nicht verifiziert','severity':'info'}
    if day>today:return {**out,'status':'Kerzendatum liegt in der Zukunft','severity':'warning'}
    if day<completed:return {**out,'status':f'Abgeschlossene Tageskerze fehlt (erwartet {completed:%d.%m.%Y})','severity':'warning'}
    if day==today:
        if not eligible:return {**out,'status':'Kerze an geschlossenem US-Handelstag – Quelle prüfen','severity':'warning'}
        if ny<opened:return {**out,'status':'Heutige Kerze vor US-Kernhandel – Quelle prüfen','severity':'info'}
        if ny<confirmed_after:return {**out,'status':'Laufende / noch nicht final bestätigte Tageskerze','severity':'none'}
    if day==completed:
        fetched=stamp(fetched_at)
        cutoff=pd.Timestamp(close_at(day).to_pydatetime()+timedelta(minutes=30))
        if fetched is None:
            return {**out,'status':'Kerzendatum passt; Abrufzeit nicht belegt','severity':'info'}
        if fetched<cutoff:
            return {**out,'status':'Schlussstand noch nicht durch frischen Abruf bestätigt','severity':'warning'}
        if eligible and ny>=opened and ny<confirmed_after and day<today:
            return {**out,'status':'Letzter Schlusskurs; heutige Tageskerze noch nicht geliefert',
                    'severity':'info','missing_current':True}
        return {**out,'status':'Letzte abgeschlossene US-Tageskerze','severity':'none'}
    return {**out,'status':'Kerzendatum / Sitzungszuordnung prüfen','severity':'info'}


def format_stamp(value):
    p=stamp(value)
    return p.tz_convert('Europe/Berlin').strftime('%d.%m.%Y %H:%M:%S %Z') if p is not None else 'Unbekannt'


def freshness_table(frame,*,now=None):
    rows=[]
    for r in (frame.to_dict('records') if isinstance(frame,pd.DataFrame) else []):
        value=r.get('Kursdaten-Kerze') or r.get('__diag_setup_data_date') or ''
        c=classify_date(r.get('Ticker'),value,now=now,fetched_at=r.get('Kursdaten-Abruf'))
        rows.append({'Ticker':str(r.get('Ticker') or '-'),
            'Kerzendatum':c['date'].strftime('%d.%m.%Y') if c['date'] else 'Unbekannt',
            'Letzter Provider-Abruf':format_stamp(r.get('Kursdaten-Abruf')),
            'Herkunft':str(r.get('Kursdaten-Quelle') or 'Unbekannt'),
            'Einordnung':c['status'],'__severity':c['severity'],'__missing_current':c['missing_current']})
    return pd.DataFrame(rows)


def render_freshness_summary(st,frame,*,now=None):
    table=freshness_table(frame,now=now)
    if table.empty:return
    warnings=int((table['__severity']=='warning').sum())
    missing=int(table['__missing_current'].sum())
    counts=table['Kerzendatum'].value_counts()
    dates=' · '.join(f'{d}: {n}' for d,n in counts.head(4).items())
    if warnings:st.warning(f'Datenstand: {warnings}/{len(table)} Werte mit fehlendem oder nicht frisch bestätigtem Schlussstand · {dates}')
    elif missing:st.info(f'Datenstand: {missing} US-Werte noch ohne heutige Tageskerze; letzter Schlusskurs vorhanden · {dates}')
    else:st.caption('Kerzendaten · '+dates)
    with st.expander('📅 Datenstand & Abrufe',expanded=False):
        st.dataframe(table.drop(columns=['__severity','__missing_current']),hide_index=True,use_container_width=True)
        st.caption('Abrufzeiten in Berliner Zeit. Ein Provider-Abruf garantiert keine neue Kerze oder Echtzeitkurse. Laufende Tageskerzen sind keine bestätigten Schlusskurse. US-Kalender 2026–2027; andere Börsen werden hier nicht nach US-Zeiten bewertet.')
