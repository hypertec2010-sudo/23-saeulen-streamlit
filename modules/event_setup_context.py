"""Daily event-setups: descriptive chart context, never an operative signal.

Rules here are explicit first-pass heuristics, not a validated trading strategy.
Only completed sessions before as_of are inspected; the current daily bar is
excluded. No provider requests, news inference, production stops or mutations.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import math
import pandas as pd


def build_event_setup_context(df, *, as_of=None):
    unavailable = {"available": False, "event": None,
                   "summary": "Tagesdaten für Event-Setups nicht ausreichend."}
    if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
        return unavailable
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(c in df.columns for c in required) or isinstance(df.columns, pd.MultiIndex):
        return unavailable
    cutoff = pd.Timestamp(as_of).date() if as_of is not None else datetime.now(ZoneInfo('Europe/Berlin')).date()
    data = df[required].copy().sort_index()
    data = data.loc[[not pd.isna(i) and i.date() < cutoff for i in data.index]]
    if len(data) < 21 or pd.Index(data.index.date).duplicated().any():
        return unavailable
    data = data.apply(pd.to_numeric, errors='coerce')
    valid = data.apply(lambda c: c.map(lambda v: pd.notna(v) and math.isfinite(v))).all(axis=1)
    valid &= (data[required] > 0).all(axis=1)
    valid &= (data['High'] >= data[['Open','Close','Low']].max(axis=1))
    valid &= (data['Low'] <= data[['Open','Close','High']].min(axis=1))
    if not bool(valid.iloc[-1]):
        return {**unavailable, 'summary': 'Letzte abgeschlossene Tageskerze ist unvollständig oder ungültig.'}
    prior_volume = data['Volume'].where(valid).shift(1).rolling(20, min_periods=20).mean()
    gap = (data['Open'] / data['Close'].shift(1) - 1) * 100
    rvol = data['Volume'] / prior_volume
    candidates = valid & valid.shift(1, fill_value=False) & (gap >= 10.0) & (rvol >= 2.0)
    positions = [i for i in range(max(20,len(data)-20),len(data)) if bool(candidates.iloc[i])]
    base = {'available': True, 'event': None, 'basis_date': data.index[-1].date().isoformat(),
            'summary': 'Kein Gap-/Volumenimpuls im Standardfilter der letzten 20 abgeschlossenen Handelstage.'}
    if not positions:
        return base
    i = positions[-1]; event = data.iloc[i]; following = data.iloc[i+1:]
    event_close = float(event['Close']); latest = float(data['Close'].iloc[-1])
    if following.empty:
        hvc_status = 'Impulstag abgeschlossen; spätere Reaktion abwarten'
    elif not valid.iloc[i+1:].all():
        hvc_status = 'Folgedaten unvollständig; keine Reaktion bestätigen'
    elif latest > event_close and float(data['Close'].iloc[-2]) <= event_close:
        hvc_status = 'HVC im letzten Tagesschluss zurückerobert'
    elif latest > event_close:
        hvc_status = 'Tagesschluss oberhalb HVC; keine neue Rückeroberung'
    else:
        hvc_status = 'Tagesschluss noch nicht über HVC'
    flag_status = 'Noch keine ausreichend lange Konsolidierung'
    flag_low = flag_high = None
    # The last close is evaluated against a band from PRIOR completed bars.
    # Including the breakout bar in its own trigger range would prevent detection.
    consolidation = following.iloc[:-1]
    if len(consolidation) >= 3:
        if not valid.iloc[i+1:].all():
            flag_status = 'Folgedaten unvollständig; Flagge nicht beurteilbar'
        elif len(consolidation) > 15:
            flag_status = 'Konsolidierung länger als das kurze Flaggenfenster'
        else:
            flag_low = float(consolidation['Low'].min()); flag_high = float(consolidation['High'].max())
            width_pct = (flag_high / flag_low - 1) * 100
            volume_contracted = float(consolidation['Volume'].mean()) <= float(event['Volume']) * 0.75
            gap_held = min(float(event['Low']), float(following['Low'].min())) > float(data['Close'].iloc[i-1])
            if not gap_held:
                flag_status = 'Gap vollständig angelaufen; keine enge Flagge bestätigt'
            elif width_pct > 8 or not volume_contracted:
                flag_status = 'Konsolidierung noch nicht eng / Volumen noch nicht beruhigt'
            elif latest > flag_high:
                flag_status = 'Mögliche Flagge: letzter Tagesschluss über Konsolidierungshoch'
            elif latest < flag_low:
                flag_status = 'Konsolidierung nach unten verlassen'
            else:
                flag_status = 'Mögliche enge Flagge; Ausbruch per Tagesschluss noch offen'
    ema10 = float(data['Close'].ewm(span=10,adjust=False).mean().iloc[-1]) if valid.all() else None
    return {**base, 'event': {
        'date': data.index[i].date().isoformat(), 'gap_pct': float(gap.iloc[i]),
        'relative_volume': float(rvol.iloc[i]), 'hvc': event_close,
        'event_low': float(event['Low']), 'hvc_status': hvc_status,
        'flag_status': flag_status, 'flag_low': flag_low, 'flag_high': flag_high,
        'ema10': ema10, 'last_close': latest,
    }, 'summary': 'Technischer Ereigniskandidat; Nachrichten-/Earnings-Auslöser nicht verifiziert.'}


def render_event_setup_context(st, package, *, ccy=''):
    """Compact, opt-in chart rubric; receives only a descriptive package."""
    with st.container(border=True):
        st.markdown('**Event-Setups · High Volume Close & Flagge**')
        if package.get('basis_date'):
            st.caption('Datenstand: '+package['basis_date'])
        if not package.get('available') or not package.get('event'):
            st.caption(package['summary'])
        else:
            e = package['event']; suffix = (' '+str(ccy)) if ccy else ''
            st.markdown(f"**Impuls {e['date']}** · Gap {e['gap_pct']:+.1f}% · Volumen {e['relative_volume']:.2f}×")
            st.write(f"High Volume Close: {e['hvc']:.2f}{suffix} · {e['hvc_status']}")
            st.write(e['flag_status'])
            if e['flag_high'] is not None:
                st.caption(f"Konsolidierung: {e['flag_low']:.2f}–{e['flag_high']:.2f}{suffix}")
            st.caption(f"Tief des Impulstages: {e['event_low']:.2f}{suffix} · Strukturreferenz, kein automatisch gesetzter Stop.")
            if e['ema10'] is not None:
                st.caption(f"EMA10: {e['ema10']:.2f}{suffix} · letzter Tagesschluss {e['last_close']:.2f}{suffix}")
            st.caption(package['summary'])
        st.caption('Informativ: Gap ≥10% und Volumen ≥2× dem Mittel der 20 vorherigen Handelstage. Nur abgeschlossene Tageskerzen vor heute; kein Intraday-Signal.')
        st.caption('Flaggenheuristik: 3–15 vorherige Folgetage, Bandbreite ≤8%, Volumen ≤75% des Impulstages und Gap nicht vollständig angelaufen. Keine Wirkung auf Score, Ampel, CRV oder Freigabe.')
