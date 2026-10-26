"""Risk and position sizing helpers extracted in v25.0.

This module contains calculation logic. App-specific analysis callbacks are
registered once by app.py via ``configure_context``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# App callbacks, configured after the analysis engine has been defined.
build_professional_radar_decision_v18 = None
build_radar_entry_rr_package_v182 = None
_v210_alert_price = None
_radar_v182_entry_zone_text = None
_radar_v182_parse_zone = None
_radar_v182_stop_value = None
_radar_v209_main_target_value = None

def configure_context(**callbacks):
    globals().update({k: v for k, v in callbacks.items() if v is not None})

def _v230_safe_float(val, default=None):
    try:
        if val in [None, "", "-", "n/a", "nan"]:
            return default
        f = float(str(val).replace("%", "").replace(",", ".").strip())
        if np.isfinite(f) and not pd.isna(f):
            return f
    except Exception:
        pass
    return default


def _v230_price_text(val, digits=2):
    f = _v230_safe_float(val, default=None)
    if f is None:
        return "n/a"
    try:
        return f"{f:.{int(digits)}f}"
    except Exception:
        return str(round(f, 2))




def _v2410_infer_quote_currency(ticker, row=None, result=None, fallback="USD"):
    """Quote-Waehrung fuer Live-/Risiko-Rechner robust ableiten.

    Der Risiko-Rechner rechnet in der Kurswaehrung des ausgewaehlten Titels.
    US-Ticker sollen daher USD zeigen, europaeische Suffixe typischerweise EUR usw.
    """
    tk = str(ticker or "").strip().upper()
    sources = []
    for obj in (row, result):
        if isinstance(obj, dict):
            sources.append(obj)
            info = obj.get("info")
            if isinstance(info, dict):
                sources.append(info)
    for obj in sources:
        for key in ("currency", "Waehrung", "Währung", "quoteCurrency", "financialCurrency"):
            val = obj.get(key)
            if val is not None and str(val).strip() and str(val).strip().lower() not in {"nan", "none", "-", "n/a"}:
                return str(val).strip().upper()
    # Suffix-basierte Fallbacks. Yahoo-Suffixe sind fuer den Risiko-Rechner
    # ausreichend, falls die Analyse kein currency-Feld liefert.
    if tk.endswith((".DE", ".F", ".BE", ".DU", ".HM", ".HA", ".MU", ".SG", ".MI", ".PA", ".AS", ".MC", ".LS", ".VI", ".BR", ".HE", ".IR", ".AT", ".OL")):
        return "EUR"
    if tk.endswith(".L"):
        return "GBP"
    if tk.endswith(".SW"):
        return "CHF"
    if tk.endswith(".ST"):
        return "SEK"
    if tk.endswith(".CO"):
        return "DKK"
    if tk.endswith(".TO") or tk.endswith(".V"):
        return "CAD"
    if tk.endswith(".AX"):
        return "AUD"
    if tk.endswith(".HK"):
        return "HKD"
    if tk.endswith(".T"):
        return "JPY"
    return str(fallback or "USD").upper()

def _v230_extract_position_inputs(result, style_name="Ausgewogen"):
    """Robuste Entry/Stop/Ziel-Basis fuer den Positionsgroessen-Rechner.

    Der Rechner soll keine neue Kaufempfehlung erzeugen, sondern aus einem bereits
    interessanten Live-/Chart-Setup die Risiko- und Stueckzahlfrage beantworten.
    """
    r = result or {}
    try:
        decision = build_professional_radar_decision_v18(r, style_name)
    except Exception:
        decision = {}
    try:
        rr = build_radar_entry_rr_package_v182(r)
    except Exception:
        rr = {}

    price = _v230_safe_float(rr.get("price"), default=None)
    if price is None:
        try:
            price = _v230_safe_float(_v210_alert_price(r), default=None)
        except Exception:
            price = None
    if price is None:
        price = _v230_safe_float(r.get("price") or r.get("Aktueller_Kurs") or r.get("regularMarketPrice") or r.get("currentPrice"), default=None)

    entry_zone = str(rr.get("entry_zone") or decision.get("entry_zone") or _radar_v182_entry_zone_text(r) or "-").strip()
    try:
        zone_low, zone_high = _radar_v182_parse_zone(entry_zone)
    except Exception:
        zone_low, zone_high = None, None

    stop = _v230_safe_float(rr.get("stop"), default=None)
    if stop is None:
        try:
            stop = _radar_v182_stop_value(r)
        except Exception:
            stop = None
    stop = _v230_safe_float(stop, default=None)

    target = _v230_safe_float(rr.get("tp1"), default=None)
    target_source = str(rr.get("target_source") or "strukturelles Hauptziel").strip()
    if target is None:
        try:
            target, target_source = _radar_v209_main_target_value(r, price=price)
        except Exception:
            target, target_source = None, target_source
    target = _v230_safe_float(target, default=None)

    # Default-Entry: aktueller Kurs, weil Positionsgroesse meist fuer eine jetzt
    # gepruefte Order berechnet wird. Die Entry-Zone bleibt sichtbar und kann
    # manuell uebersteuert werden.
    entry_default = price
    if entry_default is None and zone_high is not None:
        entry_default = zone_high

    status_hint = str(decision.get("bucket") or "-").strip()
    grade = str(decision.get("grade") or "-").strip()
    action = str(decision.get("next_step") or "-").strip()
    crv = _v230_safe_float(rr.get("crv"), default=None)

    return {
        "price": price,
        "entry_default": entry_default,
        "entry_zone": entry_zone,
        "zone_low": _v230_safe_float(zone_low, default=None),
        "zone_high": _v230_safe_float(zone_high, default=None),
        "stop": stop,
        "target": target,
        "target_source": target_source or "Ziel",
        "bucket": status_hint,
        "grade": grade,
        "action": action,
        "crv": crv,
    }


def _v230_calculate_position_size(entry, stop, target, account_size, risk_pct, max_position_pct=None):
    entry = _v230_safe_float(entry, default=None)
    stop = _v230_safe_float(stop, default=None)
    target = _v230_safe_float(target, default=None)
    account_size = _v230_safe_float(account_size, default=None)
    risk_pct = _v230_safe_float(risk_pct, default=None)
    max_position_pct = _v230_safe_float(max_position_pct, default=None)

    if entry is None or stop is None or account_size is None or risk_pct is None:
        return {"ok": False, "error": "Entry, Stop, Depotgroesse oder Risiko fehlen."}
    if entry <= 0 or stop <= 0 or account_size <= 0 or risk_pct <= 0:
        return {"ok": False, "error": "Entry, Stop, Depotgroesse und Risiko muessen groesser als 0 sein."}
    unit_risk = entry - stop
    if unit_risk <= 0:
        return {"ok": False, "error": "Stop/Invalidierung liegt nicht unter dem Entry. Bitte Stop manuell pruefen."}

    risk_amount = account_size * (risk_pct / 100.0)
    shares_raw = risk_amount / unit_risk
    shares_floor = int(max(0, np.floor(shares_raw)))
    position_value = shares_floor * entry
    actual_risk = shares_floor * unit_risk
    actual_risk_pct = (actual_risk / account_size * 100.0) if account_size else None
    stop_distance_pct = unit_risk / entry * 100.0

    max_position_value = None
    max_position_shares = None
    capped = False
    if max_position_pct is not None and max_position_pct > 0:
        max_position_value = account_size * (max_position_pct / 100.0)
        max_position_shares = int(max(0, np.floor(max_position_value / entry)))
        if shares_floor > max_position_shares:
            shares_floor = max_position_shares
            position_value = shares_floor * entry
            actual_risk = shares_floor * unit_risk
            actual_risk_pct = (actual_risk / account_size * 100.0) if account_size else None
            capped = True

    reward = None
    crv = None
    if target is not None and target > entry:
        reward = target - entry
        crv = reward / unit_risk if unit_risk > 0 else None

    return {
        "ok": True,
        "risk_amount": risk_amount,
        "unit_risk": unit_risk,
        "shares_raw": shares_raw,
        "shares": shares_floor,
        "position_value": position_value,
        "actual_risk": actual_risk,
        "actual_risk_pct": actual_risk_pct,
        "stop_distance_pct": stop_distance_pct,
        "target": target,
        "reward": reward,
        "crv": crv,
        "max_position_value": max_position_value,
        "max_position_shares": max_position_shares,
        "capped": capped,
    }


# ---------- v24.5: Positions-/Exit-Monitor mit Persistenz ----------


RISK_BASIS_VERSION = 'v30.21bv'


def build_analysis_risk_basis(result, *, generated_at=''):
    """Immutable, provider-free calculator inputs from ONE central analysis."""
    import hashlib
    import json
    from modules.technical_crv import build_technical_crv_package, parse_price_zone
    r=result or {}
    pkg=build_technical_crv_package(r)
    n=lambda value:_v230_safe_float(value,default=None)
    price=n(pkg.get('technical_crv_price_reference'))
    stop=n(pkg.get('technical_crv_stop_now'))
    target=n(pkg.get('technical_crv_target'))
    atr=n(r.get('atr',r.get('ATR')))
    atr_pct=atr/price*100 if atr is not None and price and price>0 else n(r.get('atr_pct'))
    structure=(pkg.get('technical_crv_stop_diagnostics') or {}).get('structure') or {}
    confirmed=pkg.get('technical_crv_stop_quality')=='confirmed'
    source=pkg.get('technical_crv_stop_source') or 'Stop nicht belegt'
    zone=r.get('suggested_entry_zone') or r.get('entry_zone') or '-'
    low,high=parse_price_zone(zone)
    basis=dict(version=RISK_BASIS_VERSION,ticker=str(r.get('ticker') or r.get('Ticker') or '').upper(),
        price=price,entry_default=price,entry_zone=str(zone),zone_low=low,zone_high=high,
        stop=stop,target=target,target_source=pkg.get('technical_crv_target_source'),
        crv=n(pkg.get('technical_crv_now')),crv_entry=n(pkg.get('technical_crv_entry')),
        crv_eligible=bool(confirmed and pkg.get('technical_crv_target_kind') not in ('missing',None)),
        stop_quality=pkg.get('technical_crv_stop_quality'),stop_source=source,
        risk_stop_source=source,pre_atr_stop_source=source,pre_atr_stop=stop,
        technical_stop=stop,legacy_technical_stop=stop,atr=atr,atr_pct=atr_pct,
        chart_invalidation_stop=n(structure.get('stop')) if confirmed else None,
        chart_invalidation_source=structure.get('source') if confirmed else 'Keine passende bestätigte Setup-Struktur',
        chart_invalidation_kind='setup_structure' if confirmed else 'missing',
        overarching_structure=(pkg.get('technical_crv_stop_diagnostics') or {}).get('overarching_structure') or {},
        data_date=str(r.get('ts') or ''),generated_at=str(generated_at),
        currency=str(r.get('currency') or r.get('ccy') or (r.get('info') or {}).get('currency') or ''),
        stop_reason=str((r.get('stop_selection') or {}).get('reason') or ''),
        setup_type=str(r.get('setup_type') or '-'))
    basis['fingerprint']=hashlib.sha256(json.dumps(basis,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()[:20]
    return basis


def risk_basis_from_row(row):
    """Read a scan snapshot only; never blend quotes or launch another analysis."""
    from copy import deepcopy
    basis=(row or {}).get('__risk_basis')
    if not isinstance(basis,dict) or basis.get('version')!=RISK_BASIS_VERSION:
        return None
    if str(basis.get('ticker') or '').upper()!=str((row or {}).get('Ticker') or '').upper():
        return None
    if not basis.get('fingerprint') or _v230_safe_float(basis.get('price'),0)<=0:
        return None
    return deepcopy(basis)


def atr_stop_scenario(basis):
    """Optional wider ATR comparison; the canonical analysis stop is immutable."""
    ref=_v230_safe_float(basis.get('price'))
    atr_pct=_v230_safe_float(basis.get('atr_pct'))
    stop=_v230_safe_float(basis.get('stop'))
    if not ref or ref<=0 or not atr_pct or atr_pct<=0:
        return {'stop':None,'guard_stop':None,'multiple':None}
    mult=2.2 if atr_pct>=10 else 2.0 if atr_pct>=7 else 1.8 if atr_pct>=5 else 1.7 if atr_pct>=3.5 else 1.6
    guard=ref*(1-min(80,max(3.5,atr_pct*mult))/100)
    return {'stop':min(stop,guard) if stop is not None and 0<stop<ref else guard,
            'guard_stop':guard,'multiple':mult}


def render_snapshot_calculator(st, row, *, currency='USD'):
    """Standard = exact scan basis; ATR/manual values are explicit scenarios."""
    basis=risk_basis_from_row(row)
    if basis is None:
        st.info('Für diesen Stand fehlt die gemeinsame Analyse-Risikobasis. Bitte den Wert oder die Watchlist neu scannen.')
        return
    ticker=basis['ticker'];prefix='risk_bv::'+ticker+'::'+basis['fingerprint']
    revision_key=prefix+'::revision'
    if st.button('Auf Analysewerte zurücksetzen',key=prefix+'::reset'):
        st.session_state[revision_key]=int(st.session_state.get(revision_key,0))+1
        st.rerun()
    key=prefix+'::'+str(st.session_state.get(revision_key,0))
    mode=st.selectbox('Berechnungsbasis',['Analysewerte (Standard)','ATR-Vergleich','Manuelles Szenario'],key=key+'::mode')
    manual=mode=='Manuelles Szenario';atr_mode=mode=='ATR-Vergleich'
    scenario=atr_stop_scenario(basis)
    selected_stop=scenario['stop'] if atr_mode else basis.get('stop')
    st.caption(f"Gemeinsame Analysebasis · {ticker} · Kursdaten {basis.get('data_date') or 'nicht ausgewiesen'} · Scan {basis.get('generated_at') or 'nicht ausgewiesen'}")
    st.caption('Analyse-Stop: '+_v230_price_text(basis.get('stop'))+' '+currency+' · '+str(basis.get('stop_source') or '-')+
               ' · Ziel: '+_v230_price_text(basis.get('target'))+' · '+str(basis.get('target_source') or '-'))
    if basis.get('stop_reason'):st.caption(basis['stop_reason'])
    overarching=basis.get('overarching_structure') or {}
    if overarching.get('stop') is not None:
        st.caption('Übergeordnete Referenz: Unterstützung '+_v230_price_text(overarching.get('support'))+
                   ' · gepufferter Stop '+_v230_price_text(overarching.get('stop'))+' '+currency+'; nicht automatisch der Trade-Stop.')
    quote=_v230_safe_float((row or {}).get('Kurs'))
    if quote and abs(quote-basis['price'])>0.005:
        st.caption('Screenerkurs '+_v230_price_text(quote)+' · CRV-/Rechner-Referenz '+_v230_price_text(basis['price'])+'; keine Mischung der Kursbasen.')
    if basis.get('stop_quality')!='confirmed':
        st.info('Kein passender bestätigter Strukturstop für den aktuellen Trade. Analyse-Stop und CRV sind als Ersatz-/Näherungsszenario einzuordnen.')
    if atr_mode:
        st.info('ATR-Vergleich: separates Szenario. Analyse-Stop und technische Ampel bleiben unverändert.')
        if scenario.get('multiple') is not None:
            st.caption(f"ATR-Schutz {scenario['multiple']:.1f}x · rohe ATR-Marke {_v230_price_text(scenario.get('guard_stop'))}; ein weiterer Analyse-Stop wird beibehalten.")
    elif manual:st.info('Manuelles Szenario: Entry, Stop und Ziel gelten nur für diese Rechnung.')
    c1,c2,c3=st.columns(3)
    entry=c1.number_input('Geplanter Entry',min_value=0.0,value=float(basis.get('price') or 0),format='%.4f',step=.01,key=key+'::'+mode+'::entry',disabled=not manual)
    stop=c2.number_input('Stop',min_value=0.0,value=float(selected_stop or 0),format='%.4f',step=.01,key=key+'::'+mode+'::stop',disabled=not manual)
    target=c3.number_input('Ziel',min_value=0.0,value=float(basis.get('target') or 0),format='%.4f',step=.01,key=key+'::'+mode+'::target',disabled=not manual)
    c1,c2,c3=st.columns(3)
    account=c1.number_input('Depotgröße',min_value=0.0,value=float(st.session_state.get('v230_account_size',50000)),step=1000.,key='risk_bv_account')
    risk=c2.number_input('Risiko pro Trade (%)',min_value=.01,max_value=10.0,value=float(st.session_state.get('v230_risk_pct',.5)),step=.05,key='risk_bv_risk')
    cap=c3.number_input('Maximaler Positionsanteil (%)',min_value=0.0,max_value=100.,value=float(st.session_state.get('v230_max_position_pct',12.)),step=1.,key='risk_bv_cap')
    st.session_state.update(v230_account_size=account,v230_risk_pct=risk,v230_max_position_pct=cap)
    if atr_mode and selected_stop is None:
        st.warning('ATR-Vergleich nicht verfügbar: belastbarer ATR fehlt.')
        return
    calc=_v230_calculate_position_size(entry,stop,target,account,risk,max_position_pct=cap)
    if not calc.get('ok'):
        st.warning(calc.get('error') or 'Berechnung nicht verfügbar.')
        return
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Stückzahl',str(calc['shares']))
    c2.metric('Positionswert',f"{calc['position_value']:.2f} {currency}")
    c3.metric('Rechnerisches Risiko',f"{calc['actual_risk']:.2f} {currency}")
    crv=calc.get('crv')
    crv_label='CRV auf Ziel' if mode=='Analysewerte (Standard)' and basis.get('crv_eligible') else 'CRV auf Ziel (Szenario)'
    c4.metric(crv_label,f'{crv:.2f}' if crv is not None else 'n/a')
    st.caption(f"Risiko je Aktie {calc['unit_risk']:.4f} {currency} · Stop-Abstand {calc['stop_distance_pct']:.1f}% · Berechnungsbasis: {mode}")
    if calc.get('capped'):st.caption('Stückzahl durch den maximalen Positionsanteil begrenzt.')
    st.caption('CRV separat beurteilen. Berechnete Stops sind keine gesetzten Brokerorders; Ausführungsabweichungen sind nicht eingerechnet.')
