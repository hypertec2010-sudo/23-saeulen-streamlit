"""Read-only evidence for an existing decision; never grants or revokes entry."""
from __future__ import annotations
import math


def finite(value):
    try:
        number=float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def build_invalidation_diagnostics(result, decision, *, price, used_level):
    """Describe the EXACT level and quote consumed by the existing alert path.

    The alert's text-parsed reference is not silently replaced by another level.
    A separate numeric field is exposed for discrepancy detection only.
    """
    r=result or {};d=decision or {};wave=r.get('wave_structure_pkg') or {}
    if not isinstance(wave,dict):wave={}
    p=finite(price);level=finite(used_level)
    if p is not None and p<=0:p=None
    if level is not None and level<=0:level=None
    numeric=finite(wave.get('wave_invalidation_price'))
    if numeric is None:numeric=finite(d.get('wave_invalidation_price'))
    quote_source='Provider-/Callback-Referenz'
    if p is not None:
        for key in ('live_price','Aktueller_Kurs','analysis_price','Analyse_Kursbasis','price',
                    'regularMarketPrice','regular_market_price','current_price','currentPrice',
                    'last_price','lastPrice','previousClose','Kurs'):
            val=finite(r.get(key))
            if val is not None and abs(val-p)<=max(abs(p)*1e-9,1e-9):
                quote_source=key;break
    breached=bool(p is not None and level is not None and p<=level)
    distance=(p/level-1)*100 if p is not None and level is not None else None
    mismatch=bool(level is not None and numeric is not None and abs(level-numeric)>max(abs(level)*1e-6,0.005))
    raw=str(d.get('wave_invalidation') or '-')
    detail=(f"Wave-/Setup-Modelllevel {level:.2f}; Vergleichskurs {p:.2f} ({quote_source}); "
            f"Abstand {distance:+.2f}%; Strukturstop nicht allein dadurch bestätigt"
            if level is not None and p is not None else
            'Wave-/Setup-Modell: verwendetes Level oder Vergleichskurs fehlt')
    if mismatch:detail+=f'; numerisches Modellfeld weicht ab ({numeric:.2f}) – Quelle prüfen'
    return {'level':level,'price':p,'price_source':quote_source,'distance_pct':distance,
            'source':'Wave-/Setup-Modell, bestehender Alertpfad',
            'quality':'Modellreferenz; technische Stop-Bestätigung separat prüfen',
            'raw_reference':raw,'numeric_model_level':numeric,'mismatch':mismatch,
            'breached':breached,'detail':detail}


def explain_components(components, available, *, setup_strength, action_score,
                       invalidated=False, hard_gate=False):
    """Keep missing/default components out of positive and negative evidence."""
    valid=[(name,finite(value)) for name,value in components.items()
           if available.get(name,False) and finite(value) is not None]
    drivers=sorted([(n,v) for n,v in valid if v>=60],key=lambda x:x[1],reverse=True)
    brakes=sorted([(n,v) for n,v in valid if v<60],key=lambda x:x[1])
    fmt=lambda seq:'; '.join(f'{n} {v:.0f}/100' for n,v in seq[:3])
    neutral='; '.join(f'{n}: nicht bewertet / nicht bestätigt ({finite(v):.0f} als Rechenwert)'
                     for n,v in components.items() if not available.get(n,False) and finite(v) is not None)
    reasons=[]
    if invalidated:reasons.append('Invalidierungsabzug −45')
    if hard_gate:reasons.append('Hard-Gate-Abzug −24')
    reason='; '.join(reasons) or 'keine Invalidierungs-/Hard-Gate-Abzüge'
    drivers_text=fmt(drivers) or 'Keine ausgeprägten positiven Komponenten'
    brakes_text=fmt(brakes) or 'Keine deutliche bewertete Komponenten-Bremse'
    detail=(f'Setup-Stärke vor Freigabe-Abzügen {setup_strength}/100; '
            f'bisheriger Handlungs-/Live-Score {action_score}/100. {reason}. '
            'Weitere Risiko-Abzüge und Statusbegrenzungen können hinzukommen. '
            f'Treiber: {drivers_text}. Bremsen: {brakes_text}.')
    if neutral:detail+=' Nicht bewertet: '+neutral+'.'
    return {'drivers':drivers_text,'brakes':brakes_text,'neutral':neutral,'explanation':detail}
