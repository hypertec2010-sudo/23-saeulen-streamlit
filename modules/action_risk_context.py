"""Descriptive entry warning and structural risk; no decision changes."""
from modules.technical_crv import parse_price_zone
from modules.decision_diagnostics import finite


def build_action_risk_context(result, *, entry_zone='', ccy='', invalidated=False,
                              invalidation_diagnostics=None, stop_diagnostics=None):
    r=result or {}
    low,high=parse_price_zone(entry_zone)
    suffix=(' '+str(ccy).strip()) if str(ccy or '').strip() else ''
    parts=[]
    if low is not None and high is not None and 0 < low <= high:
        parts.append(f'Entry-Warnsignal: Unter {low:.2f}{suffix} Einstieg neu prüfen. '
                     'Bewegungen innerhalb der Entry-Zone sind keine automatische Invalidierung.')
    else:
        parts.append('Entry-Warnsignal: Keine belastbare Entry-Untergrenze verfügbar.')
    stop=finite(r.get('technical_crv_stop_now'))
    if r.get('technical_crv_stop_quality')=='confirmed' and stop is not None and stop>0:
        parts.append(f'Struktur-Risiko: Bestätigter technischer Stop bei {stop:.2f}{suffix}; '
                     'separat von der Entry-Zone beachten.')
    elif ((stop_diagnostics or r.get('technical_crv_stop_diagnostics') or {}).get('structure') or {}).get('quality')=='confirmed_swing':
        parts.append('Struktur: Bestätigte Support-Referenz vorhanden; verwendeter Stop nicht als Strukturstop bestätigt. Keine automatische Übernahme der Referenz.')
    else:
        parts.append('Struktur: Keine bestätigte technische Invalidierungsmarke aus dem Strukturstop verfügbar.')
    diag=invalidation_diagnostics or {}
    if invalidated:
        level=finite(diag.get('level'))
        reference=(f' {level:.2f}{suffix}' if level is not None and level>0 else '')
        parts.append('Bestehende Sperre: Wave-/Setup-Modelllevel'+reference+
                     ' erreicht/unterschritten; Freigabe-Check beachten. Das bestätigt keinen Strukturstop.')
    return {'label':'Entry-Warnsignal / Struktur-Risiko','text':' '.join(parts)}
