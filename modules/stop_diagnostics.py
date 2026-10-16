"""Display existing stop evidence without calculating or selecting new stops."""
from modules.structural_trade_plan import finite


def build_stop_diagnostic_rows(diagnostics, *, ccy=''):
    package=diagnostics or {}
    structure=package.get('structure') or {}
    used=package.get('used') or {}
    suffix=(' '+str(ccy).strip()) if str(ccy or '').strip() else ''
    def price(value):
        val=finite(value)
        return f'{val:.2f}{suffix}' if val is not None and val>0 else 'Nicht verfügbar'
    confirmed=structure.get('quality')=='confirmed_swing'
    support=structure.get('support') if confirmed else structure.get('candidate_support')
    stop=structure.get('stop') if confirmed else structure.get('candidate_stop')
    support_note=(str(structure.get('source') or 'Bestätigte Swing-Unterstützung') if confirmed else
                  'Einzelnes Swing-Tief; Mehrfachbestätigung fehlt' if structure.get('candidate_quality')=='single_pivot' else
                  'Mehrfach getestetes Niveau; Stop-Prüfung nicht bestanden' if support is not None else
                  'Keine nutzbare Swing-Unterstützung nach den Strukturstop-Regeln belegt')
    rows=[{'Baustein':'Erkannte Unterstützung','Kurs':price(support),'Einordnung':support_note},
          {'Baustein':'Abgeleiteter Strukturstop','Kurs':price(stop),
           'Einordnung':('Bestätigte Referenz mit Puffer; verwendeter Stop ist bestätigt'
                          if used.get('quality')=='confirmed' else
                          'Bestätigte Referenz mit Puffer; nicht automatisch verwendet') if confirmed else
                        'Verworfener Kandidat; keine Stop-Freigabe' if stop is not None else
                        'Nicht bestätigt'},
          {'Baustein':'Tatsächlich verwendeter Stop','Kurs':price(used.get('used_stop')),
           'Einordnung':str(used.get('used_stop_source') or 'Quelle nicht verfügbar')+' · '+
                        str(used.get('label') or 'Qualität nicht belegt')}]
    return rows


def render_stop_diagnostics(st, diagnostics, *, ccy=''):
    package=diagnostics or {}
    st.table(build_stop_diagnostic_rows(package,ccy=ccy))
    structure=package.get('structure') or {}
    used=package.get('used') or {}
    st.caption('Strukturprüfung: '+str(structure.get('reason') or 'Diagnose fehlt; frisch analysieren.'))
    st.caption('Verwendung im CRV: '+str(used.get('reason') or 'Diagnose fehlt; frisch analysieren.'))
    selection=package.get('selection') or {}
    if selection.get('reason'):
        st.caption('Stop-Auswahl: '+str(selection['reason']))
