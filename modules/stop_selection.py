"""Choose the productive stop before sizing/targets in both analysis engines."""
from modules.structural_trade_plan import confirmed_support_stop, trade_support_stop, TRADE_STOP_POLICY, finite, stop_evidence
from modules.technical_crv import parse_price_zone

LONG_SETUPS = frozenset({'Breakout','Breakout-Retest','Pullback an MA20',
                         'Pullback an MA50','Rebound','Range-Breakout','Trendfolge'})


def select_productive_stop(df, *, price, entry_zone, atr, setup_type,
                           fallback_stop, fallback_source, position_mode=False):
    result={'stop':fallback_stop,'source':fallback_source,'selected':'existing',
            'previous_stop':fallback_stop,'previous_source':fallback_source}
    if position_mode:
        return {**result,'reason':'Positionsmodus: bestehende Stop-Führung beibehalten.'}
    if setup_type not in LONG_SETUPS:
        return {**result,'reason':'Kein passendes erkanntes Long-Setup; bisherige Stop-Auswahl beibehalten.'}
    low,high=parse_price_zone(entry_zone)
    p=finite(price)
    if p is None or p<=0 or low is None or high is None or not 0<low<=high:
        return {**result,'reason':'Kurs oder vollständige Entry-Zone fehlt; bisherige Stop-Auswahl beibehalten.'}
    result["policy"] = TRADE_STOP_POLICY
    result["overarching_structure"] = confirmed_support_stop(df,current_price=p,entry_reference=high,atr=atr)
    structure=trade_support_stop(df,current_price=p,entry_reference=high,atr=atr)
    stop=finite(structure.get('stop'))
    result['structure']=structure
    if structure.get('quality')!='confirmed_swing' or stop is None:
        # No suitable recent structure: disclose an ATR proxy instead of
        # silently substituting a distant long-term support as the trade stop.
        av=finite(atr)
        proxy=round(p-1.8*av,2) if av is not None and av>0 else None
        if proxy is not None and 0<proxy<min(p,low):
            return {**result,'stop':proxy,'source':'ATR-Risikostop (1,8x; keine passende Setup-Struktur)',
                    'selected':'atr_proxy','reason':(structure.get('reason') or 'Kein bestätigter Strukturstop.')+
                    ' ATR-Ersatz für den Trade; übergeordnete Referenz wird getrennt ausgewiesen.'}
        return {**result,'reason':structure.get('reason') or 'Kein bestätigter Strukturstop.'}
    # Whole entry zone must remain above the stop, including after cent rounding.
    selected=round(stop,2)
    if not 0<selected<min(p,low) or not stop<min(p,low):
        return {**result,'reason':'Strukturstop liegt nicht unter Kurs und gesamter Entry-Zone; bisherige Stop-Auswahl beibehalten.'}
    if stop_evidence(selected,structure['source'],structure)['quality']!='confirmed':
        return {**result,'reason':'Cent-Rundung passt nicht zur bestehenden Strukturstop-Toleranz; bisherige Stop-Auswahl beibehalten.'}
    return {**result,'stop':selected,'source':structure['source'],
            'selected':'confirmed_structure',
            'reason':'Bestätigter Strukturstop für erkanntes Long-Setup unter Kurs und gesamter Entry-Zone bevorzugt.'}
