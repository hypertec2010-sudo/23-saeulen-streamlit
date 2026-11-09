"""Whole-universe daily pattern discovery; no CHSM finalist or trade gate."""
from uuid import uuid4

from .chart_patterns import build_chart_patterns, KINDS, PHASES


def scan_patterns(*, symbols, resolution, universe, entries, model_version,
                  source, start, history_loader, clock, progress=None):
    from . import candidate_radar as radar
    rows, errors = [], []
    aborted = False
    for i, ticker in enumerate(symbols):
        try:
            frame = history_loader(ticker)  # Exactly one existing daily-data call per symbol.
            evidence = build_chart_patterns({'ticker': ticker, 'df': frame}, now=clock())
            rows.append(dict(ticker=ticker, grade=None, status='Musterprüfung', gates=[],
                crv=None, entry_distance_pct=None, target_provenance={},
                quote_date=evidence.get('basis_date'), analyzed_at=(radar.parse_time(clock()) or radar.utcnow()).isoformat(),
                decision={}, rr_package={}, chart_patterns=evidence))
        except Exception as exc:
            info = radar.safe_scan_error(exc)
            errors.append(dict(ticker=ticker, stage='Chartmuster', **info))
            aborted = info.get('category') == 'Provider-Limit'
        if progress:
            progress(i+1, len(symbols), 'Chartmuster · '+ticker)
        if aborted:
            break
    end = radar.parse_time(clock()) or radar.utcnow()
    processed = len(rows)+len(errors)
    return dict(schema_version=radar.SCHEMA_VERSION, radar_version=radar.RADAR_VERSION,
        model_version=model_version, catalog_version=radar.CATALOG_VERSION,
        key=radar.request_key(universe, 'Chartmuster', entries, model_version),
        scan_id='radar-patterns-'+start.strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8],
        universe=universe, style='Chartmuster', source=source,
        started_at=start.isoformat(), completed_at=end.isoformat(), completed=not aborted,
        requested=len(symbols), processed=processed, skipped=len(symbols)-processed,
        skipped_symbols=symbols[processed:], abort_reason='provider_rate_limit' if aborted else '',
        symbols=list(symbols), symbols_hash=radar.universe_digest(symbols), resolution=resolution,
        rows=rows, errors=errors, light_requested=len(symbols), light_processed=processed,
        light_errors=errors, preselected_symbols=[], full_analysis_count=0,
        provider_strategy='whole-universe-patterns')


def pattern_hits(payload, *, phase='Alle', direction='Alle', kind='Alle'):
    """One row per active pattern, deterministic order; all universe rows retained."""
    hits=[]
    for row in payload.get('rows') or []:
        evidence=row.get('chart_patterns') or {}
        for pattern in evidence.get('patterns') or []:
            if pattern.get('phase') not in {'forming','confirmed'}:
                continue
            if phase!='Alle' and pattern['phase']!=phase:continue
            if direction!='Alle' and pattern.get('direction')!=direction:continue
            if kind!='Alle' and pattern.get('kind')!=kind:continue
            hits.append(dict(ticker=row['ticker'], pattern=pattern, evidence=evidence))
    return sorted(hits,key=lambda h:(h['pattern']['phase']!='confirmed',h['ticker'],h['pattern']['kind']))


def render_pattern_radar(st, payload, *, fresh, limit, navigate, manager=None):
    from .chart_pattern_ui import render_chart_patterns, pattern_detail
    rows=payload.get('rows') or []
    all_hits=pattern_hits(payload)
    blocked=sum(not (r.get('chart_patterns') or {}).get('available') for r in rows)
    stale=sum(bool((r.get('chart_patterns') or {}).get('stale')) for r in rows)
    st.write(f"**Chartmuster-Prüfung: {payload.get('processed',0)}/{payload.get('requested',0)} Werte** · keine Vollanalysen")
    st.caption(f"{sum(h['pattern']['phase']=='confirmed' for h in all_hits)} bestätigte und "
               f"{sum(h['pattern']['phase']=='forming' for h in all_hits)} entstehende Muster · "
               f"{blocked} Werte nicht prüfbar · {stale} mit älterer Tagesbasis · {len(payload.get('errors') or [])} Abruffehler.")
    st.caption('Unabhängige Tagesmuster, einschließlich bärischer Formationen. Keine Kauf-/Short- oder Wächter-Freigabe. Bestätigung ist keine Gewinnwahrscheinlichkeit.')
    phase=st.selectbox('Musterstatus',['Alle','confirmed','forming'],format_func=lambda v:PHASES.get(v,v),key='radar_patterns_phase')
    direction=st.selectbox('Musterrichtung',['Alle',1,-1],format_func=lambda v:{1:'Bullisch',-1:'Bärisch'}.get(v,v),key='radar_patterns_direction')
    kind=st.selectbox('Formation',['Alle']+list(KINDS),format_func=lambda v:KINDS[v][0] if v in KINDS else v,key='radar_patterns_kind')
    hits=pattern_hits(payload,phase=phase,direction=direction,kind=kind)
    table=[dict(Ticker=h['ticker'],Muster=h['pattern']['name'],Richtung='Bullisch' if h['pattern']['direction']==1 else 'Bärisch',
        Status=PHASES[h['pattern']['phase']],Tagesbasis=h['evidence'].get('basis_date'),
        Datenstand='Historisch' if h['evidence'].get('stale') or not fresh else 'Aktuell',
        Bezugslinie=h['pattern'].get('boundary_now'),
        Bestätigungsschwelle=pattern_detail(h['evidence'],h['pattern'])['threshold'],
        Nächster_Prüfschritt=pattern_detail(h['evidence'],h['pattern'])['action'],
        Begründung=h['pattern'].get('reason')) for h in hits]
    if table:st.dataframe(table,use_container_width=True,hide_index=True)
    else:st.info('Keine entstehenden oder bestätigten Muster für diese Filter. Die Prüfdiagnose zeigt auch verworfene Muster und fehlende Daten.')
    # Cards are limited; table, selection and diagnostics always cover the entire pool.
    shown=[]
    for h in hits:
        if h['ticker'] not in shown:shown.append(h['ticker'])
    detail_ticker=st.selectbox('Details einer Trefferaktie öffnen', ['Kartenauswahl']+shown,
        key='radar_patterns_detail_ticker')
    detail_tickers=shown[:limit] if detail_ticker=='Kartenauswahl' else [detail_ticker]
    for ticker in detail_tickers:
        st.write('**'+ticker+'**')
        render_chart_patterns(st,next(h['evidence'] for h in hits if h['ticker']==ticker),key='radar_pattern_'+ticker)
    eligible=sorted({h['ticker'] for h in hits if fresh and not h['evidence'].get('stale')})
    st.caption('Gewünschte Treffer auswählen und zur frischen Sofortanalyse öffnen. Die Übergabe startet keine Analyse; anschließend bewusst neu prüfen.')
    chosen=st.multiselect('Zur frischen Sofortanalyse übernehmen',eligible,key='radar_patterns_selection')
    if st.button('Auswahl in Sofortanalyse öffnen',disabled=not chosen,key='radar_patterns_open'):
        navigate(st,chosen)
    from .pattern_watch_ui import render_capture
    render_capture(st,payload,hits,manager=manager,fresh=fresh)
    with st.expander('Prüfdiagnose · gesamte Kandidatenliste'):
        diagnostic=[]
        for r in rows:
            e=r.get('chart_patterns') or {}
            for p in e.get('patterns') or []:
                diagnostic.append(dict(Ticker=r['ticker'],Muster=p['name'],Status=PHASES[p['phase']],Grund=p['reason'],Tagesbasis=e.get('basis_date')))
        if diagnostic:st.dataframe(diagnostic,use_container_width=True,hide_index=True)
        if payload.get('errors'):
            from .candidate_radar_ui import _render_error_diagnostics
            _render_error_diagnostics(st,payload['errors'],title='Abruffehler',expanded=True)
