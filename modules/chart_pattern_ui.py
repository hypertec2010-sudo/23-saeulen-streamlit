"""Display-only daily chart-pattern evidence, separate from trade execution."""
import pandas as pd

from modules.chart_patterns import POLICY,PHASES,patterns_brief


def _fmt(value):
    return '–' if value is None else f'{value:.2f}'.replace('.',',')


def pattern_rows(package):
    return [{'Muster':p['name'],'Richtung':'Bullisch' if p['direction']==1 else 'Bärisch',
             'Status':PHASES[p['phase']],'Bezugslinie':_fmt(p.get('boundary_now')),
             'Einordnung':p['reason']+(' Projektionsreferenz bereits berührt.' if p.get('projection_touched') else '')}
            for p in (package or {}).get('patterns',[])]


def pattern_figure(package,pattern):
    import plotly.graph_objects as go
    chart=package.get('chart') or []
    fig=go.Figure(go.Candlestick(x=[b['date'] for b in chart],open=[b['open'] for b in chart],
        high=[b['high'] for b in chart],low=[b['low'] for b in chart],close=[b['close'] for b in chart],name='Tageskerzen'))
    points=pattern.get('points') or []
    if points:
        fig.add_trace(go.Scatter(x=[p['date'] for p in points],y=[p['price'] for p in points],
            mode='lines+markers',text=[p['role'] for p in points],name='Strukturpunkte',hovertemplate='%{text}: %{y:.2f}<extra></extra>'))
    line=pattern.get('boundary')
    if line and chart:
        start=max(line['anchor_index'],chart[0]['index']);end=chart[-1]['index']
        dates={b['index']:b['date'] for b in chart}
        if start in dates:
            fig.add_trace(go.Scatter(x=[dates[start],dates[end]],
                y=[line['value']+line['slope']*(i-line['anchor_index']) for i in [start,end]],
                mode='lines',line=dict(dash='dash'),name='Nacken-/Flaggenlinie'))
    for field,name in [('invalidation','Muster-Invalidierung'),('projection','Projektionsreferenz')]:
        if pattern.get(field) is not None:
            fig.add_hline(y=pattern[field],line_dash='dot',annotation_text=name)
    fig.update_layout(height=400,xaxis_rangeslider_visible=False,yaxis_title=package.get('currency',''),
                      margin=dict(l=20,r=20,t=30,b=25),legend=dict(orientation='h'))
    return fig


def render_chart_patterns(st,package,*,key='chart_patterns'):
    p=package or {}
    with st.expander('Chartmuster · unabhängig vom Trade-Wächter',expanded=False):
        if not p.get('available'):
            st.info(p.get('reason') or 'Musterbeleg fehlt; neu analysieren.')
            if p.get('patterns'):st.dataframe(pd.DataFrame(pattern_rows(p)),hide_index=True,use_container_width=True)
            return
        if p.get('stale'):st.warning('Historische Tagesbasis '+p.get('basis_date','')+'; aktuelle Bestätigung nicht belegt.')
        st.caption(f"{p.get('bars',0)} Tageskerzen bis {p.get('basis_date','')} · {p.get('currency','')} · Schlussbestätigung nach Börsenkalender, Provider-Finalität ungeprüft.")
        st.dataframe(pd.DataFrame(pattern_rows(p)),hide_index=True,use_container_width=True)
        visible=[v for v in p['patterns'] if v['phase'] not in {'none','blocked'}]
        if visible:
            item=st.selectbox('Muster mit Levels und Belegen',range(len(visible)),
                format_func=lambda i:visible[i]['name']+' · '+PHASES[visible[i]['phase']],key=key+'_choice')
            selected=visible[item]
            st.table(pd.DataFrame([{'Level':'Bezugslinie zur letzten Tageskerze','Wert':_fmt(selected['boundary_now'])},
                {'Level':'Bestätigungspuffer','Wert':_fmt(selected.get('confirmation_buffer'))},
                {'Level':'Feste Muster-Invalidierung','Wert':_fmt(selected['invalidation'])},
                {'Level':'Projektionsreferenz · kein verbindliches Kursziel','Wert':_fmt(selected['projection'])}]))
            st.caption('Struktur belegbar ab '+str(selected.get('structure_known_at') or '–')+
                       ' · Tagesbestätigung '+str(selected.get('confirmed_at') or 'fehlt')+
                       ' · Ausbruchsvolumen '+(f"{selected['volume_ratio']:.2f}×" if selected.get('volume_ratio') is not None else 'nicht geprüft'))
            st.caption(selected.get('volume_note',''))
            st.plotly_chart(pattern_figure(p,selected),use_container_width=True,key=key+'_figure')
        st.caption('Eigene Startheuristiken, keine validierte Trefferquote. Musterbestätigung ist keine Kauf-/Short- oder Wächter-Freigabe. Ampel, Analyse-Stop und bestehende Handelsregeln bleiben unverändert.')


def render_screener_patterns(st,frame):
    if not isinstance(frame,pd.DataFrame) or frame.empty or '__chart_patterns' not in frame:return
    rows=[]
    for row in frame.to_dict('records'):
        p=row.get('__chart_patterns')
        if not isinstance(p,dict) or p.get('policy')!=POLICY:
            rows.append({'Ticker':row.get('Ticker'),'Muster':'Neu scannen','Status':'Nicht prüfbar'});continue
        for record in pattern_rows(p):rows.append({'Ticker':row.get('Ticker'),**record,'Tagesbasis':p.get('basis_date'),'Historisch':bool(p.get('stale'))})
    with st.expander('Chartmuster · alle analysierten Werte',expanded=False):
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption('Bärische und bullische Tagesmuster unabhängig von der Trendampel. Fehlende alte Belege benötigen einen neuen Scan; keine zusätzlichen Providerabfragen.')
