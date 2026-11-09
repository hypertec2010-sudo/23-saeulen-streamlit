"""Display-only daily chart-pattern evidence, separate from trade execution."""
import pandas as pd
import json
from modules.trade_watch import number

from modules.chart_patterns import POLICY,PHASES,patterns_brief


def _fmt(value):
    return '–' if value is None else f'{value:.2f}'.replace('.',',')


def pattern_rows(package):
    return [{'Muster':p['name'],'Richtung':'Bullisch' if p['direction']==1 else 'Bärisch',
             'Status':PHASES[p['phase']],'Bezugslinie':_fmt(p.get('boundary_now')),
             'Einordnung':p['reason']+(' Projektionsreferenz bereits berührt.' if p.get('projection_touched') else '')}
            for p in (package or {}).get('patterns',[])]


def pattern_detail(package,pattern):
    """Explain saved daily evidence; never manufacture an entry or trade order."""
    chart=package.get('chart') or []
    close=number(chart[-1].get('close')) if chart else None
    line=number(pattern.get('boundary_now'));pad=number(pattern.get('confirmation_buffer'))
    direction=pattern.get('direction',1)
    threshold=line+direction*pad if line is not None and pad is not None else None
    distance=(threshold-close)*direction/close*100 if threshold is not None and close and close>0 else None
    phase=pattern.get('phase')
    if package.get('stale'):
        action='Zuerst Tagesdaten neu scannen. Dieser historische Beleg bestätigt keinen aktuellen Einstieg.'
    elif phase=='forming':
        action='Abgeschlossenen Tagesschluss '+('oberhalb' if direction==1 else 'unterhalb')+' der Bestätigungsschwelle abwarten. '
        action+=('Zusätzlich muss der erforderliche Volumenbeleg erfüllt sein.' if pattern.get('volume_required') else 'Volumen ist bei diesem Muster ein ergänzender Befund.')
    elif phase=='confirmed':
        action='Frische Sofortanalyse öffnen und Kurslage, Invalidierung sowie verbleibenden Zielspielraum prüfen. Eine Tagesbestätigung allein liefert keinen ausführbaren Einstieg.'
    elif phase=='invalid':
        action='Diesen Musterbeleg verwerfen. Erst eine neue Struktur mit eigenem Beleg erneut prüfen.'
    else:action='Kein aktiver Musterplan; neue Tagesdaten bzw. einen belastbaren Strukturbeleg abwarten.'
    if phase=='forming' and not package.get('stale') and distance is not None and distance<0:
        action='Der letzte Tagesschluss liegt bereits jenseits der aktuellen Schwelle, die vollständige Bestätigung fehlt aber noch. Fehlenden Volumen-/Strukturbeleg laut Befund abwarten; den Kursbruch allein nicht als Freigabe verwenden.'
    if pattern.get('projection_touched'):
        action+=' Die Projektionsreferenz wurde bereits berührt; sie ist kein noch offenes Ziel.'
    if pattern.get('break_before_known'):
        action+=' Der erste Bruch lag vor vollständiger Pivot-Bestätigung; kein rechtzeitig bekanntes historisches Einstiegssignal.'
    return dict(close=close,threshold=threshold,distance_pct=distance,action=action,
        rule=('Tagesschluss > ' if direction==1 else 'Tagesschluss < ')+_fmt(threshold),
        direction='Bullisch' if direction==1 else 'Bärisch')


def pattern_figure(package,pattern,*,window='Musterfokus',show_levels=True,show_labels=True,height=680):
    import plotly.graph_objects as go
    chart=package.get('chart') or []
    points=pattern.get('points') or []
    if chart:
        start=max(chart[0]['index'],min((p['index'] for p in points),default=chart[-1]['index']-50)-8) if window=='Musterfokus' else chart[-1]['index']-int(window)+1 if window.isdigit() else chart[0]['index']
        chart=[b for b in chart if b['index']>=start]
    fig=go.Figure(go.Candlestick(x=[b['date'] for b in chart],open=[b['open'] for b in chart],
        high=[b['high'] for b in chart],low=[b['low'] for b in chart],close=[b['close'] for b in chart],name='Tageskerzen',
        increasing_line_color='#079669',decreasing_line_color='#dc3848'))
    visible_dates={b['date'] for b in chart}
    points=[p for p in points if p['date'] in visible_dates]
    if points:
        fig.add_trace(go.Scatter(x=[p['date'] for p in points],y=[p['price'] for p in points],
            mode='lines+markers+text' if show_labels else 'lines+markers',text=[p['role'] for p in points],
            textposition='top center',textfont=dict(size=13),marker=dict(size=10,color='#7c3aed'),
            line=dict(color='#7c3aed',width=2),name='Strukturpunkte',hovertemplate='%{text}<br>%{x}: %{y:.2f}<extra></extra>'))
    line=pattern.get('boundary')
    if line and chart:
        start=max(line['anchor_index'],chart[0]['index']);end=chart[-1]['index']
        dates={b['index']:b['date'] for b in chart}
        if start in dates:
            fig.add_trace(go.Scatter(x=[dates[start],dates[end]],
                y=[line['value']+line['slope']*(i-line['anchor_index']) for i in [start,end]],
                mode='lines',line=dict(dash='dash',color='#d58a00',width=3),name='Nacken-/Flaggenlinie'))
    if show_levels:
        for field,name,color in [('invalidation','Invalidierung','#dc3848'),('projection','Projektion','#079669')]:
            if pattern.get(field) is not None:
                fig.add_hline(y=pattern[field],line_dash='dot',line_color=color,
                    annotation_text=name+' '+_fmt(pattern[field]),annotation_position='right')
    fig.update_layout(height=height,xaxis_rangeslider_visible=False,yaxis_title=package.get('currency',''),
        title=dict(text=package.get('ticker','')+' · '+pattern.get('name','')+' · Tagesbasis',font=dict(size=18)),
        dragmode='pan',hovermode='x unified',margin=dict(l=35,r=135,t=75,b=75),
        legend=dict(orientation='h',y=-.12),font=dict(size=13),template='plotly_white')
    fig.update_xaxes(type='category',nticks=10,showspikes=True,spikemode='across',spikesnap='cursor')
    fig.update_yaxes(fixedrange=False,showspikes=True,tickformat='.2f',side='right')
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
        visible=sorted([v for v in p['patterns'] if v['phase'] not in {'none','blocked'}],
            key=lambda v:({'confirmed':0,'forming':1,'invalid':2}.get(v['phase'],3),-v.get('end_index',-1)))
        if visible:
            item=st.selectbox('Muster mit Levels und Belegen',range(len(visible)),
                format_func=lambda i:visible[i]['name']+' · '+PHASES[visible[i]['phase']],key=key+'_choice')
            selected=visible[item]
            detail=pattern_detail(p,selected)
            st.markdown('**'+detail['direction']+' · '+PHASES[selected['phase']]+' · nächster Prüfschritt**')
            st.info(detail['action'])
            st.write('**Befund:** '+selected['reason'])
            descriptions={
                'bull_flag':'Aufwärtsimpuls mit enger, leicht gegenläufiger Konsolidierung. Entscheidend ist der bestätigte Bruch der oberen Flaggenbegrenzung.',
                'bear_flag':'Abwärtsimpuls mit enger Erholung/Konsolidierung. Entscheidend ist der bestätigte Bruch der unteren Flaggenbegrenzung.',
                'head_shoulders':'Drei Hochs mit höherem mittleren Kopf. Erst der bestätigte Bruch unter die Nackenlinie belegt die bärische Umkehr.',
                'inverse_head_shoulders':'Drei Tiefs mit tieferem mittleren Kopf. Erst der bestätigte Bruch über die Nackenlinie belegt die bullische Umkehr.',
                'double_bottom':'Zwei ähnlich tiefe Tiefpunkte mit Zwischenhoch. Die bullische Umkehr benötigt den bestätigten Bruch über dieses Zwischenhoch.',
                'double_top':'Zwei ähnlich hohe Hochpunkte mit Zwischentief. Die bärische Umkehr benötigt den bestätigten Bruch unter dieses Zwischentief.'}
            st.write(descriptions.get(selected.get('kind'),''))
            st.table(pd.DataFrame([
                {'Level':'Letzter abgeschlossener Tagesschluss','Wert':_fmt(detail['close'])},
                {'Level':'Nacken-/Flaggenlinie zur letzten Tageskerze','Wert':_fmt(selected['boundary_now'])},
                {'Level':'Bestätigungsschwelle inkl. Puffer · kein Entry','Wert':detail['rule']},
                {'Level':'Gerichteter Abstand zur Schwelle · negativ = bereits jenseits','Wert':_fmt(detail['distance_pct'])+' %' if detail['distance_pct'] is not None else '–'},
                {'Level':'Feste Muster-Invalidierung','Wert':_fmt(selected['invalidation'])},
                {'Level':'Projektionsreferenz · kein verbindliches Kursziel','Wert':_fmt(selected['projection'])}]))
            st.caption('Schwelle = aktuelle Bezugslinie plus Puffer bei bullischen bzw. minus Puffer bei bärischen Mustern. Bei schrägen Linien verändert sie sich mit jeder Tageskerze; gezeigter Wert gilt nur zur gespeicherten Tagesbasis.')
            st.caption('Struktur belegbar ab '+str(selected.get('structure_known_at') or '–')+
                       ' · Tagesbestätigung '+str(selected.get('confirmed_at') or 'fehlt')+
                       ' · Ausbruchsvolumen '+(f"{selected['volume_ratio']:.2f}×" if selected.get('volume_ratio') is not None else 'nicht geprüft'))
            st.caption(selected.get('volume_note',''))
            if selected.get('invalidated_at'):st.warning('Muster verworfen am '+selected['invalidated_at'])
            evidence=selected.get('evidence') or {}
            labels_map={'shape_span_bars':('Strukturspanne · Handelstage',1),
                'prior_trend_pct':('Vorlauf in Richtung des alten Trends · %',1),
                'pole_days':('Impulsdauer · Handelstage',1),'flag_days':('Flaggendauer · Handelstage',1),
                'impulse_volume_ratio':('Impulsvolumen / vorherige 20 Tage · ×',1),
                'flag_volume_ratio':('Flaggenvolumen / Impulsvolumen · ×',1)}
            evidence_rows=[{'Strukturmerkmal':label,'Befund':_fmt(number(evidence[field]))}
                for field,(label,_) in labels_map.items() if number(evidence.get(field)) is not None]
            if evidence_rows:
                with st.expander('Musteraufbau und Volumen im Detail'):st.table(pd.DataFrame(evidence_rows))
            st.caption('Invalidierung gilt für diese Formation und ist kein persönlicher Stop. Die Projektion ist geometrisch abgeleitet und kein geprüfter Handelsplan. Bärische Muster können bei gehaltenen Aktien Anlass für eine Risikoprüfung sein; sie sind keine automatische Short-Freigabe.')
            window=st.selectbox('Chartausschnitt',['Musterfokus','60','120','Gesamter Beleg'],
                format_func=lambda v:v+' Tageskerzen' if v.isdigit() else v,key=key+'_window')
            labels=st.checkbox('Strukturpunkte beschriften',value=True,key=key+'_labels')
            levels=st.checkbox('Invalidierung und Projektion im Chart zeigen',value=True,key=key+'_levels')
            st.plotly_chart(pattern_figure(p,selected,window=window,show_levels=levels,show_labels=labels),
                use_container_width=True,key=key+'_figure',config=dict(scrollZoom=True,displaylogo=False,
                modeBarButtonsToAdd=['drawline','drawrect','eraseshape'],toImageButtonOptions=dict(format='png',scale=2)))
            st.caption('Mit Mausrad zoomen, ziehen zum Verschieben; Doppelklick setzt die Ansicht zurück. Zoom, Bildexport und Zeichenwerkzeuge stehen in der Chartleiste. Zeichnungen dienen nur der Ansicht und werden nicht gespeichert. Bei gedrängter Skala die Referenzlinien ausblenden.')
            with st.expander('Strukturpunkte und Prüfbefund exportieren'):
                st.table(pd.DataFrame(selected.get('points') or []))
                export=dict(ticker=p.get('ticker'),basis_date=p.get('basis_date'),currency=p.get('currency'),
                    stale=p.get('stale'),pattern=selected,review=detail,note='Analysebeleg, kein Handelsauftrag oder Trade-Wächter-Plan.')
                summary='\n'.join([str(p.get('ticker',''))+' · '+selected['name'],
                    'Tagesbasis: '+str(p.get('basis_date','')),PHASES[selected['phase']]+' · '+detail['direction'],
                    selected['reason'],'Bestätigung zur gespeicherten Tagesbasis: '+detail['rule'],
                    'Invalidierung: '+_fmt(selected.get('invalidation')),
                    'Projektionsreferenz: '+_fmt(selected.get('projection')),detail['action'],
                    'Analysebeleg, kein ausführbarer Handelsplan oder Auftrag.'])
                st.download_button('Prüfnotiz als Text herunterladen',summary,
                    file_name=str(p.get('ticker','Muster'))+'_'+selected['kind']+'_Pruefnotiz.txt',mime='text/plain',key=key+'_note')
                st.download_button('Musterbeleg als JSON herunterladen',json.dumps(export,ensure_ascii=False,indent=2),
                    file_name=str(p.get('ticker','Muster'))+'_'+selected['kind']+'_Beleg.json',mime='application/json',key=key+'_export')
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
