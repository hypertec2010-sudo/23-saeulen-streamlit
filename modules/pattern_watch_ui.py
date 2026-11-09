"""User-scoped saved pattern list; monitoring only while this panel is open."""
from modules import pattern_watch as watch
from modules.trade_watch_ui import shown_time


def render_capture(st,payload,hits,*,manager,fresh):
    if manager is None:return
    eligible=[h for h in hits if fresh and not h['evidence'].get('stale') and not h['pattern'].get('projection_touched')]
    if not eligible:return
    st.markdown('**Formationen für den Muster-Wächter vormerken**')
    selected=st.multiselect('Konkrete Muster auswählen',list(range(len(eligible))),
        format_func=lambda i:eligible[i]['ticker']+' · '+eligible[i]['pattern']['name'],key='pattern_watch_capture')
    days=st.selectbox('Beobachtungsdauer',[14,7,30],format_func=lambda d:f'{d} Kalendertage',key='pattern_watch_days')
    if st.button('Ausgewählte Muster beobachten',disabled=not selected,key='pattern_watch_add'):
        try:
            items=watch.load(manager);messages=[]
            for i in selected:
                h=eligible[i]
                try:items=watch.create(items,h['evidence'],h['pattern'],days=days)
                except ValueError as exc:messages.append(str(exc))
            watch.save(manager,items)
            st.success('Musterliste gespeichert. Oben im Muster-Wächter gezielt aktualisieren.')
            for message in set(messages):st.warning(message)
            st.rerun()
        except (ValueError,RuntimeError) as exc:st.error(str(exc))


def render_watcher(st,*,manager,provider_factory=None,navigate=None):
    with st.expander('Muster-Wächter · vorgemerkte Formationen',expanded=False):
        st.caption('Nur vorgemerkte Aktien prüfen · kein Radar und keine Vollanalyse · keine Brokerorders.')
        st.info('Automatische Prüfungen laufen nur bei geöffneter Radar-Seite und verbundener Sitzung. Keine Hintergrundüberwachung bei geschlossenem Browser. Minutenkurse sind nicht garantiert in Echtzeit.')
        auto=st.toggle('Muster automatisch prüfen (solange geöffnet)',value=False,key='pattern_watch_auto',disabled=provider_factory is None)
        interval=st.selectbox('Muster-Prüfintervall',[300,60,900],format_func=lambda s:f'{s//60} Minuten',key='pattern_watch_interval')
        st.caption('Automatik im regulären Handelsfenster sowie 60 Minuten nach Schluss für die Tagesbestätigung. Außerhalb manuell prüfbar. Tagesbestätigung frühestens 30 Minuten nach Schluss. Tageshistorie nutzt den Provider-Cache; Minutenkurse werden gezielt aktualisiert.')
        def panel():
            if callable(getattr(manager,'begin_run',None)):manager.begin_run()
            try:items=watch.load(manager)
            except (ValueError,RuntimeError) as exc:st.warning(str(exc));return
            manual=st.button('Nur vorgemerkte Aktien jetzt aktualisieren',key='pattern_watch_refresh',disabled=provider_factory is None or not any(watch.active(i) for i in items))
            if manual or auto:
                try:
                    updated=watch.poll(items,provider_factory(),force=manual)
                    if updated!=items:
                        watch.save(manager,updated)
                        before={i['id']:i['state'] for i in items}
                        for item in updated:
                            if item['state']!=before.get(item['id']):st.toast(item['ticker']+': '+watch.LABELS.get(item['state'],item['state']))
                    items=updated
                except Exception as exc:st.error('Aktualisierung/Speicherung fehlgeschlagen: '+str(exc))
            try:storage_status=manager.status()
            except Exception:storage_status={}
            if storage_status.get('degraded') or str(storage_status.get('last_backend','')).startswith('local'):
                st.warning('Musterliste nur lokal verfügbar; geräteübergreifende Speicherung nicht bestätigt.')
            if not items:
                st.info('Noch keine Formation vorgemerkt. Im Radar-Suchprofil Chartmuster gewünschte Treffer mit „Muster beobachten“ übernehmen. Danach ist kein neuer Radar nötig.')
                return
            archive=st.checkbox('Archivierte Muster zeigen',key='pattern_watch_show_archive')
            visible=[i for i in items if archive or not i.get('archived')]
            if not visible:st.info('Alle Formationen archiviert.');return
            rows=[]
            for i in visible:
                checked=i.get('last_checked_at')
                old=checked is None or (watch.utc()-watch.utc(checked)).total_seconds()>900
                rows.append({'Ticker':i['ticker'],'Muster':i['name'],'Richtung':'Bullisch' if i['original'].get('direction')==1 else 'Bärisch',
                    'Zustand':'Nicht aktuell prüfbar' if i.get('data_blocked') else watch.LABELS.get(i['state'],i['state']),
                    'Pause':bool(i.get('paused')),'Kurs':i.get('price'),'Kursbasis':i.get('price_source','Noch nicht geprüft'),
                    'Schwelle':i.get('threshold'),'Abstand zur Schwelle %':None if i.get('data_blocked') else i.get('distance_pct'),
                    'Jenseits der Schwelle %':None if i.get('data_blocked') else i.get('beyond_threshold_pct'),'Raum zur Projektion %':None if i.get('data_blocked') else i.get('projection_room_pct'),
                    'Tagesbasis':i.get('basis_date'),'Kurszeit':shown_time(i.get('quote_at')) if 'T' in str(i.get('quote_at','')) else i.get('quote_at'),
                    'Letzte Prüfung':shown_time(checked),'Prüfung älter als 15 Minuten':old,'Datenhinweis':i.get('health')})
            st.dataframe(rows,use_container_width=True,hide_index=True)
            st.caption('Nähe: höchstens 1 % vor der Schwelle (Anzeigeheuristik). Intraday-Bruch ist nur ein früher Hinweis. Bestätigung erfordert abgeschlossene Tagesdaten und alle gespeicherten Volumen-/Strukturregeln. Negative Restprojektion bzw. bereits berührte Projektion sind kein offenes Ziel.')
            ident=st.selectbox('Beobachtete Formation / Verlauf',[i['id'] for i in visible],
                format_func=lambda ident:next(i['ticker']+' · '+i['name'] for i in visible if i['id']==ident),key='pattern_watch_detail')
            selected=next(i for i in items if i['id']==ident)
            st.write(selected.get('health',''))
            if selected.get('data_blocked'):st.warning('Letzter Musterzustand ist historisch. Aktuelle Bestätigung nicht belegt.')
            if selected.get('current',{}).get('projection_touched') or selected.get('intraday_projection_touched'):st.warning('Projektionsreferenz bereits berührt. Kein noch offenes Ziel.')
            if st.button('Beobachtung fortsetzen' if selected.get('paused') else 'Beobachtung pausieren',
                key='pattern_watch_pause',disabled=selected.get('state') in watch.TERMINAL or bool(selected.get('archived'))):
                selected['paused']=not selected.get('paused');watch.event(selected,'pause',watch.utc(),'Pause '+str(selected['paused']))
                try:watch.save(manager,items);st.rerun()
                except RuntimeError as exc:st.error(str(exc))
            if st.button('Formation archivieren',key='pattern_watch_archive',disabled=bool(selected.get('archived'))):
                selected['archived']=True;watch.event(selected,'archive',watch.utc(),'Formation archiviert.')
                try:watch.save(manager,items);st.rerun()
                except RuntimeError as exc:st.error(str(exc))
            if navigate and st.button('Diese Aktie frisch in der Sofortanalyse prüfen',key='pattern_watch_analyze'):
                navigate(st,[selected['ticker']])
            from modules.chart_pattern_ui import render_chart_patterns
            display_package=dict(selected.get('current_package') or dict(selected['source'],patterns=[selected['original']]))
            if selected.get('data_blocked'):display_package['stale']=True
            render_chart_patterns(st,display_package,key='pattern_watch_chart_'+ident)
            st.caption('Vorgemerkt '+shown_time(selected['created_at'])+' · Ablauf '+shown_time(selected['expires_at'])+'. Alte Struktur und Invalidierung bleiben fest. Geänderte historische Kursbasis verlangt einen neuen Beleg.')
            st.dataframe([dict(Zeit=shown_time(e['at']),Ereignis=e['kind'],Details=e['text']) for e in reversed(selected.get('events',[]))],hide_index=True,use_container_width=True)
        if hasattr(st,'fragment'):st.fragment(run_every=interval if auto else None)(panel)()
        else:
            if auto:st.warning('Automatik hier nicht verfügbar; bitte manuell aktualisieren.')
            panel()
