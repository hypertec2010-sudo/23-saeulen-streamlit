# v28.4.2 Mobile Screener Snapshot & Reconnect Fix

Diese Version baut auf **v28.4.1** auf. Multipage-Navigation, Supabase,
Repositories, Trade-Journal, R-Multiple-Berechnung und Desktop-Live-Screener
bleiben kompatibel.

## Mobile Live-Screener

Der letzte abgeschlossene Live-Scan wird als Snapshot in Supabase und im lokalen
Fallback gespeichert. Nach einer Display-Pause oder einem WebSocket-Reconnect
wird dieser Stand sofort angezeigt, anstatt die gesamte Watchlist erneut ab
Ticker 1 zu analysieren.

Im neuen `📱 Mobile-Modus`:

- kompakte Karten statt breiter Tabelle
- vertikale Cockpit-Navigation
- größere, untereinander angeordnete Einstellungen
- Auto-Scan standardmäßig aus
- manueller Scan jederzeit über `Jetzt prüfen`
- optionaler Auto-Scan bei aktiver Browser-Sitzung
- 120 Sekunden Wiederverbindungs-Schutz vor einem sofortigen Vollscan

Der Snapshot wird im bestehenden Storage-Namespace-System unter
`live_screener_snapshots` abgelegt. Eine neue Supabase-Tabelle ist nicht nötig.

## Qualitätssicherung

Bei jedem Push auf `main` oder `master` sowie bei Pull Requests führt GitHub
automatisch aus:

```bash
python verify_deployment.py
pytest -q
```

Zusätzlich geprüft werden Snapshot-Serialisierung, DataFrame-Roundtrip,
Cache-Key-Trennung und Begrenzung der gespeicherten Snapshot-Anzahl.

## Upgrade

1. Vollständigen Inhalt des ZIP-Pakets ins Repository übernehmen.
2. Vorhandene Streamlit-Secrets unverändert lassen.
3. Keine SQL-Migration ausführen.
4. GitHub-Commit pushen und den Workflow **v28.4.2 Quality Gate** prüfen.
5. App auf dem Smartphone öffnen und `📱 Mobile-Modus` aktivieren.
6. Einmal `Jetzt prüfen` ausführen, damit der erste Snapshot gespeichert wird.
7. Display sperren, wieder entsperren und den wiederhergestellten Stand prüfen.

## Lokale Prüfung

```bash
python -m pip install -r requirements-ci.txt
python verify_deployment.py
pytest -q
streamlit run app.py
```

Details: `RELEASE_NOTES_v28_4_2.md`.
