# v28.4.2 – Mobile Screener Snapshot & Reconnect Fix

## Problem

Auf Smartphones kann das Betriebssystem den Browser und die Streamlit-WebSocket-Verbindung pausieren, sobald das Display ausgeschaltet wird. Nach dem Entsperren wurde dadurch häufig ein neuer kompletter Watchlist-Scan gestartet.

## Änderungen

- Der letzte vollständig abgeschlossene Live-Screener-Lauf wird im zentralen Storage gespiegelt.
- Verwendeter Namespace: `live_screener_snapshots`.
- Nach einem Browser-/WebSocket-Reconnect wird der passende Snapshot für Watchlist, Ticker, Stil und Horizont geladen.
- Eine Schutzpause von 120 Sekunden verhindert einen sofortigen Vollscan nach der Wiederverbindung.
- Neuer persistenter `📱 Mobile-Modus`.
- Im Mobile-Modus ist der automatische Scan standardmäßig ausgeschaltet; `Jetzt prüfen` bleibt jederzeit verfügbar.
- Optional kann der Auto-Scan auf dem Smartphone wieder aktiviert werden.
- Kompakte Kartenansicht statt breiter Tabelle.
- Cockpit-Navigation wird im Mobile-Modus vertikal dargestellt.
- Desktop-Ansicht und bestehender 60-Sekunden-Heartbeat bleiben unverändert.

## Daten und Deployment

- Keine SQL-Migration erforderlich.
- Keine Änderung an Streamlit-Secrets erforderlich.
- Bestehende Supabase-Tabelle `app_state` wird weiterverwendet.
- Der lokale Spiegel bleibt als Fallback aktiv.

## Technische Dateien

```text
modules/live_screener_snapshot.py
tests/test_live_screener_snapshot.py
```
