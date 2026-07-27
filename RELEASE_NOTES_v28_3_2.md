# v28.3.2 – Live-Screener Auto-Refresh Heartbeat Fix

## Behoben

- Der Live-Screener prüft den Refresh-Zeitplan jetzt mit einem stabilen 60-Sekunden-Heartbeat.
- Fälligkeit wird am Zeitstempel des letzten erfolgreichen Live-Scans gemessen.
- Native Seiten- und Widget-Reruns setzen den Refresh-Zeitplan nicht mehr ungewollt zurück.
- Legacy-URL-Parameter werden in der nativen Multipage-App nicht mehr bei jedem Lauf geschrieben oder gelöscht.
- Ein sichtbarer Hinweis zeigt die ungefähre nächste automatische Prüfung.
- Manuelle Scans setzen den nächsten Auto-Refresh korrekt neu.
- Doppelte Refresh-Reruns werden für 120 Sekunden unterdrückt.

## Unverändert

- Supabase-Schema und Secrets
- Repository- und Datenmodell-Schicht
- Watchlists, Positionen und Trade-Journal
- Cockpit-Navigationsfix aus v28.3.1

## Betriebsbedingung

Der native Streamlit-Refresh läuft, solange die Browser-Sitzung verbunden und der Bereich **Live-Screener** geöffnet ist.
