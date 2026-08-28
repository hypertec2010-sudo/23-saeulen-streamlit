# Changelog

## v28.6e6
- aggressives globales Cache-Clear aus v28.6e5 entfernt
- manueller Vollrefresh provider-sicher gemacht
- letzter gueltiger Tickerstand bleibt bei temporaeren 429-Fehlern sichtbar
- manuelle Vollpruefung laeuft weiterhin ueber alle Ticker

## v28.7
- Shadow Performance Tracking mit 1T/3T/5T/10T/20T Forward Returns.
- Shadow-Ereignisse werden dedupliziert persistent protokolliert.
- Performance-Auswertung getrennt nach Aufwertung/Abwertung.
- Kursnachladen nur per explizitem Button, um Provider-Rate-Limits zu vermeiden.
- Keine Änderung an Live-Ampel, Shadow-Entscheidungslogik, Scores oder Guardrails.
