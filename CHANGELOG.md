# Changelog

## v28.7a
- Zweistufiger, bestaetigungspflichtiger Workflow fuer vollstaendige Positionsschliessungen.
- Vorschau vor dem Exit mit Ticker, Stueckzahl, Exit-Kurs und berechnetem P/L.
- Plausibilitaetswarnung und zweite Bestaetigung bei auffaelligem Ausstiegskurs.
- Neue Undo-Funktion fuer versehentlich geschlossene Trades.
- Ab v28.7a wird vor jedem Full-Close ein kompletter Positions-Snapshot fuer verlustfreies Undo gespeichert.
- Legacy-Schliessungen werden aus Journal- und Event-Historie rekonstruiert.
- Rueckgaengig gemachte Abschluesse werden aus P/L-, Trefferquoten- und Closed-Trade-Statistik neutralisiert, bleiben aber als Audit-Historie erhalten.
- Keine Aenderung an Screener-, Shadow-, Score-, Guardrail- oder Benchmark-Logik.

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
