# Release Notes v28.1

## Zweck

v28.1 ist ein Architektur-Release. Die sichtbaren Funktionen bleiben bewusst stabil, waehrend der Datenzugriff intern klarer getrennt wird.

## Aenderungen

- neue Domain-Modelle fuer Position, Journal, Event und Watchlist
- neue Repository Registry als zentraler Zugriffspunkt
- Position Monitor nutzt `PositionRepository`
- Trade-Journal nutzt `TradeJournalRepository`
- Event-Log nutzt `EventRepository`
- Watchlist-Repository wird ueber `modules.repositories` bereitgestellt
- alte Supabase-Payloads werden beim Lesen normalisiert
- unbekannte Legacy-Felder bleiben erhalten
- Speicheranzeige im Positionsmonitor korrigiert
- Regressionstest um Repository- und Modelltests erweitert

## Upgrade-Risiko

Niedrig. Das Datenbankschema und die Namespace-Namen bleiben unveraendert. Vorhandene Daten muessen nicht erneut importiert werden.

## Kontrolltest nach Deployment

1. Eine vorhandene Watchlist oeffnen.
2. Eine bestehende Position laden und speichern.
3. Eine Trade-Notiz anlegen.
4. App neu starten.
5. Kontrollieren, dass Position und Journal-Eintrag weiterhin vorhanden sind.
