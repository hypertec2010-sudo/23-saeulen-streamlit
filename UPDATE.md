# v30.5b - Berlin Timezone / MEZ-MESZ Timestamp Fix

v30.5b vereinheitlicht sichtbare Zeitstempel auf die IANA-Zeitzone `Europe/Berlin`.

## Behoben
- Der Zeitstempel `Letzter vollständig abgeschlossener Live-Stand` wurde auf Streamlit Cloud teilweise in Server-UTC angezeigt und lag dadurch in Deutschland um 1 bzw. 2 Stunden daneben.
- Gespeicherte Legacy-Scan-Zeitstempel ohne Zeitzonenoffset werden bei der Anzeige als UTC interpretiert und korrekt nach Berlin konvertiert.
- Sommer-/Winterzeit wird automatisch berücksichtigt:
  - Winter: MEZ (UTC+1)
  - Sommer: MESZ (UTC+2)
- Wiederhergestellte Snapshot-Zeitstempel werden ebenfalls in Berlin-Zeit angezeigt.
- Die sichtbare `Scan-Zeit` in Live-Monitor-Details wird nach Berlin konvertiert, auch bei bereits vorhandenen alten Snapshots.
- User-facing Backtest-, Review-, Export- und Auto-Run-Zeitstempel werden nun direkt mit Berlin-Zeit erzeugt.
- Dateinamen/Run-IDs der Einzelanalyse verwenden ebenfalls Berlin-Zeit.

## Kompatibilität
- Interne Refresh-/Snapshot-Anker bleiben im bisherigen Zeitformat, damit Auto-Refresh, Cache-Alter und Atomic-Scan-Kompatibilität nicht verändert werden.
- Die Konvertierung betrifft bewusst die sichtbare Zeitdarstellung und nutzerbezogene Logs.
- Alte Snapshots müssen nicht neu gescannt werden, damit der angezeigte Zeitpunkt korrigiert wird.

## Unverändert
- Keine Änderung an Watchlist-, WTI-/Commodity-, Harvest-/Chop-, TP1/TP2/TP3-, Live-/Shadow-/Exit- oder Providerlogik.
- Keine zusätzlichen Provider-Abfragen.
