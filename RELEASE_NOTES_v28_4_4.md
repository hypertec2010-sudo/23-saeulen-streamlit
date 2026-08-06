# Release Notes v28.4.4 – Complete Batch Scanning

## Behoben

Der Live-Screener hat bisher nur die ersten 40 eindeutigen Watchlist-Werte analysiert. Weitere Werte wurden still abgeschnitten und erschienen weder in den Ergebnissen noch unter den nicht analysierbaren Tickern.

## Neu

- sichtbarer **Scan-Umfang**: 40, 80, 120 oder alle Werte
- Standardauswahl **Alle Werte**
- klare Zähler für Watchlist, eindeutige Ticker, Scan-Menge und ausstehende Werte
- kontrollierte Verarbeitung in 20er-Batches
- Fortschrittsanzeige während des Scans
- persistenter Checkpoint nach jedem Batch
- Fortsetzung eines unterbrochenen Scans nach Browser-/Display-Unterbrechung
- getrennte Darstellung von:
  - erfolgreich analysiert
  - nicht analysierbar
  - bewusst ausstehend wegen gewählter Scan-Grenze
  - doppelte Watchlist-Einträge
- vollständiger Scan-Snapshot in Supabase und lokalem Spiegel

## Verhalten

Bei **Alle Werte** wird jeder eindeutige Ticker der Watchlist aufgerufen. Bei einer kleineren Grenze werden die übrigen Ticker ausdrücklich als ausstehend angezeigt. Sie gelten nicht als Datenfehler.

Ein unterbrochener Scan speichert die bereits abgeschlossenen Batches. Bei aktivem Auto-Scan wird nach der Wiederverbindungs-Schutzpause fortgesetzt. Bei pausiertem Mobile Auto-Scan setzt **Jetzt prüfen** den offenen Scan fort.

## Migration

- keine SQL-Migration
- keine Änderung an Streamlit-Secrets
- bestehende Supabase-Daten bleiben kompatibel
