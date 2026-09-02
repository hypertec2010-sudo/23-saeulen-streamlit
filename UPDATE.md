# v28.7b - Atomic Complete Scan

Dieses Update baut die Live-Screener-Scanlogik grundlegend um. Ziel: lieber etwas langsamer, dafuer immer ein eindeutig vollstaendiger und nachvollziehbarer Stand.

## Behobene Ursachen
1. **Teil-Batches wurden bisher bereits in Session/Snapshot gespeichert.** Dadurch konnte ein Reconnect oder Rerun einen halbfertigen Scan als sichtbaren Stand wiederherstellen.
2. **Manuelle Scans starteten mit alten Ergebniszeilen als Fallback.** Erfolgreiche neue Ticker ersetzten diese nur tickerweise; bei Fehlern konnte eine alte Zeile im vermeintlich neuen Scan stehen bleiben.
3. **Der Auto-Refresh-Heartbeat verglich unterschiedliche Cache-Keys.** Der echte Cache enthielt ein `schema`-Feld, der Heartbeat-Key nicht. Dadurch konnte ein gueltiger Cache bei jedem Heartbeat als unpassend gelten und unnoetige Scans/Reruns ausloesen.
4. **Unvollstaendige Batch-Checkpoints wurden fortgesetzt.** Das machte schwer erkennbar, welche Zeile aus welchem Lauf stammt.
5. **Scan-Grenzen 40/80/120 konnten bewusst Teilmengen erzeugen.** Fuer den Live-Screener ist jetzt immer die komplette eindeutige Watchlist aktiv.

## Neue Logik: Atomic Complete Scan
- Jeder Live-Scan verarbeitet **alle eindeutigen Ticker** der Watchlist.
- Batches existieren nur noch intern fuer Fortschritt und Provider-Schutz.
- **Kein Teil-Batch wird mehr als Live-Stand gespeichert.**
- Session-Cache und persistenter Snapshot werden erst nach dem gesamten Lauf aktualisiert.
- Wird ein Lauf technisch abgebrochen, bleibt der letzte vollstaendig abgeschlossene Atomic-Stand erhalten.
- Alte v28.6/v28.7a-Snapshots werden beim ersten Start bewusst nicht als Atomic-Stand vertraut. Nach dem Update deshalb einmal einen neuen Vollscan ausfuehren.

## Frische Daten ohne 429-Sturm
- `Jetzt vollständig aktualisieren` erzeugt einen frischen Analyse-Key, ohne den globalen Streamlit-/Yahoo-Cache zu leeren.
- Der Lauf wird bewusst gedrosselt: kleine interne Batches und kurze Pausen zwischen Titeln.
- Bei 429/temporären Providerfehlern gibt es bis zu zwei spaetere Retry-Runden mit Cooldown.
- Ein versehentlicher Doppelklick innerhalb von zwei Minuten nutzt denselben Freshness-Key und startet keinen zweiten Provider-Sturm.
- Auto-Vollscans nutzen ebenfalls einen frischen, intervallgebundenen Analyse-Key.

## Keine Mischdaten mehr
Ein Ticker hat im neuen Stand genau zwei Moeglichkeiten:
- aktuelle Ergebniszeile aus diesem Vollscan, oder
- aktueller Fehlerstatus aus diesem Vollscan.

Ein fehlgeschlagener Ticker wird **nicht** mit seiner alten Ergebniszeile aufgefuellt. Fehler werden immer sichtbar ausgewiesen.

## Sichtbarer Datenstand
Nach jedem Lauf zeigt die App:
- Zeit des letzten vollstaendig abgeschlossenen Scans,
- Alter des Datenstands,
- verarbeitet / erfolgreich / aktuelle Fehler,
- Scan-Dauer,
- Scan-Modus,
- Kennzeichnung `kein Mischstand`.

## Auto-Refresh
- Heartbeat und Cache verwenden jetzt denselben Schema-Key.
- Waehren eines Vollscans darf der Minuten-Heartbeat keinen zweiten Scan triggern.
- Nach einem komplett fehlgeschlagenen Auto-Lauf gilt eine 5-Minuten-Schutzpause; ein manueller Vollscan bleibt jederzeit moeglich.

Keine Aenderung an Live-/Shadow-Ampel, Scores, Guardrails, regionalen Benchmarks, Positionen oder Trade-Journal-Logik.
Keine SQL-/Secrets-Aenderung erforderlich.
