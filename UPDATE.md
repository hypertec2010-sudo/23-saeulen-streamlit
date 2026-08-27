# v28.6e6 – Rate-Limit-Safe Refresh

Ursache:
- v28.6e5 leerte bei `Jetzt prüfen` den kompletten Live-Analyzer-Cache.
- Bei grossen Watchlists wurden dadurch sehr viele Yahoo-Abfragen auf einmal erzwungen.
- Nach dem ersten HTTP 429 liefen oft fast alle folgenden Ticker ebenfalls in das Rate-Limit.

Fix:
- `Jetzt prüfen` startet weiterhin einen Vollscan aller Ticker.
- Der zentrale Provider-/Analyzer-Cache wird dabei nicht mehr global geloescht.
- Der letzte gueltige Screenerstand bleibt waehrend des manuellen Refreshs als Fallback sichtbar.
- Erfolgreich neu berechnete Ticker ersetzen ihren alten Stand tickerweise.
- Ein temporaerer 429-Fehler loescht keinen zuletzt gueltigen Tickerstand mehr aus der Tabelle.
- Alte Teil-Batches gelten bei manuellem Scan trotzdem nicht als abgeschlossen: alle Ticker werden erneut durchlaufen.

Keine Aenderung an Live-/Shadow-Ampel, Scores, Benchmarks, Guardrails, SQL oder Secrets.

Nach Upload Streamlit einmal rebooten. Falls Yahoo aktuell bereits global gedrosselt hat, einige Minuten warten und dann einmal `Jetzt prüfen` ausfuehren.
