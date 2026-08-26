# v28.6e5 – Manual Full Refresh Fix

## Ursache
`Jetzt prüfen` startete zwar einen neuen Screenerlauf, aber der zentrale Live-Analyzer konnte bis zu 15 Minuten alte Tickeranalysen aus seinem Cache zurückgeben. Bei einem zuvor unvollständigen Batch wurde außerdem nur der Rest fortgesetzt; bereits abgeschlossene alte Zeilen blieben bestehen.

## Fix
- `Jetzt prüfen` leert gezielt nur den Cache von `analyze_stock_live_cached_v2414`.
- Ein manueller Scan startet immer mit leerem Live-DataFrame und prüft alle ausgewählten Ticker neu.
- Nur der automatische Batch-Scan darf einen unvollständigen vorherigen Lauf fortsetzen.
- Der persistente `Last Visible Stand` bleibt weiterhin als Fallback erhalten, überschreibt aber keinen manuellen Vollrefresh.
- Cache-Schema auf v28.6e5 angehoben.

Keine Änderungen an Score, Ampel, Shadow, Benchmarks, Guardrails, SQL oder Secrets.
