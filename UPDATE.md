# v28.6e2 – Regional Benchmark Priority Fix

## Ursache
v28.6e1 startete den regionalen Benchmark-Recovery nur, wenn 21T/63T-RS fehlten.
Bei IPS.PA, SU.PA und ADDT-B.ST lieferte die alte Analyse-Engine bereits RS gegen EURO STOXX 50.
Dadurch wurden CAC 40 bzw. OMX Stockholm nie versucht und die Diagnose zeigte `Nicht benötigt`.

## Fix
- Für europäische Ticker hat der definierte Landesbenchmark jetzt Vorrang vor einem bereits vorhandenen generischen Europa-Benchmark.
- `.PA` versucht primär CAC 40 (`^FCHI`).
- `.ST` versucht primär OMX Stockholm 30 (`^OMX`).
- `.DE` bleibt bei DAX (`^GDAXI`), wenn dieser bereits aktiv ist.
- Erst bei echtem Fehler/zu wenig Daten wird auf EURO STOXX 50 zurückgefallen.
- Diagnose zeigt nun den tatsächlichen Abrufstatus des Primärbenchmarks.

Keine Änderung an Live-Ampel, Shadow-Ampel, Guardrails oder Basis-Score.

## Test
Nach Reboot frisch scannen: IPS.PA, SU.PA, ADDT-B.ST sowie SAP.DE als Kontrollwert.
