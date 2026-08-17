# v28.4.5b3 - Consolidated New-Listing Fix

Dieses Update ersetzt die vier zusammengehoerigen Dateien gemeinsam, damit keine alte v28.4.4-Analyse-Engine mit dem neuen Provider gemischt laeuft.

Geaendert:
- legacy_app.py
- modules/provider_manager.py
- modules/ticker_resolver.py
- modules/legacy_analysis_core.py
- VERSION.txt

Wichtig nach Upload:
1. Alle Dateien ueberschreiben.
2. Streamlit App rebooten.
3. SKHY und SPCX erneut testen.

Erwartete Fehlermeldung, falls weiterhin zu wenig Daten geliefert werden:
`Noch zu wenig Kursdaten (X Handelstage)...`

Wenn stattdessen weiterhin `Nicht genug Kursdaten fuer belastbare Analyse` erscheint, laeuft noch eine alte legacy_analysis_core.py im Repository/Deployment.
