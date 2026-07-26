# Release Notes v28.3 – Analysis Core Extraction

## Änderungen

- Die zentrale, rund 2.600 Zeilen große Legacy-Analyse-Pipeline wurde aus `legacy_app.py` entfernt.
- Neue Datei: `modules/legacy_analysis_core.py`.
- Die Analyse-Facade in `modules/analysis_engine.py` bleibt der einheitliche Einstiegspunkt.
- Legacy-Hilfsfunktionen werden unmittelbar vor dem Analyseaufruf über einen expliziten Kontext gebunden.
- Der extrahierte Core importiert keine Streamlit-Oberfläche und kann separat kompiliert und geprüft werden.
- Ein bislang ungeschützter `signal_conflict_label`-Zugriff besitzt nun einen neutralen Kompatibilitätswert, statt in einer seltenen Positionsmanagement-Verzweigung einen `NameError` auszulösen.
- Deployment- und Regressionstests prüfen, dass der Analyse-Core nicht wieder in `legacy_app.py` zurückwandert.

## Kompatibilität

- Kein neues Supabase-Schema.
- Keine Änderung an Streamlit-Secrets.
- Kein erneuter Datenimport.
- Bestehende Analyse-Ergebnisfelder und UI-Aufrufe bleiben erhalten.

## Upgrade

Den vollständigen Inhalt des Pakets ins Repository übernehmen und die App neu starten.
