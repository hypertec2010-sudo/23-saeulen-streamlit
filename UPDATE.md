# v30.3f - Rotation Drilldown Pandas-Series Crash Fix

v30.3f behebt den unmittelbar nach v30.3e sichtbaren Streamlit-Crash beim Ausfuehren eines Aktien-Drilldowns.

## Fehlerbild
Nach Auswahl einer Rotation und Klick auf **Top-Kandidaten fuer Auswahl pruefen** konnte Streamlit mit folgendem Fehler abbrechen:

`ValueError: The truth value of a Series is ambiguous`

Der Trace zeigte auf die Speicherung des aktuellen Radar-Kontexts fuer den Drilldown.

## Ursache
Der Hilfsblock `_v303c_rotation_drilldown_context()` legte die Zeilen des aktuellen Radar-Frames als `pandas.Series` im internen `row_map` ab. Beim erfolgreichen Drilldown wurde anschliessend sinngemaess `row_map.get(...) or {}` verwendet.

Python versucht bei `or` den Wahrheitswert des linken Objekts auszuwerten. Fuer eine pandas-Series ist dieser Wahrheitswert absichtlich nicht eindeutig; pandas wirft deshalb den ValueError.

## Fix
- Jede Radar-Kontextzeile wird bereits beim Aufbau des `row_map` in ein normales Python-Dictionary konvertiert.
- Die spaetere Speicherstelle verwendet kein `or {}` mehr auf einem potenziellen pandas-Objekt.
- Als zusaetzliche Schutzschicht wird eine unerwartete Series dort nochmals explizit per `.to_dict()` konvertiert.

## Unveraendert
Das v30.3e Universe-Alignment bleibt erhalten. Es gibt keine Aenderung an Rotation-/Leadership-Berechnung, Phasenlogik, Aktienkorb-Zuordnung, Live-/Shadow-Ampel, Early-Profit-/Exit-Engine oder Provider-Cadence.
