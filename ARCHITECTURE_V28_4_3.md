# Architektur v28.4.3 – Status Change Explainer

`modules/live_change_explainer.py` ist eine reine, testbare Vergleichsschicht. Sie
kennt Streamlit nicht und erhaelt nur den vorherigen sowie den aktuellen
Ticker-Snapshot. `modules/live_monitor.py` sammelt die Diagnosekomponenten, legt sie
im persistenten Live-History-State ab und schreibt die erzeugte Erklaerung in
Tabelle, Historie und Event-Log.

Interne Diagnosefelder beginnen waehrend der Berechnung mit `__` und werden vor der
UI-Ausgabe entfernt. Nur die zusammengefasste Spalte `Warum geändert?` bleibt
sichtbar.
