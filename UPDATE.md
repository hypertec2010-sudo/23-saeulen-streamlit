# v28.8 - Engine Calibration & Backtest

v28.8 baut auf dem stabilen Atomic Complete Scan aus v28.7b auf. Die produktive Live-Ampel, Score-Schwellen und Guardrails bleiben unveraendert. Neu ist eine belastbare, event-basierte Kalibrierung der Shadow-Engine gegen spaetere Kursentwicklung.

## Kalibrierungsprinzip
- Shadow-Aufwertungen gelten als Treffer, wenn der Kurs danach steigt.
- Shadow-Abwertungen gelten als Treffer, wenn der Kurs danach faellt.
- Daraus entsteht der **Shadow-Edge**: Forward-Return mit der Richtung des Shadow-Signals normiert. Positiver Edge bedeutet, dass die Shadow-Abweichung gegenueber der Live-Ampel in die richtige Richtung zeigte.
- Auswertung fuer 1T / 3T / 5T / 10T / 20T.
- Mehrere Score-Aenderungen innerhalb derselben laufenden Divergenz werden zu einer Episode zusammengefasst, damit ein einzelner Ticker die Statistik nicht kuenstlich aufblaeht.

## Neue Auswertungen
- Trefferquote und durchschnittlicher/medianer Shadow-Edge je Horizont.
- Directional MFE/MAE: bestes und schlechtestes Kursfenster nach dem Signal, jeweils in Shadow-Richtung normiert.
- Kalibrierung der aktuellen Guarded-Score-Baender: Rot <28, Weiss 28-54, Gelb 55-71, Gruen ab 72.
- Segmentierung nach Guardrails, RS-Dynamik, Marktregime und Volatilitaetsregime.
- Separater Guardrail-Backtest, wenn Raw Engine Score und Guarded Engine Score fuer ein Ereignis vorliegen.
- Stichprobenstatus von 'Zu klein' bis 'Breiter' statt vorschneller Schlussfolgerungen.
- Kalibrierungsdiagnose mit Beobachtung und Konsequenz - rein analytisch, keine automatische Parameter-Aenderung.

## Verbesserte Datengrundlage ab v28.8
Neue Shadow-Ereignisse speichern zusaetzlich den damaligen Kontext: Raw Engine Score, Kontext-Anpassung, Kontext-Verlaesslichkeit, Guardrail, RS-Dynamik, Markt-/Volatilitaetsregime, Benchmark, aktive Gates sowie technische Komponenten. Alte Ereignisse werden nicht nachtraeglich mit erfundenen Daten aufgefuellt; fehlende Legacy-Metadaten bleiben sichtbar.

## Persistenz
Shadow-Performance und Kalibrierungsdaten nutzen nun bevorzugt die zentrale Storage-Schicht (`shadow_performance_v288`). Die bisherige lokale JSON-Datei bleibt als Fallback/Migrationsquelle erhalten. Keine SQL- oder Secrets-Aenderung erforderlich.

## Forward-Performance Fixes
- Trefferquote fuer Abwertungen korrigiert: fallende Kurse sind dort jetzt korrekt ein Treffer.
- Nicht-Handelstage werden bei 1T/3T/... korrekt indiziert.
- Performance-Aktualisierung bleibt explizit per Button; kein zusaetzlicher Yahoo-Traffic durch Auto-Scans.

## Wichtig
v28.8 ist weiterhin Beobachtungsmodus. Auch eine gute Statistik aendert die produktive Live-Ampel nicht. Ein spaeterer Cutover kommt erst nach ausreichend Daten und der geplanten Validierung.
