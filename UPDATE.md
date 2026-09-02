# v29.0 - Trading Journal & Learning Engine

v29.0 baut auf dem stabilen v28.9 Positions-/Exit-Engine-Stand auf. Die neue Learning Engine ist bewusst beobachtend: Sie wertet reale Trades und Management-Ereignisse aus, veraendert aber keine produktive Ampel, Score-Schwelle, Gewichtung, Position oder Order automatisch.

## Entry-Kontext wird ab jetzt mit dem Trade gespeichert
Neue Positionen erhalten beim erstmaligen Speichern einen strukturierten Snapshot des bereits vorhandenen Atomic-Screener-Kontexts. Dazu gehoeren unter anderem:
- Live- und Shadow-Ampel
- Live-, Engine- und Guarded Score
- Engine-Empfehlung und Guardrail
- Markt- und Volatilitaetsregime
- RS-Dynamik und Relative Staerke
- Radar-Bucket, Grade und CRV
- Benchmark, Entry-Abstand, Setup-Alert und aktive Gates

Dafuer wird kein neuer Provider-Call gestartet. Die Daten stammen aus dem bereits vollstaendig abgeschlossenen Atomic-Scan.

Wichtig: Bei alten, bereits vor v29.0 angelegten Positionen wird der historische Entry-Kontext nicht mit heutigen Daten aufgefuellt. Er bleibt als unbekannt markiert. So werden keine kuenstlichen Backtest-Daten erzeugt.

## Journal-Datensatz pro geschlossenem Trade
Die Learning Engine baut aus dem bestehenden Journal einen Trade-Datensatz mit genau einer Zeile pro gueltig geschlossenem Trade-Zyklus. Teilverkaeufe werden dem spaeteren Full-Close zugerechnet. Rueckgaengig gemachte Fehlschliessungen bleiben Audit-Historie und zaehlen nicht als abgeschlossener Trade.

Ausgewertet werden unter anderem:
- Gesamt P/L
- Gesamt-R
- rekonstruierte Kapitalrendite, sofern Entry und Initial-Stueckzahl belastbar vorliegen
- Haltedauer, sofern ein echter Entry-Zeitpunkt vorhanden ist
- Win Rate und Profit Factor
- Entry-Kontext und Shadow-vs-Live-Beziehung

## Segment-Lernen
Trades koennen im Journal-Dashboard nach folgenden Segmenten verglichen werden:
- Setup / Radar-Bucket
- Marktregime
- Volatilitaetsregime
- RS-Dynamik
- Live-Ampel
- Shadow vs Live
- Live-Score-Band
- Guarded-Score-Band
- Guardrail
- Grade

Jedes Segment zeigt Trades, Trefferquote, durchschnittliches und medianes R, durchschnittliche Kapitalrendite, Profit Factor und Stichprobenstatus.

## Exit Engine 2.0 Lerncheck
Die v28.9 Exit-Engine-Events werden konservativ mit spaeter geschlossenen Trades verknuepft. Ein Match erfolgt nur, wenn Entry- und Exit-Zeitpunkt sicher bekannt sind. Ausgewertet werden:
- erste Exit-Engine-Aktion im Trade
- maximaler Exit-Druck
- R und P/L bei Erstwarnung
- Schluss-R
- R-Veraenderung nach der Erstwarnung
- Vorlauf bis zum tatsaechlichen Exit

Die R-Veraenderung danach ist kein hypothetischer Backtest-Exit. Sie zeigt nur, ob sich der Trade nach der Warnung weiter verbessert oder verschlechtert hat.

## Manuelle Erkenntnisse
Selbst eingetragene Erkenntnis-Texte werden zusaetzlich nach wiederkehrenden Themen wie Entry/Timing, Stop/Risiko, Exit, FOMO, Positionsgroesse, Marktumfeld, Earnings und Disziplin gezaehlt. Es erfolgt keine automatische Interpretation oder Regelanpassung.

## Stichproben-Guard
Die Learning Engine kennzeichnet Datenmengen als Zu klein, Fruehphase, Mittel, Gut oder Breiter. Beobachtungen unter 5 Trades werden nicht als Kalibrierungsbasis dargestellt. Historische Trades ohne Entry-Kontext bleiben fuer P/L/R nutzbar, werden aber nicht in Setup-/Regime-Vergleiche hineinerfunden.

## Technisch unveraendert
- Atomic Complete Scan / Rate-Limit-Schutz
- Live- und Shadow-Ampel
- v28.8 Shadow Calibration
- v28.9 Exit Engine 2.0
- v28.7a Trade-Close Undo
- keine SQL- oder Secrets-Aenderung
- Live-Screener-Cache-Schema bleibt absichtlich auf dem v28.9-Datenschema, weil v29.0 keine neuen Live-Rohfelder benoetigt
