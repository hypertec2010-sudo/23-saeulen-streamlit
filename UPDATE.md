# v30.3 - Early Profit Learning & Calibration

v30.3 schliesst den in v30.2 bewusst noch offenen Learning-Kreis: Early-Profit-Warnungen werden jetzt mit spaeter real geschlossenen Trades verknuepft und darauf geprueft, ob der Gewinnschutz im Nachhinein tatsaechlich sinnvoll war oder ob die Aktie danach noch deutlich weiterlief.

## Neue Auswertung im Trade-Journal
Unter **Early Profit Protection · Lern- & Kalibrierungscheck** wird pro geschlossenem Trade maximal ein unabhaengiger Lernfall verwendet: die erste sicher zuordenbare Early-Profit-Warnung innerhalb des realen Entry-/Exit-Zeitfensters.

Mehrere spaetere Warnungen desselben Trades werden zwar als Kontext gezaehlt, uebergewichten die Statistik aber nicht.

## Was gemessen wird
- R-Multiple zum Zeitpunkt der ersten Early-Profit-Warnung,
- final realisiertes Gesamt-R des Trade-Zyklus,
- Delta-R nach der Warnung,
- realer Giveback in R,
- Profit Velocity am Warnzeitpunkt,
- Exhaustion Risk am Warnzeitpunkt,
- damaliger Historical Giveback Risk,
- erste und staerkste Early-Profit-Empfehlung im Trade.

## Lernklassifikation
- **Gewinnschutz bestaetigt**: final realisiertes R liegt mindestens 0,25R unter dem R am ersten Warnzeitpunkt.
- **Gewinnschutz stark bestaetigt**: mindestens 0,75R Giveback.
- **Laufenlassen besser**: final realisiertes R liegt mindestens 0,25R ueber dem Warnzeitpunkt.
- **Laufenlassen klar besser**: mindestens +0,75R danach.
- Dazwischen bleibt der Fall neutral.

Diese Schwellen sind nur fuer den Lerncheck. Sie veraendern die v30.2-Empfehlung nicht.

## Neue Kalibrierungen
Die Learning Engine zeigt:
- Trefferbild nach damaliger Early-Profit-Empfehlung,
- Ergebnis nach Profit-Velocity-Band,
- Ergebnis nach Exhaustion-Risk-Band,
- Kalibrierung des historischen Giveback-Risikos gegen real beobachtete Trade-Ausgaenge.

Damit wird sichtbar, ob z. B. sehr hohe Exhaustion-Werte bei den eigenen Trades tatsaechlich haeufiger in Givebacks enden oder ob die Warnungen bislang zu frueh ausloesen.

## Stichproben-Guard
- <5: Zu klein
- 5-9: Fruehphase
- 10-19: Mittel / fruehe Kalibrierung
- 20-39: Gut / beobachtbar kalibriert
- >=40: Breiter

Kleine Stichproben bleiben ausdruecklich als solche markiert. Es gibt keine automatische Regel-, Score-, Stop- oder Order-Aenderung.

## Methodischer Hinweis
Der Lerncheck ist bewusst **kein hypothetischer Sofortverkaufs-Backtest**. Er vergleicht das R am Warnzeitpunkt mit dem spaeter tatsaechlich realisierten Gesamt-R des Trades. Teilverkaeufe und reale Positionsfuehrung bleiben dadurch Teil des echten Ergebnisses.

## Provider-Schutz
Keine neuen Kurs- oder Historienabfragen. v30.3 arbeitet nur mit bereits vorhandenen Trade-Journal- und Event-Log-Daten.

## Unveraendert
- v30.2 Early Profit Protection & Giveback Engine,
- produktive Live-/Shadow-Ampel,
- Exit Engine 2.0,
- Rotation Radar,
- Atomic Complete Scan,
- Portfolio Engine,
- SQL / Secrets.
