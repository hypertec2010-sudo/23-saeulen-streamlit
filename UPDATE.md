# v28.9 - Positions-/Exit-Engine 2.0

v28.9 baut auf v28.8 und dem Atomic Complete Scan auf. Die produktive Live-/Shadow-Screenerlogik bleibt unveraendert. Neu ist eine eigene, konservative Management-Engine fuer bereits offene Positionen.

## Grundprinzip
- Positionsentscheidungen werden nicht mehr nur aus R-Multiple oder einem einzelnen Exit-Score abgeleitet.
- Exit Engine 2.0 kombiniert Exit-Druck, taktisches Ruecksetzerrisiko, Trendbruch, Momentum-Abbau, Distribution und relative Schwaeche.
- Marktregime, Volatilitaetsregime und RS-Dynamik wirken moderat als Kontext.
- Der reale Positionspuffer entscheidet, wie ein Warnsignal umgesetzt werden soll: Gewinner werden eher geschuetzt/teilrealisiert, Verlierer mit gleichem Druck eher reduziert.
- Stop-Verletzungen und starke Trendbruch-/Exit-Signale besitzen harte Score-Floors und koennen durch ein positives Marktumfeld nicht neutralisiert werden.

## Fuehrungsaktionen
Die Engine unterscheidet jetzt zwischen:
- Halten / laufen lassen
- Gewinnschutz nachziehen
- Stop enger / eng beobachten
- Teilgewinn pruefen
- Teilgewinn / Risiko reduzieren
- Risiko reduzieren / Exit vorbereiten
- Exit / deutlichen Risikoabbau pruefen

Es werden keine Orders ausgefuehrt und keine Stops automatisch veraendert.

## Positionspuffer
Fuer jede offene Position werden P/L, R-Multiple, aktueller Stop und Ziel gemeinsam betrachtet. Die Anzeige zeigt zusaetzlich:
- Exit-Ampel 2.0
- Fuehrungsaktion 2.0
- Exit-Druck 0-100
- Gewinnpuffer in Prozent und R
- Stop-Status
- Datenbasis/Konfidenz
- konkrete Haupttreiber

## Atomic-Live-Daten statt Extra-Abfragen
Die benoetigten technischen Rohsignale werden ab v28.9 direkt im bestehenden Atomic-Vollscan mitgefuehrt. Der Positionsmonitor startet dafuer keine zusaetzlichen Yahoo-/Provider-Abfragen. Der Live-Cache-Schema-Key wurde auf v28.9 angehoben, damit nach dem Update kein alter Snapshot ohne diese Felder als neue Positionsbasis verwendet wird.

## Wichtiger Filter-Fix
Der UI-Filter **nur aktive / gruen-gelb** gilt ab v28.9 ausschliesslich fuer die Screener-Anzeige. Der Positions-/Exit-Monitor verwendet immer den kompletten letzten Atomic-Stand. Eine rote oder weisse offene Position kann dadurch nicht mehr ausgerechnet aus der Exit-Ueberwachung verschwinden.

## Event-Grundlage fuer v29.0
Orange/rote Exit-Engine-Zustaende werden dedupliziert im bestehenden Event-Log protokolliert. Damit kann die geplante Journal & Learning Engine spaeter auswerten, ob Managementwarnungen tatsaechlich nuetzlich waren.

## Unveraendert
- Live-Ampel und Shadow-Ampel
- Guardrails und Score-Schwellen
- Regional Benchmarks und RS-Kalibrierung
- Atomic Complete Scan und Rate-Limit-Schutz
- Trade-Close Undo/Sicherheitsworkflow aus v28.7a
- keine SQL- oder Secrets-Aenderung
