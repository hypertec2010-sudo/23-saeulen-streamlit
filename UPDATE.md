# v30.1 - Investment Rotation Radar

v30.1 baut auf dem stabilen v30.0 Controlled Cutover auf und fuegt einen eigenstaendigen, beobachtenden Branchen-/Kapitalfluss-Radar hinzu. Ziel ist nicht, den staerksten Sektor der Vergangenheit zu markieren, sondern moeglichst frueh zu erkennen, wo sich relative Marktführerschaft beschleunigt oder abkuehlt.

## Neuer Trading-Cockpit-Bereich
Im Trading-Cockpit gibt es `🧭 Rotation Radar`.

Die erste Version beobachtet drei Ebenen:
- Investmentklassen / Regionen
- US-Sektoren
- liquide Branchen- und Themen-ETFs

Europa und Deutschland sind als eigene Aktienregionen enthalten; die detaillierte Sektor-/Branchenrotation startet bewusst mit besonders liquiden US-/globalen ETFs, um die Datenqualitaet und Provider-Stabilitaet hoch zu halten.

## Drei Kernwerte
### Leadership Score
0-100 aus:
- 63T Relative Staerke
- 21T Relative Staerke
- 21T absolute Performance
- Trendstruktur ueber MA20 / MA50 / MA200

Leadership beantwortet: `Wo ist die bestehende Marktführerschaft?`

### Rotation Momentum
0-100 aus der Beschleunigung der 5T-, 21T- und 63T-Relative-Staerke sowie der kurzfristigen eigenen Performance.

Rotation beantwortet: `Wo verbessert oder verschlechtert sich die Marktposition gerade?`

Dadurch kann ein noch mittelmaessiger Sektor frueh als Rotation erscheinen, bevor sein 63T-Leadership-Wert bereits an der Spitze steht.

### Breadth Confirmation
Breadth ist bewusst eine zweite, manuell ausgeloeste Stufe. Fuer ausgewaehlte Sektoren/Branchen werden repraesentative Einzelwerte betrachtet:
- Anteil ueber MA20
- Anteil ueber MA50
- Anteil mit positiver 21T-Performance
- Anteil mit positiver 21T-RS gegen SPY

Das vermeidet, dass bei jedem Radar-Aufruf dutzende zusaetzliche Aktien abgefragt werden.

## Rotationsphasen
Der Radar unterscheidet:
- 🟣 Emerging
- 🟢 Leading
- 🟡 Mature
- 🟠 Cooling
- 🔴 Rotating Out

Zusätzlich werden Rangveraenderungen und Score-Deltas gegen 1T / 5T / 20T historische Marktstände berechnet. Diese historischen Vergleiche werden direkt aus demselben 2-Jahres-Daily-Datensatz rekonstruiert und sind daher nicht davon abhaengig, dass die App an jedem Handelstag geoeffnet war.

## Frueher Trendshift statt nur Rueckspiegel
Die Tabelle zeigt unter anderem:
- aktuellen Peer-Rang
- Rang Δ5T
- Leadership Δ5T
- Rotation Δ5T
- RS 5T / 21T / 63T
- Trendshift-Lesart

Ein Sektor kann damit beispielsweise noch kein etablierter Leader sein, aber bereits als `Emerging` mit starkem Rangaufstieg sichtbar werden.

## Atomic Radar Snapshot
Wie beim Live-Screener wird kein halb fertiger neuer Stand als aktuell verkauft:
- der Kern-Radar wird als kompletter Batch geladen
- mindestens 85 Prozent Datenabdeckung sind fuer die Veroeffentlichung eines neuen Snapshots erforderlich
- ein unvollstaendiger Lauf ersetzt den letzten gueltigen Snapshot nicht
- fehlende Gruppen werden nicht mit alten Einzelzeilen in einen neuen Stand gemischt

Der letzte vollstaendige Radar-Snapshot wird ueber die zentrale Storage-Schicht gespeichert.

## Provider-Schutz
- Hauptscan nur auf liquide ETFs/Indizes
- yfinance Batch-Download mit `threads=False`
- maximal zwei Batch-Versuche
- Einzel-Fallback nur bei hoechstens sechs fehlenden Symbolen
- 30-Minuten-Daten-Cache
- Breadth nur explizit fuer ausgewaehlte Kandidaten
- kein automatischer Radar-Scan bei jedem Streamlit-Rerun

## Noch kein produktiver Score-Faktor
v30.1 ist Beobachtungsmodus. Rotation, Breadth und Branchenphase veraendern noch nicht:
- Live-Ampel
- Shadow-Ampel
- Guarded Engine Score
- Entry-Guardrails
- Positionen oder Orders

Der Radar schafft zuerst eine eigenstaendige Datengrundlage. Ein spaeterer Schritt kann pruefen, ob Aktien aus `Emerging/Leading` Branchen tatsaechlich bessere Forward-Returns liefern als vergleichbare Setups aus `Cooling/Rotating Out` Branchen.

Keine SQL-/Secrets-Aenderung erforderlich. Das Atomic-Live-Screener-Cache-Schema bleibt unveraendert.
