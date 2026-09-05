# v30.4 - Short-Term Trader & Profit Harvest Engine

v30.4 ergaenzt die bestehende klassische TP-/Trendlogik um einen zweiten, taktischen Gewinnpfad fuer schwankende Marktphasen. Die bisherige Logik bleibt unveraendert und ist weiterhin der primaere Pfad fuer saubere Trendbewegungen.

## Neu
- Neue providerfreie Komponente `modules/short_term_trader.py`.
- Dynamisches `Kurzfrist-Trader-Ziel` aus aktuellem Kurs/Entry, ATR, Markt-/Volatilitaetskontext, RS-Dynamik und bereits vorhandenen technischen Exit-/Risikosignalen.
- `Chop Risk` 0-100 als Kontextindikator fuer unruhige/gemischte Markt- und Techniklagen.
- `Harvest Score` 0-100 fuer die Frage, wie relevant ein frueheres taktisches Monetarisieren eines positiven Puffers ist.
- Zusaetzliches Feld `Gewinn sichern ab` unterhalb des eigentlichen Kurzfrist-Ziels.
- Dynamischer taktischer Horizont: typischerweise 1-3T bei hohem Chop, 2-5T bei gemischtem Kontext und 3-8T bei geordneterem Trend.
- Teilgewinn-Idee 25/33/40/50% nur als Beratung; kein automatischer Verkauf.
- Restpositions-Regel trennt taktische Gewinnmitnahme bewusst vom klassischen Trendpfad.

## Live-Screener
- Der abgeschlossene Atomic-Live-Frame wird nach dem Scan ohne weitere Provider-Abfrage um folgende Spalten ergaenzt:
  - Trader-Ziel
  - Trader-Sicherung ab
  - Harvest-Score
  - Chop-Risk
  - Trader-Modus
  - Trader-Horizont
- Die operative Haupttabelle zeigt Trader-Ziel, Harvest-Score und Trader-Modus direkt neben dem Live-Score.
- Mobile Karten zeigen dieselben taktischen Kerninformationen.
- Die Berechnung wird bei jedem Render aus exakt dem vorhandenen Atomic-Stand neu aufgebaut; kein eigener Marktprovider-Cache und kein versteckter Zusatzrequest.

## Trade-Plan / Einzelanalyse
- TP1, TP2 und TP3 bleiben exakt bestehen.
- Daneben erscheint ein eigener Block mit Kurzfrist-Trader-Ziel, Gewinn-sichern-Schwelle, Harvest Score und Trader-Modus.
- Das neue Ziel ersetzt weder 1R-Ziel noch Hauptziel oder technisches Setup-Ziel.

## Offene Positionen
- Positions-/Exit-Monitor zeigt zusaetzlich Trader-Harvest, Trader-Ziel, Harvest Score, Chop Risk und Sicherungsschwelle.
- Die Positionsbewertung verbindet den aktuellen Atomic-Kontext mit bereits berechnetem Profit Velocity, Exhaustion Risk und - bei ausreichender Stichprobe - historischem Giveback Risk aus v30.2.
- In einem starken, bestaetigten Trend wird Harvesting bewusst heruntergewichtet, damit gute Leader nicht allein wegen eines schnellen kleinen Gewinns abgeschnitten werden.
- In choppy/volatilen Situationen mit positivem Gewinnpuffer kann ein Teilgewinn von 25-50% pruefbar werden, waehrend die Restposition separat ueber Trend/Exit gefuehrt wird.

## Learning-Vorbereitung
- Aktive taktische Harvest-Hinweise werden dedupliziert als Event `Short-Term Profit Harvest` gespeichert.
- Payload enthaelt Harvest Score, Chop Risk, Trader-Ziel, Sicherungsschwelle, Teilgewinn-Idee, Haltedauer, P/L, Profit Velocity, Exhaustion und Giveback Risk.
- v30.4 veraendert daraus noch keine Schwellen automatisch. Die Events schaffen nur die Datenbasis fuer eine spaetere reale Kalibrierung gegen geschlossene Trades.

## Sicherheits-/Datenregeln
- Keine Orders, keine Stop-Aenderung und kein automatischer Teilverkauf.
- Keine Aenderung an produktiver Live-Ampel, Shadow-Ampel, Guardrails, Validated Engine oder Exit Engine 2.0.
- Keine Aenderung an der klassischen TP1/TP2/TP3-Formel.
- Kein Short-Term-Ziel aus einem gespeicherten Alt-Kurs: ohne aktuellen Atomic-Kurs bleibt die Positionsentscheidung neutral.
- Ohne ATR-Basis wird kein kuenstlich praezises Kurzfrist-Ziel erzeugt.
- Historical Giveback beeinflusst den Harvest Score erst ab einer Mindeststichprobe von n>=5.
- Keine neuen Yahoo-/Marktprovider-Abfragen.

## Methodik-Hinweis
`Chop Risk` ist kein eigenstaendiger Marktindex und kein Backtest der letzten Tageswechsel. Der Score fasst den bereits vorhandenen aktuellen Markt-/Volatilitaets-/RS-/Exit-Kontext zusammen. Er dient nur zur taktischen Einordnung, ob ein kleiner positiver Puffer in der aktuellen Lage wertvoller als in einem sauberen Trendmarkt ist.
