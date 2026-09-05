# v30.4a - Trader-Ziel & Harvest-Ampel Highlighting

v30.4a ist ein reiner UI-/Lesbarkeits-Patch fuer die in v30.4 eingefuehrte Short-Term-Trader-/Profit-Harvest-Schicht. Die Berechnung selbst bleibt unveraendert.

## Live-Screener
- Trader-Ziel, Harvest-Ampel und Trader-Modus stehen jetzt direkt hinter Ticker, Name und Kurs und damit vor den breiten Shadow-/Diagnosespalten.
- Das Trader-Ziel wird in der operativen Tabelle als `⚡ +x.x% @ Kurs` hervorgehoben.
- Der bisherige Harvest-Score bekommt eine eigene Ampelanzeige:
  - `🟠` ab 75/100: Profit-Harvest priorisieren.
  - `🟡` ab 60/100: Kurzfrist-Ziel relevant.
  - `🟢` unter 60/100: Hybrid-/Trendpfad bleibt dominant.
- Die Spaltenkoepfe werden kompakt als `⚡ Trader-Ziel`, `Harvest-Ampel` und `Trader-Modus` dargestellt.

## Mobile / Ticker-Details
- Mobile Live-Karten erhalten einen eigenen hervorgehobenen Trader-Block oberhalb der normalen Kontextfelder.
- Hohe Harvest-Relevanz wird im Block zusaetzlich optisch betont.
- Die Ticker-Detailansicht zeigt den Trader-Pfad als eigene Infozeile mit Ziel, Harvest-Ampel und Modus.

## Unveraendert
- Keine Aenderung an `modules/short_term_trader.py` oder den v30.4-Schwellen.
- Keine Aenderung an Live-/Shadow-Ampel, Guardrails, Exit Engine, Positionen oder klassischen TP1/TP2/TP3.
- Keine neuen Provider-, Yahoo- oder FX-Abfragen.
- Die Hervorhebung arbeitet nur auf dem Display-DataFrame; der produktive Atomic-Live-Frame bleibt unveraendert.
