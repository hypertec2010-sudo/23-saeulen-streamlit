# v30.8a - Decision Confidence Consistency Fix

v30.8a ist ein gezielter Konsistenz-Patch fuer die in v30.8 eingefuehrte Decision-Confidence-/Evidence-Schicht. Die Trading-Entscheidungen selbst bleiben unveraendert; verbessert wird nur, wie belastbar, aktuell und asset-gerecht die Datenbasis beschrieben wird.

## Behoben
- `Aktualitaet` zeigt im Live-Screener, in Ticker-Details, in Positionen und im Portfolio jetzt bevorzugt den **echten Berlin-Scan-Zeitstempel** statt nur generischer Texte wie `vollstaendig abgeschlossener Atomic-Scan` oder `aktuelle Session`.
- Portfolio-Freshness nutzt den bereits vorhandenen Atomic-Vollscan-Zeitpunkt und erzeugt dafuer keinen Provider-Call.
- Positions-Konfidenz wird ohne aktuellen Atomic-Stand konsequent auf **Niedrig** begrenzt. Ein fehlender belastbarer Stop-Plan kann eine ansonsten hohe Konfidenz auf **Mittel** deckeln.
- Einzelanalyse ist jetzt **asset-aware**:
  - Aktien nutzen weiterhin Fundamental-Coverage als Teil der Evidenz.
  - Commodities/Rohstoffe, ETFs und Indizes werden nicht mehr wegen bewusst fehlender Unternehmens-Fundamentals als schwach belastbar dargestellt.
  - Fuer diese Instrumente basiert die Decision-Confidence auf bereits geladener Kurs-/Historienbasis; eine Historie unter 63 Handelstagen wird transparent als Grenze gezeigt.
- Bei Aktien verhindert ein fehlender belastbarer Benchmark-Kontext jetzt eine `Hoch`-Konfidenz, statt nur als Text unter `Grenzen` aufzutauchen.
- Mobile Ticker-Details zeigen `Keine harten Einstiegsgates aktiv` nicht mehr als Warnbox `Aktive Einstiegsgates`.

## Transparenz
- Es wird weiterhin **kein neuer Trading-Score** berechnet.
- Die Konfidenz beeinflusst weder Ampeln noch Scores, Gates, Harvest/Chop, Stops, TP-Ziele, Portfolio-Risiko oder Orders.
- Fuer die Freshness-Anzeige werden ausschliesslich bereits vorhandene Scan-Zeitstempel verwendet.
- Commodity-/ETF-/Index-Confidence ist eine Datenbasis-Einschaetzung und keine Aussage ueber Richtung oder Kaufqualitaet.

## Unveraendert
- Keine neuen Provider-Abfragen.
- Keine Aenderung an Atomic-Scan-/Snapshot-Semantik.
- Keine Aenderung an v30.6 Harvest-Learning.
- Keine Aenderung an v30.7 Decision Summary oder produktiver Trading-Logik.
