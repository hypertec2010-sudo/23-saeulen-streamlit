# v30.10 - Action Queue Outcome Validation

v30.10 ergänzt die v30.9 `Decision Action Queue` um eine rein beobachtende Outcome-Validierung. Ziel ist nicht, die Queue automatisch zu verändern, sondern nach einigen Handelstagen messbar zu prüfen, ob `Jetzt prüfen`, `Beobachten` und `Blockiert` tatsächlich unterschiedliche Folgeergebnisse zeigen.

## Neu
- Neue providerfreie Persistenz `decision_action_queue_learning_v3010`.
- Pro Watchlist wird nur der **letzte vollständig abgeschlossene Atomic-Queue-Stand pro Berlin-Tag** gespeichert.
- Streamlit-Reruns bzw. derselbe Scan werden dedupliziert; ein späterer kompletter Scan desselben Tages ersetzt den früheren Tagesstand.
- 1T-/3T-/5T-Outcomes werden nur gewertet, wenn am exakten Mo-Fr-Zieltag wieder ein vollständiger Atomic-Scan vorhanden ist. Fehlende Tage bleiben offen und werden nicht mit einem späteren Kurs ersetzt.
- Für jeden auswertbaren Fall werden u. a. gespeichert/abgeleitet:
  - ursprüngliche Queue-Kategorie,
  - Decision-Confidence,
  - Live-Score,
  - Folge-Return,
  - beobachtetes Scan-Max / Scan-Min,
  - ob im beobachteten Pfad mindestens +2% bzw. -2% erreicht wurden,
  - Queue-Kategorie am Zieltag und Kategorie-Wechsel.
- Neue Ansicht `Action Queue · Outcome Validation` direkt unter der Decision Action Queue.
- Tab `Kategorie 1/3/5T`: Median Return, positive Quote, +2%-/ -2%-Pfadquote sowie Scan-Max/Scan-Min je Queue-Kategorie.
- Tab `Confidence · 3T`: separate Beobachtung, ob `Hoch` / `Mittel` / `Niedrig` später unterschiedliche Outcomes zeigen.
- Tab `Kategorie-Wechsel · 3T`: zeigt, wie stabil bzw. wechselhaft die Queue-Einstufung über drei Handelstage bleibt.
- Automatische Text-Hinweise werden erst ab ausreichender Teilstichprobe erzeugt und bleiben rein diagnostisch.
- CSV-Export der Outcome-Einzelfälle.
- Glossar um `Action Queue Outcome Validation`, `Positive Rate` und `Scan-Max / Scan-Min` ergänzt.

## Interpretationsgrenzen
- `Jetzt prüfen` bedeutet weiterhin **nicht automatisch kaufen**.
- Scan-Max / Scan-Min beruhen auf vorhandenen vollständigen Scanpunkten, nicht auf Intraday-Hochs oder Intraday-Tiefs. Zwischen den Scans können größere Bewegungen stattgefunden haben.
- Mo-Fr wird ohne zusätzlichen Börsenkalender als Zieltag verwendet. Fällt ein Zieltag auf einen Feiertag oder fehlt ein vollständiger Scan, wird der Fall nicht künstlich aufgefüllt.
- Kleine Stichproben werden ausdrücklich als `Zu klein` bzw. `Früh` markiert.
- Die Outcome-Auswertung ist eine Qualitätskontrolle der Priorisierung und keine Backtest-Garantie.

## Unverändert
- Keine Änderung an der v30.9 Queue-Kategorisierung oder Sortierung.
- Keine automatische Anpassung an Live-Score, Decision-Confidence, Harvest/Chop, Gates oder Engine-Regeln.
- Keine Änderung an TP1/TP2/TP3, Stops, Orders oder Positionsgrößen.
- Keine neuen Provider-Abfragen.
