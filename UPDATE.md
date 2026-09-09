# v30.6 - Harvest Outcome & Learning Validation

v30.6 macht aus der neuen Short-Term-Trader-/Harvest-Schicht erstmals eine systematisch beobachtbare Kalibrierung. Die App speichert dafür pro Watchlist und Berlin-Tag nur den neuesten vollständig abgeschlossenen Atomic-Vollscan und vergleicht die damalige Harvest-/Chop-Einschätzung später mit real beobachteten 1T/3T/5T-Folgeergebnissen.

## Neu
- Neue providerfreie Persistenz `harvest_outcome_learning_v306`.
- Pro Watchlist wird nur ein letzter vollständiger Scan pro Berlin-Tag gehalten; mehrere Streamlit-Reruns oder Intraday-Scans übergewichten einen Tag nicht.
- Speicherung nur aus `complete + atomic` Vollscans. Teilstände und gemischte Frames werden nicht gelernt.
- 1/3/5T-Outcomes werden nur berechnet, wenn am exakten Mo-Fr-Zieltag ein vollständiger Folge-Scan vorliegt. Fehlende Tage bleiben offen statt mit einem späteren Kurs ersetzt zu werden.
- Für jeden auswertbaren Fall: Forward Return, Scan-Max, Scan-Min, Giveback vom beobachteten Scan-Peak, Trader-Ziel erreicht ja/nein und rein beobachtende Bewertung `Teilgewinn eher sinnvoll / Laufenlassen eher besser / gemischt`.
- Vergleich nach Harvest-Bändern `<45`, `45-59`, `60-74`, `75-100` sowie Chop-Bändern.
- Separater Check echter deduplizierter `Short-Term Profit Harvest`-Positionsereignisse aus dem bestehenden Eventlog.
- Neue Trade-Journal-Ansicht `Short-Term Harvest · Outcome & Validation · v30.6` mit 1/3/5T-, Harvest-, Chop- und Positions-Harvest-Tabs sowie CSV-Export.
- Stichproben-Reifegrad verhindert, dass kleine Datenmengen als belastbare Schwellenkalibrierung erscheinen.
- Glossar um Outcome, Validation, Forward Return, MFE und MAE ergänzt.

## Transparenz / Grenzen
- Scan-Max und Giveback basieren auf vorhandenen vollständigen Vollscan-Punkten, nicht auf Intraday-Hochs/-Tiefs; echte Zwischenbewegungen können daher größer gewesen sein.
- Börsentage werden ohne zusätzlichen Kalenderprovider als Mo-Fr-Zieltage behandelt. Fehlt wegen Feiertag oder ausgelassenem Scan der exakte Zieltag, wird kein Ersatzwert erfunden.
- Die Klassifikation dient ausschließlich der späteren persönlichen Kalibrierung.

## Unverändert
- Keine automatische Änderung an Harvest-/Chop-Schwellen.
- Keine Änderung an Trader-Zielen, TP1/TP2/TP3, Stops, Orders oder Positionsgrößen.
- Keine Änderung an Live-, Shadow-, Guarded-, Exit- oder Portfolio-Logik.
- Keine neuen Provider-Abfragen.
