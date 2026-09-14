# v30.11 - Calibration Advisor

v30.11 schließt den nächsten Schritt der Lernkette: Nach Harvest-Outcome-Validation (v30.6) und Action-Queue-Outcome-Validation (v30.10) werden die bereits beobachteten 3T-Ergebnisse jetzt in einen **Shadow-only Calibration Advisor** überführt.

Der Advisor darf ausschließlich Hinweise formulieren. Er verändert keine produktive Schwelle, Queue-Kategorie, Decision-Confidence-Regel, Einstiegsgates, Stops, TP-Ziele, Positionsgrößen oder Orders.

## Neu
- Neuer providerfreier Modulbaustein `modules/calibration_advisor.py`.
- Neuer Bereich im Trade-Journal: `Calibration Advisor · v30.11 · Shadow only`.
- Verbindlicher Stichproben-Guard: Gruppenvergleiche werden erst ab mindestens **15 auswertbaren 3T-Fällen je Vergleichsgruppe** als Kalibrierungshinweis interpretiert.
- Action Queue: vergleicht `Jetzt prüfen` gegen `Beobachten` anhand Median-Return, positiver Quote und +2%-Pfadquote.
- Blockierungs-Check: vergleicht `Blockiert` gegen `Jetzt prüfen` anhand 3T-Return und -2%-Pfadquote. Ein später steigender blockierter Wert führt ausdrücklich nicht automatisch zu einer Gate-Lockerung.
- Decision Confidence: prüft, ob `Hoch` gegenüber `Mittel/Niedrig` tatsächlich ein stabileres 3T-Profil zeigt.
- Harvest: prüft die aktuelle Warnschwelle 60 gegen Fälle unter 60 anhand späterem Giveback, bestätigten Teilgewinn-Situationen und `Laufenlassen besser`.
- Chop: prüft beobachtend, ob Chop >=60 tatsächlich mit höherem späterem Giveback einhergeht.
- Shadow-Tabelle für Harvest-Schwellen 55/60/65/70/75/80. Die App zeigt die Profile nebeneinander, wählt aber bewusst **keine automatische Bestschwelle**.
- CSV-Export der Advisor-Empfehlungen.
- Glossar ergänzt um `Calibration Advisor`, `Shadow-Kalibrierung` und `Stichproben-Guard`.

## Mögliche Advisor-Status
- `Halten` / `bestätigt`: die aktuelle Trennung wird von der bisherigen Stichprobe gestützt.
- `Weiter beobachten`: noch kein klares Trennbild.
- `Stichprobe aufbauen`: Mindestanzahl je Vergleichsgruppe noch nicht erreicht.
- `Shadow prüfen` / `Gate-Audit`: genug Daten für eine gezielte analytische Prüfung, aber **keine produktive Änderung**.

## Schutz vor Überoptimierung
- Keine Empfehlung aus kleinen Gruppen unter 15 Fällen.
- Kein automatisches Ranking einer vermeintlich "besten" Harvest-Schwelle.
- Keine nachträgliche Rekonstruktion fehlender Outcomes.
- Grundlage bleiben ausschließlich bereits gespeicherte exakte 3T-Folge-Outcomes aus den bestehenden Lernschichten.

## Unverändert
- Live-/Shadow-/Guarded-Engine-Scores und Schwellen.
- Decision Action Queue Kategorien und Sortierung.
- Harvest-/Chop-Produktivschwellen.
- TP1/TP2/TP3, Stops, Entry-Gates, Portfolio- und Exit-Engine.
- Keine zusätzlichen Yahoo-/HTTP-Provider-Abfragen.
