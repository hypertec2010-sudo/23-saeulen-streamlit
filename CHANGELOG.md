## v28.6
- Shadow Mode eingeführt: Live-Ampel und virtuelle Engine-Ampel laufen parallel.
- Shadow-Ampel basiert ausschließlich auf dem Guarded Engine-Score; produktive Ampel bleibt unverändert.
- Abweichungen werden als Aufwertung/Abwertung/Gleich gekennzeichnet.
- Deduplizierte persistente Shadow-Historie mit Zeitstempel ergänzt.
- Desktop, Mobile und Ticker-Details zeigen den Shadow-Status.

## v28.5b
- RS-Dynamik 21T/63T als Beobachtungswert ergänzt.
- Dynamikschwellen: ab +5 Prozentpunkten verbessert, bis -5 verschlechtert, dazwischen stabil.
- Keine Score-/Ampelwirkung in diesem Release.
- Kontext-Confidence in der mobilen Darstellung klarer benannt.
- Cache-Schema auf v28.5b angehoben.

## v28.5a
- Trading Engine 2.0 im Beobachtungsmodus eingeführt.
- Context-Adjusted Score zusätzlich zum unveränderten Basis-Live-Score.
- Transparente Beiträge für Relative Stärke, Marktregime und Volatilitätsregime.
- Desktop, Mobile und Ticker-Detailansicht um Engine-Vergleich erweitert.
- Bestehende Ampel bleibt vollständig am bisherigen Basis-Score.

# Changelog

## v28.4.8.1
- Behoben: gemeinsamer NameError im Trading-Context-Validierungspfad.
- ATR-/Volatilitäts-Erklärung wird erst nach Berechnung der benötigten Werte erzeugt.
- Cache-Schema erneuert, um fehlerhafte alte Live-Scans zu verwerfen.

## v28.5a1
- Context-Engine-RS feiner kalibriert.
- Ungültiges `Score 0/100`-Fallback bei fehlender Benchmark-RS entfernt.
- Fehlende Benchmark-RS ist neutral und wird transparent als n/a angezeigt.

## v28.5a2
- Context Confidence im Engine-Beobachtungsmodus ergänzt.
- Missing-Data Guard: unvollständiger Kontext darf keinen positiven Bonus erzeugen.
- Kontext-Verlässlichkeit in Desktop, Mobile und Engine-Erklärung sichtbar.

## v28.5c
- RS-Level + RS-Dynamik gemeinsam im Engine-Beobachtungsscore kalibriert.
- RS-Dynamik modifiziert den Level-Beitrag statt eines pauschalen Add-ons.
- Context Confidence eindeutig als `Kontext ★...` gekennzeichnet.
- Ampel bleibt weiterhin am Basis-Score.

## v28.5d
- Engine Guardrails im Beobachtungsmodus ergänzt.
- Kontext kann fehlende Trigger/hart blockierte Setups nicht hochstufen.
- Chart-/CRV-Bremsen begrenzen positive Kontextboni.
- Guarded Engine-Score und Engine-Empfehlung in Desktop/Mobile/Detailansicht.

## v28.6a
- Shadow Validation Dashboard hinzugefuegt.
- Shadow-Episoden mit Dauer, Richtung und Kursentwicklung auswertbar.
- Neue Shadow-Events enthalten den Ereigniskurs.
- Produktive Trading-Logik bleibt unveraendert.
