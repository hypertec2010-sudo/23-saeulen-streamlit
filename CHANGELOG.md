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
