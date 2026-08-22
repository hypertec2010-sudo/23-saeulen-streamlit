# v28.5a1 – Engine Calibration & EU Benchmark Fallback Fix

Geändert:
- Relative-Stärke-Gewichtung feiner kalibriert.
- +2 bis +8 % 63T-Outperformance gibt nur noch +1 RS-Punkt.
- +8 bis +15 % gibt +3, ab +15 % +4.
- Negative RS wird analog stärker abgestuft.
- Fehlt die konkrete 63T-Benchmark-RS, wird nicht mehr `Score 0/100` als Ersatz verwendet.
- Stattdessen: `n/a · Benchmarkdaten fehlen` und RS-Beitrag = 0.
- Cache-Schema angehoben, damit die neue Kalibrierung sofort sichtbar wird.

Wichtig:
- Ampel und Basis-Score bleiben unverändert (Beobachtungsmodus).
- Keine SQL-/Supabase-/Secrets-Änderung.

Testwerte nach frischem Scan vergleichen: UNP, DT, MSFT, LITE, ACN, IPS.PA, SAP.DE.
