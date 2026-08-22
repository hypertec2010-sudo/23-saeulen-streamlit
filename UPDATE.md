# v28.5b – RS-Dynamik 21T/63T

Neu:
- RS-Dynamik vergleicht 21T- mit 63T-Benchmark-Outperformance.
- Anzeige: Verbessert, Stabil oder Verschlechtert inklusive Differenz in Prozentpunkten.
- Vollständige Berechnungsbasis in Mobile/Ticker-Details.
- Mobile Bezeichnung `Kontext-Confidence` zur klaren Abgrenzung von `Datenqualität`.
- Neues Live-Cache-Schema erzwingt einmalig einen frischen Scan.

Wichtig:
- RS-Dynamik ist in v28.5b rein informativ.
- Sie verändert weder Kontext-Anpassung/Engine-Score noch die bestehende Ampel.
- Missing-Data-Guard aus v28.5a2 bleibt aktiv.
- Keine Änderungen an Supabase, SQL oder Secrets.

Test:
1. Dateien in GitHub überschreiben.
2. Streamlit komplett rebooten.
3. Live-Screener frisch scannen.
4. QRVO sowie UNP/DT/MSFT/LITE prüfen.
5. SAP.DE/IPS.PA sollen bei fehlenden Vergleichsdaten `n/a · Vergleichsdaten fehlen` zeigen.
