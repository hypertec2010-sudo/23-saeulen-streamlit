# UPDATE v28.4.8 – Trading Context Validation

Geändert:
- `modules/live_monitor.py`
- `legacy_app.py`
- `VERSION.txt`

Neu:
- Relative Stärke zeigt 63T-Aktienperformance, Benchmark-Performance und Outperformance.
- Volatilitätsregime zeigt ATR(14)% und die verwendeten Schwellen.
- Marktregime zeigt Benchmark, Kurs, MA50/MA200, 1T/5T-Performance und den Regime-Grund.
- Mobile und Ticker-Details zeigen die vollständigen Berechnungen aufklappbar.
- Score und Ampel bleiben unverändert.

Nach dem Upload:
1. Dateien überschreiben.
2. Streamlit einmal rebooten.
3. Live-Screener frisch scannen.
4. Einen US-Wert, einen EU-Wert und ein New Listing prüfen.

Keine Änderungen an Supabase, SQL oder Secrets.
