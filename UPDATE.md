# v28.4.5b1 – New-Listing History Fix

Geändert:
- `modules/ticker_resolver.py`
- `modules/provider_manager.py`
- `modules/legacy_analysis_core.py`
- `VERSION.txt`

Wichtig:
- SKHY lädt Historie ab 13.07.2026.
- SPCX lädt Historie ab 12.06.2026, damit die frühere ETF-Historie unter demselben Symbol nicht vermischt wird.
- Bei unerwartet leerer/kurzer Historie wird zusätzlich `yf.download()` versucht.
- Für normale Ticker bleibt das bisherige Verhalten bestehen.

Keine Änderungen an Supabase, SQL oder Secrets.

Nach Upload testen: SKHY, SPCX, SPX, AAPL.
