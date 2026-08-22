# v28.4.8.1 – Live-Screener Context Hotfix

## Ursache
In v28.4.8 wurde `Volatilitäts-Details` berechnet, bevor `atr_pct_live` und `volatility_regime` erzeugt wurden. Dadurch scheiterte jeder Live-Screener-Ticker am selben gemeinsamen Codepfad.

## Fix
- Volatilitäts-Validierung hinter die ATR-/Regime-Berechnung verschoben.
- Live-Cache-Schema angehoben, damit der fehlerhafte 59/59-Scan nicht erneut geladen wird.
- Keine Änderung an Score, Ampel oder Trading-Regeln.

## Nach Upload
1. Dateien überschreiben.
2. Streamlit komplett rebooten.
3. Live-Screener öffnen.
4. Einmal `Jetzt prüfen` starten.

Keine Änderung an Supabase-Schema, SQL oder Secrets.
