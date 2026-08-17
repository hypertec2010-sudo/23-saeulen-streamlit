# Update v28.4.5b

## Geändert
- Smart Ticker Resolver: SPX, NDX, VIX, DJI, RUT und Share-Class-Symbole wie BRK.B.
- Gültige normale Ticker wie SKHY und SPCX bleiben unverändert.
- New-Listing-Unterstützung: ab 20 Handelstagen reduzierte Kurzfristanalyse statt pauschaler Ablehnung wegen fehlender MA200-Historie.
- Langfristige Indikatoren, für die noch keine Historie existiert, bleiben neutral/nicht verfügbar und liefern keine künstlichen Pluspunkte.

## Upload
Die Dateien dieses Patch-ZIPs über die vorhandenen Dateien im Repository legen. Es sind nur 6 Dateien.

## Keine Änderung nötig
- Supabase
- SQL
- Streamlit Secrets
- Watchlists / Positionen / Journal

## Praxistest
Nach Deployment bitte nacheinander prüfen: SKHY, SPCX, SPX, AAPL, SAP.DE, BRK.B.
