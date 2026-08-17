# v28.4.5b4 – New-Listing Core Routing Fix

## Ursache
Die App ruft zuerst einen älteren `analysis_core` auf. Dieser beendet SKHY/SPCX mit
`Nicht genug Kursdaten für belastbare Analyse`, bevor die neue New-Listing-Logik erreicht wird.

## Fix
`modules/analysis_engine.py` erkennt genau diesen Mindesthistorien-Fehler und leitet den Ticker
an `legacy_analysis_core.py` weiter. Dort greift die reduzierte New-Listing-/Momentum-Analyse.

## Hochladen
Alle Dateien dieses ZIPs über die vorhandenen Dateien im Repository kopieren und Streamlit rebooten.
Keine SQL-/Supabase-/Secrets-Änderung erforderlich.

## Test
SKHY, SPCX, AAPL, SPX
