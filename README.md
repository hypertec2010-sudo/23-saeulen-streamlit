# v27.1 Trade-Journal Navigation/Scan Fix

Der Trade-Journal-Bereich nutzt beim Wechsel nur noch den letzten Live-Scan aus dem Cache. Ein automatischer Vollscan der Watchlist wird ausschließlich im aktiven Bereich **Live-Screener** ausgelöst.

# v27.0 Trade-Journal

Neu in dieser Version:

- Teilverkäufe aus offenen Positionen dokumentieren
- Positionen vollständig schließen
- Ausstiegskurs, Datum und Schließungsgrund speichern
- realisierten Gewinn/Verlust und R-Multiple berechnen
- Stop-Anpassungen mit Historie protokollieren
- Trade-Notizen und Erkenntnisse speichern
- persistentes Trade-Journal je Watchlist
- Journal-Kennzahlen: geschlossene Trades, Teilverkäufe, realisiertes P/L, Trefferquote und durchschnittliches R
- CSV-Export des Trade-Journals

Neues Modul:

- `modules/trade_journal.py`

## Deployment

Den vollständigen Inhalt dieses Ordners ins Repository übernehmen. Insbesondere muss
`modules/trade_journal.py` direkt neben den übrigen Moduldateien liegen.

Optional prüfen:

```bash
python verify_deployment.py
```

Start:

```bash
streamlit run app.py
```
