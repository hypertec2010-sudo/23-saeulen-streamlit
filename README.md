# v26.0 Modularisierung Phase 4

Neu ausgelagert:

- `modules/analysis_engine.py` – zentrale Core-/Fallback-Analyse und Asset-Nachbearbeitung
- `modules/cache_layer.py` – gemeinsame Analyse-Cache-Schicht und Marktzeit-Buckets
- `modules/market_data.py` – robuste Preis-/ATR-Helfer
- `modules/ticker_resolver.py` – Ticker-Normalisierung und Kandidatenvarianten
- `modules/scoring_engine.py` – gemeinsame Score-Normalisierung

Die bestehenden UI-, Live-Monitor-, Watchlist-, Risiko-, Positions- und Event-Module bleiben erhalten.

## Deployment

Den vollständigen Inhalt dieses Ordners ins Repository übernehmen. Danach optional ausführen:

```bash
python verify_deployment.py
```

Start:

```bash
streamlit run app.py
```
