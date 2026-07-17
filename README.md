# Capital Hill Score Modell v25.1 – Modular Deployment-Fix

## Wichtig für Streamlit Cloud

Nicht nur `app.py` hochladen. Im GitHub-Repository müssen diese Dateien auf derselben Ebene liegen:

```text
app.py
modules/
  __init__.py
  risk_calculator.py
  position_monitor.py
  event_log.py
```

Am einfachsten den Inhalt dieses ZIPs vollständig in das Repository entpacken und committen.

Startdatei in Streamlit Cloud: `app.py`

Lokaler Start:

```bash
streamlit run app.py
```


## v25.2 – Modularisierung Phase 2

Zusaetzliche Module:

- `modules/live_monitor.py`: Live-Screener, Ampellogik, Hysterese, Trade-State und Statushistorie
- `modules/watchlist_storage.py`: Startkurse, Batch-Queue und Watchlist-Speicherung

Beim GitHub-Upload muss weiterhin der komplette Ordner `modules/` neben `app.py` liegen.
