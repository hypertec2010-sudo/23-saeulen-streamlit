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
