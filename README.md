# v28.3.2 Cockpit Navigation State Fix

Diese Version baut auf v28.3 auf. Die native Multipage-Navigation, Supabase-Speicherung, Repositories, Datenmodelle und die extrahierte Analyse-Pipeline bleiben kompatibel.

## Navigations-Fix v28.3.2

Der Startbereich einer nativen Seite wird nur noch beim tatsächlichen Seitenwechsel gesetzt. Manuelle Wechsel im Trading-Cockpit zwischen Live-Screener, Risiko-Rechner, Positionen / Exit, Trade-Journal und Historie bleiben dadurch bei Streamlit-Reruns erhalten.

## Wichtigste Änderung

Die große Analyse-Pipeline befindet sich jetzt in:

```text
modules/legacy_analysis_core.py
```

`legacy_app.py` enthält diese rund 2.600 Zeilen nicht mehr. Der Aufruf erfolgt weiterhin über `modules/analysis_engine.py`, sodass die bestehende Oberfläche und das Ergebnisformat unverändert bleiben.

## Upgrade

1. Vollständigen Inhalt des ZIP-Pakets ins Repository übernehmen.
2. Vorhandene Streamlit-Secrets unverändert lassen.
3. Keine SQL-Migration ausführen.
4. App neu starten.
5. Eine Sofortanalyse, den Radar und eine Watchlist-Analyse testen.

## Prüfung

```bash
python verify_deployment.py
streamlit run app.py
```

Weitere Details: `RELEASE_NOTES_v28_3_1.md`, `ARCHITECTURE_V28_3.md` und `RELEASE_NOTES_v28_3.md`.


## v28.3.2

Der Live-Screener nutzt einen 60-Sekunden-Heartbeat und den Cache-Zeitstempel für zuverlässige automatische Scans. Details: `RELEASE_NOTES_v28_3_2.md`.
