# v28.3 Analysis Core Extraction

Diese Version baut auf v28.2 auf. Die native Multipage-Navigation, Supabase-Speicherung, Repositories und Datenmodelle bleiben kompatibel.

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

Weitere Details: `ARCHITECTURE_V28_3.md` und `RELEASE_NOTES_v28_3.md`.
