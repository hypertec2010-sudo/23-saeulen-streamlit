# Architektur v28.3 – Analyse-Core

## Aufrufkette

```text
Streamlit-Seite
  -> legacy_app.analyze_stock(...)
  -> modules.analysis_engine.analyze_stock(...)
       -> analysis_core.analyze_stock (bevorzugter Core)
       -> modules.legacy_analysis_core.legacy_analyze_stock (Kompatibilitäts-Fallback)
```

`legacy_app.py` bindet nur die von der extrahierten Pipeline benötigten Hilfsfunktionen und Konstanten. Der Core selbst liegt nicht mehr im UI-Skript.

## Abgrenzung

- `modules/analysis_engine.py`: Facade, Fallback-Steuerung und Asset-Nachbearbeitung
- `modules/legacy_analysis_core.py`: bewährte vollständige Analyse-Pipeline und Kontextdiagnose
- `modules/market_data.py`: kleine wiederverwendbare Marktdaten-Helfer
- `modules/scoring_engine.py`: allgemeine Score-Helfer
- `legacy_app.py`: verbleibende UI- und Kompatibilitätsfunktionen

## Nächste Extraktionsstufe

Die noch in `legacy_app.py` liegenden Analyse-Helfer können anschließend nach Fachbereichen verschoben werden, etwa technische Struktur, Fundamentals, FOMO und Radar. Die wichtigste monolithische Pipeline ist mit v28.3 bereits aus dem UI-Skript entfernt.
