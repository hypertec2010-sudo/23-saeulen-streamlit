# Architektur v28.4.4 – Live-Screener Batch Pipeline

## Ziel

Das frühere implizite 40er-Limit wird durch eine explizite, testbare Scan-Pipeline ersetzt.

## Komponenten

### `modules/live_scan_batches.py`

Reine Logik für:

- Ticker-Normalisierung und Duplikaterkennung
- sichtbare Scan-Grenzen
- Batch-Aufteilung
- Checkpoint-Metadaten
- Zusammenführen und globale Sortierung der Resultate

### `modules/live_monitor.py`

`build_live_watchlist_monitor_v212` analysiert ohne explizites Limit alle übergebenen eindeutigen Ticker. Ein Limit wird nur noch bewusst vom Aufrufer gesetzt.

### `modules/live_screener_snapshot.py`

Snapshot-Version 2 speichert zusätzlich `scan_meta`:

- vollständiger/unvollständiger Scan
- verarbeitete Ticker
- noch offene Ticker
- bewusst ausstehende Ticker
- Duplikate
- Batch-Größe

### `legacy_app.py`

Die Streamlit-Oberfläche:

1. erstellt einen sichtbaren Scan-Plan,
2. verarbeitet die Auswahl in 20er-Batches,
3. speichert nach jedem Batch einen Checkpoint,
4. setzt einen unvollständigen Checkpoint fort,
5. zeigt Fehler und Scan-Grenzen getrennt an.

## Cache-Identität

Die Cache- und Snapshot-Identität enthält die tatsächlich ausgewählten Ticker. Ein Wechsel von 40 auf alle Werte erzeugt deshalb einen neuen, passenden Scan-Zustand.
