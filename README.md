# v28.1 Repository- und Datenmodell-Schicht

Diese Version baut auf der funktionierenden Supabase-Migration aus v28.0 auf. Die bestehende Tabelle `app_state`, die Streamlit-Secrets und bereits gespeicherte Daten bleiben voll kompatibel.

## Neu in v28.1

- verbindliche Domain-Modelle fuer Positionen, Journal-Eintraege, Signal-Events und Watchlist-Eintraege
- zentrale Repository Registry
- eigene Repositories fuer Positionen, Trade-Journal und Event-Log
- Watchlist-Repository ueber die neue Repository-API exportiert
- Dependency Injection in die bestehenden Fachmodule
- automatische Normalisierung persistierter Datensaetze
- weiterhin lokaler Spiegel und Supabase-Fallback
- erweiterte Regressionspruefungen fuer Modelle und Repositories

## Neue Struktur

```text
modules/
    domain/
        __init__.py
        models.py
    repositories/
        __init__.py
        base.py
        position_repository.py
        trade_journal_repository.py
        event_repository.py
        registry.py
    storage/
        ... unveraendert aus v28.0
```

Weitere Details stehen in `ARCHITECTURE_V28_1.md`.

## Upgrade von v28.0

1. Den vollstaendigen Inhalt dieses Pakets in das Repository uebernehmen.
2. Die vorhandenen Streamlit-Secrets unveraendert lassen.
3. Keine neue SQL-Datei ausfuehren; das vorhandene `app_state`-Schema bleibt gueltig.
4. Die App neu starten.
5. Unter **Hilfen & Verwaltung > Speicherung v28.1** den Speichertest ausfuehren.
6. Positionen, Trade-Journal und Watchlists kurz auf vorhandene Daten pruefen.

## Deployment pruefen

```bash
python verify_deployment.py
```

Start:

```bash
streamlit run app.py
```
