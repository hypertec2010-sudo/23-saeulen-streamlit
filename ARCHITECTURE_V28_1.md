# Architektur v28.1

v28.1 trennt die fachlichen Daten von der technischen Speicherung.

```text
Streamlit UI / app.py
        |
        v
Fachmodule (Positionen, Journal, Events)
        |
        v
Repository Registry
        |
        +-- PositionRepository
        +-- TradeJournalRepository
        +-- EventRepository
        +-- WatchlistRepository
        |
        v
StorageManager
        |
        +-- SupabaseBackend
        +-- LocalJsonBackend (Spiegel/Fallback)
```

## Domain-Modelle

`modules/domain/models.py` enthaelt die verbindlichen Modelle:

- `Position`
- `JournalEntry`
- `SignalEvent`
- `WatchlistItem`

Die Modelle validieren und normalisieren persistierte Daten. Gleichzeitig bleiben die bisherigen Dictionary-Feldnamen erhalten, damit vorhandene Supabase-Daten und die bestehende UI ohne Migration weiter funktionieren.

## Repositories

`modules/repositories/` kapselt die Namespaces der zentralen Speicherung:

- `PositionRepository` -> `positions`
- `TradeJournalRepository` -> `trade_journal`
- `EventRepository` -> `event_log`
- `WatchlistRepository` -> `watchlists`

Die Streamlit-Fachmodule erhalten ihr jeweiliges Repository per Dependency Injection. Direkte Namespace-Zugriffe bleiben nur als Rueckwaertskompatibilitaets-Fallback erhalten.

## Datenkompatibilitaet

Es ist keine neue Supabase-Tabelle und keine SQL-Migration erforderlich. `app_state` sowie alle vorhandenen Namespaces bleiben unveraendert.
