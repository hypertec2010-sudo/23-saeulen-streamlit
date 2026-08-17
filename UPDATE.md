# Update v28.4.5a – Smart Provider Manager

## Geändert

- `modules/provider_manager.py` neu: zentraler Marktdaten-Zugriff.
- `legacy_app.py`: Daily-Historie, 60m-Historie und Benchmark-Historie nutzen den Provider Manager.
- App-Version auf `v28.4.5a` gesetzt.

## Bewusst noch nicht enthalten

- Keine zweite Datenquelle / kein Stooq-Fallback.
- Noch kein automatischer 429-Retry mit Wartequeue.
- Noch keine neue Datenqualitäts-Spalte.
- Noch keine erweiterten Index-Aliasse.

Diese Punkte folgen in v28.4.5b–d. So bleibt der erste Infrastruktur-Schritt klein und risikoarm.

## Keine Änderungen erforderlich

- Supabase-Schema
- Streamlit-Secrets
- Watchlists
- Positionen
- Trade-Journal

## Schnelltest nach Upload

1. `AAPL` in der Sofortanalyse prüfen.
2. Live-Screener einmal manuell starten.
3. Prüfen, dass Kurse und Unternehmensdaten wie zuvor erscheinen.
4. Radar kurz starten und einen Kandidaten öffnen.
