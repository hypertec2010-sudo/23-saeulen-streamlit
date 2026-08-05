# Architektur v28.4.2 – Mobile Snapshot Layer

## Ziel

Der Streamlit-Session-Cache bleibt der schnelle Laufzeitspeicher. Zusätzlich
wird nach jedem vollständig abgeschlossenen Live-Scan ein serialisierbarer
Snapshot über den vorhandenen `StorageManager` gespeichert.

```text
Live-Scan
  -> Session-State Cache
  -> Snapshot-Serializer
  -> Supabase app_state
  -> lokaler JSON-Spiegel
```

Nach einem Browser- oder WebSocket-Reconnect gilt die umgekehrte Reihenfolge:

```text
Watchlist + Ticker + Stil + Horizont
  -> stabiler Snapshot-Key
  -> Supabase / lokaler Spiegel
  -> DataFrame-Rekonstruktion
  -> Session-State Cache
  -> sofortige Anzeige
```

## Scan-Schutz

Ein wiederhergestellter Snapshot setzt eine 120-Sekunden-Schutzpause. Während
dieser Zeit darf weder der normale Stale-Cache-Pfad noch der 60-Sekunden-
Heartbeat einen Vollscan auslösen. Manuelles `Jetzt prüfen` bleibt davon
unberührt.

Im Mobile-Modus ist Auto-Scan standardmäßig deaktiviert. Das verhindert, dass
ein Smartphone nach dem Entsperren allein wegen einer fälligen Zeitmarke einen
kompletten Scan beginnt.

## Persistenz

- Namespace: `live_screener_snapshots`
- Identität: SHA-256-Digest aus Watchlist, normalisierten Tickern, Stil und Horizont
- Inhalt: Timestamp, Live-DataFrame, Fehler-DataFrame und UI-Metadaten
- Begrenzung in der App: sechs zuletzt gespeicherte Snapshot-Konfigurationen
- keine zusätzliche SQL-Tabelle
