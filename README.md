# v28.4.4 Complete Batch Scanning

Diese Version beseitigt das bisher stille 40er-Limit des Live-Screeners.

## Wichtigste Änderungen

- **Alle Werte** ist der Standard-Scanumfang.
- Alternativ können 40, 80 oder 120 Werte bewusst gewählt werden.
- Nicht enthaltene Ticker werden sichtbar als **ausstehend** ausgewiesen.
- Der Scan läuft in 20er-Batches mit Fortschrittsanzeige.
- Nach jedem Batch wird ein Supabase-/Local-Checkpoint gespeichert.
- Nach einer Browser- oder Display-Unterbrechung kann der Scan fortgesetzt werden.
- Nicht analysierbare Ticker und nicht aufgerufene Ticker sind klar getrennt.
- Duplikate werden gezählt und nur einmal analysiert.

## Deployment

1. Vollständigen ZIP-Inhalt in das Repository übernehmen.
2. Keine SQL-Migration ausführen.
3. Streamlit-Secrets unverändert lassen.
4. App neu starten.
5. Im Live-Screener den gewünschten **Scan-Umfang** prüfen.
6. Für vollständige Abdeckung **Alle Werte** verwenden.
7. GitHub Actions unter **v28.4.4 Quality Gate** kontrollieren.

Details stehen in `RELEASE_NOTES_v28_4_4.md` und `ARCHITECTURE_V28_4_4.md`.
