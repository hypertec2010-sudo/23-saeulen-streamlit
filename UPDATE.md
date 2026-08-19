# Update v28.4.5c - Rate-Limit & Retry

Geaendert:
- `modules/provider_manager.py`: zentrale Request-Drosselung, 429-Cooldown und kontrollierte Retries.
- `modules/live_monitor.py`: Rate-Limits werden als `Temporär ausstehend` statt dauerhaft nicht analysierbar markiert.
- `modules/live_scan_batches.py`: temporaere Rate-Limit-Ticker gelten im Checkpoint nicht als abgeschlossen und werden beim naechsten Scan erneut versucht.
- `VERSION.txt`, `CHANGELOG.md`.

Keine Aenderungen an Supabase, SQL, Secrets, Watchlists, Positionen oder Journal.

Nach Update:
1. Dateien in GitHub ueberschreiben.
2. Streamlit komplett rebooten.
3. Live-Screener starten.
4. Falls Yahoo einen 429 liefert, muss der Wert als temporaer ausstehend erscheinen und bei einem Folgescan erneut versucht werden.
5. MRVL, AVGO und QRVO gezielt pruefen.
