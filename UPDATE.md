# Update v28.4.7 – Trading Context + Mobile Explainability UX

Geändert:
- `modules/live_monitor.py`: Relative Stärke, Volatilitätsregime und Marktregime als rein informativer Kontext.
- `legacy_app.py`: Kontextspalten Desktop/Mobile, vollständige Mobile-Erklärungen über „Mehr anzeigen“, Cache-Schema v28.4.7.
- `VERSION.txt`

Wichtig:
1. Alle Dateien über die vorhandenen Dateien in GitHub kopieren.
2. Streamlit einmal komplett rebooten.
3. Live-Screener einmal frisch scannen lassen.
4. Desktop: neue Spalten Relative Stärke, Volatilitätsregime, Marktregime prüfen.
5. Mobile: „Mehr anzeigen“ öffnen und prüfen, dass Erklärungen vollständig lesbar sind.

Keine Änderungen an Supabase, SQL oder Secrets. Die neuen Kontextwerte verändern Score/Ampel noch nicht.
