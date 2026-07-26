# v28.2 Native Streamlit Multipage-Struktur

Diese Version baut auf v28.1 auf und ändert weder das Supabase-Schema noch bestehende Daten.

## Neue Seiten

- Sofortanalyse
- Kandidaten-Radar
- Watchlisten
- Positionen / Exit
- Trade-Journal

Die Navigation verwendet `st.navigation` und `st.Page`. `app.py` ist jetzt ein kleiner Einstiegspunkt; die bewährte Oberfläche bleibt in `legacy_app.py` erhalten und wird von den Seiten kontrolliert ausgeführt.

## Upgrade

1. Vollständigen Inhalt des ZIP-Pakets ins Repository übernehmen.
2. Bestehende Streamlit-Secrets unverändert lassen.
3. Keine SQL-Migration ausführen.
4. App neu starten.
5. Jede Seite einmal öffnen und Positionen sowie Trade-Journal prüfen.

## Prüfung

```bash
python verify_deployment.py
streamlit run app.py
```

Weitere Details: `ARCHITECTURE_V28_2.md` und `RELEASE_NOTES_v28_2.md`.
