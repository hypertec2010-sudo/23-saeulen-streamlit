# Release Notes v28.2

## Native Multipage-Struktur

- `app.py` ist jetzt ein kleiner Einstiegspunkt für Seiteneinstellungen, Login und Navigation.
- Die bisherige stabile Oberfläche liegt kompatibel in `legacy_app.py`.
- Native Navigation über `st.navigation` und `st.Page`.
- Eigene Seiten für Sofortanalyse, Radar, Watchlisten, Positionen und Trade-Journal.
- Alte Live-Monitor-Query-Parameter können die Seitennavigation nicht mehr überschreiben.
- Trade-Journal und Positionsseite öffnen direkt den passenden Trading-Cockpit-Bereich.
- Fallback-Navigation für ältere Streamlit-Versionen.
- Supabase, Repository-Schicht und Domain-Modelle bleiben unverändert.

## Upgrade

Den vollständigen Paketinhalt deployen. Keine SQL-Migration und keine Änderung an den Secrets erforderlich.
