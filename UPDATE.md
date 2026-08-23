# v28.6b – Shadow Compare UI + Versionskonsistenz

Neu:
- Live-Ampel und Shadow-Ampel stehen in der Desktop-Tabelle direkt nebeneinander ganz vorne.
- Shadow-Abweichung folgt direkt dahinter, danach erst Ticker/Name/Kurs.
- Live-Score und Guarded Engine-Score stehen ebenfalls weiter vorne für den schnellen Vergleich.
- Zentrale App-Version auf v28.6b angehoben.
- Hauptbereiche zeigen konsistent dieselbe aktuelle Release-Version.
- Veraltete sichtbare Versionspräfixe in Hinweistexen wurden bereinigt.
- Cache-Schema angehoben, damit die neue Tabellenreihenfolge nach dem Update sofort greift.

Unverändert:
- Live-Ampel-Logik
- Shadow-/Engine-Regeln
- Scores und Guardrails
- Supabase/SQL/Secrets

Nach Upload: Streamlit einmal rebooten und einen frischen Scan starten.
