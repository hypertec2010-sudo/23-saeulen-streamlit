# v28.4.5d - Datenqualitaet

Geaendert:
- modules/live_monitor.py
- legacy_app.py
- VERSION.txt
- UPDATE.md
- CHANGELOG.md

Neu:
- Datenqualitaet getrennt vom Trading-Score
- 1-5 Sterne anhand Historienlaenge und technischer Datenbasis
- New Listings werden sichtbar als reduzierte Datenbasis gekennzeichnet
- Datenqualitaet auch in der Mobile-Kartenansicht
- Detailfeld `Datenbasis` bleibt in Historie/Details verfuegbar

Keine Aenderungen:
- Supabase / SQL
- Secrets
- Watchlists
- Positionen / Trade-Journal

Test nach Upload:
1. Streamlit rebooten.
2. Live-Screener starten.
3. AAPL und einen lang gelisteten Titel sollten typischerweise hohe Datenqualitaet zeigen.
4. SKHY/SPCX muessen analysierbar bleiben und eine reduzierte/New-Listing-Datenqualitaet zeigen.
5. Trading-Score/Ampel darf durch die neue Qualitaetsanzeige nicht veraendert werden.
