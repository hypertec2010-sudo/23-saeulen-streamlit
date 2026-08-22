# v28.4.5d1 – Datenqualitaet Cache-Fix

Geaendert:
- Live-Screener-Cache und Supabase-Snapshot erhalten eine neue Schema-Version.
- Alte v28.4.5c-Scans ohne Datenqualitaets-Spalte werden nicht mehr wiederverwendet.
- Nach dem Update wird einmalig ein frischer Live-Scan aufgebaut.
- Danach zeigen Desktop-Tabelle und Mobile-Karten dieselben Datenqualitaetswerte.

Keine Aenderungen an:
- Supabase-Schema
- Secrets
- Watchlists
- Positionen
- Trade-Journal

Nach Update:
1. Dateien in GitHub ueberschreiben.
2. Streamlit einmal rebooten.
3. Live-Screener oeffnen und einen vollstaendigen Scan abwarten bzw. 'Jetzt pruefen' klicken.
4. Desktop und Mobile vergleichen.
