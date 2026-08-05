# Supabase Setup fuer v28.1

## 1. Tabelle anlegen

Den kompletten Inhalt von `supabase_schema.sql` im SQL Editor des Supabase-Projekts ausfuehren.

## 2. Zugangsdaten finden

In Supabase unter **Project Settings > API** werden benoetigt:

- Project URL
- Service Role Key

Der Service Role Key ist ein Server-Secret. Er darf nicht in Quellcode, Screenshots oder Browser-Ausgaben gelangen.

## 3. Streamlit Secrets konfigurieren

In Streamlit Community Cloud unter **App settings > Secrets** die Konfiguration aus `.streamlit/secrets.example.toml` einfuegen und die Platzhalter ersetzen.

## 4. Verbindung und Migration

Nach einem Neustart der App:

1. Sidebar oeffnen.
2. **Hilfen & Verwaltung** oeffnen.
3. Unter **Speicherung v28.1** auf **Speicher testen** klicken.
4. **Legacy-JSON importieren** ausfuehren.
5. **Google-Watchlists importieren** ausfuehren, sofern die bisherigen Google-Secrets noch vorhanden sind.
6. Watchlist, Position und Journal testweise aendern und nach einem Reload kontrollieren.

## 5. Rueckfallmodus

Ohne gueltige Supabase-Konfiguration bleibt die App lauffaehig. Sie verwendet dann `.app_storage/` als lokalen JSON-Speicher. Auf kurzlebigen Cloud-Dateisystemen ist dieser Modus nur als Notfall-Fallback gedacht.


Hinweis v28.1: Gegenueber v28.0 ist keine SQL-Aenderung erforderlich. Die Repository- und Domain-Schicht nutzt dieselbe Tabelle `app_state`.

## v28.4.2 Live-Screener-Snapshots

Für die mobile Wiederherstellung ist keine neue Tabelle erforderlich. Die App
speichert den letzten Live-Screener-Stand im vorhandenen `app_state`-Schema
unter dem Namespace `live_screener_snapshots`. Nach dem ersten abgeschlossenen
Scan erscheint dafür automatisch ein Eintrag im Supabase Table Editor.
