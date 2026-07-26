# v28.0 Supabase Storage Migration

Diese Version fuehrt eine zentrale Speicherabstraktion ein. Ohne Supabase-Konfiguration arbeitet die App weiterhin mit einem atomaren lokalen JSON-Fallback. Sobald Supabase in den Streamlit-Secrets konfiguriert ist, werden die Daten remote gespeichert und zusaetzlich lokal gespiegelt.

## Migrierte Bereiche

- Watchlists und Watchlist-Einstellungen
- offene Positionen
- Trade-Journal
- Signal-/Trade-Event-Log
- Watchlist-Startkurse
- Live-Monitor-Statushistorie

Google OIDC kann weiterhin fuer die Anmeldung verwendet werden. Die Anmeldung und die Datenspeicherung sind voneinander getrennt. Bestehende Google-Sheets-Analyse- und Auto-Run-Logs bleiben in v28.0 noch optional bestehen; die operativen Watchlists koennen bereits auf den neuen Speicher umgestellt werden.

## Neue Dateien

```text
modules/storage/
    __init__.py
    base.py
    local_backend.py
    manager.py
    migration.py
    supabase_backend.py
    watchlist_repository.py

supabase_schema.sql
migrate_storage.py
.streamlit/secrets.example.toml
```

## Supabase einrichten

1. Ein Supabase-Projekt anlegen.
2. `supabase_schema.sql` im Supabase SQL Editor ausfuehren.
3. Die Werte aus `.streamlit/secrets.example.toml` in die Streamlit-App-Secrets uebernehmen.
4. Die App neu starten.
5. In der Sidebar unter **Hilfen & Verwaltung > Speicherung v28.0** den Speichertest ausfuehren.
6. Zuerst **Legacy-JSON importieren**, danach bei Bedarf **Google-Watchlists importieren**.

Der `service_role_key` darf nur in den serverseitigen Streamlit-Secrets liegen und niemals in GitHub eingecheckt oder in der Oberflaeche angezeigt werden.

## Beispiel fuer Streamlit Secrets

```toml
[storage]
backend = "supabase"
use_for_watchlists = true
mirror_local = true
user_scope = "email_hash"
local_dir = ".app_storage"

[supabase]
url = "https://YOUR_PROJECT.supabase.co"
service_role_key = "YOUR_SUPABASE_SERVICE_ROLE_KEY"
table = "app_state"
timeout_seconds = 10
```

## Verhalten bei Ausfall

Schlaegt ein Supabase-Zugriff fehl, speichert die App weiter in `.app_storage/`. Der Status wird in der Sidebar als degradierter Remote-Speicher angezeigt. Beim naechsten erfolgreichen Remote-Schreibvorgang wird Supabase wieder zur primaeren Ablage.

## Deployment pruefen

```bash
python verify_deployment.py
```

Start:

```bash
streamlit run app.py
```
