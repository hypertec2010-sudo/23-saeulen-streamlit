# v28.4 – AppTests und GitHub CI

Diese Version baut auf dem stabilen Stand v28.3.2 auf. Die fachliche Oberfläche,
Supabase-Struktur und gespeicherten Daten bleiben kompatibel.

## Neu

- GitHub-Actions-Workflow unter `.github/workflows/quality.yml`
- Pytest-Suite für Navigation, Storage-Fallback und Trade-Journal
- Streamlit `AppTest` für den echten Login-Einstiegspunkt
- Streamlit `AppTest` für Cockpit-Wechsel und normale Reruns
- testbare Live-Refresh-Policy in `modules/live_refresh_policy.py`
- deterministische Tests für Cache-Fälligkeit und Trigger-Deduplizierung
- Secret-Guard gegen versehentlich eingecheckte produktive Schlüssel
- separate CI-Abhängigkeiten in `requirements-ci.txt`

## Live-Refresh

Die Berechnung der Auto-Refresh-Fälligkeit wurde ohne Funktionsänderung aus dem
UI-Fragment in eine reine Policy-Schicht ausgelagert. Das Streamlit-Fragment
zeigt weiterhin den Status an und löst bei Fälligkeit den App-Rerun aus.

## Deployment

Keine SQL-Migration und keine Änderung der Streamlit-Secrets erforderlich.
Den vollständigen Paketinhalt ins Repository übernehmen. Nach einem Push auf
`main` oder `master` startet der Workflow automatisch.

Lokal:

```bash
python -m pip install -r requirements-ci.txt
python verify_deployment.py
pytest -q
```
