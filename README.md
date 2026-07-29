# v28.4.1 R-Multiple Initial-Risk Fix

Diese Version baut auf **v28.3.2** auf. Multipage-Navigation, Supabase,
Repositories, Trade-Journal und Live-Screener bleiben kompatibel.

## Qualitätssicherung

Bei jedem Push auf `main` oder `master` sowie bei Pull Requests führt GitHub
automatisch aus:

```bash
python verify_deployment.py
pytest -q
```

Geprüft werden unter anderem:

- Syntax und Deployment-Struktur
- Cockpit-Navigation über normale Streamlit-Reruns
- Live-Refresh-Fälligkeit aus dem letzten erfolgreichen Scan
- Schutz vor doppelten Refresh-Triggern
- Supabase-Ausfall mit lokalem Spiegel
- Teilverkauf und vollständiger Trade-Abschluss
- echter unauthentifizierter App-Einstieg über Streamlit AppTest
- mögliche versehentlich eingecheckte produktive Secrets

## Wichtige Dateien

```text
.github/workflows/quality.yml
requirements-ci.txt
pytest.ini
tests/
modules/live_refresh_policy.py
```

## Upgrade

1. Vollständigen Inhalt des ZIP-Pakets ins Repository übernehmen.
2. Vorhandene Streamlit-Secrets unverändert lassen.
3. Keine SQL-Migration ausführen.
4. GitHub-Commit pushen und den Workflow **v28.4.1 Quality Gate** prüfen.
5. App neu starten und Live-Screener, Exit und Trade-Journal kurz testen.

## Lokale Prüfung

```bash
python -m pip install -r requirements-ci.txt
python verify_deployment.py
pytest -q
streamlit run app.py
```

Details: `RELEASE_NOTES_v28_4.md` und `RELEASE_NOTES_v28_4_1.md`.
