# v28.6 – Shadow Mode

Geändert:
- `modules/live_monitor.py`
- `legacy_app.py`
- `VERSION.txt`
- `UPDATE.md`
- `CHANGELOG.md`

## Neu
- Virtuelle `Shadow-Ampel` aus dem **Guarded Engine-Score**.
- Schwellen im Shadow-Modell: Grün ab 72, Gelb ab 55, Weiß ab 28, Rot darunter.
- `Shadow-Abweichung`: Aufwertung / Abwertung / Gleich gegenüber der produktiven Live-Ampel.
- Persistente Shadow-Historie im Storage-Namespace `shadow_mode_history`.
- Historie ist dedupliziert: identische Auto-Refreshes erzeugen keine neuen Events.
- Erste echte Live/Shadow-Abweichung wird gespeichert; spätere Zustandsänderungen bzw. Rückkehr zu Gleich ebenfalls.
- Desktop, Mobile und Ticker-Details zeigen Live- und Shadow-Ampel parallel.

## Wichtig
Die produktive Live-Ampel, der Basis-Score und bestehende Trading-Regeln werden **nicht verändert**.

## Nach Upload
1. Alle Dateien über die vorhandenen Dateien in GitHub kopieren.
2. Streamlit einmal komplett rebooten.
3. Live-Screener frisch scannen.
4. Spalten `Shadow-Ampel` und `Shadow-Abweichung` prüfen.
5. `Shadow-Mode Historie` öffnen und kontrollieren, ob Abweichungen protokolliert werden.
6. Einen Auto-Refresh abwarten: identische Zustände dürfen keine Duplikate erzeugen.

Keine SQL-/Secrets-Änderung erforderlich.
