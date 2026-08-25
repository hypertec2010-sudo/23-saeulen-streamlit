# v28.6e4 – Persistent Last Visible Stand

Behoben:
- Bei pausiertem Auto-Scan war trotz Hinweis gelegentlich kein gespeicherter Screener-Stand sichtbar.
- Neben dem strikten Scan-Cache wird jetzt ein stabiler `Last Visible Stand` gespeichert.
- Der stabile Key normalisiert die Ticker-Reihenfolge und ist nicht an kleine UI-/Patch-Schemawechsel gebunden.
- Bei Reconnect/Desktop-Rerun wird zuerst der exakte Snapshot, danach der stabile sichtbare Stand versucht.
- Die Pause-Meldung sagt nur noch, dass ein Stand sichtbar ist, wenn wirklich ein Cache/Snapshot geladen wurde.
- Falls noch nie ein kompatibler Stand gespeichert wurde, fordert die UI einmalig zu `Jetzt prüfen` auf.

Keine Änderungen an Scores, Live-/Shadow-Ampel, Benchmarks, Guardrails oder Trading-Logik.

Nach Upload:
1. Streamlit rebooten.
2. Einmal `Jetzt prüfen` vollständig durchlaufen lassen.
3. Auto-Scan pausieren bzw. Browser/Display schlafen lassen.
4. Zurückkehren: der letzte Live-Stand muss ohne neuen Scan sichtbar sein.
