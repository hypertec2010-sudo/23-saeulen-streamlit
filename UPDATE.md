# v28.6a – Shadow Validation Dashboard

Neu:
- Dashboard fuer Live-Ampel vs. Shadow-Ampel.
- Anzahl Shadow-Episoden, offene Abweichungen, Auf-/Abwertungen.
- Kursentwicklung seit Shadow-Ereignis, soweit Kursdaten vorhanden sind.
- Episodendauer und aktueller/abgeschlossener Status.
- Neue Shadow-Ereignisse speichern den Kurs zum Ereigniszeitpunkt.
- Bestehende alte Shadow-Historie bleibt lesbar; alte Eintraege koennen bei Kursauswertung `n/a` zeigen.

Unveraendert:
- produktive Live-Ampel
- Basis-Score
- Engine-/Guardrail-Regeln
- Supabase-Schema / SQL / Secrets

Nach Update:
1. Dateien in GitHub ueberschreiben.
2. Streamlit rebooten.
3. Frischen Live-Scan starten.
4. Im Screener `Shadow Validation Dashboard` oeffnen.
