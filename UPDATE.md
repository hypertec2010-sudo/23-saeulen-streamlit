# v28.5a – Trading Engine 2.0 / Beobachtungsmodus

Neu:
- Basis-Live-Score bleibt unverändert und steuert weiterhin die bestehende Ampel.
- Zusätzlicher Context-Adjusted Engine-Score für den Vergleich.
- Kontext-Anpassung wird transparent aufgeteilt in Relative Stärke, Marktregime und Volatilitätsregime.
- Desktop zeigt Basis-Score, Kontext-Anpassung und Engine-Score nebeneinander.
- Mobile zeigt `Basis → Engine` und die vollständige Erklärung im vorhandenen „Mehr anzeigen“-Bereich.
- Detailansicht zeigt Basis, Kontext und Engine direkt unter dem Trading Context.

Gewichtung im Beobachtungsmodus:
- Relative Stärke: -4 bis +4 Punkte.
- Marktregime: -3 bis +3 Punkte.
- Volatilitätsregime: 0 bis -2 Punkte; niedrig -1, hoch -2.
- Maximaler theoretischer Kontextbereich aktuell: -9 bis +7 Punkte.

Wichtig:
- Ampel/Status werden NICHT vom Engine-Score verändert.
- Keine Änderung an Supabase, SQL, Secrets, Positionen oder Journal.
- Nach Upload Streamlit einmal rebooten und einen frischen Live-Scan starten.
