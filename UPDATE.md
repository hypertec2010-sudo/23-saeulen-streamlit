# v28.6c1 – Gate Transparency Hotfix

Behoben:
- `entry_distance` wurde in einzelnen harten Gate-Pfaden vor seiner Initialisierung verwendet.
- Die Gate-Transparenz liest Entry-Abstand und Risiko jetzt lokal direkt aus Decision/Result.
- Damit brechen Ticker wie MRNA, HOOD, AXON, DHR und ZS nicht mehr ab.

Unverändert:
- Score-Logik
- Live-/Shadow-Ampel
- Guardrails
- Supabase / SQL / Secrets

Nach Upload Streamlit einmal rebooten und den Live-Screener frisch starten.
