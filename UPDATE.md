# v28.5a2 - Context Confidence + Missing-Data Guard

Neu:
- Kontext-Verlässlichkeit mit Sterneanzeige.
- Fehlende Relative-Stärke-/Benchmarkdaten werden neutral behandelt.
- Ein positiver Context-Adjustment wird bei unvollständigem Kontext neutralisiert.
- Negative Risikoanpassungen bleiben auch bei fehlenden Daten möglich.
- Basis-Score und echte Ampel bleiben unverändert (Beobachtungsmodus).

Erwarteter Test:
- UNP / DT / MSFT / ACN / LITE: vollständiger Kontext, normale Engine-Anpassung.
- IPS.PA / SAP.DE bei fehlender Benchmark-RS: kein positiver Kontextbonus; Engine-Score entspricht mindestens in diesen positiven Markt-Fällen dem Basis-Score.

Nach Upload Streamlit rebooten und einen frischen Live-Scan starten.
Keine SQL-/Supabase-/Secrets-Aenderung.
