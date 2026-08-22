# v28.5d – Engine Guardrails

Neu:
- Guarded Engine-Score parallel zum bisherigen Roh-Engine-Score.
- Positive Kontextboni koennen fehlende technische Trigger nicht ersetzen.
- Hartes Einstiegsgate/Invalidierung blockiert eine positive Aufwertung.
- Schwache Chart-/CRV-Komponenten begrenzen positive Kontext-Aufwertungen.
- Neue Engine-Empfehlung: Aufwertung möglich, Aufwertung begrenzt, Keine Aufwertung · Guardrail, Abwertung, Abwertung / blockiert oder Bestätigt / keine Änderung.
- Die echte Ampel und der Basis-Live-Score bleiben weiterhin unverändert.

Test:
1. Dateien in GitHub überschreiben.
2. Streamlit rebooten.
3. Live-Screener frisch starten.
4. UNP, QRVO, DT, MSFT, LITE und SAP.DE prüfen.
5. Besonders auf Guarded Engine-Score und Engine-Empfehlung achten.

Keine Änderungen an Supabase, SQL oder Secrets.
