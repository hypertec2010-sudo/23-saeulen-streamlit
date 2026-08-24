# v28.6e – Regional Benchmark Engine

## Neu
- Deutsche Werte (`.DE` usw.) nutzen DAX (`^GDAXI`) als Primaerbenchmark.
- Frankreich (`.PA`) nutzt CAC 40 (`^FCHI`).
- Schweden (`.ST`) nutzt OMX Stockholm 30 (`^OMX`).
- Weitere Zuordnungen: AEX, BEL 20, FTSE MIB, IBEX 35, SMI, FTSE 100, OMX Copenhagen, Oslo, Helsinki, ATX und PSI.
- Wenn ein europaeischer Primaerbenchmark keine belastbaren Daten liefert, versucht der Live-Screener `^STOXX50E` als Europa-Fallback.
- 21T- und 63T-RS werden gegen denselben tatsaechlich verwendeten Benchmark berechnet.
- Der verwendete Benchmark wird in Desktop, Mobile und Ticker-Detailansicht sichtbar.
- Bei New Listings ohne 63 Handelstage bleibt der Missing-Data-Guard aktiv; es werden keine historischen RS-Werte erfunden.

## Unveraendert
- Live-Ampel
- Shadow-Ampel-Regeln
- Basis-Score
- Guardrails
- Supabase/SQL/Secrets

## Test nach Upload
1. Dateien in GitHub ueberschreiben.
2. Streamlit komplett rebooten.
3. Einen frischen Screener-Scan starten.
4. SAP.DE, IFX.DE, ADS.DE: Benchmark DAX pruefen.
5. IPS.PA, SU.PA, EL.PA: Benchmark CAC 40 pruefen.
6. ADDT-B.ST: OMX Stockholm 30 pruefen.
7. Kontrollieren, dass 21T/63T-RS und RS-Dynamik nicht mehr `n/a` sind, sofern genug Aktienhistorie vorhanden ist.
