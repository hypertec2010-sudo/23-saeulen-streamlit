# v28.4.3 – Live Status Change Transparency

## Problem

Die Ampel konnte zwischen zwei Scans von Gelb auf Rot wechseln, obwohl der in der
Tabelle gerundete Kurs gleich aussah. Die bisherige Anzeige zeigte zwar den neuen
Status und den allgemeinen Grund, aber nicht, welcher Eingangswert sich gegenueber
dem vorherigen Scan geaendert hatte.

## Loesung

Der Live-Monitor speichert je Ticker zusaetzlich die entscheidenden Diagnosewerte:

- Timing
- Konfluenz
- Chartbewertung
- Trigger-Komponente
- Trend-Komponente
- CRV-Komponente
- Radar-Bucket
- finale Freigabe
- hartes Einstiegsgate
- Invalidierung
- Entry-/Wave-/Nahe-am-Trigger-Zustand

Bei einem Statuswechsel werden diese Werte mit dem vorherigen Scan verglichen. Die
App zeigt daraus eine kurze Erklaerung in **Warum geändert?**. Eine nahezu fehlende
Kursbewegung wird ausdruecklich genannt, damit klar ist, dass der Wechsel aus der
Indikator-/Gate-Logik stammt.

## Kompatibilitaet

- keine neue Supabase-Tabelle
- keine SQL-Migration
- keine Aenderung der Secrets
- alte Statushistorien bleiben lesbar
- aussagekraeftige Komponentenvergleiche stehen ab dem zweiten Scan mit v28.4.3 bereit
