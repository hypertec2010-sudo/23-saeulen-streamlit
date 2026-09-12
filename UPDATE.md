# v30.8 - Decision Confidence & Evidence Layer

v30.8 ergaenzt die in v30.7 vereinheitlichte Entscheidungs-Zusammenfassung um eine transparente zweite Ebene: **Wie belastbar ist diese Aussage eigentlich?**. Die neue Schicht veraendert keine Trading-Entscheidung, sondern zeigt nur, auf welcher Datenbasis die vorhandene Entscheidung beruht und wo Grenzen liegen.

## Neu
- Einheitlicher Streifen `Entscheidungs-Konfidenz` mit `Hoch / Mittel / Niedrig / Nicht bewertet`.
- Direkt daneben werden – sofern vorhanden – `Evidenz`, `Aktualitaet` und `Grenzen` angezeigt.
- Fehlende Daten werden nicht still als ausreichend interpretiert; die Konfidenz wird bei kritischen Luecken begrenzt.
- Live-Screener / Ticker-Details: nutzt bereits vorhandene Datenqualitaet, Kontext-Verlaesslichkeit und Benchmarkstatus. Auf Desktop gibt es zusaetzlich einen kompakten Ticker-Auswahlexpander mit derselben Entscheidungs-Zusammenfassung wie mobil.
- Rotation Drilldown: Konfidenz basiert nur auf vorhandenen Kernmetriken (Sektor-RS, RS-Beschleunigung, Trend, Entry-Readiness), Engine-Bestaetigung und darauf, ob der Radar-Snapshot Legacy oder aktuell ist.
- Portfolio: zeigt Kursabdeckung, Stop-Abdeckung und FX-Vollstaendigkeit als Evidenz. Fehlende FX-Umrechnung bleibt sichtbar und verhindert eine zu hohe Konfidenz.
- Positionen: zeigt, ob ein aktueller Atomic-Kursstand vorliegt, welche Exit-Engine-Konfidenz vorhanden ist und ob ein belastbarer Stop-Plan existiert.
- Einzelanalyse: nutzt die bereits vorhandene Analyse-/Fundamental-Coverage, geladene Felder und abgeleitete Felder als transparente Evidenz.

## Wichtige Grenzen
- Die neue Konfidenz ist **kein neuer Trading-Score** und wird nicht in Live-, Shadow-, Harvest-, Chop-, Exit-, Portfolio- oder Rotation-Scores eingerechnet.
- `Hoch` bedeutet nur: die aktuell angezeigte Entscheidung hat eine vollstaendigere/sauberere Datenbasis. Es bedeutet nicht automatisch `Kaufen` oder `Halten`.
- Bei fehlenden Vergleichs- oder Benchmarkdaten wird keine Information erfunden.
- Keine neuen Provider-Abfragen; es werden ausschliesslich bereits geladene Werte verwendet.

## Unveraendert
- Keine Aenderung an TP1/TP2/TP3, Stops, Positionsgroessen, Orders oder Einstiegsgates.
- Keine Aenderung an Harvest-/Chop-Schwellen oder v30.6-Lernlogik.
- Keine Aenderung an Atomic-Scan- oder Snapshot-Semantik.
