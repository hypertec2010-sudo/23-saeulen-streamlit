# v30.5c - Live-Screener Decision Summary Cleanup

v30.5c bereinigt die Explainability-/Detaildarstellung im Live-Screener. Die bisherige Ansicht wiederholte in `Mehr anzeigen · Warum TICKER?` dieselben Score-Treiber und Bremsen mehrfach und zeigte viele technische Rohdetails, ohne die wichtigste Frage klar zu beantworten: Was bedeutet das jetzt fuer diesen Wert?

## Neu
- Der bisherige Expander wird zu `Details · TICKER · Entscheidung & Änderung`.
- Neues handlungsnahes `Kurzfazit` aus bereits vorhandenen Live-Feldern.
- Neue Zeile `Nächste Handlung` mit klarer Lesart fuer Einstieg/Beobachten/Gates und ggf. Harvest-Kontext.
- `Seit letztem Scan` erscheint nur noch bei einer echten Aenderung; `Unverändert` erzeugt keine unnoetige Erklaerungsbox.
- `Treiber` und `Bremse` werden unter `Warum dieser Score?` genau einmal angezeigt.
- Aktive Einstiegsgates werden nur angezeigt, wenn wirklich ein Gate aktiv ist; `Keine harten Einstiegsgates aktiv` wird nicht nochmals wiederholt.
- Engine-, RS-, Volatilitaets- und Marktregime-Rohdetails sind standardmaessig verborgen und koennen mit `Berechnungsdetails anzeigen` gezielt eingeblendet werden.
- Mobile Karten zeigen statt des langen Rohtexts ein kurzes `Kurzfazit`.
- Auch die Ticker-Detailansicht nutzt Kurzfazit + naechste Handlung statt des redundanten Score-Rohtexts.

## Beispiel der neuen Lesart
Statt:
`Treiber: Trigger 82/100; Timing 82/100; Trend 82/100. Bremsen: ...`
und direkt darunter erneut dieselben Treiber/Bremsen,

erscheint zuerst eine Aussage wie:
`Setup technisch aktiv bzw. triggernah; der Trendpfad bleibt primaer. RS-Dynamik schwaecht sich ab.`

Danach folgt die konkrete naechste Handlung und nur einmal die eigentliche Score-Herleitung.

## Unveraendert
- Keine Aenderung an Live-Score, Shadow, Guarded Engine, Gates oder Radar.
- Keine Aenderung an Harvest-/Chop-Berechnung oder Trader-Zielen.
- Keine Aenderung an TP1/TP2/TP3, Exit Engine, Positionen, FX oder WTI-Logik.
- Keine neuen Provider-Abfragen.
