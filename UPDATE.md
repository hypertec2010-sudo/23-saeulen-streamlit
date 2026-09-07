# v30.5 - WTI Commodity Context Layer

v30.5 erweitert den bestehenden Commodity-/Rohstoffmodus gezielt fuer WTI Oel (`CL=F`). Die klassische Aktien-/TP-/Live-/Shadow-/Exit-Logik bleibt unveraendert; hinzu kommt eine transparente, rohstoffspezifische Zusatzsicht.

## Neu fuer WTI
- Eigener Block `WTI Oel · Commodity-Kontext` in der Einzelanalyse.
- WTI-Trend ueber 5, 21 und 63 Handelstage.
- Eigenes Trendbild: breit aufwaerts/abwaerts, kurzfristiger Ruecksetzer/Erholung oder gemischt.
- Oel-Volatilitaet aus ATR-% mit Regime niedrig / normal / erhoeht / hoch.
- Bestehender Kurzfrist-Trader-/Harvest-Pfad wird im WTI-Kontext direkt mit Trader-Ziel, Harvest, Chop und Horizont angezeigt.

## Optionaler externer Oel-Kontext
- Brent (`BZ=F`) fuer den Brent-WTI-Spread.
- XLE als liquider US-Energieaktien-Proxy fuer Relative-Performance-Vergleiche.
- DXY (`DX-Y.NYB`) fuer Dollar-Richtung und WTI-DXY-Korrelation.
- Dollar-Effekt wird nur als Kontextsignal interpretiert, nicht als Kausalitaet oder eigenstaendiges Handelssignal.

## Provider-Schutz
- Der normale Atomic-/Watchlist-Scan erzeugt keinerlei neue Requests.
- Brent/XLE/DXY werden erst nach einem expliziten Klick im WTI-Expander geladen.
- Ein Klick startet genau einen Batch-Request fuer alle drei Vergleichsreihen.
- Erfolgreiche Daten werden ueber Streamlit 6 Stunden gecacht; der Aktualisieren-Button kann bewusst einen neuen Batch erzwingen.
- Fehlende Vergleichsreihen werden sichtbar ausgewiesen statt still geschaetzt.

## Transparenz-Hinweise
- Brent und WTI basieren auf Yahoo-Front-Month-Futures; Futures-Rollwechsel koennen den Spread beeinflussen.
- XLE ist ein Energieaktien-Proxy und kein Ersatz fuer physisches Rohöl oder den WTI-Future.
- Der Dollar-Effekt nutzt DXY-Richtung und die juengste WTI-DXY-Korrelation. Ein statistischer Zusammenhang kann sich veraendern.

## Fachbegriffe-Legende
Neu hinzugekommen sind u. a. WTI, Brent, Brent-WTI Spread, Front-Month Future, XLE, DXY / US-Dollar-Index, Dollar-Effekt und Barrel.

## Unveraendert
- Keine Aenderung an TP1/TP2/TP3.
- Keine Aenderung an Live-/Shadow-Ampel, Guardrails oder Exit Engine.
- Keine Aenderung an v30.4b Harvest-/Chop-Kalibrierung.
- Keine automatischen Orders, Stops oder Teilverkaeufe.
