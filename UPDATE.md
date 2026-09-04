# v30.3e - Rotation Drilldown Universe Alignment Fix

v30.3e behebt die konkrete Abweichung, dass die Investment-Rotation-Uebersicht z. B. **4 Emerging** meldet, im Feld **Rotation fuer Aktien-Drilldown** aber nur zwei dieser Emerging-Gruppen erscheinen.

## Tatsaechliche Ursache
Das war kein weiterer Cache-/Persistenzfehler. Die beiden Bereiche nutzten unterschiedliche Universen:
- Die Radar-Uebersicht zaehlt alle Ebenen: Investmentklassen, Regionen, Sektoren und Branchen/Themen.
- Der v30.1d Aktien-Drilldown enthielt urspruenglich nur Sektor-/Themen-Gruppen mit fest hinterlegtem Aktienkorb.

Damit konnte der Emerging-Zaehler korrekt 4 anzeigen, obwohl nur 2 dieser vier Gruppen im Drilldown-Universum lagen.

## Jetzt ein gemeinsames Universum
Die Drilldown-Auswahl wird ab v30.3e aus exakt derselben aktuellen Radar-Population wie die Hauptuebersicht gebaut. Wenn oben vier Gruppen Emerging sind, stehen auch vier Emerging-Gruppen in der Auswahl.

Jede Option zeigt jetzt zusaetzlich ihre **Radar-Ebene**, Phase, Rotation und Leadership. Damit ist sofort sichtbar, ob es sich um eine Region, einen Sektor, ein Thema oder eine andere Investmentklasse handelt.

## Neue repraesentative Aktienkoerbe
Fuer bisher nicht abgedeckte, aber sinnvoll in Aktien zerlegbare Gruppen wurden provider-schonende kompakte Koerbe ergaenzt:
- SPY / breiter US-Markt
- QQQ / Nasdaq
- VGK / Europa
- EWG / Deutschland
- EEM / Emerging Markets

Fuer GLD, DBC, USO und CPER werden Aktienkoerbe ausdruecklich als **Aktien-Proxy** markiert. Hier sind die Einzelaktien nicht der ETF selbst, sondern repraesentative Unternehmen mit direktem Exposure zum jeweiligen Rohstoff-/Assetthema.

## Bond-/Credit-Gruppen
TLT und HYG bleiben jetzt ebenfalls in der Auswahl sichtbar. Sie erhalten bewusst **keinen direkten Aktienkorb**, weil ein Aktienranking gegen eine Treasury-/High-Yield-Rotation fachlich eine andere Aussage waere. Die Radar-Phase und Rotation werden dennoch vollstaendig angezeigt.

## Transparenz
Direkt ueber der Auswahl steht nun ein Kontrollsatz nach dem Muster:

`Emerging oben: 4 · Emerging hier: 4 · davon 3 mit Aktienkorb`

Damit ist die Population sofort pruefbar und eine spaetere Differenz nicht mehr unsichtbar.

## Provider-Schutz
Der normale Rotation-Hauptscan wird nicht um Einzelaktien erweitert. Aktien des Drilldowns werden weiterhin nur auf expliziten Klick fuer genau eine ausgewaehlte Gruppe geladen.

## Unveraendert
Keine Aenderung an Rotation-Score-/Phase-Formel, Live-/Shadow-Ampel, Positions-/Exit-Engine, Early-Profit-Learning, Portfolio Engine, SQL oder Secrets.
