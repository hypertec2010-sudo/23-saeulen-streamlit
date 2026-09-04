# v30.1d - Rotation Stock Drilldown

v30.1d schliesst die Luecke zwischen Branchenrotation und konkreter Aktienauswahl.

## Neu im Rotation Radar
Unter **Top-Kandidaten aus dieser Rotation** kann eine Sektor-/Branchenrotation ausgewaehlt und per Klick in einen kompakten repraesentativen Aktienkorb aufgebohrt werden.

Der Drilldown bewertet jede Aktie relativ zum ausgewaehlten Sektor-/Branchen-ETF mit:
- Sektor-RS 5T / 21T / 63T,
- RS-Beschleunigung,
- Trend-Score,
- Entry-Readiness,
- Abstand zu MA20 und 20T-Hoch,
- Kandidaten-Score 0-100.

## Kandidatenstatus
Die Anzeige unterscheidet u. a.:
- Early Leader,
- Confirmed Leader,
- Rotation beschleunigt,
- technisch bereit,
- Leader aber ueberdehnt.

## Verbindung zur bestehenden Engine
Wenn ein Kandidat bereits im letzten vollstaendigen Atomic-Screener-Stand enthalten ist, werden Live-/Shadow-Ampel, Live-Score, Guarded Engine-Score, CRV, RS-Dynamik, Setup-Alert und aktive Einstiegsgates lesend ergaenzt.

Fehlt ein Kandidat dort, wird das klar als **nicht im Atomic Live-Scan** angezeigt. Das ist keine negative Bewertung.

## Provider-Schutz
Der Stock Drilldown laeuft nur auf expliziten Klick und nur fuer eine ausgewaehlte Gruppe. Der normale Rotation-Radar-Hauptscan wird dadurch nicht vergroessert. Ein unvollstaendiger Drilldown ersetzt keinen vorhandenen letzten Drilldown-Stand der Sitzung.

## Unveraendert
Keine Aenderung an produktiver Live-Ampel, Shadow-Ampel, Atomic Complete Scan, v30.0 Cutover-Gates, Portfolio Engine, SQL oder Secrets.
