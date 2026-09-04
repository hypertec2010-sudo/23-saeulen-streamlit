# v30.3c - Rotation Drilldown Snapshot Sync Fix

v30.3c behebt veraltete bzw. unvollstaendige Werte in **Rotation fuer Aktien-Drilldown**.

## Ursache
Die Drilldown-Selectbox verwendete bisher stabile Ticker-Optionen plus `format_func`. Bei einem neuen Radar-Snapshot konnten Phase/Rotation/Leadership im sichtbaren Selectbox-Label hinter dem aktuellen Radar-Frame zurueckbleiben. Ausserdem durfte ein neuer Radar-Snapshot bereits ab 85% Gesamt-Abdeckung publiziert werden; dadurch konnten einzelne drilldown-faehige Sektor-/Themen-Gruppen fehlen.

## Fix
- Drilldown-Labels werden nun direkt aus dem aktuell sichtbaren Radar-Frame gebaut.
- Die sichtbaren Optionen enthalten selbst Phase, Name, Ticker, Rotation und Leadership; aendert sich der Radar-Kontext, aendert sich damit die Selectbox-Optionsliste und ein altes Label kann nicht weiterverwendet werden.
- Die zuletzt ausgewaehlte Gruppe bleibt ueber einen separaten Ticker-State erhalten.
- Ein Snapshot-Fingerprint erkennt Wechsel von Phase/Rotation/Leadership und synchronisiert die Auswahl sofort neu.
- Alle hinterlegten Drilldown-Gruppen bleiben sichtbar. Fehlt eine Gruppe in einem alten/partiellen Snapshot, erscheint sie transparent als `Daten im aktuellen Radar-Stand fehlen` statt einfach zu verschwinden.
- Neue Radar-Snapshots werden nur noch publiziert, wenn neben der bisherigen Gesamt-Abdeckung auch **alle drilldown-faehigen Sektor-/Themen-Gruppen** vorhanden sind.
- Ein Aktien-Drilldown, der vor einem neuen Radar-Kontext berechnet wurde, wird sichtbar als veraltet markiert; fuer neue Aktienkennzahlen muss er bewusst erneut gestartet werden. Dadurch entstehen keine versteckten Provider-Requests.

## Unveraendert
Keine Aenderung an Leadership-/Rotation-Formeln, Phasenlogik, Breadth-Formel, Live-/Shadow-Ampel, Positions-/Exit-Engine, Early-Profit-Engine oder automatischen Orders.
