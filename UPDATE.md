# v30.14a - Depot Import Mindest-Transaktionsvolumen

v30.14a ist ein kleiner Zusatz zum Depot-Excel-Import aus v30.14. Er ist speziell für Broker-Exporte wie Trading 212 gedacht, in denen neben den bewusst im Tool geführten Positionen auch viele kleine Pie-/Fractional-Transaktionen enthalten sein können.

## Neu
- Optionaler Schalter `Nur Kauf-/Verkaufstransaktionen ab Mindestvolumen importieren`.
- Standardwert des Mindestvolumens: **500 EUR**; der Wert ist im Importdialog anpassbar.
- Standardmäßig ist der Filter **aus**, sodass sich das bisherige Importverhalten nicht verändert.
- Der Filter wirkt pro einzelner BUY-/SELL-Transaktion bereits **vor**:
  - Import-Vorschau,
  - v30.14 Bestands-Abgleich,
  - Rebuild-/Oversell-Prüfung,
  - Positionsbuchung,
  - Trade-Journal.
- Ausgeschlossene Kleintransaktionen werden separat in einem aufklappbaren Bereich angezeigt und nicht als importiert markiert.
- Die Import-Vorschau zeigt zusätzlich `Transaktionsvolumen EUR`.

## EUR-Ermittlung
Das Transaktionsvolumen wird providerfrei aus der Brokerdatei bestimmt. Priorität:
1. absoluter `Net Total`, wenn dessen Währung EUR ist,
2. absoluter `Gross Total`, wenn dessen Währung EUR ist,
3. `Stück × Preis/Aktie`, wenn die Preiswährung EUR ist.

Es wird bewusst keine FX-Richtung geraten und keine externe Kursabfrage durchgeführt. Kann das EUR-Volumen einer Kauf-/Verkaufszeile nicht eindeutig bestimmt werden, bleibt die Zeile aus Sicherheitsgründen im Import enthalten und die UI weist darauf hin.

## Beispiel Trading 212 Pie
Bei aktivem Filter mit 500 EUR:
- Kauf über 42 EUR -> wird aus Positionsimport und Journal ausgeschlossen.
- Kauf über 499,99 EUR -> wird ausgeschlossen.
- Kauf über 600 EUR -> wird normal verarbeitet.
- Eine Dividende/Zins-/sonstige Archiv-Action bleibt vom Mindestvolumenfilter unberührt, da sie ohnehin keine offene Aktienposition verändert.

## Unverändert
- v30.14 Reconciliation Guard und dessen Bestätigungs-/Blockierlogik.
- Weighted Average Entry, Fractional Shares, Teil-/Vollverkäufe und Dublettenschutz.
- Storage-Namespace und bestehendes Broker-ID-/Hash-Ledger.
- Stops, Targets, Trading-Scores, Gates, Harvest/Chop und Orders.
- Keine neuen Provider-Abfragen.
