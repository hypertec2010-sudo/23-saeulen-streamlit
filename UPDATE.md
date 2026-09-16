# v30.13 - Depot Excel Transaction Import

v30.13 ergänzt den Positions-/Exit-Bereich um einen providerfreien Broker-/Depot-Transaktionsimport. Ziel ist, Käufe, Verkäufe und andere Depotbewegungen aus einem bestehenden Excel-/CSV-Export nachzuziehen, wenn die manuelle Pflege im Tool nicht vollständig geschafft wurde.

## Unterstütztes Importformat
Die Importlogik erkennt die vom Nutzer genannte Spaltenstruktur, insbesondere:
- Action
- Time (UTC)
- ISIN
- Ticker
- Name
- Notes
- ID
- No. of shares
- Price / share
- Currency (Price / share)
- Exchange rate
- Result / Currency (Result)
- Gross Total / Currency (Gross Total)
- Withholding tax / Currency (Withholding tax)
- Currency conversion fee / zugehörige Währung
- Taxes / Currency (Taxes)
- Net Total / Currency (Net Total)

Pflichtfelder für eine Positionsbuchung sind `Action`, `Time (UTC)`, `Ticker`, `No. of shares` und `Price / share`.

## Neu
- Neuer Expander `Depot-Excel importieren` im Bereich `Positionen / Exit`.
- Unterstützt `.xlsx`, `.xlsm`, `.csv` und `.txt`.
- UTC-Zeitstempel werden automatisch nach `Europe/Berlin` konvertiert.
- Erkennt Buy-/Sell-Varianten wie `Market buy`, `Limit buy`, `Market sell`, `Limit sell` sowie einfache deutsche/englische Kauf-/Verkaufsbegriffe.
- Negative Broker-Stückzahlen bei eindeutigem Buy/Sell werden als Betragsstückzahl normalisiert.
- Käufe erhöhen die Position und bilden einen gewichteten Durchschnitts-Entry (`Weighted Average Entry`).
- Verkäufe reduzieren offene Stücke; vollständige Verkäufe schließen die Position.
- Teilverkäufe und vollständige Schließungen werden zusätzlich in das bestehende Trade-Journal geschrieben.
- Broker-Result, Result-Währung, Preis-Währung, Notizen und Import-ID bleiben in den importierten Journal-/Archivdaten erhalten.
- Dividenden, Zinsen, Ein-/Auszahlungen und sonstige Actions werden archiviert, verändern aber bewusst keine Aktienposition.
- Neue offene Broker-Positionen werden in die aktuelle Positions-Watchlist synchronisiert, ohne dafür Marktdaten abzurufen.

## Zwei Importmodi
### Nur neue Transaktionen anwenden
Für regelmäßige Folgeexporte. Bereits importierte Broker-IDs werden nicht nochmals gebucht. Fehlt eine Broker-ID, wird ein deterministischer Transaktions-Hash verwendet.

### Enthaltene Ticker aus Datei neu aufbauen
Für eine vollständige Historie. Alle Buy-/Sell-Zeilen der Datei werden für die enthaltenen Ticker chronologisch ab Null rekonstruiert. Bereits manuell gepflegte Management-Metadaten wie Stop, Ziel, Portfolio-Gruppe und vorhandener Entry-Kontext werden bei weiterhin offenen Positionen nach Möglichkeit erhalten.

## Sicherheitslogik
- Vorschau vor jeder Buchung.
- Explizite Bestätigung erforderlich.
- Verkauf größer als bekannte offene Stückzahl blockiert den gesamten Import und wird als Anomalie angezeigt.
- Doppelte Zeilen innerhalb derselben Datei werden nicht gebucht.
- Wiederholter Import derselben Datei führt im Modus `Nur neue` nicht zu Doppelbuchungen.
- Neue Stop-/Target-Werte werden nicht aus der Brokerdatei erfunden.
- Broker-importierte Schließungen werden nicht im manuellen `Schließung rückgängig machen` angeboten; Korrekturen erfolgen über einen korrigierten Export + Rebuild.

## Fractional Shares
- Der Import verarbeitet Bruchstücke wie `0.25` exakt.
- Die sichtbare Positions-Tabelle zeigt importierte Fractional Shares ohne Integer-Rundung.
- Einige ältere manuelle Teilverkaufs-/Schließungsdialoge arbeiten weiterhin mit ganzen Stückzahlen. Bei Fractional-Positionen werden manuelle Verkäufe deshalb blockiert und sollen erneut über den Depot-Import eingespielt werden.
- Stop-Anpassungen und Notizen bleiben auch bei Fractional Shares manuell nutzbar.

## Unverändert
- Keine Orderausführung.
- Keine automatische Kauf-/Verkaufsentscheidung.
- Keine Änderung an Live-/Shadow-/Guarded-, Exit-, Harvest-, Chop-, Queue-, Confidence- oder Calibration-Logik.
- Keine neuen Yahoo-/Web-/Provider-Abfragen.
