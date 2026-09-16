# v30.14 - Depot Import Reconciliation Guard

v30.14 erweitert den Depot-Excel-Import aus v30.13 um einen expliziten Bestands-Abgleich für den typischen Mischfall: Im Tool existieren bereits manuell gepflegte Positionen, während anschließend nur ein begrenzter Broker-Zeitraum, z. B. die letzten zwei Monate, importiert werden soll.

## Neu
- Neuer Bereich `Bestands-Abgleich · manuelle Positionen vs. Datei` direkt in der Import-Vorschau.
- Prüfung pro Ticker auf:
  - aktuell offene Tool-Stückzahl,
  - Tool-Entry,
  - Herkunft der Position (`Depot-Excel` oder manuell/nicht markiert),
  - gespeicherten Positionsbeginn,
  - ersten und letzten Datei-Zeitpunkt,
  - erste Broker-Aktion,
  - Datei-Käufe, Datei-Verkäufe und Netto-Stückzahl,
  - Replay-Fähigkeit der Datei `ab Null`.
- Manuell gepflegte offene Position + noch nicht importierte Brokerzeilen wird nicht mehr still angewendet. Der Ticker erhält `ABGLEICH` und benötigt eine zweite explizite Bestätigung.
- Im Modus `Nur neue Transaktionen anwenden` muss bestätigt werden, dass der aktuelle Tool-Bestand dem Stand unmittelbar **vor** der ersten noch nicht importierten Broker-Transaktion entspricht.
- Im Modus `Enthaltene Ticker aus Datei neu aufbauen` muss bei markierten manuellen Positionen bestätigt werden, dass die Datei den vollständigen Kauf-/Verkaufszyklus enthält.
- Rebuild-Historien werden pro Ticker chronologisch ab Stückzahl Null geprüft. Ein Verkauf, für den innerhalb der Datei zuvor nicht genügend Stück aufgebaut wurden, erzeugt `BLOCKIERT` und kann nicht über eine Checkbox übergangen werden.
- Bereits mit `broker_source = Depot-Excel` geführte Positionen werden erkannt; der bestehende Broker-ID/Hash-Dublettenschutz bleibt maßgeblich.
- Bereits verarbeitete Broker-IDs werden im inkrementellen Modus aus dem neuen Abgleich herausgenommen und erzeugen keine unnötige Warnung.

## Beispiel: nur die letzten zwei Monate importieren
Wenn im Tool am Beginn des Importzeitraums bereits 10 Stück manuell korrekt hinterlegt waren und die Datei danach einen Kauf von 5 Stück enthält, kann `Nur neue Transaktionen anwenden` nach expliziter Abgleich-Bestätigung daraus 15 Stück mit gewichteter Entry-Berechnung machen.

Wenn die manuell hinterlegten 15 Stück den Kauf aus der Datei jedoch bereits enthalten, warnt v30.14 vor dem möglichen Doppelzählen. Die App kann diese semantische Überschneidung nicht automatisch über Broker-ID erkennen, weil die manuelle Position keine Broker-Transaktions-ID besitzt.

## Kompatibilität
- Der vorhandene Storage-Namespace `depot_transaction_import_v3013` bleibt absichtlich erhalten. Dadurch bleiben bereits gespeicherte Broker-IDs, Hashes und Importarchive nach dem Update gültig.
- Weighted Average Entry, Fractional Shares, Teil-/Vollverkäufe, Trade-Journal und UTC->Berlin-Konvertierung aus v30.13 bleiben unverändert.
- Manuelle Stops, Targets, Gruppen und Kontexte bleiben weiterhin erhalten.

## Unverändert
- Keine Provider-Abfrage durch den Depot-Import oder den neuen Guard.
- Keine Orderausführung.
- Keine automatische Erfindung fehlender Anfangsbestände.
- Kein automatisches Überschreiben einer Warnung ohne ausdrückliche Benutzerbestätigung.
