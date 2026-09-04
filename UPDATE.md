# v30.3d - Rotation Radar Hard-Fresh Snapshot Fix

v30.3d behebt den Fall, dass Investment Rotation Radar und Aktien-Drilldown nach Refresh oder Reboot weiterhin alte Phasen-/Rotationswerte anzeigen.

## Ursache
Der bisherige manuelle Radar-Refresh verwendete einen nur in der Session hochgezaehlten Cache-Token. Nach einem Session-/Browser-Neustart konnte derselbe `st.cache_data`-Key erneut entstehen und innerhalb der Cache-TTL einen aelteren Kursframe liefern. Zusaetzlich pruefte die Provider-Schicht bisher vor allem die Historienlaenge; ein Ticker mit vielen Bars, aber einem veralteten letzten Handelstag konnte als vorhanden gelten. Der persistierte Radar-Snapshot wurde ausserdem nicht unmittelbar nach dem Schreiben rueckgelesen und verifiziert.

## Hard-Fresh Refresh
- Jeder explizite Radar-Refresh nutzt jetzt einen global eindeutigen `time_ns`-Nonce statt eines wiederverwendbaren Session-Zaehlers.
- Dasselbe gilt fuer Breadth- und Stock-Drilldown-Abrufe auf Klick.
- Der 30-Minuten-Cache bleibt provider-schonend bestehen, kann einen manuellen Refresh aber nicht mehr mit einem alten Key bedienen.

## Daily-Freshness je Drilldown-Gruppe
- Fuer alle Aktien-Drilldown-Sektor-/Themen-ETFs wird der letzte vorhandene Daily-Handelstag geprueft.
- Eine Serie mit ausreichender Historie, aber veraltetem letzten Handelstag, gilt jetzt als stale und wird wie eine Datenluecke behandelt.
- Bei wenigen stale/fehlenden Tickers darf weiterhin der bestehende provider-schonende Einzel-Fallback greifen.
- Ein neuer Radar-Snapshot wird nur publiziert, wenn alle Pflicht-Drilldown-Gruppen vollstaendig UND auf demselben belastbaren Handelstag sind.

## Neuer verifizierter Persistenz-Namespace
- Neuer Namespace: `rotation_radar_snapshot_v303d`.
- Schema: `rotation-v30.3d-hard-fresh`.
- Ein neuer Radar-Stand gilt erst als gespeichert, wenn Snapshot-ID und Frame-Fingerprint unmittelbar aus dem zentralen Storage wieder identisch gelesen wurden.
- Nach erfolgreichem Refresh rendert die UI aus genau diesem rueckgelesenen persistenten Frame, nicht aus einem separaten In-Memory-Zwischenstand.
- Alte v30.1/v30.3c-Snapshots werden nur als klar markierter Legacy-Fallback angezeigt, bis einmal ein erfolgreicher v30.3d-Hard-Refresh erfolgt ist.

## Transparenz
Die Radar-Ansicht zeigt fuer den verifizierten Snapshot explizit:
- Daten bis (letzter gemeinsamer Handelstag),
- Snapshot-ID,
- Persistenzquelle.

Damit ist nach einem Reboot eindeutig pruefbar, ob wirklich derselbe frisch gespeicherte Radar-Stand geladen wurde.

## Unveraendert
Keine Aenderung an Live-/Shadow-Ampel, Rotation-Score-Formel, Phase-Logik, Positions-/Exit-Engine, Early-Profit-Learning, Portfolio Engine, SQL oder Secrets.
