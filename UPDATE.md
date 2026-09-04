# v30.3i - Automatic ECB FX Layer

v30.3i automatisiert die Fremdwaehrungs-Umrechnung der Portfolio & Risk Engine. USD/EUR und weitere von der EZB publizierte Referenzwaehrungen muessen im Normalfall nicht mehr manuell eingetragen werden.

## Neu
- Automatischer Abruf der offiziellen ECB-Euro-Referenzkurstabelle ueber einen einzigen kompakten XML-Request.
- 12-Stunden-Cache: kein FX-Request pro Position und kein wiederholter Abruf bei normalen Streamlit-Reruns.
- Automatische Cross-Rate-Berechnung fuer jede von der ECB gelieferte Waehrung gegen die gewaehlte Depot-Basiswaehrung.
- Damit funktionieren u. a. USD, GBP, CHF, SEK, NOK, DKK, JPY, CAD, AUD, PLN und CZK automatisch, sofern die jeweilige Waehrung im aktuellen ECB-Snapshot enthalten ist.
- ECB-Stand/Alter wird im Portfolio sichtbar angezeigt.
- Expliziter Button `ECB-FX jetzt aktualisieren` kann den 12h-Cache gezielt umgehen.

## Last-Good / Ausfallschutz
- Jeder erfolgreiche ECB-Snapshot wird ueber die zentrale Storage-Schicht als `portfolio_fx_ecb_last_good_v303i` gespeichert.
- Ist ECB temporaer nicht erreichbar, wird der letzte gespeicherte Snapshot nur als klar markierter `ECB Last-Good`-Fallback verwendet.
- Stand und Alter des Fallbacks bleiben sichtbar.
- Ein Last-Good-Snapshot aelter als 7 Kalendertage wird nicht mehr automatisch fuer die Portfolio-Aggregation verwendet.
- Fehlt danach weiterhin ein Kurs, bleibt die Basiswaehrungs-Aggregation bewusst unvollstaendig statt einen Kurs zu schaetzen.

## Manuelle Overrides
- Bereits gespeicherte manuelle FX-Werte werden als Overrides erhalten.
- Pro Waehrung kann `Manuell ueberschreiben` aktiviert werden.
- Portfolio-Einstellungen speichern nur die manuellen Overrides; automatische ECB-Kurse werden separat ueber ihren eigenen Last-Good-Speicher verwaltet.
- Ein manueller Wert hat Vorrang vor dem automatischen ECB-Kurs.

## Portfolio-Integration
- Die von v30.3h getrennten Gates bleiben erhalten: Kursabdeckung, Stop-Abdeckung und FX-Abdeckung sind weiterhin unabhaengig.
- Sobald fuer alle offenen Fremdwaehrungspositionen ein automatischer ECB-Kurs oder manueller Override vorliegt, werden Investiert, Exposure, Cash und Risiko bis Stop automatisch in der Depot-Basiswaehrung berechnet.
- Keine Aenderung an Positionen, Stops oder Orders.

## Provider-Schutz
- Keine neuen Yahoo-/Aktienprovider-Abfragen.
- Ein ECB-Abruf liefert alle Waehrungen gemeinsam.
- Standard-TTL 12 Stunden.
- Manueller Refresh nur auf expliziten Klick.
- ECB-Referenzkurse werden als Bewertungs-/Informationskurse behandelt, nicht als garantierte Ausfuehrungskurse.

## Unveraendert
Live-/Shadow-Ampel, Rotation Radar, Exit Engine 2.0, Early Profit Protection, Trading Journal, Positionen und Orders bleiben unveraendert.
