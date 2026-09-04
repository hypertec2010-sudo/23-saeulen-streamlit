# v30.3h - Portfolio Coverage / FX Separation Fix

v30.3h behebt die widerspruechliche Portfolio-Anzeige, bei der oben z. B. `3 Positionen · 3 aktuelle Atomic-Kurse` erkannt wurden, waehrend die Portfolio Engine darunter `Aktuelle Kursabdeckung 0%` und `Stop-Abdeckung 0%` meldete.

## Ursache
Die v29.1 Portfolio Engine fuehrte Teile der Datenabdeckung wertgewichtet in der Depot-Basiswaehrung. Wenn fuer eine Fremdwaehrung noch kein expliziter FX-Kurs vorhanden war, konnte die Basiswaehrungs-Abdeckung deshalb auf 0 fallen, obwohl alle Positionen einen aktuellen Atomic-Kurs und gueltige Stops hatten. Marktdaten-, Stop- und FX-Abdeckung wurden damit in der UI vermischt.

## Neu
- Marktdaten-Abdeckung wird positionsbasiert direkt aus den offenen Positionen + vorhandenem Atomic Complete Scan berechnet.
- Stop-Abdeckung wird unabhaengig von FX positionsbasiert berechnet.
- FX-Abdeckung ist ein eigenes Gate und blockiert nur Basiswaehrungs-Summen.
- Bei 3 von 3 frischen Atomic-Kursen zeigt die Kursabdeckung jetzt 100%, auch wenn USD->EUR noch fehlt.
- Bei vollstaendigem explizitem FX-Pfad werden Investiert, Exposure, Cash und Risiko bis Stop aus exakt derselben Positions-/Kursbasis wie die Diagnosezeilen reconciled.
- Veraltete v29.1-Texte wie `Kursabdeckung 0%` oder `Stop-Abdeckung 0%` werden entfernt, wenn die native Positionsbasis das Gegenteil belegt.
- Solange FX fehlt, wird der numerische Portfolio-Risk-Score nur noch als vorlaeufig angezeigt und nicht als vollstaendig freigegebene Ampel ausgegeben.
- Bei nur einer Positionswaehrung werden auch ohne FX der native Investitionswert und das native Risiko bis Stop direkt oberhalb der Portfolio-Kennzahlen angezeigt.
- Der FX-Expander oeffnet sich automatisch, solange eine benoetigte Umrechnung fehlt. Ein positiver manueller FX-Wert wirkt sofort; Speichern macht ihn reboot-fest.

## Provider-Schutz
Keine neuen Yahoo-/Marktprovider-Requests und keine automatische FX-Schaetzung. Alle Berechnungen verwenden ausschliesslich bereits vorhandene Positions- und Atomic-Daten plus explizit eingegebene FX-Kurse.

## Unveraendert
Live-/Shadow-Ampel, Exit Engine 2.0, Early Profit Protection, Rotation Radar, Positionen und Orders werden nicht automatisch veraendert.
