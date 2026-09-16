# v30.15a - Chart-Stop Source Transparency

v30.15a korrigiert die missverstaendliche Bezeichnung des bisherigen 3,5%-Mindestabstands als `Technischer Stop` und macht die Stop-Herkunft im `📐 Risiko-Rechner` nachvollziehbar. Die produktive Setup-, Exit-, Portfolio-, Harvest-/Chop- und Orderlogik bleibt unveraendert.

## Neu
- Vier getrennte Werte: `Chart-Invalidierung`, `3,5%-Fallback`, `ATR-Schutz` und `Empfohlener Risiko-Stop`.
- Chart-Invalidierungen werden nur aus belastbaren numerischen Strukturfeldern abgeleitet.
- Prioritaet: Swing-/Higher-Low-Invalidierung, Unterkante einer bestaetigten Support-/aktiven Chartzone, explizites Supportniveau, Unterkante der Entry-/Reclaim-Zone.
- Die konkrete Chartquelle wird direkt angezeigt.
- Textbeschreibungen werden nicht nach beliebigen Zahlen durchsucht; Prozentwerte koennen dadurch nicht versehentlich als Kursniveau interpretiert werden.
- Fehlt ein belastbares numerisches Chartniveau, erscheint der 3,5%-Wert ausdruecklich als `Fallback` und nicht mehr als technische Chartmarke.
- Liegt eine echte Chart-Invalidierung naeher als 3,5% am Entry, bleibt der 3,5%-Mindestabstand als defensivere Vor-ATR-Basis erhalten.
- Liegt die Chart-Invalidierung weiter entfernt, bestimmt die Charttechnik die Vor-ATR-Basis.
- Danach greift unveraendert der v30.15-ATR-Schutz; der final empfohlene Long-Stop ist der weiter entfernte Wert.
- Bei manuell geaendertem Entry werden Fallback und ATR-Schutz neu berechnet, waehrend die absolute Chartmarke sichtbar bleibt.

## Beispiel
Entry 100, Swing-Low 92, 3,5%-Fallback 96,50 und ATR-Schutz 89,20:
- Chart-Invalidierung: 92,00,
- Vor-ATR-Basis: 92,00 aus der Swing-Struktur,
- ATR-Schutz: 89,20,
- Empfohlener Risiko-Stop: 89,20.

Wenn keine belastbare Chartmarke vorhanden ist, zeigt die App stattdessen:
- Chart-Invalidierung: n/a,
- 3,5%-Fallback: 96,50,
- klarer Warnhinweis, dass 96,50 keine Chartmarke ist.

## Sicherheitsgrenzen
- Keine automatische Aenderung bestehender Positions-Stops.
- Keine Aenderung an TP1/TP2/TP3, Exit-Engine, Entry-Gates, Trade-Journal oder Provider-Aufrufen.
- Der Stop bleibt manuell ueberschreibbar.
- Das maximale Depotrisiko bleibt unveraendert; ein weiterer Stop reduziert weiterhin die Positionsgroesse.
