# v30.0 - Validated Trading Engine / Controlled Cutover

v30.0 baut auf dem stabilen v29.1 Portfolio & Risk Engine auf. Diese Version ist bewusst kein automatisches Umschalten der produktiven Ampel. Sie fuehrt einen transparenten Release-Gate-Prozess ein, der aus real gespeicherten Shadow-Forward-Returns, Guardrail-Ereignissen, Regime-Metadaten, Trading-Journal-Lerndaten und der Portfolio-Konfiguration ableitet, welche Engine-Bausteine fuer einen spaeteren kontrollierten Cutover ausreichend belegt sind.

## Neue Cutover-Zentrale
Im Live-Screener / Shadow-Bereich gibt es den neuen Expander `Validated Trading Engine / Controlled Cutover`. Dort werden die vorhandenen Daten in eine Freigabeentscheidung uebersetzt.

Die produktive Betriebsart bleibt in v30.0 immer die bestehende Live-Ampel. Auch ein gruener Validierungsstatus schaltet keine Schwelle, Gewichtung oder Ampel automatisch um.

## Harte Release-Gates
Ein Voll-Cutover bleibt gesperrt, solange eines der harten Gates offen ist:
- mindestens 40 auswertbare Divergenz-Episoden auf dem automatisch gewaehlten Primaerhorizont
- Shadow-Trefferquote mindestens 56 Prozent
- durchschnittliche Shadow-Edge mindestens +0,40 Prozent und positiver Median
- mindestens 10 Aufwertungen und 10 Abwertungen mit jeweils mindestens 52 Prozent Richtungs-Trefferquote
- mindestens zwei stabile Forward-Horizonte mit n>=10
- mindestens 70 Prozent Kontextabdeckung ueber Guardrail, RS-Dynamik, Markt- und Volatilitaetsregime
- mindestens 10 messbar gebremste Guardrail-Ereignisse und mindestens 55 Prozent defensiv bestaetigte Verlaeufe
- mindestens zwei Marktregime mit jeweils mindestens fuenf auswertbaren Episoden

Der Validation Score von 0 bis 100 dient nur als Orientierung. Er kann kein offenes hartes Gate ueberstimmen.

## Freigabe-Matrix pro Engine-Baustein
v30.0 bewertet separat:
- Guarded Engine Score
- Shadow-Aufwertungen
- Shadow-Abwertungen
- Engine Guardrails
- RS-Dynamik / Kontext
- Markt- und Volatilitaetsregime
- Trading Journal & Learning Engine
- Exit Engine 2.0
- Portfolio-Risikogate

Moegliche Stati sind `Shadow only`, `Daten sammeln`, `Teilfreigabe-Kandidat`, `Freigabereif` sowie beim Portfolio-Gate `Beratend aktiv`.

## Mehrere Horizonte statt Einzelwert
Die Freigabeentscheidung betrachtet 1T / 3T / 5T / 10T / 20T gemeinsam. Der Primaerhorizont wird automatisch nach Datenreife gewaehlt, wobei ein sinnvoller Swing-Horizont gegenueber einem rauschenden 1T-Ergebnis bevorzugt wird.

Mehrere Score-Aenderungen innerhalb derselben laufenden Live-vs-Shadow-Divergenz werden weiterhin nicht als unabhaengige Stichproben doppelt gezaehlt.

## Reale Trading-Evidenz als zusaetzliche Stuetze
Die v29.0 Learning Engine wird als zusaetzliche, aber nicht harte Freigabe-Evidenz genutzt. Beruecksichtigt werden Anzahl geschlossener Trades, durchschnittliches R und Entry-Kontextabdeckung. Die Exit Engine 2.0 bekommt einen eigenen Reifegrad auf Basis sicher zugeordneter Warnung-vs-Exit-Verlaeufe.

Historische Trades oder Shadow-Events ohne damals gespeicherten Kontext werden weiterhin nicht mit heutigen Daten aufgefuellt.

## Portfolio-Kontext
Das v29.1 Portfolio-Risikogate wird im Cutover-Prozess als operative Schutzschicht sichtbar. Es bleibt bewusst beratend: v30.0 verwendet das Portfolio-Gate nicht fuer automatische Orders oder zum Umschreiben der Live-/Shadow-Ampel.

## Keine neuen Provider-Abfragen
Der Cutover-Report verwendet ausschliesslich bereits gespeicherte Shadow-Performance-, Journal-, Event- und Portfolio-Daten. Er startet keine zusaetzlichen Yahoo-/Marktprovider-Abfragen und veraendert das Atomic-Screener-Cache-Schema nicht.

## Naechster Schritt bei echter Freigabereife
Erst wenn alle harten Gates erfuellt sind, meldet v30.0 `Kontrollierter Cutover-Kandidat`. Selbst dann bleibt die Live-Ampel produktiv. Ein spaeterer v30.x-Schritt kann daraus einen begrenzten A/B-Cutover einzelner freigegebener Komponenten machen, statt die bestehende Engine sofort vollstaendig zu ersetzen.

Keine SQL-/Secrets-Aenderung erforderlich.
