# v29.1 - Portfolio & Risk Engine

v29.1 baut auf dem stabilen v29.0 Trading Journal & Learning Engine auf. Die neue Portfolio Engine betrachtet erstmals das Gesamtrisiko der offenen Positionen statt nur den einzelnen Trade. Sie ist bewusst beratend: Es werden keine Positionen, Orders, Live-/Shadow-Ampeln oder Entry-/Exit-Schwellen automatisch veraendert.

## Eigener Portfolio-Risiko-Bereich
Im Trading-Cockpit gibt es den neuen Bereich `Portfolio-Risiko`. Dort kann zwischen dem gesamten Positionsspeicher und der aktuell ausgewaehlten Positions-Watchlist gewechselt werden.

Die Engine berechnet unter anderem:
- investierten Positionswert und Exposure relativ zum Gesamtdepot
- rechnerische Cash-/Reservequote
- groesste Einzelposition und Top-3-Konzentration
- Cluster-/Sektor-Konzentration
- Risiko bis zu den aktuell gesetzten Stops
- Stop-Abdeckung des Portfolios
- Anteil des Exposures mit orange/roter Exit Engine 2.0
- aktuelle Atomic-Kursabdeckung und FX-Abdeckung
- unrealisiertes P/L auf Portfolioebene, sofern die Waehrungsumrechnung vollstaendig ist

## Portfolio-Risikoampel
Eine separate Portfolio-Ampel bewertet Exposure, Einzelpositions-Klumpen, Cluster-Konzentration, Stop-Risiko, Stop-Verletzungen und defensiven Exit-Druck. Die Ampel ist ein separates Risikogate und veraendert weder Live-Score noch Shadow-Score.

Die Engine zeigt neben dem Score konkrete Risikotreiber und priorisierte Portfolio-Massnahmen, zum Beispiel:
- kein weiteres Risiko in einem bereits uebergewichteten Cluster
- groesste Einzelposition pruefen
- fehlende Stops/Invalidierungen ergaenzen
- neue Kaeufe pausieren, solange ein grosser Teil des bestehenden Exposures bereits Exit-Druck zeigt
- Exposure/Cash normalisieren

## Missing-Data-Guard auf Portfolioebene
v29.1 rechnet keine alten oder unvollstaendigen Daten schoen:
- Nur Ticker aus dem aktuell abgeschlossenen Atomic-Scan gelten als aktuelle Kursbasis.
- Positionen anderer Watchlists koennen im Gesamtdepot mit ihrem gespeicherten letzten Kurs sichtbar bleiben, werden aber als nicht aktuell markiert.
- Bei zu niedriger aktueller Kursabdeckung wird keine gruene Portfolio-Freigabe gezeigt.
- Doppelte Ticker in mehreren Positions-Watchlists werden im Gesamtdepot nicht doppelt gezaehlt; verwendet wird der zuletzt aktualisierte Positionsdatensatz und der Konflikt wird sichtbar gemeldet.

## Korrekte Mehrwaehrungslogik
US-, Europa-, Schweiz-, Schweden- oder UK-Positionen werden nicht stillschweigend in einer gemeinsamen Waehrung addiert. Die Basiswaehrung des Depots ist einstellbar. Fuer Fremdwaehrungen wird eine explizite Umrechnung `1 Fremdwaehrung = x Basiswaehrung` verwendet.

Wichtig: v29.1 ruft dafuer bewusst keinen weiteren Marktprovider automatisch auf. Fehlt ein FX-Kurs, wird die betroffene Position nicht in Depotwert, Cash oder Exposure hineingeschaetzt und die FX-Abdeckung sinkt sichtbar.

Depot-Basiswaehrung, Gesamtdepotwert und eingegebene FX-Raten koennen ueber die bestehende zentrale Storage-Schicht persistent gespeichert werden. Dafuer ist keine neue SQL-Tabelle erforderlich.

## Portfolio-Gruppen / Cluster
Offene Positionen erhalten ein neues persistentes Feld `portfolio_group`. Beim Speichern kann eine Portfolio-Gruppe bzw. ein Sektor manuell gesetzt oder die konservative automatische Zuordnung uebernommen werden.

Manuelle Gruppen haben immer Vorrang. Nur klar zuordenbare Titel werden automatisch in breite Gruppen wie Halbleiter, Cloud/Cyber/Software, Mega-Cap Tech, Finanzen, Industrie, Gesundheit oder Konsum eingeordnet. Unklare Titel bleiben transparent als `Sonstige/Unbekannt` markiert.

## Pre-Trade Portfolio Guard
Der neue Abschnitt `Neuen Trade gegen Portfolio pruefen` simuliert einen geplanten Positionswert, ohne einen Trade anzulegen. Gezeigt werden:
- Anteil der neuen Position am Gesamtdepot
- Exposure nach dem Kauf
- Cluster-Konzentration nach dem Kauf
- Portfolio-Ampel fuer die geplante Groesse

Damit kann ein technisch guter Einzeltrade trotzdem als zu gross oder zu stark mit bestehenden Positionen korreliert/gebuendelt erkannt werden.

## Technisch unveraendert
- v28.7b Atomic Complete Scan und Rate-Limit-Schutz
- v28.8 Shadow Calibration / Backtest
- v28.9 Positions-/Exit-Engine 2.0
- v29.0 Trading Journal & Learning Engine
- v28.7a Trade-Close Undo
- keine SQL- oder Secrets-Aenderung
- keine zusaetzlichen Provider-Abfragen durch die Portfolio Engine
- Live-Screener-Cache-Schema bleibt unveraendert, da v29.1 keine neuen Live-Rohfelder benoetigt
