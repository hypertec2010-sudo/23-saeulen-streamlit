# Changelog

## v30.0
- Validated Trading Engine / Controlled Cutover als transparente Release-Gate-Schicht eingefuehrt.
- Produktive Live-Ampel bleibt Kontrollgruppe; v30.0 schaltet keine Engine automatisch um.
- Automatische Auswahl eines reifen Primaerhorizonts aus 1T / 3T / 5T / 10T / 20T.
- Harte Gates fuer Stichprobe, Gesamt-Edge, Richtungsbalance, Horizont-Stabilitaet, Kontextabdeckung, Guardrail-Nachweis und Marktregime-Abdeckung.
- Validation Score 0-100 als Orientierung; offene harte Gates koennen dadurch nicht ueberstimmt werden.
- Separate Freigabe-Matrix fuer Guarded Engine Score, Aufwertungen, Abwertungen, Guardrails, RS-Kontext, Regime, Trading Learning, Exit Engine 2.0 und Portfolio-Risikogate.
- Reale v29.0 Trade-Lerndaten werden als zusaetzliche Freigabe-Evidenz eingebunden, ohne alte Trades mit aktuellem Kontext aufzufuellen.
- Exit Engine 2.0 erhaelt einen eigenen Reifegrad anhand sicher zugeordneter Warnung-vs-Exit-Verlaeufe.
- v29.1 Portfolio-Gate wird als operative beratende Schutzschicht in der Cutover-Matrix sichtbar.
- Regime-Stabilitaet und Horizont-Stabilitaet werden separat im Dashboard angezeigt.
- Keine neuen Provider-Abfragen, keine Aenderung am Atomic-Screener-Cache-Schema, keine SQL-/Secrets-Aenderung.

## v29.1
- Portfolio & Risk Engine als eigener Trading-Cockpit-Bereich eingefuehrt.
- Gesamtdepot- und aktuelle-Watchlist-Sicht fuer offene Positionen.
- Exposure, Cash/Reserve, Einzelpositionsgewicht, Top-3- und Cluster-Konzentration.
- Portfolio-Risiko bis Stop sowie Stop-Abdeckung und Stop-Verletzungen.
- Exit Engine 2.0 wird als bestehender Positions-Risikokontext auf Portfolioebene aggregiert.
- Missing-Data-Guard verhindert gruene Portfolio-Freigabe bei unvollstaendiger Kurs-/FX-Basis.
- Doppelte Ticker ueber Positions-Watchlists werden im Gesamtdepot dedupliziert statt doppelt gezaehlt.
- Explizite Mehrwaehrungslogik; fehlende FX-Raten werden nicht geschaetzt.
- Depot-Basis, Depotwert und FX-Raten koennen ueber die zentrale Storage-Schicht persistent gespeichert werden.
- Persistente Portfolio-Gruppe/Sektor pro Position mit manueller Ueberschreibbarkeit.
- Konservative Cluster-Heuristik fuer klar erkennbare Gruppen; Unbekannt bleibt sichtbar.
- Pre-Trade Portfolio Guard simuliert Positionsgroesse, Exposure und Cluster-Konzentration vor einem neuen Trade.
- Keine automatische Aenderung an Positionen, Orders, Live-/Shadow-Ampeln oder Scores.
- Keine SQL-/Secrets-Aenderung und keine neuen Provider-Abfragen.

## v29.0
- Trading Journal & Learning Engine im Beobachtungsmodus eingefuehrt.
- Neue Positionen speichern einen strukturierten Entry-Kontext aus dem bereits abgeschlossenen Atomic-Live-Scan; keine Extra-Provider-Abfragen.
- Historische Positionen ohne Entry-Kontext werden nicht mit aktuellen Daten rueckwirkend aufgefuellt.
- Journalzeilen enthalten ab v29.0 Live-/Shadow-/Score-/Regime-/RS-/Guardrail-/Setup-Kontext als eigenstaendige Exportfelder.
- Learning-Datensatz mit einer Zeile pro gueltig geschlossenem Trade-Zyklus; Teilverkaeufe werden dem Full-Close zugerechnet.
- Rueckgaengig gemachte Fehlschliessungen bleiben Audit-Historie und werden nicht als abgeschlossene Trades gezaehlt.
- Kennzahlen: Trefferquote, Gesamt P/L, durchschnittliches/medianes R, Profit Factor, Kapitalrendite, Haltedauer und Kontextabdeckung.
- Segmentauswertung nach Radar-Bucket, Marktregime, Volatilitaet, RS-Dynamik, Live-Ampel, Shadow-vs-Live, Score-Baendern, Guardrail und Grade.
- Exit Engine 2.0 Lerncheck verknuepft Warnungen nur bei sicherem Entry-/Exit-Zeitfenster mit dem spaeteren Trade-Ergebnis.
- R-Veraenderung nach Erstwarnung und Warnvorlauf als neue Management-Lernmetriken.
- Wiederkehrende Themen aus manuellen Erkenntnis-Texten werden transparent gezaehlt.
- Stichproben-Guard verhindert, dass kleine Datenmengen als belastbare Kalibrierung erscheinen.
- Learning-Datensatz als CSV exportierbar.
- Keine automatische Aenderung an Live-/Shadow-Ampel, Scores, Guardrails, Positionen oder Orders.
- Keine SQL-/Secrets-Aenderung; Live-Cache-Schema absichtlich unveraendert.

## v28.9
- Positions-/Exit-Engine 2.0 fuer offene Long-Positionen eingefuehrt.
- Exit-Druck kombiniert Exit-Score, taktisches Risiko, Trendbruch, Momentum-Abbau, Distribution und relative Schwaeche.
- Marktregime, Volatilitaetsregime und RS-Dynamik als moderater Positionskontext integriert.
- Positionspuffer (P/L und R-Multiple) steuert, ob Warnungen eher Gewinnschutz/Teilgewinn oder Risikoabbau/Exit ausloesen.
- Harte Floors fuer Stop-Verletzung sowie sehr starken Exit-/Trendbruchdruck.
- Neue Fuehrungsstufen: Halten, Gewinnschutz, Stop enger, Teilgewinn, Risiko reduzieren, Exit pruefen.
- Stop-Status, Stop-Plan, Gewinnschutz-Plan, Add-on-Bremse und Datenkonfidenz transparent sichtbar.
- Technische Exit-Rohfelder werden ohne zusaetzliche Provider-Abfragen aus dem Atomic-Live-Scan an den Positionsmonitor weitergereicht.
- Kritischer UI-Fix: 'nur aktive' filtert nur noch die Screener-Ansicht; Positions-/Exit-Monitor sieht immer den vollstaendigen Atomic-Stand.
- Orange/rote Exit-Engine-Zustaende werden dedupliziert fuer die spaetere Learning Engine protokolliert.
- Live-Cache-Schema fuer die neuen Positionsfelder auf v28.9 angehoben.
- Keine Aenderung an Live-/Shadow-Ampel, Entry-Guardrails oder produktiven Score-Schwellen.

## v28.8
- Engine Calibration & Backtest als neuer Analysebereich im Shadow-Dashboard.
- Shadow-Edge eingefuehrt: Aufwertung erwartet positive, Abwertung negative Forward-Returns.
- Trefferquote fuer Shadow-Abwertungen korrigiert.
- 1T/3T/5T/10T/20T Horizon-Vergleich mit durchschnittlichem und medianem Edge.
- Directional MFE/MAE fuer favorable/adverse excursion ergaenzt.
- Divergenz-Zustandsaenderungen werden fuer die Kalibrierung zu Episoden de-clustert.
- Guarded-Score-Baender 0-27 / 28-54 / 55-71 / 72-100 separat auswertbar.
- Segmentauswertung nach Guardrails, RS-Dynamik, Marktregime und Volatilitaetsregime.
- Guardrail-Backtest fuer messbare Differenz zwischen Raw Engine Score und Guarded Engine Score.
- Kalibrierungsurteil mit Stichprobenstatus; keinerlei automatische Aenderung an Live-Logik.
- Neue Shadow-Ereignisse speichern erweiterten Kontext und technische Komponenten fuer spaetere Kalibrierung.
- Shadow-Performance bevorzugt nun die zentrale Storage-Schicht; lokales JSON bleibt Fallback.
- Nicht-Handelstag-Indizierung im Forward-Return-Tracking korrigiert.
- Performance-Refresh bleibt manuell und rate-limit-sicher.
- Atomic Complete Scan aus v28.7b unveraendert beibehalten.

## v28.7b
- Live-Screener auf Atomic Complete Scan umgestellt.
- Live-Screener scannt immer die komplette eindeutige Watchlist; 40/80/120-Teilmengen im Live-Screener entfernt.
- Teil-Batches werden nicht mehr in Session oder persistente Snapshots geschrieben.
- Alte Ergebniszeilen werden nicht mehr tickerweise in einen neuen Lauf hineingemischt.
- Persistente Restore-Logik akzeptiert nur v28.7b-Atomic-Vollstaende.
- Status-/Hysterese-Historie wird erst nach komplettem Rohscan atomar aktualisiert.
- Manueller und automatischer Vollscan erhalten frische Analyse-Keys ohne globales Cache-Clear.
- Provider-Drosselung mit kleineren Batches, Per-Ticker-Pause und zwei Cooldown-Retry-Runden fuer temporaere/429-Fehler.
- Heartbeat-Cache-Key korrigiert: `schema` wird nun identisch verglichen; verhindert faelschlich dauernd faellige Auto-Scans.
- Scan-Lock verhindert Auto-Rerun waehrend eines laufenden Vollscans.
- 5-Minuten-Auto-Cooldown nach komplett fehlgeschlagenem Lauf.
- Sichtbarer Scan-Status mit Datenzeit, Alter, Erfolg/Fehler, Dauer und `kein Mischstand`.
- Keine Aenderung an Ampel-, Score-, Shadow-, Guardrail-, Benchmark-, Positions- oder Journal-Logik.

## v28.7a
- Zweistufiger, bestaetigungspflichtiger Workflow fuer vollstaendige Positionsschliessungen.
- Vorschau vor dem Exit mit Ticker, Stueckzahl, Exit-Kurs und berechnetem P/L.
- Plausibilitaetswarnung und zweite Bestaetigung bei auffaelligem Ausstiegskurs.
- Neue Undo-Funktion fuer versehentlich geschlossene Trades.
- Ab v28.7a wird vor jedem Full-Close ein kompletter Positions-Snapshot fuer verlustfreies Undo gespeichert.
- Legacy-Schliessungen werden aus Journal- und Event-Historie rekonstruiert.
- Rueckgaengig gemachte Abschluesse werden aus P/L-, Trefferquoten- und Closed-Trade-Statistik neutralisiert, bleiben aber als Audit-Historie erhalten.
- Keine Aenderung an Screener-, Shadow-, Score-, Guardrail- oder Benchmark-Logik.

## v28.6e6
- aggressives globales Cache-Clear aus v28.6e5 entfernt
- manueller Vollrefresh provider-sicher gemacht
- letzter gueltiger Tickerstand bleibt bei temporaeren 429-Fehlern sichtbar
- manuelle Vollpruefung laeuft weiterhin ueber alle Ticker

## v28.7
- Shadow Performance Tracking mit 1T/3T/5T/10T/20T Forward Returns.
- Shadow-Ereignisse werden dedupliziert persistent protokolliert.
- Performance-Auswertung getrennt nach Aufwertung/Abwertung.
- Kursnachladen nur per explizitem Button, um Provider-Rate-Limits zu vermeiden.
- Keine Änderung an Live-Ampel, Shadow-Entscheidungslogik, Scores oder Guardrails.
