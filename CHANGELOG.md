# Changelog

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
