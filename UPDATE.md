# v30.7 - Unified Decision Summary

v30.7 vereinheitlicht die Lesart der wichtigsten Arbeitsbereiche. Statt je Bereich andere Überschriften, doppelte Kurzfassungen oder reine Roh-Scores zu zeigen, folgt die Oberfläche jetzt möglichst demselben Muster: **Was sehe ich? · Was hat sich geändert? · Nächste Handlung · Warum?**. Die Änderung ist rein darstellend und nutzt ausschließlich bereits berechnete Daten.

## Neu
- Gemeinsame, kompakte `Entscheidungs-Zusammenfassung` als wiederverwendbare UI-Schicht.
- Live-Screener: Begriffe aus v30.5c vereinheitlicht. `Kurzfazit` heißt jetzt `Was sehe ich?`, `Seit letztem Scan` wird zu `Was hat sich geändert?`, und Treiber/Bremse stehen einmal unter `Warum?`.
- Rotation Stock Drilldown: der aktuell führende Kandidat erhält eine kurze Entscheidungs-Zusammenfassung mit Kandidaten-Score, Engine-Bestätigung, Sektor-RS, RS-Beschleunigung, Trend und Entry-Readiness. Das Radar bleibt ausdrücklich Beobachtungslogik; Live-/Shadow-Ampeln und Gates bleiben maßgeblich.
- Portfolio-Risiko: Status, nächste Handlung und wichtigste Risikotreiber erscheinen als eine kompakte Zusammenfassung. Vollständige Treiber und Maßnahmen liegen darunter in einem optionalen Detail-Expander.
- Positionen / Exit: der ausgewählte offene Trade erhält vor den Engine-Details eine Management-Zusammenfassung aus Exit-Druck, aktueller Atomic-Kursbasis, Stop-Plan und – falls relevant – Harvest-/Teilgewinn-Hinweis.
- Einzelanalyse: die bisherige `Nächste Handlung`-Box wurde zur vierteiligen Entscheidungs-Zusammenfassung erweitert: `Was sehe ich?`, `Nächste Handlung`, konkreter Trigger und Invalidierung/defensiver Punkt. Das spätere doppelte Kurzfazit im Überblick entfällt.
- Feature-Überschriften mit sichtbaren Versionsnummern verwenden wieder `APP_VERSION`, damit die Oberfläche nicht gleichzeitig alte Versionslabels zeigt.

## Transparenz / Grenzen
- `Was hat sich geändert?` wird nur angezeigt, wenn im jeweiligen Bereich tatsächlich ein belastbarer Vergleichswert vorhanden ist; fehlende Historie wird nicht erfunden.
- Die Rotation-Zusammenfassung bewertet nur bereits vorhandene Drilldown-/Atomic-Daten und startet keinen zusätzlichen Provider-Request.
- Portfolio- und Positions-Zusammenfassungen verändern keine darunterliegenden Berechnungen. Die Detail-Engines bleiben vollständig sichtbar.

## Unverändert
- Keine Änderung an Live-, Shadow-, Guarded-, Rotation-, Exit-, Portfolio-, Harvest- oder Chop-Scores.
- Keine Änderung an TP1/TP2/TP3, Stops, Orders, Positionsgrößen oder Einstiegsgates.
- Keine automatische Kalibrierung aus v30.6.
- Keine neuen Provider-Abfragen.
