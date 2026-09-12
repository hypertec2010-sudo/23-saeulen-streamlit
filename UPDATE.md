# v30.9 - Decision Action Queue

v30.9 ergänzt den Live-Screener um eine kompakte, providerfreie Watchlist-Triage. Ziel ist nicht ein weiterer Score, sondern eine schnellere Antwort auf die operative Frage: **Welche Werte muss ich jetzt wirklich ansehen?**

## Neu
- Neuer Bereich `Decision Action Queue · Watchlist-Priorisierung` oberhalb der ausführlichen Live-Screener-Darstellung.
- Drei klare Kategorien:
  - `🎯 Jetzt prüfen`: grünes Setup mit bestehender Trigger-/Entry-/Armed-Evidenz und ohne hartes Einstiegsgate.
  - `👀 Beobachten`: noch keine unmittelbare Prüffreigabe bzw. neutralere/gelbe/weiße Setups.
  - `⛔ Blockiert`: hartes Einstiegsgate, Engine-Blockierung oder Invalidierung.
- Vier Sofortmetriken: Anzahl `Jetzt prüfen`, `Beobachten`, `Blockiert` sowie Anzahl Werte mit hoher Decision-Confidence.
- Jede Kategorie besitzt eine kompakte Tabelle mit Ticker, Name, Ampel, Live-Score, Decision-Confidence, Status, Trade-State, CRV, Entry-Abstand, Harvest, Veränderung, Fokus-Grund und nächster Handlung.
- Unter jeder Kategorie können `Evidenz / Aktualität / Grenzen` separat eingeblendet werden.
- Die Queue verwendet bewusst den **vollständigen angereicherten Atomic-Stand der Watchlist**, auch wenn in der Haupttabelle der UI-Filter `nur aktive` eingeschaltet ist.
- Geänderte Werte werden im Fokus-Grund bevorzugt mit dem vorhandenen `Warum geändert?` erklärt; unveränderte Werte nutzen bestehende Score-Treiber bzw. Statusinformationen.
- Glossar um `Decision Action Queue` und `Triage` erweitert.

## Sortierung ohne neuen Trading-Score
Die Queue berechnet keinen eigenen Kauf-/Trading-Score. Innerhalb der bestehenden Kategorien wird transparent sortiert nach:
1. Kategorie (`Jetzt prüfen` → `Beobachten` → `Blockiert`),
2. Decision-Confidence (`Hoch` → `Mittel` → `Niedrig`),
3. vorhandenem Live-Score absteigend,
4. echter Änderung seit letztem Scan,
5. Ticker.

## Sicherheits- und Interpretationsgrenzen
- `Jetzt prüfen` bedeutet ausdrücklich **nicht automatisch kaufen**.
- Harte Einstiegsgates bleiben vollständig blockierend.
- CRV, Entry-Regeln, Live-/Shadow-Logik und Guardrails bleiben maßgeblich.
- Decision-Confidence ist Daten-/Evidenzvertrauen und kein Renditeversprechen.

## Unverändert
- Keine Änderung an Live-, Shadow-, Guarded-, Rotation-, Exit-, Portfolio-, Harvest- oder Chop-Scores.
- Keine Änderung an TP1/TP2/TP3, Stops, Orders, Positionsgrößen oder Einstiegsgates.
- Keine automatische Kalibrierung aus v30.6.
- Keine neuen Provider-Abfragen.
