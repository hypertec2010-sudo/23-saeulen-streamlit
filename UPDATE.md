# v30.12 - Calibration Stability & Experiment Tracker

v30.12 erweitert den Shadow-only Calibration Advisor aus v30.11 um eine Stabilitäts- und Experimenthistorie. Der Advisor soll nicht nur einen punktuellen Kalibrierungshinweis liefern, sondern zeigen, ob dieselbe Empfehlung bei wachsender Outcome-Evidenz tatsächlich wiederholt bestehen bleibt oder zwischen den Auswertungsständen kippt.

## Neu
- Neue providerfreie Persistenz `calibration_stability_v3012`.
- Pro Watchlist wird höchstens ein Advisor-Stand pro Berlin-Tag gespeichert.
- Zusätzlich schützt ein Evidenz-Fingerprint vor Scheinstabilität: Wird die App an einem späteren Tag mit exakt derselben Advisor-Evidenz erneut geöffnet, entsteht **kein** neuer Stabilitätsstand.
- Ein neuer unabhängiger Evidenzstand entsteht erst, wenn sich z. B. Stichprobe, Outcome-Metriken, Status, Aussage oder zugrunde liegende Advisor-Evidenz verändert.
- Für jeden Calibration-Advisor-Bereich werden gespeichert und ausgewertet:
  - aktuelle Empfehlung / normalisierte Stance,
  - gleiche Empfehlung in Folge,
  - Trefferquote der aktuellen Empfehlung in den letzten bis zu 8 Evidenzständen,
  - Anzahl der Wechsel in den letzten bis zu 8 Evidenzständen,
  - aktuelle Mindest-Stichprobe,
  - erster und letzter Historienstand.
- Stabilitätsstufen:
  - `Noch zu kurz / Daten sammeln`,
  - `Beobachten`,
  - `Wechselhaft`,
  - `Vorläufig stabil`,
  - `Stabil`.
- Ein Shadow-Hinweis wird erst als **`Manuell prüfbarer Kandidat`** markiert, wenn:
  - mindestens 5 unabhängige Evidenzstände in Folge dieselbe Shadow-Prüfempfehlung tragen,
  - die Empfehlung in den letzten Ständen eine hohe Stabilität zeigt,
  - und die aktuelle Mindest-Stichprobe mindestens 30 Fälle je Vergleichsgruppe erreicht.
- Neue Metriken im Calibration Advisor: `Evidenzstände`, `Stabile Hinweise`, `Manuell prüfbar`, `Wechselhaft`.
- Vollständige Calibration-Historie kann angezeigt und als CSV exportiert werden.
- Glossar ergänzt um `Calibration Stability` und `Manuell prüfbarer Kandidat`.

## Schutz vor Overfitting
- Ein stabiler Hinweis ist **keine** automatische Freigabe zur Regeländerung.
- `Manuell prüfbar` bedeutet nur: Der Shadow-Hinweis ist inzwischen wiederholt und mit stärkerer Stichprobe sichtbar genug, um ihn bewusst zu untersuchen.
- Gleiche Daten an mehreren Kalendertagen zählen nicht mehrfach.
- Wechselhafte Empfehlungen werden ausdrücklich als solche sichtbar und nicht zu einem Änderungsvorschlag verdichtet.

## Unverändert
- Keine automatische Änderung an Harvest-/Chop-Schwellen.
- Keine Änderung an Action Queue, Decision Confidence, Live-/Shadow-Score oder Einstiegsgates.
- Keine Änderung an Stops, TP1/TP2/TP3, Orders oder Positionsgrößen.
- Keine neuen Provider-Abfragen.
