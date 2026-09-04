# v30.2 - Early Profit Protection & Giveback Engine

v30.2 ergaenzt den Positions-/Exit-Bereich um eine separate Fruehgewinn-Schutzlogik fuer Trades, die kurz nach dem Entry ungewoehnlich schnell steigen.

## Neue Kernwerte
- **Profit Velocity 0-100**: bewertet Gewinn seit Entry relativ zu Haltedauer, ATR und R-Multiple.
- **Exhaustion Risk 0-100**: trennt gesunde Beschleunigung von Ueberdehnung/Ermuedung anhand von MA10-Abstand, Exit-/Trend-/Momentum-/Distribution-/RS-Signalen, Markt und Volatilitaet.
- **Historical Giveback Risk**: misst auf expliziten Klick, wie oft aehnliche schnelle Moves in der Aktie bzw. kompakten Vergleichsaktien innerhalb der folgenden 5 Handelstage mindestens die Haelfte des Impulses wieder abgegeben haben.

## Empfehlungen
Die Zusatzengine unterscheidet u. a.:
- Healthy Acceleration / laufen lassen,
- Gewinnschutz pruefen,
- Teilgewinn 25-50% pruefen,
- in extremen, technisch bestaetigten Faellen Teilgewinn / Exit pruefen.

Ein schneller Kursanstieg allein erzeugt bewusst kein Verkaufssignal. Positive RS-/Markt-/Akkumulationsbestaetigung kann einen schnellen Move weiterhin als konstruktiv einstufen.

## Historische Fast-Move-Analyse
Im Bereich **Positionen / Exit -> Offenen Trade verwalten** kann fuer die ausgewaehlte Position die Historie manuell aktualisiert werden.

Die Analyse:
- verwendet ein zum aktuellen Trade passendes 1-3T-Fast-Move-Fenster,
- kalibriert Mindestbewegung in Prozent und ATR an den aktuellen Move,
- untersucht die gleiche Aktie ueber bis zu 5 Jahre,
- ergaenzt bei passender Rotation-/Portfolio-Gruppe maximal vier repraesentative Vergleichsaktien,
- misst Giveback >=50%, Ruecklauf zum Move-Start, direkten Follow-through, medianen Ruecksetzer und medianen weiteren Lauf,
- kennzeichnet die Stichprobe als Zu klein / Fruehphase / Mittel / Gut / Breiter.

Ein gespeichertes historisches Profil beeinflusst die aktuelle Empfehlung nur, solange Tempo und Haltefenster noch ausreichend zum damaligen Fast-Move-Profil passen. Bei deutlicher Veraenderung wird es nur noch angezeigt und eine Aktualisierung empfohlen.

## Provider-Schutz
Die normale Positionsanzeige und der Atomic-/Auto-Scan erzeugen **keine zusaetzlichen Historien-Requests**. Historie wird nur per explizitem Button geladen, gebuendelt fuer die Position plus maximal vier Peers. Ein einzelner Target-Fallback ist nur erlaubt, wenn die Batch-Abfrage die Position selbst nicht geliefert hat.

## Learning / Events
Gelbe, orange und rote Early-Profit-Zustaende werden dedupliziert als **Early Profit Protection** Event protokolliert. Historische Aktualisierungen werden separat dokumentiert. Damit kann die bestehende Learning Engine spaeter pruefen, ob frueher Gewinnschutz tatsaechlich vorteilhaft war.

## Unveraendert
- keine automatische Order,
- keine automatische Stop-Aenderung,
- keine Aenderung an Live-/Shadow-Ampel,
- keine Aenderung an Exit Engine 2.0,
- keine Aenderung am Atomic-Screener-Cache-Schema,
- keine SQL-/Secrets-Aenderung.
