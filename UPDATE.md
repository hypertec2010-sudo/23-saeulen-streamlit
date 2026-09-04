# v30.3g - Portfolio Data Bridge & Missing-Data Display Fix

v30.3g behebt die irrefuehrende Nullwert-Darstellung im Portfolio-&-Risk-Bereich.

## Fehlerbild
Bei vorhandenen offenen Positionen konnten gleichzeitig `Investiert 0`, `Exposure 0%`, `Cash = Gesamtdepot`, `Risiko bis Stop 0%`, `Kursabdeckung 0%` und `FX-Abdeckung 0%` erscheinen.

Das waren nicht zwingend echte Nullwerte. Bei fehlender FX-Umrechnung blockiert die bestehende v29.1-Logik eine Basiswaehrungs-Aggregation bewusst. Bei fehlendem Atomic-Scan darf zudem ein gespeicherter `last_price` nicht als frischer Kurs ausgegeben werden.

## Neu
- Fehlende FX-Aggregation wird jetzt als **n/a** statt als 0 dargestellt.
- Ein Datenstatus zeigt, wie viele offene Positionen gefunden wurden, wie viele einen aktuellen Atomic-Kurs haben und wie viele nur einen gespeicherten Kurs besitzen.
- Unter **Native Positionswerte / Datenbasis** bleiben die vorhandenen Werte trotzdem sichtbar, ohne unzulaessige EUR-/FX-Schaetzung.
- Die Einzelpositionstabelle zeigt Entry, Stop, Stueck, native Waehrung, Kursbasis, Positionswert und Stop-Risiko.
- Wenn ein Ticker im bereits vorhandenen Atomic Complete Scan einen aktuellen Kurs besitzt, wird dieser ohne neuen Provider-Abruf in den Positionsspeicher gespiegelt.

## Wichtig
Ein gespeicherter Kurs bleibt als `gespeicherter Kurs` markiert und erhoeht die aktuelle Kursabdeckung nicht. Fehlende FX-Kurse werden weiterhin nicht automatisch geschaetzt oder versteckt vom Marktprovider geladen.

## Unveraendert
Portfolio-Risk-Score, Portfolio-Ampel, Live-/Shadow-Ampel, Exit Engine 2.0, Early Profit Protection und Rotation Radar bleiben fachlich unveraendert.
