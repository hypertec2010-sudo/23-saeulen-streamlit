# v30.3j - Compact Responsive Board Metrics

v30.3j verbessert die Lesbarkeit der kompakten Kennzahlenkarten im gesamten Dashboard. Lange Feldnamen sollen auch in dichten 5-/6-Spalten-Boards vollstaendig sichtbar bleiben, statt abgeschnitten oder unleserlich gross dargestellt zu werden.

## Neu
- Globale responsive Darstellung fuer Streamlit-Metric-Karten (`st.metric`) im Dashboard.
- Kleinere Label-Schrift mit sauberem mehrzeiligem Umbruch fuer lange Bezeichnungen wie `Aktuelle Kursabdeckung`, `Risiko bis Stop`, `Cash / Reserve` oder `Setup-Confidence`.
- Werte bleiben hervorgehoben, werden aber auf dichten Desktop-Boards leicht kompakter dargestellt.
- Metric-Delta/Hinweistext wird ebenfalls kleiner und darf umbrechen.
- Einheitliche Mindesthoehe der Desktop-Labels sorgt dafuer, dass Werte auch bei ein- und zweizeiligen Feldnamen optisch ausgerichtet bleiben.
- Auf schmalen/mobile Layouts wird die Mindesthoehe wieder geloest, damit die Karten nicht unnoetig hoch werden.

## Geltungsbereich
- Die Anpassung gilt bewusst fuer aehnliche `st.metric`-Felder im gesamten Board und nicht nur fuer die sechs Portfolio-Kennzahlen.
- Portfolio & Risk Engine profitiert insbesondere bei `Investiert`, `Exposure`, `Cash / Reserve`, `Risiko bis Stop`, `Aktuelle Kursabdeckung` und `Stop-Abdeckung`.

## Unveraendert
- Keine Aenderung an Portfolio-Berechnung, FX, Kurs-/Stop-Abdeckung, Rotation Radar, Live-/Shadow-Logik, Exit Engine, Positionen oder Orders.
- Keine neuen Provider-Abfragen.
