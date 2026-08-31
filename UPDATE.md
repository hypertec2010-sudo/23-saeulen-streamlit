# v28.7a - Trade-Close Undo & Safety

Dieses Update behebt versehentliche oder mit falschem Kurs ausgefuehrte manuelle Trade-Schliessungen.

## Neu: Schliessung rueckgaengig machen
- Im Trade-Journal gibt es jetzt den Bereich `Versehentliche Schliessung rueckgaengig machen`.
- Eine geschlossene Position kann wieder als offene Position hergestellt werden.
- Es wird **kein kuenstlicher Gegentrade** angelegt.
- Der fehlerhafte Abschluss wird als Audit-Eintrag `Schliessung rueckgaengig` markiert und zaehlt danach nicht mehr zu realisiertem P/L, Trefferquote oder geschlossenen Trades.
- Bei ab v28.7a geschlossenen Positionen wird vor dem Schliessen ein kompletter Positions-Snapshot im Journal gespeichert. Dadurch ist spaeter eine verlustfreie Wiederherstellung moeglich.
- Aeltere Schliessungen ohne Snapshot werden aus Journal- und Event-Historie rekonstruiert. Entry, Stop, Stueckzahl und vorherige Teilverkaeufe werden soweit vorhanden wiederhergestellt; Ziel/Details sollten danach kurz kontrolliert werden.

## Neu: Zweistufige Schliessung
`Position schliessen` fuehrt den Trade nicht mehr beim ersten Klick aus.

1. Ausstiegskurs, Datum, Grund und Notizen eingeben.
2. `Schliessung pruefen` waehlen.
3. Die App zeigt Ticker, Stueckzahl, Exit-Kurs und berechnetes P/L.
4. Erst nach expliziter Bestaetigung kann `Endgueltig schliessen` ausgefuehrt werden.

## Plausibilitaetswarnung
- Bei einem zeitnahen Exit, der mindestens 12 % vom aktuellen Live-Kurs abweicht, erscheint eine deutliche Warnung.
- Ebenso bei extremen Ergebnissen gegenueber dem Entry (ab +100 % oder unter -50 %).
- In diesem Fall ist eine zweite ausdrueckliche Kursbestaetigung notwendig.

## Bestehender ACN-Fall
Der bereits versehentlich geschlossene ACN-Trade vom 31.08.2026 kann nach dem Update direkt im Trade-Journal ausgewaehlt und rueckgaengig gemacht werden. Da dieser Abschluss noch vor v28.7a entstand, wird er ueber die Legacy-Rekonstruktion wiederhergestellt. Danach bitte Entry, Stop, Ziel und Stueckzahl einmal kurz kontrollieren.

Keine Aenderung an Live-Screener, Shadow-Engine, Scores, Guardrails oder Benchmark-Logik.
