# v30.5a - Watchlist Future Ticker Fix

v30.5a behebt die Watchlist-Eingabe fuer Yahoo-Future-Ticker wie `CL=F`.

## Behoben
- `CL=F` wurde in der manuellen Watchlist-Eingabe bisher nicht als direkter Ticker erkannt, weil der lokale Tickercheck das Zeichen `=` ausgeschlossen hat.
- Dadurch fiel `CL=F` faelschlich in die Namenssuche; wenn dort kein Treffer kam, blieb die Aufloesung leer und die Queue meldete `Keine neuen Werte zum Vormerken erkannt.`
- Yahoo-Futures mit `=F` werden jetzt explizit als valide direkte Ticker akzeptiert.
- Ebenfalls robust: Yahoo-Indizes mit `^`, z. B. `^GSPC`.
- Die Watchlist-Eingabe nutzt jetzt dieselbe Commodity-/Ticker-Aufloesung wie die Analyse: Rohstoff-Aliasse wie `WTI`, `Oil`, `Gold` werden zuerst aufgeloest; echte Future-Ticker werden direkt uebernommen; erst danach wird eine Namenssuche versucht.
- Nicht aufloesbare Eingaben werden sichtbar genannt, statt still zu verschwinden.
- Das Eingabefeld nennt `CL=F` und `WTI` jetzt explizit als Beispiele.

## Beispiele
- `CL=F` -> `CL=F`
- `WTI` -> `CL=F`
- `BZ=F` -> `BZ=F`
- `GC=F` -> `GC=F`
- `NG=F` -> `NG=F`
- `^GSPC` -> `^GSPC`

## Unveraendert
- Keine Aenderung an WTI-Analyse, Commodity Context, Harvest/Chop, TP1/TP2/TP3, Live-/Shadow-/Exit-Logik.
- Keine zusaetzlichen Provider-Abfragen fuer direkt erkannte Future-Ticker.
- Watchlist-Queue und gebuendelte Speicherung bleiben unveraendert.
