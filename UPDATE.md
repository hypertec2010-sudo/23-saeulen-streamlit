# v30.3b - Positions-Watchlist Recovery Fix

v30.3b behebt den verbleibenden Fall, in dem eine bestehende Positions-Watchlist trotz v30.3a weiterhin nicht aufrufbar war.

## Tatsächliche Restursache
v30.3a nutzte zwar den echten Watchlist-Katalog, filterte im Bereich **Positionen** danach aber weiterhin hart auf den Typ `Positions-Watchlist`.

Wenn ein alter Backend-/Katalogeintrag die Liste leer, falsch oder als normale `Watchlist` typisierte, blieb sie deshalb weiter unsichtbar. Gleichzeitig konnte die Erstellfunktion den Namen weiterhin als bereits vorhanden erkennen.

## Fix 1: Keine gespeicherte Liste wird mehr versteckt
- Der Bereich **Bestehende Watchlist auswählen** zeigt jetzt alle vom Katalog bekannten Listen.
- Listen mit passendem Typ stehen zuerst.
- Der gespeicherte Typ wird direkt im Auswahltext angezeigt.
- Eine falsch typisierte Liste bleibt sichtbar und kann explizit im aktuellen Arbeitsbereich übernommen werden.

## Fix 2: Duplicate-Recovery
Wenn beim Erstellen einer Liste der Backend-Name als bereits vorhanden / existent / Duplicate bestätigt wird:
- wird **keine neue Liste erstellt**,
- der bestehende Name wird als Recovery-Eintrag übernommen,
- die gewünschte Typisierung (`Watchlist` oder `Positions-Watchlist`) wird nur für die UI-Wiederherstellung gespeichert,
- die Liste erscheint nach dem Rerun in der Auswahl.

Damit kann auch eine Liste wieder erreichbar gemacht werden, die vom Backend bei der Duplikatprüfung erkannt, aber vom Katalog nicht geliefert wird.

## Fix 3: Öffnen über dedizierte Backend-API
Nach Auswahl werden die Ticker primär über `get_watchlist_tickers(name)` geladen. `load_watchlists_df()` ist nur noch Fallback. Damit hängt das Öffnen einer Liste nicht mehr davon ab, ob ihre Ticker-Zeilen im globalen Watchlist-DataFrame korrekt sichtbar sind.

## Persistenz
Recovery-Einträge werden über die bestehende zentrale Storage-Schicht gespeichert und bleiben über Reruns/Neustarts erhalten. Beim echten Löschen einer Watchlist wird der Recovery-Eintrag ebenfalls entfernt.

## Einmalige Wiederherstellung einer unsichtbaren bestehenden Liste
Falls die bestehende Positionswatchlist auch nach Installation noch nicht sofort auftaucht:
1. im Bereich **Positionen -> Watchlist verwalten** denselben bestehenden Namen eintragen,
2. Typ `Positions-Watchlist` wählen,
3. einmal **Watchlist erstellen** drücken.

Wenn das Backend meldet, dass die Liste bereits existiert, übernimmt v30.3b sie automatisch in die Auswahl. Es wird dabei keine zweite Liste erzeugt.

## Unverändert
Keine Änderung an Early Profit Protection/Learning, Exit Engine, Live-/Shadow-Ampel, Rotation Radar, Atomic Complete Scan, Portfolio Engine, SQL oder Secrets.
