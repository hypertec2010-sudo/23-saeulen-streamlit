# v30.1b - Rotation Radar Navigation Fix

v30.1b behebt den Ruecksprung vom neuen `🧭 Rotation Radar` auf den `📡 Live-Screener`.

## Ursache
Die Trading-Cockpit-Navigation war historisch innerhalb des Live-Screener-Cache-Renderpfads aufgebaut. Der neue Rotation Radar ist jedoch ein eigenstaendiger Arbeitsbereich. Bei einem Streamlit-Rerun konnte der bedingt gerenderte Radio-Widget-State verschwinden bzw. neu mit dem Default `Live-Screener` initialisiert werden. Zusaetzlich war der Radar indirekt davon abhaengig, dass ein kompatibler Live-Screener-Stand vorhanden war.

## Fix
- Die Cockpit-Auswahl wird jetzt vor dem Live-Cache-Gate gerendert.
- Die Auswahl besitzt einen separaten persistenten Session-State, der kein Widget-Key ist.
- Der alte Cockpit-Key bleibt nur als Kompatibilitaetswert fuer bestehende Refresh-Logik erhalten.
- Auto-Refresh erkennt `Rotation Radar` ueber den persistenten State und fuehrt dort keinen Live-Screener-Autoscan aus.
- `Rotation Radar` kann auch ohne gueltigen Live-Snapshot geoeffnet werden.
- Eine leere Aktien-Watchlist blockiert den Rotation Radar nicht.

## Unveraendert
Keine Aenderung an Rotation-Berechnung, Leadership/Rotation/Breadth, Live-/Shadow-Ampeln, Scores, Guardrails, Atomic-Scan-Schema, Provider-Drosselung, SQL oder Secrets.
