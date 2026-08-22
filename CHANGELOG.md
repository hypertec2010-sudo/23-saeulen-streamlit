# Changelog

## v28.4.5c
- Zentrale Yahoo-Request-Drosselung eingefuehrt.
- Exponentieller kurzer Cooldown bei HTTP 429 / Too Many Requests.
- History-Aufrufe werden kontrolliert bis zu drei Mal versucht.
- Info-Aufrufe erhalten einen kurzen Retry-Pfad.
- Rate-Limit-Ticker werden im Live-Screener als temporaer ausstehend klassifiziert.
- Temporaere Fehler bleiben im Batch-Checkpoint offen und werden bei einem spaeteren Scan erneut verarbeitet.
- New-Listing- und Ticker-Routing aus v28.4.5b4 bleiben erhalten.

## v28.4.5d
- Datenqualitaetsanzeige im Live-Screener eingefuehrt.
- Datenqualitaet strikt vom Trading-Score getrennt.
- Historienlaenge, MA-Verfuegbarkeit, ATR und Volumen werden fuer die Qualitaetsstufe beruecksichtigt.
- Mobile Karten zeigen die Datenqualitaet direkt an.
