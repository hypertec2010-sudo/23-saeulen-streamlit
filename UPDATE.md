# v28.6e3 - Explainability Display Fix

Fix:
- `Warum dieser Score?` wird nur noch angezeigt, wenn tatsaechlich Erklaerungstext vorhanden ist.
- Falls das zusammengesetzte Feld in einem alten/teilweisen Snapshot fehlt, wird die Erklaerung automatisch aus `Score-Treiber` und `Score-Bremsen` rekonstruiert.
- Derselbe Fallback gilt in der Ticker-Detailansicht.
- Cache-Schema angehoben, damit nach dem Reboot ein frischer Stand erzeugt wird.

Unveraendert:
- Live-/Shadow-Ampel
- Basis-, Engine- und Guarded-Score
- Regional Benchmarks
- Guardrails / Einstiegsgates
- Supabase / SQL / Secrets
