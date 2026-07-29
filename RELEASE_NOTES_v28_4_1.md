# v28.4.1 – R-Multiple Initial-Risk Fix

## Behoben

Nach einer Stop-Anpassung über den Entry wurde das R-Multiple als `n/a` angezeigt, obwohl Entry, Kurs und Stop vorhanden waren. Ursache war, dass der aktuelle nachgezogene Stop fälschlich als Risikobasis verwendet wurde.

## Neues Verhalten

- R-Multiple = `(aktueller Kurs - Entry) / (Entry - Initial-Stop)`.
- Der aktuelle Stop bleibt für Stop-/Exit-Erkennung zuständig.
- Ein Stop über Entry wird als Gewinnschutz erkannt.
- Die Tabelle zeigt den `Initial-Stop (R-Basis)` separat.
- Bei älteren Positionen wird der ursprüngliche Stop aus der Stop-Historie rekonstruiert, sofern möglich.
- Fehlt die ursprüngliche Risikobasis wirklich, lautet die Meldung nun `Initialrisiko fehlt` statt fälschlich `Entry und Stop werden benötigt`.

Keine Änderung an Supabase-Schema, Secrets oder bestehenden Daten erforderlich.
