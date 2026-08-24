## v28.6e
- Regional Benchmark Engine fuer europaeische Aktien.
- Landesbenchmarks mit Europa-Fallback.
- 21T/63T Relative Staerke und RS-Dynamik werden bei fehlenden Core-Daten im Live-Monitor nachberechnet.
- Tatsaechlich verwendeter Benchmark wird transparent angezeigt.
- Keine Aenderung an Live-/Shadow-Ampel oder Guardrails.

## v28.6c1
- Hotfix fuer uninitialisierte `entry_distance`-Variable in Gate Transparency.
- Risiko-/Entry-Istwerte werden innerhalb der Transparenzfunktion robust aufgeloest.

# Changelog

## v28.6c
- Gate Transparency: konkrete Einstiegsgates inklusive Detailursache und Schwellenwerten.
- Gate-Information in Desktop, Mobile und Ticker-Details sichtbar.
- Keine Aenderung der Bewertungslogik.

## v28.6e1 - Regional Benchmark Fallback Diagnostics
- Zeigt Primaerbenchmark, Abrufstatus, Fallback-Grund und Diagnosekette.
- Keine Aenderung an Score-, Ampel-, Shadow- oder Guardrail-Logik.

## v28.6e2
- Regionaler Landesbenchmark erhält Vorrang vor bereits vorhandenen generischen Europa-RS-Daten.
- CAC 40 / OMX Stockholm werden bei .PA / .ST nun tatsächlich versucht.
- Fallback-Diagnose zeigt echten Primärabruf statt `Nicht benötigt`.
