# v28.4.3 Live Status Change Transparency

Diese Version macht Ampelwechsel im Live-Screener nachvollziehbar. Ein Wert kann
auch bei nahezu unveraendertem Kurs von Gelb auf Rot wechseln, wenn sich Trigger,
Timing, Konfluenz, Radar-Bucket, finale Freigabe oder ein hartes Einstiegsgate
aendern. Der konkrete Ausloeser wird jetzt direkt angezeigt und historisiert.

## Neu

- Spalte **Warum geändert?** in der Live-Screener-Haupttabelle
- Erklaerung direkt auf mobilen Screener-Karten
- Vergleich von Kurs und Live-Score zum vorherigen Scan
- Vergleich der Komponenten Timing, Konfluenz, Chart, Trigger, Trend und CRV
- klare Kennzeichnung neuer Invalidierungen und harter Einstiegsgates
- Statuswechsel-Historie enthaelt dieselbe Erklaerung
- Event-Log speichert den konkreten Ausloeser
- bestehende Hysterese, Mobile-Snapshots und Supabase-Speicherung bleiben erhalten

## Beispiel

```text
Kurs nahezu unverändert (+0,03 %). Auslöser: ein hartes Einstiegsgate wurde aktiv;
Radar-Bucket Nahe am Trigger→Warnsignale / meiden; finale Freigabe ist entfallen;
Live-Score 62→38 (-24); Timing 68→41.
```

## Deployment

1. Vollstaendigen ZIP-Inhalt in das Repository uebernehmen.
2. Keine SQL-Migration ausfuehren.
3. Streamlit-Secrets unveraendert lassen.
4. Nach zwei Live-Scans stehen Vergleichsdaten fuer die neue Erklaerung bereit.
5. GitHub Actions unter **v28.4.3 Quality Gate** kontrollieren.

Details: `RELEASE_NOTES_v28_4_3.md`.
