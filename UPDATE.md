# v30.1c - Strict Undercut & Rally Fix

v30.1c haertet den bestehenden Undercut-&-Rally-/Support-Reclaim-Baustein.

## Pflicht-Gate
Ein U&R ist nur noch aktiv, wenn tatsaechlich beides vorliegt:
- kurzer Undercut eines relevanten Swing-Lows/Supports,
- anschliessender Schlusskurs-Reclaim ueber dieser Referenz.

Bullischer Tag, unterer Docht, Volumen oder Relative Staerke duerfen das U&R nicht mehr alleine ueber die Aktivierungsschwelle heben.

## Relevanter Support statt nur starres 20T-Tief
- Bevorzugt wird ein lokales Swing-Low mit Struktur links/rechts.
- Nur kurze Shakeouts von ca. 0,3% bis 6,0% unter der Referenz gelten als U&R-Kandidat.
- Der Schlusskurs muss mindestens ca. 0,2% ueber der Referenz zurueckliegen.
- Falls kein sauberes Swing-Low verfuegbar ist, bleibt das bisherige 20T-Tief als transparenter Fallback erhalten.

## Qualitaetsbewertung nach bestandenem Pflicht-Gate
Erst nach echtem U&R werden Zusatzpunkte vergeben fuer:
- starken Schluss im Tagesrange,
- unteren Docht,
- bullische Tagesreaktion,
- Volumenbestaetigung,
- RS/Leadership-Bestaetigung,
- klaren Reclaim-Abstand.

## Folgetag
Ein U&R des Vortags bleibt nur dann als aktive Fortsetzung sichtbar, wenn der aktuelle Tag echte Folgestaerke zeigt. Ein Bruch ueber das Vortageshoch kann die Qualitaet weiter bestaetigen.

## Transparenz
Im Spezialmuster-Kontext werden U&R-Pflichtgate, Modus, Referenztyp, Referenzkurs, Undercut-Tiefe und Reclaim-Abstand getrennt gefuehrt. Die Orientierung wechselt bei aktivem U&R auf die konkrete Reclaim-/Swing-Low-Zone.

## Unveraendert
Keine Aenderung an produktiver Live-Ampel, Shadow-Ampel, Atomic Scan, Rotation Radar, Portfolio Engine, SQL oder Secrets. Der Spezialmuster-Baustein bleibt ein weicher Chart-/Trigger-Kontext.
