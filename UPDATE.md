# v30.15 - Volatility-Aware Risk Stop

v30.15 verbessert ausschließlich den Stop-Vorschlag im Bereich `📐 Risiko-Rechner`. Die produktive Setup-, Exit-, Portfolio-, Harvest-/Chop- und Orderlogik bleibt unverändert.

## Hintergrund
Bei volatilen Aktien konnte ein technisch plausibler Stop trotzdem innerhalb der normalen täglichen Schwankungsbreite liegen. Der Risiko-Rechner übernahm diesen engen technischen Stop direkt für die Positionsgrößenberechnung. Dadurch konnte die rechnerische Stückzahl relativ groß werden, während gleichzeitig die Wahrscheinlichkeit eines normalen Volatilitäts-Stopouts höher war.

## Neu
- Der bestehende technische Stop bleibt als `Technischer Stop` sichtbar.
- Zusätzlich berechnet der Risiko-Rechner einen `ATR-Schutz` aus ATR(14) in Prozent des Kurses.
- Der vorgeschlagene `Empfohlene Risiko-Stop` verwendet für Long-Setups immer den **weiter entfernten** Wert aus technischer Invalidierung und ATR-Schutz. Ein bereits weiter entfernter technischer Stop wird niemals künstlich enger gesetzt.
- Dynamischer ATR-Multiplikator:
  - ATR < 3,5%: 1,6 ATR,
  - ATR 3,5–<5%: 1,7 ATR,
  - ATR 5–<7%: 1,8 ATR,
  - ATR 7–<10%: 2,0 ATR,
  - ATR >=10%: 2,2 ATR.
- Zusätzlich bleibt ein Mindestpuffer von 3,5% im Risiko-Rechner bestehen.
- Der ATR-Schutz wird auf den tatsächlich gewählten `Geplanten Entry` neu berechnet.
- Neue Schaltfläche `Empfohlenen Risiko-Stop übernehmen` für den Fall, dass der Entry oder der Stop manuell verändert wurde.
- Wird ein Stop manuell enger als der ATR-Schutz gewählt, zeigt die App eine klare Warnung, überschreibt den Benutzerwert aber nicht.
- Wenn der ATR-Schutz den technischen Stop erweitert und der empfohlene Stop verwendet wird, zeigt die App den Effekt auf die Positionsgröße: enge technische Stückzahl -> ATR-geschützte Stückzahl.
- Ab 18% notwendigem Stop-Abstand wird nicht mehr empfohlen, den Stop nur für mehr Stücke enger zu setzen. Stattdessen: kleinere Positionsgröße oder besseren Entry/Pullback abwarten.
- Bei fehlendem ATR bleibt der technische Stop unverändert.
- Glossar ergänzt um `Volatility-Aware Risk Stop` und `ATR-Schutz / ATR-Puffer`.

## Beispiel
Entry 100, technischer Stop 96, ATR 6%:
- technischer Abstand: 4%,
- ATR-Multiplikator: 1,8,
- ATR-Schutzabstand: 10,8%,
- ATR-Schutzstop: 89,20,
- vorgeschlagener Risiko-Stop: 89,20.

Das maximale Depotrisiko bleibt z. B. bei 0,5%. Der weitere Stop führt daher zu weniger Stück statt zu mehr absolutem Risiko.

## Sicherheitsgrenzen
- v30.15 ändert keine bestehenden Positions-Stops automatisch.
- Keine Änderung an TP1/TP2/TP3, Exit-Engine, Entry-Gates oder Trade-Journal.
- Der Stop bleibt im Risiko-Rechner manuell überschreibbar.
- Extrem hohe rechnerische ATR-Abstände werden für die Darstellung bei 80% gekappt und als sehr volatil markiert.
- Keine neuen Provider-Aufrufe; die Risiko-Basis wird weiterhin nur beim expliziten Öffnen/Nutzen des Risiko-Rechners geladen und tickerbezogen gecacht.
