## v30.4b - Harvest / Chop Calibration
- Recalibrated the Short-Term Trader / Profit Harvest layer after real-scan review showed overly compressed green Harvest values.
- Added provider-free cross-sectional `Scan-Chop` from the complete Atomic watchlist frame: active breadth, RS deterioration breadth, weak/neutral RS breadth, volatility and signal stability.
- Added positive-market/weak-breadth divergence so a constructive headline regime cannot hide choppy participation underneath.
- RS deterioration now scales with the actual PP change; severe negative RS dynamics increase Chop/Harvest more strongly.
- Clean-trend Live-score discount is now guarded by RS, risk pressure and signal stability instead of applying too broadly.
- Harvest color thresholds remain unchanged at yellow >=60 and orange >=75; calibration changes the score inputs rather than moving the goalposts.
- Live Screener now exposes `Chop / Schwankung` next to the Harvest ampelfield and shows one full-scan Scan-Chop summary.
- Position Harvest receives the same scan context through a separate tactical frame; productive Live/Shadow/Portfolio semantics stay unchanged.
- Regression on the pasted 49-row scan: Scan-Chop 55/100, 7 yellow Harvest rows from visible fields only, while clean leaders stayed green; no orange state is forced.
- No new provider calls, orders, stops, productive score changes or automatic cutover.

## v30.4a - Trader-Ziel & Harvest-Ampel Highlighting
- Reiner UI-Patch fuer den v30.4 Short-Term-Trader-/Profit-Harvest-Pfad; Berechnungslogik unveraendert.
- Live-Screener ordnet Trader-Ziel, Harvest-Ampel und Trader-Modus direkt hinter Ticker/Name/Kurs ein.
- Trader-Ziel wird mit `⚡` hervorgehoben; Harvest-Score wird als eigene 🟠/🟡/🟢 Ampel dargestellt.
- Desktop-Spalten erhalten kompaktere Trader-/Harvest-Header.
- Mobile Live-Karten erhalten einen separaten hervorgehobenen Trader-Block.
- Ticker-Details zeigen Trader-Ziel, Harvest-Ampel und Trader-Modus in einer eigenen Infozeile.
- Keine Aenderung an Live-/Shadow-/Exit-/TP-Logik und keine zusaetzlichen Provider-Requests.

## v30.4 - Short-Term Trader & Profit Harvest Engine
- Neue additive, providerfreie Kurzfrist-Trader-/Profit-Harvest-Schicht; klassische TP-/Trendlogik bleibt unveraendert.
- Dynamisches Kurzfrist-Trader-Ziel aus Kurs/Entry, ATR, Markt-/Volatilitaetskontext, RS-Dynamik und bestehenden technischen Risikofeldern.
- Neue Scores `Harvest Score` und `Chop Risk` 0-100 sowie `Gewinn sichern ab`, Trader-Horizont und Teilgewinn-Idee.
- Live-Screener wird nach dem abgeschlossenen Atomic-Scan um Trader-Ziel, Sicherung, Harvest, Chop, Modus und Horizont angereichert; keine Zusatzrequests.
- Haupttabelle und Mobile-Karten zeigen den taktischen Pfad direkt neben der bestehenden Live-Sicht.
- Einzelanalyse/Trade-Plan zeigt Kurzfrist-Trader-Ziel parallel zu TP1/TP2/TP3; bestehende Ziel-/CRV-Berechnung wird nicht ersetzt.
- Positions-/Exit-Monitor verbindet den taktischen Pfad mit Profit Velocity, Exhaustion Risk und ausreichend belastbarem Giveback Risk.
- Starke Trendqualitaet reduziert Harvest-Druck; choppy/volatile Gewinnlagen koennen Teilgewinn 25-50% pruefbar machen, waehrend die Restposition separat weiterlaeuft.
- Aktive Hinweise werden dedupliziert als `Short-Term Profit Harvest` Event gespeichert und schaffen eine spaetere Lern-/Kalibrierungsbasis.
- Missing-data guard: kein aktueller Atomic-Kurs oder keine ATR-Basis erzeugt kein kuenstlich praezises taktisches Ziel.
- Keine Orders, keine Stop-Aenderung, keine automatische Schwellenkalibrierung und keine Aenderung an Live-/Shadow-/Exit-/Validated-Logik.

## v30.3j - Compact Responsive Board Metrics
- Globale `st.metric`-Darstellung fuer dichte Dashboard-/Board-Karten kompakter und responsiver gemacht.
- Lange Labels umbrechen jetzt mehrzeilig und werden kleiner dargestellt, statt in 5-/6-Spalten-Layouts abgeschnitten zu werden.
- Metric-Werte bleiben hervorgehoben, nutzen aber eine leicht reduzierte responsive Schriftgroesse.
- Delta-/Hinweiszeilen duerfen ebenfalls umbrechen und werden kompakter dargestellt.
- Desktop-Labels erhalten eine einheitliche Mindesthoehe fuer bessere Ausrichtung; auf Mobile wird diese wieder geloest.
- Wirkt app-weit auf vergleichbare `st.metric`-Felder, ohne Berechnungs-, Daten- oder Providerlogik zu veraendern.

## v30.3i - Automatic ECB FX Layer
- Portfolio-FX wird standardmaessig automatisch aus der offiziellen ECB-Euro-Referenzkurstabelle geladen.
- Ein einziger ECB-XML-Abruf deckt alle unterstuetzten Waehrungen ab; keine Einzelabfrage pro Position.
- 12-Stunden-Streamlit-Cache mit explizitem `ECB-FX jetzt aktualisieren`-Bypass.
- Cross-Rates gegen die gewaehlte Depot-Basiswaehrung werden aus den EUR-Referenzkursen berechnet.
- Erfolgreiche ECB-Snapshots werden als `portfolio_fx_ecb_last_good_v303i` persistent gespeichert.
- Bei ECB-Ausfall darf ein bis zu 7 Kalendertage alter Last-Good-Snapshot als klar markierter Fallback dienen; aeltere Snapshots werden nicht automatisch aggregiert.
- ECB-Referenzdatum, Alter und Quelle sind im Portfolio sichtbar; stale Fallback wird nicht stillschweigend als frisch dargestellt.
- Manuelle FX-Werte bleiben als explizite Overrides erhalten und haben Vorrang vor ECB-Automatik.
- Portfolio-Einstellungen speichern nur manuelle Overrides; automatische ECB-Kurse bleiben in einer separaten Last-Good-Schicht.
- Keine Yahoo-/Aktienprovider-Zusatzlast und keine Aenderung an Live-/Shadow-/Exit-/Positionslogik.

## v30.3h - Portfolio Coverage / FX Separation Fix
- Portfolio-Marktdaten-, Stop- und FX-Abdeckung werden nicht mehr miteinander vermischt.
- 3/3 aktuelle Atomic-Kurse ergeben 100% aktuelle Kursabdeckung, auch wenn eine Fremdwaehrungs-Umrechnung noch fehlt.
- Stop-Abdeckung wird direkt aus den offenen Positionen berechnet und bleibt von FX unabhaengig.
- Bei vollstaendigem explizitem FX-Pfad werden Investiert, Exposure, Cash und Risiko bis Stop aus derselben Positions-/Atomic-Basis reconciled.
- Falsche Alt-Treiber/Aktionen mit `Kursabdeckung 0%` oder `Stop-Abdeckung 0%` werden gegen die reale Positionsbasis korrigiert.
- Fehlendes FX blockiert nur Basiswaehrungs-Aggregate; der Portfolio-Risk-Score wird bis dahin als vorlaeufig statt als voll freigegebene Ampel dargestellt.
- Einwaehrungs-Portfolios zeigen auch ohne FX native Investitions- und Stop-Risikowerte.
- FX-Eingabebereich oeffnet sich automatisch, solange eine benoetigte Umrechnung fehlt; positive Eingaben wirken sofort.
- Keine zusaetzlichen Provider-Requests, keine FX-Schaetzung und keine Aenderung an produktiver Live-/Shadow-/Exit-Logik.

## v30.3g - Portfolio Data Bridge & Missing-Data Display Fix
- Portfolio-&-Risk-Anzeige stellt fehlende FX-/Kursdaten nicht mehr als scheinbar echte Nullwerte dar.
- Bei fehlender FX-Umrechnung werden Investiert, Exposure, Cash und Risiko bis Stop als `n/a` angezeigt statt irrefuehrend als 0.
- Neuer Datenstatus zeigt Anzahl offener Positionen, aktuelle Atomic-Kurse und Positionen mit nur gespeichertem Kurs.
- Native Positionswerte bleiben auch ohne FX transparent sichtbar, getrennt nach Waehrung und Kursbasis.
- Neue native Einzelpositionstabelle zeigt Watchlist, Ticker, Waehrung, Stueck, Entry, Stop, Kurs, Kursbasis, Positionswert und Stop-Risiko.
- Bereits vorhandene Atomic-Kurse werden ohne zusaetzlichen Provider-Call als `last_price`/Waehrung in den Positionsspeicher gespiegelt; fehlende Atomic-Kurse bleiben klar stale.
- Waehrungserkennung des Portfolio-Moduls wird mit der Positionssicht abgeglichen, damit fremde Waehrungen nicht verschwinden.
- Keine FX-Schaetzung, keine versteckten Yahoo-Requests und keine Aenderung an Portfolio-Risk-Score, Live-/Shadow-Ampel oder Exit-Engine.

## v30.3f - Rotation Drilldown Pandas-Series Crash Fix
- Streamlit-Crash beim Start eines Rotation-Aktien-Drilldowns behoben (`ValueError: The truth value of a Series is ambiguous`).
- Ursache: `_v303c_rotation_drilldown_context()` speicherte pandas-Series als Werte im internen `row_map`; spaeter wurde eine solche Series durch `... or {}` implizit als bool ausgewertet.
- `row_map` enthaelt jetzt ausschliesslich normale Python-Dictionaries.
- Die Drilldown-Speicherstelle konvertiert unerwartete Series zusaetzlich defensiv in Dictionaries und verwendet keine Bool-Auswertung von pandas-Objekten mehr.
- v30.3e Universe-Alignment bleibt unveraendert: alle Radar-Gruppen bleiben sichtbar; Gruppen mit Aktienkorb/Proxy koennen weiterhin on-demand geprueft werden.
- Keine Aenderung an Rotation-Score, Phase, Kandidaten-Score, Live-/Shadow-Ampel, Provider-Refresh oder Positions-/Exit-Logik.

## v30.3e - Rotation Drilldown Universe Alignment Fix
- Ursache der Abweichung 4x Emerging in der Radar-Uebersicht vs. nur 2x Emerging im Aktien-Drilldown behoben.
- Hauptuebersicht und Drilldown-Auswahl verwenden jetzt dieselbe komplette Radar-Population ueber alle Ebenen.
- Auswahl zeigt zusaetzlich die Radar-Ebene, damit Investmentklasse/Region/Sektor/Thema nicht verwechselt werden.
- Breiter Markt/Regionen SPY, QQQ, VGK, EWG und EEM erhalten repraesentative liquide Aktienkoerbe.
- GLD, DBC, USO und CPER erhalten klar als Aktien-Proxy markierte Koerbe; der Vergleich bleibt relativ zum jeweiligen Asset-/Commodity-ETF.
- Reine Bond-/Credit-Gruppen wie TLT/HYG bleiben sichtbar, werden aber bewusst als `kein direkter Aktienkorb` markiert statt fachlich fragwuerdige Aktienkandidaten zu erfinden.
- UI zeigt einen expliziten Radar-Abgleich `Emerging oben` vs. `Emerging hier` sowie Anzahl mit Aktienkorb.
- Bestehende Sektor-/Themenkoerbe werden nicht ueberschrieben; Erweiterungen greifen nur fuer bisher fehlende Gruppen.
- Keine Aenderung an Rotation-Score-/Phase-Formel, produktiver Live-/Shadow-Ampel, Exit-/Entry-Logik oder automatischen Provider-Requests.

## v30.3d - Rotation Radar Hard-Fresh Snapshot Fix
- Manueller Rotation-Radar-Refresh verwendet jetzt einen global eindeutigen `time_ns`-Nonce statt eines nach Reboot wiederverwendbaren Session-Zaehlers.
- Breadth- und Stock-Drilldown-Refresh erhalten ebenfalls eindeutige Nonces.
- Provider-Vollstaendigkeit prueft jetzt neben >=150 Bars auch den letzten Daily-Handelstag je Pflicht-Drilldown-Gruppe.
- Stale Sektor-/Themen-ETFs werden wie echte Datenluecken behandelt und bei kleiner Zahl provider-schonend einzeln nachgeladen.
- Neuer Radar-Snapshot wird nur bei vollstaendigen UND frischen Drilldown-Gruppen publiziert.
- Neuer persistenter Namespace `rotation_radar_snapshot_v303d` mit Schema `rotation-v30.3d-hard-fresh`.
- Snapshot-ID und Radar-Frame-Fingerprint werden nach jedem Schreiben sofort aus Storage rueckgelesen und verifiziert.
- UI rendert nach erfolgreichem Refresh aus dem verifizierten persistenten Readback-Frame.
- Legacy-Snapshots bleiben nur als klar markierter Fallback sichtbar, bis ein erfolgreicher v30.3d-Hard-Refresh erfolgt.
- Radar zeigt Daten-bis-Datum, Snapshot-ID und Persistenzquelle zur eindeutigen Reboot-Kontrolle.
- Keine Aenderung an produktiver Live-/Shadow-Ampel oder Rotation-Score-/Phase-Formel.

## v30.3c - Rotation Drilldown Snapshot Sync Fix
- `Rotation fuer Aktien-Drilldown` liest Phase/Rotation/Leadership jetzt garantiert aus demselben aktuell sichtbaren Radar-Frame wie die Haupttabelle.
- Selectbox nutzt dynamische echte Anzeige-Optionen statt stabiler Ticker plus `format_func`; Phasenwechsel wie Gelb -> Gruen werden sofort sichtbar.
- Separater persistenter Ticker-State behaelt die gewaehlte Gruppe trotz geaendertem Radar-Label bei.
- Snapshot-Fingerprint erkennt geaenderten Radar-Kontext und synchronisiert den Widget-State.
- Alle hinterlegten Drilldown-Gruppen bleiben in der Auswahl sichtbar; fehlende Gruppen werden explizit als Datenluecke markiert.
- Neuer Publish-Gate: kein neuer Radar-Snapshot, wenn eine drilldown-faehige Sektor-/Themengruppe fehlt, auch wenn die Gesamt-Abdeckung bereits >=85% ist.
- Bereits berechnete Aktien-Drilldowns werden bei geaendertem Radar-Kontext als veraltet markiert statt still als aktuell dargestellt.
- Keine zusaetzlichen automatischen Provider-Requests und keine Aenderung an Scoring-/Phasenformeln.

## v30.3b - Positions-Watchlist Recovery Fix
- Verbleibenden Positions-Watchlist-Fehler behoben: Typfilter versteckt keine gespeicherten Listen mehr.
- Alle bekannten Watchlists bleiben sichtbar; passende Typen werden nur noch priorisiert statt hart gefiltert.
- Auswahl zeigt den gespeicherten Watchlist-Typ direkt an.
- Falsch typisierte Altlisten koennen explizit im aktuellen Arbeitsbereich uebernommen werden.
- Neue persistente Recovery-Schicht fuer den Zustand "Backend bestaetigt Duplikat, Katalog liefert Liste aber nicht".
- Bei bestaetigtem Duplicate wird keine neue Liste erzeugt; der bestehende Name wird automatisch fuer die UI wiederhergestellt und ausgewaehlt.
- Ticker einer ausgewaehlten Liste werden primaer ueber `get_watchlist_tickers` geladen; globales `load_watchlists_df` ist nur Fallback.
- Recovery-Eintraege werden beim echten Loeschen der Watchlist ebenfalls entfernt.
- Keine Aenderung an Trading-, Exit-, Learning- oder Providerlogik.

## v30.3a - Positions-Watchlist Catalog Fix
- Operative Watchlist-Auswahl auf den echten Watchlist-Katalog (`get_watchlist_catalog_df`) umgestellt.
- Bereits angelegte, aber leere Positions-Watchlists bleiben sichtbar und auswählbar.
- `load_watchlists_df` bleibt nur als Legacy-/Fallback-Quelle fuer vorhandene Ticker-Zeilen.
- Historische/abweichende Typwerte werden auf `Watchlist` bzw. `Positions-Watchlist` normalisiert.
- Bei leerem altem Katalog-Typ kann ein expliziter Typ aus Ticker-Zeilen den Typ ergaenzen; explizite Katalogwerte haben sonst Vorrang.
- Getrennte Streamlit-Auswahlkeys fuer Watchlisten- und Positionsmodus verhindern State-Leaks beim Moduswechsel.
- Kandidaten-Radar nutzt fuer Ziel-Watchlists ebenfalls den robusten Katalog, inklusive leerer Listen.
- Keine Aenderung an Trading-/Exit-/Learning-Logik oder Providerzugriffen.

## v30.3 - Early Profit Learning & Calibration
- v30.2 Early-Profit-Warnungen werden sicher per Entry-/Warn-/Exit-Zeitfenster real geschlossenen Trades zugeordnet.
- Pro Trade zaehlt fuer die Statistik nur die erste Early-Profit-Warnung; spaetere Events uebergewichten den Trade nicht.
- Vergleich von Warn-R gegen final realisiertes Gesamt-R mit Delta-R und Giveback-R.
- Lernklassen: Gewinnschutz bestaetigt/stark bestaetigt, neutral, Laufenlassen besser/klar besser.
- Neue Segmentauswertung nach Early-Profit-Empfehlung, Profit-Velocity-Band und Exhaustion-Risk-Band.
- Historical Giveback Risk wird gegen den spaeter real beobachteten R-Giveback kalibriert.
- Persoenliche Lernhinweise markieren, ob Warnungen bisher eher hilfreich, zu frueh oder gemischt waren.
- Stichproben-Guard verhindert belastbare Aussagen bei zu wenigen Faellen.
- Keine Provider-Abfragen, keine hypothetischen Kurspfade und keine automatische Aenderung an v30.2-Empfehlungen, Scores, Stops oder Orders.

## v30.2 - Early Profit Protection & Giveback Engine
- Neue additive Fruehgewinn-Schutzschicht im Positions-/Exit-Monitor.
- Profit Velocity 0-100 aus P/L, Haltedauer, ATR-Normalisierung und R-Multiple.
- Exhaustion Risk 0-100 trennt Healthy Acceleration von Ueberdehnung/Ermuedung.
- Bestehende Atomic-Felder wie MA10-Abstand, Exit-/Trendbruch-/Momentum-/Distribution-/RS-Scores sowie Markt-/Volatilitaetsregime werden providerfrei wiederverwendet.
- Neue Empfehlungen: Healthy Acceleration, Gewinnschutz pruefen, Teilgewinn 25-50% pruefen und nur bei extremer technischer Bestaetigung Teilgewinn/Exit pruefen.
- On-Demand 5-Jahres-Fast-Move-Historie fuer die Aktie plus maximal vier kompakte Rotation-/Sektor-Peers.
- Historische Kennzahlen: Giveback >=50%, Ruecklauf zum Move-Start, direkter Follow-through, medianer Ruecksetzer und medianer weiterer Lauf.
- Historische Evidenz erhaelt Stichprobenstatus und beeinflusst aktuelle Empfehlungen nur bei kompatiblem Tempo/Haltefenster.
- Provider-Schutz: keine Zusatzrequests im normalen Atomic-/Auto-Scan; Historie nur auf expliziten Klick, gebuendelt.
- Defensive Early-Profit-Zustaende werden dedupliziert fuer die bestehende Learning-/Event-Schicht protokolliert.
- Keine automatische Order/Stop-Aenderung, keine Aenderung an Live-/Shadow-Ampel, Exit Engine 2.0 oder Cache-Schema.
- Keine SQL-/Secrets-Aenderung.

## v30.1d - Rotation Stock Drilldown
- Neuer On-Demand-Aktien-Drilldown direkt im Investment Rotation Radar.
- Emerging/Leading-Sektoren und -Branchen koennen in einen kompakten repraesentativen Aktienkorb aufgebohrt werden.
- Kandidaten-Ranking relativ zum jeweiligen Sektor-/Branchen-ETF statt nur relativ zum Gesamtmarkt.
- Neue Kennzahlen: Kandidaten-Score, Sektor-RS 5T/21T/63T, RS-Beschleunigung, Trend-Score, Entry-Readiness, MA20-Ueberdehnung und Abstand zum 20T-Hoch.
- Kandidatenlabels: Early Leader, Confirmed Leader, Rotation beschleunigt, technisch bereit, Leader aber ueberdehnt.
- Bestehender Atomic-Live-Screener wird nur lesend zur Anreicherung mit Live-/Shadow-Ampel, Live-Score, Guarded Score, CRV, RS-Dynamik, Setup-Alert und Einstiegsgates verwendet.
- Ticker ausserhalb des aktuellen Atomic-Screener-Standes werden nicht negativ bewertet, sondern transparent als 'nicht im Atomic Live-Scan' markiert.
- Drilldown ist rein beobachtend; keine Aenderung an produktiver Live-/Shadow-Logik, Scores oder Cutover-Gates.
- Provider-sicher: nur expliziter Klick und nur ein ausgewaehlter Branchen-/Sektorkorb; keine automatische Zusatzlast im Radar-Hauptscan.

# Changelog

## v30.1c
- Undercut-&-Rally-Pflichtgate eingefuehrt: echter Undercut plus Schlusskurs-Reclaim ist zwingend.
- Bullischer Tag, Docht, Volumen und RS koennen ohne bestandenen U&R-Kern keinen U&R-Score mehr erzeugen.
- Relevante lokale Swing-Lows werden vor dem starren 20T-Tief bevorzugt; 20T-Tief bleibt transparenter Fallback.
- Kurzer Shakeout wird auf ca. 0,3% bis 6,0% Undercut-Tiefe begrenzt; Reclaim muss ca. 0,2% ueber der Referenz schliessen.
- Same-Day-Reclaim und bestaetigter Folgetag werden getrennt ausgewiesen.
- U&R-spezifischer Trigger-Score verwendet Reclaim-/Folgetag-Qualitaet statt generischer Pivot-Punkte.
- U&R-Referenz, Referenztyp, Undercut- und Reclaim-Abstand werden transparent im Musterpaket gefuehrt.
- Keine Aenderung an produktiver Live-/Shadow-Ampel oder anderen Engine-Bausteinen.

## v30.1b
- Navigation des Trading-Cockpits aus dem bedingt gerenderten Live-Cache-Pfad herausgeloest.
- Eigener persistenter, nicht an ein Widget gebundener Cockpit-State verhindert Ruecksprung auf `Live-Screener`.
- Rotation Radar bleibt auch ohne kompatiblen/aktuellen Live-Screener-Snapshot erreichbar.
- Auto-Refresh liest den persistenten Cockpit-State und startet beim offenen Rotation Radar keinen Live-Vollscan.
- Leere Aktien-Watchlist blockiert den eigenstaendigen Rotation Radar nicht mehr.
- Bestehende Live-/Shadow-/Rotation-/Score-Logik unveraendert.

## v30.1a
- Live-Screener-Firmenname-Hardening: `Name = Ticker` wird nicht mehr als gueltiger Name akzeptiert.
- Persistente Ticker->Firmenname-Registry ueber zentrale Storage-Schicht.
- Firmenname-Fallback aus dem bereits gefuellten Analyse-/`load_data`-Cache statt sofortiger Yahoo-Suche.
- Provider-freier Offline-Fallback fuer bekannte App-Ticker/Aliasse, inklusive DELL, MSFT, NVDA und NFLX.
- Atomic-Snapshot-Anzeige repariert bekannte Namen ohne versteckte Provider-Abfragen.
- Yahoo Search nur noch letzter Fallback; temporaere Search-/Rate-Limits zerstoeren bekannte Namen nicht mehr.
- Keine Aenderung an Scores, Ampeln, Rotation Radar oder Atomic-Scan-Logik.

## v30.1
- Investment Rotation Radar als neuer Trading-Cockpit-Bereich eingefuehrt.
- Hierarchisches Universum aus Investmentklassen/Regionen, US-Sektoren und liquiden Branchen-/Themen-ETFs.
- Leadership Score 0-100 aus 63T-/21T-RS, 21T-Performance und MA20/50/200-Trendstruktur.
- Rotation Momentum 0-100 auf Basis der Beschleunigung von 5T-/21T-/63T-Relative-Staerke und kurzfristiger Eigenperformance.
- Rotationsphasen `Emerging`, `Leading`, `Mature`, `Cooling`, `Rotating Out`.
- Historische 1T-/5T-/20T-Deltas und Peer-Rangveraenderungen direkt aus dem gleichen Daily-Datensatz.
- Zweistufige Breadth Confirmation ueber repraesentative Sektor-/Branchenmitglieder.
- Breadth misst Anteil ueber MA20/MA50, positive 21T-Performance und positive 21T-RS.
- Atomic Radar Snapshot: ein unvollstaendiger neuer Lauf ersetzt den letzten gueltigen Stand nicht.
- Zentraler Storage fuer den letzten vollstaendigen Rotation-Snapshot.
- Provider-Schutz durch Batch-Download, begrenzte Retries, kleinen Einzel-Fallback, 30-Minuten-Cache und manuelle Breadth-Stufe.
- Mobile Kurzansicht fuer die wichtigsten Rotationssignale plus komplette Tabelle im Expander.
- Beobachtungsmodus: keine Aenderung an Live-/Shadow-Ampel, Scores, Guardrails, Positionen oder Orders.
- Keine SQL-/Secrets-Aenderung und keine Aenderung am Atomic-Live-Screener-Cache-Schema.

## v30.0
- Validated Trading Engine / Controlled Cutover als transparente Release-Gate-Schicht eingefuehrt.
- Produktive Live-Ampel bleibt Kontrollgruppe; v30.0 schaltet keine Engine automatisch um.
- Automatische Auswahl eines reifen Primaerhorizonts aus 1T / 3T / 5T / 10T / 20T.
- Harte Gates fuer Stichprobe, Gesamt-Edge, Richtungsbalance, Horizont-Stabilitaet, Kontextabdeckung, Guardrail-Nachweis und Marktregime-Abdeckung.
- Validation Score 0-100 als Orientierung; offene harte Gates koennen dadurch nicht ueberstimmt werden.
- Separate Freigabe-Matrix fuer Guarded Engine Score, Aufwertungen, Abwertungen, Guardrails, RS-Kontext, Regime, Trading Learning, Exit Engine 2.0 und Portfolio-Risikogate.
- Reale v29.0 Trade-Lerndaten werden als zusaetzliche Freigabe-Evidenz eingebunden, ohne alte Trades mit aktuellem Kontext aufzufuellen.
- Exit Engine 2.0 erhaelt einen eigenen Reifegrad anhand sicher zugeordneter Warnung-vs-Exit-Verlaeufe.
- v29.1 Portfolio-Gate wird als operative beratende Schutzschicht in der Cutover-Matrix sichtbar.
- Regime-Stabilitaet und Horizont-Stabilitaet werden separat im Dashboard angezeigt.
- Keine neuen Provider-Abfragen, keine Aenderung am Atomic-Screener-Cache-Schema, keine SQL-/Secrets-Aenderung.

## v29.1
- Portfolio & Risk Engine als eigener Trading-Cockpit-Bereich eingefuehrt.
- Gesamtdepot- und aktuelle-Watchlist-Sicht fuer offene Positionen.
- Exposure, Cash/Reserve, Einzelpositionsgewicht, Top-3- und Cluster-Konzentration.
- Portfolio-Risiko bis Stop sowie Stop-Abdeckung und Stop-Verletzungen.
- Exit Engine 2.0 wird als bestehender Positions-Risikokontext auf Portfolioebene aggregiert.
- Missing-Data-Guard verhindert gruene Portfolio-Freigabe bei unvollstaendiger Kurs-/FX-Basis.
- Doppelte Ticker ueber Positions-Watchlists werden im Gesamtdepot dedupliziert statt doppelt gezaehlt.
- Explizite Mehrwaehrungslogik; fehlende FX-Raten werden nicht geschaetzt.
- Depot-Basis, Depotwert und FX-Raten koennen ueber die zentrale Storage-Schicht persistent gespeichert werden.
- Persistente Portfolio-Gruppe/Sektor pro Position mit manueller Ueberschreibbarkeit.
- Konservative Cluster-Heuristik fuer klar erkennbare Gruppen; Unbekannt bleibt sichtbar.
- Pre-Trade Portfolio Guard simuliert Positionsgroesse, Exposure und Cluster-Konzentration vor einem neuen Trade.
- Keine automatische Aenderung an Positionen, Orders, Live-/Shadow-Ampeln oder Scores.
- Keine SQL-/Secrets-Aenderung und keine neuen Provider-Abfragen.

## v29.0
- Trading Journal & Learning Engine im Beobachtungsmodus eingefuehrt.
- Neue Positionen speichern einen strukturierten Entry-Kontext aus dem bereits abgeschlossenen Atomic-Live-Scan; keine Extra-Provider-Abfragen.
- Historische Positionen ohne Entry-Kontext werden nicht mit aktuellen Daten rueckwirkend aufgefuellt.
- Journalzeilen enthalten ab v29.0 Live-/Shadow-/Score-/Regime-/RS-/Guardrail-/Setup-Kontext als eigenstaendige Exportfelder.
- Learning-Datensatz mit einer Zeile pro gueltig geschlossenem Trade-Zyklus; Teilverkaeufe werden dem Full-Close zugerechnet.
- Rueckgaengig gemachte Fehlschliessungen bleiben Audit-Historie und werden nicht als abgeschlossene Trades gezaehlt.
- Kennzahlen: Trefferquote, Gesamt P/L, durchschnittliches/medianes R, Profit Factor, Kapitalrendite, Haltedauer und Kontextabdeckung.
- Segmentauswertung nach Radar-Bucket, Marktregime, Volatilitaet, RS-Dynamik, Live-Ampel, Shadow-vs-Live, Score-Baendern, Guardrail und Grade.
- Exit Engine 2.0 Lerncheck verknuepft Warnungen nur bei sicherem Entry-/Exit-Zeitfenster mit dem spaeteren Trade-Ergebnis.
- R-Veraenderung nach Erstwarnung und Warnvorlauf als neue Management-Lernmetriken.
- Wiederkehrende Themen aus manuellen Erkenntnis-Texten werden transparent gezaehlt.
- Stichproben-Guard verhindert, dass kleine Datenmengen als belastbare Kalibrierung erscheinen.
- Learning-Datensatz als CSV exportierbar.
- Keine automatische Aenderung an Live-/Shadow-Ampel, Scores, Guardrails, Positionen oder Orders.
- Keine SQL-/Secrets-Aenderung; Live-Cache-Schema absichtlich unveraendert.

## v28.9
- Positions-/Exit-Engine 2.0 fuer offene Long-Positionen eingefuehrt.
- Exit-Druck kombiniert Exit-Score, taktisches Risiko, Trendbruch, Momentum-Abbau, Distribution und relative Schwaeche.
- Marktregime, Volatilitaetsregime und RS-Dynamik als moderater Positionskontext integriert.
- Positionspuffer (P/L und R-Multiple) steuert, ob Warnungen eher Gewinnschutz/Teilgewinn oder Risikoabbau/Exit ausloesen.
- Harte Floors fuer Stop-Verletzung sowie sehr starken Exit-/Trendbruchdruck.
- Neue Fuehrungsstufen: Halten, Gewinnschutz, Stop enger, Teilgewinn, Risiko reduzieren, Exit pruefen.
- Stop-Status, Stop-Plan, Gewinnschutz-Plan, Add-on-Bremse und Datenkonfidenz transparent sichtbar.
- Technische Exit-Rohfelder werden ohne zusaetzliche Provider-Abfragen aus dem Atomic-Live-Scan an den Positionsmonitor weitergereicht.
- Kritischer UI-Fix: 'nur aktive' filtert nur noch die Screener-Ansicht; Positions-/Exit-Monitor sieht immer den vollstaendigen Atomic-Stand.
- Orange/rote Exit-Engine-Zustaende werden dedupliziert fuer die spaetere Learning Engine protokolliert.
- Live-Cache-Schema fuer die neuen Positionsfelder auf v28.9 angehoben.
- Keine Aenderung an Live-/Shadow-Ampel, Entry-Guardrails oder produktiven Score-Schwellen.

## v28.8
- Engine Calibration & Backtest als neuer Analysebereich im Shadow-Dashboard.
- Shadow-Edge eingefuehrt: Aufwertung erwartet positive, Abwertung negative Forward-Returns.
- Trefferquote fuer Shadow-Abwertungen korrigiert.
- 1T/3T/5T/10T/20T Horizon-Vergleich mit durchschnittlichem und medianem Edge.
- Directional MFE/MAE fuer favorable/adverse excursion ergaenzt.
- Divergenz-Zustandsaenderungen werden fuer die Kalibrierung zu Episoden de-clustert.
- Guarded-Score-Baender 0-27 / 28-54 / 55-71 / 72-100 separat auswertbar.
- Segmentauswertung nach Guardrails, RS-Dynamik, Marktregime und Volatilitaetsregime.
- Guardrail-Backtest fuer messbare Differenz zwischen Raw Engine Score und Guarded Engine Score.
- Kalibrierungsurteil mit Stichprobenstatus; keinerlei automatische Aenderung an Live-Logik.
- Neue Shadow-Ereignisse speichern erweiterten Kontext und technische Komponenten fuer spaetere Kalibrierung.
- Shadow-Performance bevorzugt nun die zentrale Storage-Schicht; lokales JSON bleibt Fallback.
- Nicht-Handelstag-Indizierung im Forward-Return-Tracking korrigiert.
- Performance-Refresh bleibt manuell und rate-limit-sicher.
- Atomic Complete Scan aus v28.7b unveraendert beibehalten.

## v28.7b
- Live-Screener auf Atomic Complete Scan umgestellt.
- Live-Screener scannt immer die komplette eindeutige Watchlist; 40/80/120-Teilmengen im Live-Screener entfernt.
- Teil-Batches werden nicht mehr in Session oder persistente Snapshots geschrieben.
- Alte Ergebniszeilen werden nicht mehr tickerweise in einen neuen Lauf hineingemischt.
- Persistente Restore-Logik akzeptiert nur v28.7b-Atomic-Vollstaende.
- Status-/Hysterese-Historie wird erst nach komplettem Rohscan atomar aktualisiert.
- Manueller und automatischer Vollscan erhalten frische Analyse-Keys ohne globales Cache-Clear.
- Provider-Drosselung mit kleineren Batches, Per-Ticker-Pause und zwei Cooldown-Retry-Runden fuer temporaere/429-Fehler.
- Heartbeat-Cache-Key korrigiert: `schema` wird nun identisch verglichen; verhindert faelschlich dauernd faellige Auto-Scans.
- Scan-Lock verhindert Auto-Rerun waehrend eines laufenden Vollscans.
- 5-Minuten-Auto-Cooldown nach komplett fehlgeschlagenem Lauf.
- Sichtbarer Scan-Status mit Datenzeit, Alter, Erfolg/Fehler, Dauer und `kein Mischstand`.
- Keine Aenderung an Ampel-, Score-, Shadow-, Guardrail-, Benchmark-, Positions- oder Journal-Logik.

## v28.7a
- Zweistufiger, bestaetigungspflichtiger Workflow fuer vollstaendige Positionsschliessungen.
- Vorschau vor dem Exit mit Ticker, Stueckzahl, Exit-Kurs und berechnetem P/L.
- Plausibilitaetswarnung und zweite Bestaetigung bei auffaelligem Ausstiegskurs.
- Neue Undo-Funktion fuer versehentlich geschlossene Trades.
- Ab v28.7a wird vor jedem Full-Close ein kompletter Positions-Snapshot fuer verlustfreies Undo gespeichert.
- Legacy-Schliessungen werden aus Journal- und Event-Historie rekonstruiert.
- Rueckgaengig gemachte Abschluesse werden aus P/L-, Trefferquoten- und Closed-Trade-Statistik neutralisiert, bleiben aber als Audit-Historie erhalten.
- Keine Aenderung an Screener-, Shadow-, Score-, Guardrail- oder Benchmark-Logik.

## v28.6e6
- aggressives globales Cache-Clear aus v28.6e5 entfernt
- manueller Vollrefresh provider-sicher gemacht
- letzter gueltiger Tickerstand bleibt bei temporaeren 429-Fehlern sichtbar
- manuelle Vollpruefung laeuft weiterhin ueber alle Ticker

## v28.7
- Shadow Performance Tracking mit 1T/3T/5T/10T/20T Forward Returns.
- Shadow-Ereignisse werden dedupliziert persistent protokolliert.
- Performance-Auswertung getrennt nach Aufwertung/Abwertung.
- Kursnachladen nur per explizitem Button, um Provider-Rate-Limits zu vermeiden.
- Keine Änderung an Live-Ampel, Shadow-Entscheidungslogik, Scores oder Guardrails.
