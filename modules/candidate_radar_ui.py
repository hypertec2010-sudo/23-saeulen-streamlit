"""Text-first candidate discovery UI; explanations collapsed, decisions visible."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import re

from . import candidate_radar as radar
from .radar_universe import get_universe_map, universe_notes, CATALOG_VERSION


def _md(value):
    # Provider text is data, not executable HTML or Markdown.
    return re.sub(r"([\\`*_{}\[\]<>#])", r"\\\1", str(value or "").replace("\n", " "))


def _fmt(value, digits=2):
    value = radar.number(value)
    return "n/a" if value is None else f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _records(loader):
    try:
        result = loader()
        data, error = result if isinstance(result, tuple) else (result, None)
        if data is None:
            return [], bool(error)
        rows = data.to_dict("records") if hasattr(data, "to_dict") else list(data)
        return rows, bool(error)
    except Exception:
        return [], True


def _render_error_diagnostics(st, errors, *, title="Fehlerdiagnose", expanded=False):
    summary = radar.summarize_scan_errors(errors)
    if not summary["count"]:
        return
    if summary["categories"]:
        parts = [f"{count}x {category}" for category, count in summary["categories"][:5]]
        st.write("**H\u00e4ufigste Ursachen:** " + " \u00b7 ".join(parts))
    with st.expander(title, expanded=expanded):
        if summary["stages"]:
            st.caption("Fehlerstufe: " + " \u00b7 ".join(f"{count}x {stage}" for stage, count in summary["stages"][:4]))
        if summary["types"]:
            st.caption("Fehlerklassen: " + " \u00b7 ".join(f"{count}x {kind}" for kind, count in summary["types"][:6]))
        for row in summary["examples"]:
            st.write(
                f"**{_md(row['ticker'])}** \u00b7 {_md(row['stage'])} \u00b7 {_md(row['category'])} \u00b7 "
                f"{_md(row['detail'])} ({_md(row['error_type'])})"
            )
        remaining = summary["count"] - len(summary["examples"])
        if remaining > 0:
            st.caption(f"+ {remaining} weitere Fehler.")


def _select(st, label, options, key, default=None):
    # Stale widget choices cannot resurrect hidden inputs after a scan change.
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    idx = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=idx, key=key)


def render_result_card(st, row, membership="Neu im erfassten Bestand"):
    ticker = _md(row["ticker"])
    st.markdown(f"**{row.get('rank_at_discovery', '-')}. {ticker} \u00b7 {_md(row['name'])}**")
    st.markdown(f"**{_md(row['status'])}** \u00b7 Qualit\u00e4tsnote **{_md(row['grade'])}** \u00b7 {_md(membership)}")
    st.write("Auff\u00e4llig: " + row.get("why", ""))
    if row.get("gates"):
        st.warning("Sperre: " + "; ".join(row["gates"]))
    st.write("N\u00e4chster Schritt: " + row.get("next_step", ""))
    provenance = row.get("target_provenance") or {}
    st.write(f"Screener-CRV {_fmt(row.get('crv'))} \u00b7 Zielbasis: {provenance.get('label', 'unbekannt')} \u00b7 "
             f"Branche: {row.get('sector') or 'unbekannt'} \u00b7 {row.get('asset_type', 'Unbekannt')}")
    cov = "unbekannt" if row.get("coverage") is None else _fmt(row["coverage"] * 100, 0) + " %"
    st.caption(f"Kursdaten: {row.get('quote_date') or 'unbekannt'} \u00b7 Datenabdeckung: {cov}")


def _navigate(st, tickers):
    joined = "\n".join(tickers)
    state = st.session_state
    state["batch_input"] = joined
    state["batch_input_widget_main"] = joined
    state["analysis_mode"] = "Mehrere Aktien vergleichen"
    state["analysis_mode_run"] = "Mehrere Aktien vergleichen"
    state["analysis_mode_widget_main"] = "Mehrere Aktien vergleichen"
    state["workspace_mode"] = "Sofortanalyse"
    # Do not set run_analysis: switching pages must not launch a scan/order.
    state["run_analysis"] = False
    state["analysis_requested"] = False
    state.pop("ranking_results", None)
    state.pop("ranking_df", None)
    state["position_perspective_widget_main"] = "Pre-Entry / Watchlist"
    state["buy_in_widget_main"] = 0.0
    if callable(getattr(st, "switch_page", None)):
        st.switch_page("pages/analysis.py")
    else:
        state["fallback_navigation_v282"] = "Sofortanalyse"
        st.rerun()


def render_candidate_radar(st, *, storage, scan, catalog_loader, watchlists_loader,
                           positions_loader, queue_watchlist, model_version=radar.RADAR_VERSION,
                           clock=radar.utcnow):
    st.subheader("Kandidaten-Radar \u00b7 Ideen entdecken")
    st.caption(f"Radar {radar.RADAR_VERSION} \u00b7 Beobachtungsidee \u2192 Watchlist \u2192 frischer Live-Scan \u2192 Paketplanung")
    maps = get_universe_map()
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        universe = _select(st, "Kandidatenliste", list(maps) + ["Eigene Liste"], "radar_v3021_universe", st.session_state.get("radar_universe", "US Tech"))
    with c2:
        style = _select(st, "Suchprofil", list(radar.STYLES), "radar_v3021_style", st.session_state.get("radar_screening_style", "Leader"))
    with c3:
        limit = _select(st, "Karten anzeigen", [5, 10, 15, 20], "radar_v3021_limit", 5)
    st.session_state["radar_universe"] = universe
    st.session_state["radar_screening_style"] = style
    custom = ""
    if universe == "Eigene Liste":
        custom = st.text_area("Ticker oder Firmennamen, mit Zeilenumbruch oder Komma getrennt", key="radar_v3021_custom",
                              value=st.session_state.get("radar_custom_input", ""))
        st.session_state["radar_custom_input"] = custom
        entries = [e.strip() for e in re.split(r"[\n,;]+", custom) if e.strip()]
    else:
        entries = list(maps[universe][0])
    with st.expander("\u2139\ufe0f Suchprofil, Kandidatenliste & Methodik", expanded=False):
        st.write("Feste Ausgangslisten, keine vollst\u00e4ndige oder laufend gepflegte Marktsuche. "
                 "Bezeichnungen wie Small/Mid Cap sind Suchlisten, keine aktuellen Gr\u00f6\u00dfenfilter.")
        st.write("Leader: best\u00e4tigte St\u00e4rke. Charttechnik: Struktur und Impulse. "
                 "Turnaround: fr\u00fche Drehversuche. Ausgewogen: gemischtes Suchprofil. "
                 "Die bestehenden Noten und Scores bleiben unver\u00e4ndert und sind keine Gewinnwahrscheinlichkeiten.")
        st.write("Qualit\u00e4t ist nicht Einstiegsfreigabe. Sperren bleiben unabh\u00e4ngig von der Note wirksam. "
                 "CRV-Zielherkunft wird beschrieben, keine Ziel- oder Stopformel ge\u00e4ndert. "
                 "Nicht gelieferte MTF-/Wellenpr\u00fcfungen gelten als nicht berechnet.")
        st.write("Kursdaten d\u00fcrfen f\u00fcr die Kategorie 'Im Live-Screener pr\u00fcfen' h\u00f6chstens "
                 f"{radar.MAX_QUOTE_AGE_DAYS} Kalendertage alt sein. Das ist keine Zusicherung eines Echtzeitkurses.")
        max_age = st.number_input("Scan-Frischegrenze in Stunden", min_value=1, max_value=72, value=24, step=1, key="radar_v3021_max_age")
        st.write(f"Listenpflege: {CATALOG_VERSION} \u00b7 {len(entries)} Eingaben. "
                 "Nur dokumentierte Symbolkorrekturen; weitere Providerfehler werden ausgewiesen.")
        st.code(", ".join(entries) or "Keine Eingaben")
        for note in universe_notes(universe):
            st.write(f"{note['input']}: {note['action']} \u2014 {note['detail']}")
    st.write(f"**Scanumfang: {len(entries)} Listeneintr\u00e4ge** \u00b7 Anzeigelimit {limit} begrenzt nur die Karten, nicht die Analyse.")
    key = radar.request_key(universe, style, entries, model_version)
    owner = radar.text(getattr(storage, "user_id", None), "no-user")
    session_key = "radar_v3021_result_" + owner + "_" + key
    payload = st.session_state.get(session_key)
    if not radar.snapshot_valid(payload, key):
        payload = radar.load_snapshot(storage, key)
        if payload:
            st.session_state[session_key] = payload
    if st.button("Gespeicherten Scan neu laden (ohne Analyse)", key="radar_v3021_reload"):
        saved = radar.load_snapshot(storage, key)
        if saved is not None:
            payload = saved
            st.session_state[session_key] = saved
            st.session_state.pop(session_key + "_save", None)
        else:
            st.info("Kein passender benutzerbezogener Scan gespeichert. Der Sitzungsstand bleibt unveraendert.")
    if st.button("Kandidatenliste jetzt scannen", key="radar_v3021_run", type="primary", disabled=not entries):
        bar = st.progress(0.0)
        status_line = st.empty()
        fresh = None
        def progress(done, total, ticker):
            bar.progress(done / total)
            status_line.write(f"{done}/{total} gepr\u00fcft \u00b7 {ticker}")
        try:
            with st.spinner("Gemeinsame Aktienanalyse l\u00e4uft ..."):
                fresh = scan(universe=universe, style=style, entries=entries, source="manual", progress=progress)
            if not radar.snapshot_valid(fresh, key) or not fresh.get("rows"):
                raise ValueError("Vollst\u00e4ndiger Radar-Scan fehlt")
            payload = fresh
            st.session_state[session_key] = fresh
            ok, message = radar.save_snapshot(storage, fresh)
            st.session_state[session_key + "_save"] = (ok, message)
        except Exception as exc:
            if isinstance(fresh, Mapping) and fresh.get("errors"):
                processed = int(fresh.get("processed") or fresh.get("requested") or len(entries))
                successful = len(fresh.get("rows") or [])
                failed = len(fresh.get("errors") or [])
                st.error(
                    f"Neuer Radar-Lauf fehlgeschlagen: {successful}/{processed} Werte erfolgreich, {failed} Fehler. "
                    "Ein vorhandener Scan bleibt unver\u00e4ndert."
                )
                _render_error_diagnostics(
                    st, fresh.get("errors"), title="Fehlerbeispiele aus diesem Scan", expanded=True
                )
            else:
                info = radar.safe_scan_error(exc)
                st.error(
                    "Neuer Radar-Lauf fehlgeschlagen (" + info["error_type"] + "). "
                    "Ein vorhandener Scan bleibt unver\u00e4ndert."
                )
                st.write("**Ursache:** " + info["category"] + " \u00b7 " + info["detail"])
        finally:
            bar.empty()
            status_line.empty()
    save_status = st.session_state.get(session_key + "_save")
    if save_status:
        ok, message = save_status
        if ok and "nur lokal" not in message:
            st.success(message)
        else:
            st.warning(message)
    if payload is not None and not save_status:
        try:
            storage_status = storage.status()
        except Exception:
            storage_status = {}
        if storage_status.get("degraded") or str(storage_status.get("last_backend", "")).startswith("local"):
            st.warning("Radar aus lokalem Ersatzspeicher; dauerhafte Cloud-Speicherung nicht bestaetigt.")
    if payload is None:
        st.info("F\u00fcr diese Liste und dieses Profil liegt noch kein neuer Radar-Scan vor. "
                "Einmal bewusst scannen. Alte gemeinsame JSON-Snapshots werden aus Sicherheitsgr\u00fcnden "
                "nicht automatisch einem Nutzer zugeordnet; Watchlists und Trades bleiben erhalten.")
        return
    fresh, freshness_text = radar.snapshot_freshness(payload, now=clock(), max_scan_hours=max_age)
    st.write(f"**Scanstand:** {payload['completed_at']} \u00b7 {payload['scan_id']} \u00b7 Quelle: {payload['source']}")
    if fresh:
        st.caption(freshness_text)
    else:
        st.warning(freshness_text + " Historische Ideen bleiben lesbar, aber nicht als aktueller Einstieg bezeichnet.")
    wl_rows, wl_failed = _records(watchlists_loader)
    catalog, catalog_failed = _records(catalog_loader)
    try:
        positions = positions_loader() or {}
        positions_failed = False
    except Exception:
        positions, positions_failed = {}, True
    if wl_failed or positions_failed:
        st.warning("Bestandsabgleich unvollst\u00e4ndig: 'neu' ist nicht best\u00e4tigt. Watchlist/Positionen separat pr\u00fcfen.")
    watched, held, pending = radar.known_tickers(wl_rows, positions, st.session_state.get("pending_watchlist_adds_v228", []))
    rows = radar.view_rows(payload, now=clock(), max_scan_hours=max_age)
    counts = Counter(r["status"] for r in rows)
    cs = st.columns(5)
    for col, label, value in zip(
        cs,
        ("Ausgewertet", "Live-Pr\u00fcfung m\u00f6glich", "Trigger abwarten", "Beobachten / kein Trade-Plan", "Gesperrt / Daten fehlen"),
        (f"{len(rows)}/{payload['requested']}", counts[radar.READY], counts[radar.NEAR], counts[radar.NO_PLAN], counts[radar.BLOCKED]),
    ):
        col.metric(label, value)
    if not fresh:
        st.write(f"**Historische Beobachtungen: {len(rows)}. Keine aktuelle Radar-Freigabe.**")
    else:
        st.write(f"Weitere Beobachtungsideen: {counts[radar.WATCH]} \u00b7 Abfragefehler: {len(payload['errors'])}")
    if payload["errors"]:
        st.warning("Nicht auswertbar: " + ", ".join(e["ticker"] for e in payload["errors"]))
        _render_error_diagnostics(st, payload["errors"], title="Abfragefehler im Detail", expanded=False)
    mtf_missing = sum(not r.get("mtf_available") for r in rows)
    wave_missing = sum(not r.get("wave_available") for r in rows)
    coverage_missing = sum(r.get("coverage") is None for r in rows)
    if coverage_missing:
        st.warning(f"Datenabdeckung unbekannt: {coverage_missing}/{len(rows)} Wert(e).")
    no_plan_rows = [r for r in rows if r.get("status") == radar.NO_PLAN]
    if no_plan_rows:
        symbols = [r["ticker"] for r in no_plan_rows]
        st.write(f"**Trade-Plan noch nicht erzeugt ({len(no_plan_rows)}):** "
                 + ", ".join(symbols[:15]) + (" ..." if len(symbols) > 15 else "")
                 + " \u00b7 Setup noch nicht valide; fehlende Entry-/Stop-/CRV-Werte gelten hier nicht als Datenfehler.")
    gate_counts = Counter(g for r in rows for g in r.get("gates", []))
    for reason, count in gate_counts.most_common(3):
        symbols = [r["ticker"] for r in rows if reason in r.get("gates", [])]
        st.write(f"**{_md(reason)} ({count}):** " + ", ".join(symbols[:15]) + (" ..." if len(symbols) > 15 else ""))
    with st.expander("Scan-Details, vollst\u00e4ndige Ergebnisse & Export", expanded=False):
        st.write(f"Vollst\u00e4ndig durchlaufen: {payload['processed']}/{payload['requested']}. "
                 "Ausgewertet umfasst auch gesperrte Kandidaten; es bedeutet nicht kaufbar.")
        if rows:
            if mtf_missing == len(rows) and wave_missing == len(rows):
                st.caption("Zusatzanalysen MTF/Welle: im schnellen Radar-Scan nicht aktiv.")
            else:
                st.caption(f"Zusatzanalysen: MTF verf\u00fcgbar {len(rows)-mtf_missing}/{len(rows)} \u00b7 "
                           f"Welle verf\u00fcgbar {len(rows)-wave_missing}/{len(rows)}.")
        for error in payload["errors"]:
            st.write(f"{error['ticker']}: {error['reason']}")
        for note in payload["resolution"]:
            st.write(f"{note['input']}: {note['action']} \u2014 {note['detail']}")
        table = [{
            "Rang": r.get("rank_at_discovery"), "Ticker": r["ticker"], "Name": r["name"],
            "Status": r["status"], "Note": r["grade"], "Score": r["score"], "CRV": r["crv"],
            "Zielherkunft": r["target_provenance"]["label"], "Kursdatum": r["quote_date"],
            "Abdeckung": r["coverage"], "Sperren": "; ".join(r["gates"]),
            "Trade-Plan": "; ".join(r.get("trade_plan_reasons", [])),
            "Entry-Zone": r["entry_zone"], "Naechster Schritt": r["next_step"],
        } for r in rows]
        try:
            st.dataframe(table, use_container_width=True, hide_index=True)
        except Exception:
            st.info("Detailtabelle nicht darstellbar; Textkarten und JSON-Export bleiben verfuegbar.")
        st.download_button("Radar-Scan als JSON exportieren", data=json.dumps(radar.json_safe(payload), ensure_ascii=False, indent=2),
                           file_name=payload["scan_id"] + ".json", mime="application/json", key="radar_v3021_export")
    st.markdown("### Priorisierte Beobachtungskandidaten")
    # Rank all rows before any display or status filter. No group can consume a
    # hidden analytical limit before another group is evaluated.
    include_known = st.checkbox("Bereits erfasste / vorgemerkte Werte mit anzeigen", value=False, key="radar_v3021_include_known")
    eligible_view = rows if include_known else [r for r in rows if r["ticker"] not in watched | held | pending]
    labels = ["Alle Ergebnisse"] + [v for v in (radar.READY, radar.NEAR, radar.NO_PLAN, radar.WATCH, radar.BLOCKED, radar.HISTORY) if any(r["status"] == v for r in eligible_view)]
    status_filter = _select(st, "Ergebnisgruppe", labels, "radar_v3021_filter")
    filtered = [r for r in eligible_view if status_filter == "Alle Ergebnisse" or r["status"] == status_filter]
    st.write(f"**{len(filtered)} Treffer im Filter \u00b7 {min(limit, len(filtered))} Karten sichtbar**. "
             f"Bereits erfasst/vorgemerkt im Scan: {sum(r['ticker'] in watched | held | pending for r in rows)}.")
    if not filtered:
        st.info("Keine Kandidaten in dieser Ansicht. Bereits erfasste Werte oder andere Ergebnisgruppe einblenden.")
    for row in filtered[:limit]:
        t = row["ticker"]
        membership = "Offene Position" if t in held else "Bereits vorgemerkt" if t in pending else "Auf Watchlist" if t in watched else "Bestandsstatus ungepr\u00fcft" if wl_failed or positions_failed else "Neu im erfassten Bestand"
        with st.container(border=True):
            render_result_card(st, row, membership)
    all_tickers = [r["ticker"] for r in filtered]
    select_key = "radar_v3021_selected_" + key
    if select_key in st.session_state:
        st.session_state[select_key] = [t for t in st.session_state[select_key] if t in all_tickers]
    selected = st.multiselect("Kandidaten bewusst ausw\u00e4hlen (alle Treffer des Filters)", all_tickers, default=[], key=select_key)
    st.markdown("### Auswahl weiterverwenden")
    options = {}
    for c in catalog:
        name = radar.text(c.get("Watchlist_Name"))
        kind = radar.text(c.get("Watchlist_Type"), "Watchlist")
        if name:
            options[f"{name} \u00b7 {kind}"] = (name, kind)
    target = _select(st, "Ziel-Watchlist", list(options), "radar_v3021_target") if options else None
    if not options or catalog_failed:
        st.warning("Ziel-Watchlists fehlen oder konnten nicht vollst\u00e4ndig geladen werden.")
    if not fresh:
        st.info("Eine \u00dcbernahme historischer Ideen erzeugt nur Beobachtungseintr\u00e4ge. Vor jedem Einstieg frisch pr\u00fcfen.")
    a, b, c = st.columns(3)
    with a:
        if st.button("In Sofortanalyse laden", key="radar_v3021_analysis", disabled=not selected):
            _navigate(st, selected)
    with b:
        if st.button("F\u00fcr Watchlist vormerken", key="radar_v3021_queue", disabled=not selected or not target or wl_failed or catalog_failed):
            name, kind = options[target]
            existing = {radar.text(r.get("Ticker")).upper() for r in wl_rows if radar.text(r.get("Watchlist_Name")).lower() == name.lower()}
            queued = {radar.text(r.get("Ticker")).upper() for r in st.session_state.get("pending_watchlist_adds_v228", []) if radar.text(r.get("Watchlist_Name")).lower() == name.lower()}
            new = [t for t in selected if t not in existing | queued]
            if not new:
                st.info("Auswahl bereits auf dieser Watchlist oder vorgemerkt; keine doppelte \u00dcbernahme.")
            else:
                try:
                    ok, message = queue_watchlist(name, kind, new, source="Kandidaten-Radar | " + payload["scan_id"],
                                                 check_frequency=st.session_state.get("selected_watchlist_check_frequency", "4x t\u00e4glich"), existing_tickers=list(existing))
                except Exception:
                    ok, message = False, "Watchlist-Vormerkung fehlgeschlagen; vorhandene Queue bitte pruefen."
                if ok:
                    audit_ok = radar.save_discoveries(storage, payload, name, new)
                    st.session_state["radar_v3021_handoff_" + owner] = (message, audit_ok)
                else:
                    st.error(message)
    with c:
        if st.button("Zu Watchlisten wechseln", key="radar_v3021_watchlists"):
            if callable(getattr(st, "switch_page", None)):
                st.switch_page("pages/watchlists.py")
            else:
                st.session_state["workspace_mode"] = "Watchlisten"
                st.session_state["fallback_navigation_v282"] = "Watchlisten"
                st.rerun()
    handoff = st.session_state.get("radar_v3021_handoff_" + owner)
    if handoff:
        st.success("Letzte Radar-Übergabe: " + handoff[0])
        st.info("Nur Beobachtungseintr\u00e4ge vorgemerkt. Im Watchlisten-Bereich noch geb\u00fcndelt speichern. Keine Order, Position oder REAL-Buchung erzeugt.")
        if not handoff[1]:
            st.warning("Radar-Herkunft konnte nicht dauerhaft gesichert werden. Die Watchlist-Vormerkung bleibt in dieser Sitzung erhalten.")
