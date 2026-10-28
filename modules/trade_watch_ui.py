"""Standalone trade-watcher UI. Deliberately never imports legacy_app/analyzers."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd

from modules import trade_watch as watch
from modules.storage.manager import create_storage_manager

RULES = {
    "5-Minuten-Schluss": "close5", "15-Minuten-Schluss": "close15",
    "Kurs erreicht (1m-Momentaufnahme)": "contact",
}


def shown_time(value):
    try:
        return watch.utc(value).tz_convert("Europe/Berlin").strftime("%d.%m. %H:%M:%S") if value else "–"
    except (ValueError, TypeError):
        return "Unbekannt"


def capture_form(st, basis=None, *, key="watch_setup", manager=None):
    """Read-only structural levels; only an actual detector package can be saved."""
    from copy import deepcopy
    from modules.watch_setups import validate, form_id
    basis = basis or {}
    package = basis.get("watch_setup")
    ok, reason = validate(package, now=watch.utc(), ticker=basis.get("ticker"))
    if not ok:
        st.info("Kein überwachbares Chartsetup vorhanden.")
        st.caption(reason)
        return
    prefix = key + "_" + form_id(package)
    st.markdown("**" + package["label"] + " · " + package["ticker"] + "**")
    st.caption("Erkannt aus abgeschlossenen Tageskerzen bis " + package["basis_date"] +
               " · erneute Chartprüfung spätestens nach Ablauf am " + shown_time(package["valid_until"]))
    currency = package.get("currency") or "Kurseinheit der Analyse"
    st.table(pd.DataFrame([
        {"Regel": "Entry-Auslöser", "Level": package["entry"], "Einheit": currency},
        {"Regel": "Oberste Entry-Grenze", "Level": package["max_entry"], "Einheit": currency},
        {"Regel": "Struktur-Invalidierung", "Level": package["invalidation"], "Einheit": currency},
        {"Regel": "Technisches Ziel (optional)", "Level": package.get("target"), "Einheit": currency}]))
    st.caption("Einstieg erst nach neuem 5-Minuten-Schluss innerhalb der Entry-Grenzen. Invalidierung bei 1m-Low auf/unter dem Strukturlevel. Levels werden aus dem erkannten Setup übernommen und sind hier nicht frei änderbar.")
    st.caption("Diese Invalidierung gehört zum erkannten Chartsetup. Der allgemeine Analyse-Stop wird dadurch nicht verändert.")
    crv = package.get("crv")
    st.caption("CRV zum optionalen Ziel: " + (f"{crv:.2f}R" if crv is not None else "nicht bewertbar") +
               " · separat beurteilen; verändert die Setup-Erkennung nicht.")
    with st.expander("Warum ist das ein Chartsetup?", expanded=False):
        for check in package.get("checks", []):
            st.write("✓ " + check)
        st.caption(package.get("stop_source") or "")
        chart = package.get("chart") or []
        if chart:
            import plotly.graph_objects as go
            frame = pd.DataFrame(chart)
            fig = go.Figure(go.Candlestick(x=frame.date, open=frame.Open, high=frame.High,
                                          low=frame.Low, close=frame.Close, name="Abgeschlossene Tageskerzen"))
            for label, field, color in [("Entry", "entry", "#2e7d32"),
                                         ("Entry maximal", "max_entry", "#ef6c00"),
                                         ("Invalidierung", "invalidation", "#c62828")]:
                fig.add_hline(y=package[field], line_color=color, line_dash="dot", annotation_text=label)
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
                              yaxis_title=currency, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=prefix + "_chart")
        st.json(package.get("evidence") or {})
    st.caption("Dies ist eine eng definierte, noch nicht auf Profitabilität validierte Chart-Erkennung. Andere Muster werden derzeit nicht übernommen. Die allgemeine Analyse-Ampel wird dadurch nicht geändert.")
    with st.form(prefix + "_form"):
        note = st.text_input("Eigene Notiz (optional)", key=prefix + "_note", max_chars=500)
        reviewed = st.checkbox("Erkanntes Chartsetup und abgeleitete Regeln angesehen", key=prefix + "_reviewed")
        save = st.form_submit_button("Chartsetup im Wächter verfolgen", type="primary")
    if save:
        try:
            if not reviewed:
                raise ValueError("Bitte Chartstruktur und Regeln vor der Übernahme ansehen.")
            source = deepcopy(basis)
            source["watch_setup"] = deepcopy(package)
            idea = watch.make_idea(ticker=package["ticker"], entry=package["entry"],
                invalidation=package["invalidation"], max_entry=package["max_entry"], mode=package["mode"],
                expires_at=package["valid_until"], target=package.get("target"), currency=package.get("currency"),
                source=source, note=note)
            storage = manager or create_storage_manager(st_module=st, app_dir=Path(__file__).resolve().parents[1])
            watch.add_idea(storage, idea)
            st.success(package["ticker"] + ": erkanntes Chartsetup im Trade-Wächter gespeichert.")
            if storage.degraded:
                st.warning("Nur lokaler Ersatzspeicher verfügbar; geräteübergreifende Speicherung derzeit nicht bestätigt.")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))


def capture_analysis(st, result, *, hard_gate=False, key="watch_analysis"):
    from modules.risk_calculator import build_analysis_risk_basis
    from modules.watch_setups import detect
    basis = build_analysis_risk_basis(result)
    basis["watch_setup"] = detect(result, hard_gate=hard_gate)
    capture_form(st, basis, key=key)


def render_capture(st, frame, *, key="watch_screener"):
    """Capture from already analyzed screener rows, including restored snapshots."""
    from modules.risk_calculator import risk_basis_from_row
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Ticker" not in frame:
        return
    with st.expander("👁️ Trade-Idee verfolgen", expanded=False):
        from modules.watch_setups import validate
        rows = frame.to_dict("records")
        eligible = [row for row in rows if validate(row.get("__watch_setup"), now=watch.utc(), ticker=row.get("Ticker"))[0]]
        if not eligible:
            st.info("Kein überwachbares Chartsetup vorhanden.")
            if not any(isinstance(row.get("__watch_setup"), dict) for row in rows):
                st.caption("Dieser Scan enthält noch keine konkrete Chartsetup-Prüfung. Gewünschte Aktien einmal neu scannen.")
            else:
                with st.expander("Warum wurde kein Setup freigegeben?", expanded=False):
                    st.dataframe(pd.DataFrame([{"Ticker": row.get("Ticker"), "Grund": validate(row.get("__watch_setup"), now=watch.utc(), ticker=row.get("Ticker"))[1]} for row in rows]), hide_index=True, use_container_width=True)
            return
        index = st.selectbox("Erkanntes Chartsetup", range(len(eligible)),
            format_func=lambda i: str(eligible[i]["Ticker"]) + " · " + eligible[i]["__watch_setup"]["label"], key=key + "_row")
        row = eligible[index]
        basis = risk_basis_from_row(row)
        if basis is None:
            st.info("Die Analysebasis dieses Scans fehlt. Aktie einmal neu scannen.")
            return
        basis["watch_setup"] = row["__watch_setup"]
        capture_form(st, basis, key=key)


def table_rows(ideas, *, now=None):
    current = watch.utc(now)
    rows = []
    for idea in ideas:
        price = watch.number(idea.get("price"))
        distance = (price / idea["entry"] - 1) * 100 if price is not None else None
        state = "Archiviert" if idea.get("archived") else "Pausiert" if idea.get("paused") else watch.LABELS.get(idea["state"], idea["state"])
        if idea.get("setup_blocked"):
            state = "Kein geprüftes Chartsetup · gesperrt"
        elif current >= watch.utc(idea["expires_at"]) and idea["state"] not in watch.TERMINAL:
            state = "Abgelaufen"
        quote = watch.utc(idea["quote_at"]) if idea.get("quote_at") else None
        health = str(idea.get("health") or "Noch nicht geprüft")
        if quote is not None and current - quote > timedelta(minutes=3) and idea["state"] not in watch.TERMINAL:
            health = "Kursdaten veraltet · " + health
        rows.append({"Ticker": idea["ticker"], "Chartsetup": ((idea.get("source") or {}).get("watch_setup") or {}).get("label") or "Nicht nachgewiesen", "Status": state, "Daten": health,
                     "Kurs": price, "Einheit": idea.get("currency") or "–", "Entry": idea["entry"],
                     "Abstand %": round(distance, 2) if distance is not None else None,
                     "Entry maximal": idea["max_entry"], "Invalidierung": idea["invalidation"],
                     "Ziel": idea.get("target"), "Regel": next((k for k, v in RULES.items() if v == idea["mode"]), idea["mode"]),
                     "1m-Kerzenstart": shown_time(idea.get("quote_at")),
                     "Provider-Abruf": shown_time(idea.get("last_fetch_at")),
                     "Letzte Prüfung": shown_time(idea.get("last_attempt_at")),
                     "Signal gespeichert": shown_time(idea.get("triggered_at")),
                     "Gültig bis": shown_time(idea["expires_at"])})
    return pd.DataFrame(rows)


def render_monitor(st, *, manager=None, provider_factory=None):
    from modules.provider_manager import get_market_data_provider
    manager = manager or create_storage_manager(st_module=st, app_dir=Path(__file__).resolve().parents[1])
    provider_factory = provider_factory or get_market_data_provider
    st.title("👁️ Trade-Wächter")
    st.caption("Gezielte Überwachung gespeicherter Long-Ideen · keine neue Vollanalyse · keine Brokerorders")
    st.info("Automatische Prüfungen laufen nur, solange diese Seite geöffnet und die Sitzung verbunden ist. Keine Hintergrundüberwachung bei geschlossenem Browser. Yahoo liefert keine garantierten Echtzeit-/Tickdaten.")
    st.caption("Geprüft werden ausschließlich die gespeicherten Kursregeln. Nachrichten, Earnings und die ursprüngliche technische Bewertung werden nicht erneut analysiert; das Ziel dient hier als Referenz.")
    st.caption("Neue Ideen ausschließlich über ein erkanntes Chartsetup im Screener oder in der Sofortanalyse übernehmen. Keine freie Eingabe von Kurslevels.")
    a, b = st.columns(2)
    auto = a.toggle("Automatisch prüfen (solange geöffnet)", value=False, key="watch_auto")
    interval = b.selectbox("Prüfintervall", [60, 120, 30], format_func=lambda x: f"{x} Sekunden", key="watch_interval")
    st.caption("Bis zu 10 laufende Ideen. US-Kernhandel (Kalender 2026–2027) nach New Yorker Zeit; .DE im Xetra-Fenster 09:00–17:30. Kein Vor-/Nachhandel. Xetra-Feiertage sind nicht vollständig hinterlegt; fehlende Daten sperren neue Trigger. Für den Wächter möglichst nur eine aktive Browserseite verwenden.")

    def panel():
        try:
            ideas = watch.load_ideas(manager)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        manual = st.button("Jetzt gezielt prüfen", key="watch_poll", type="primary", disabled=not any(watch.active(i) for i in ideas))
        if manual or auto:
            try:
                now = watch.utc()
                needs_quotes = any(watch.active(i) and now < watch.utc(i["expires_at"]) and
                                   watch.in_session(now, i["market"]) for i in ideas)
                updated = watch.poll(ideas, provider_factory() if needs_quotes else None, force=manual)
                if updated != ideas:
                    watch.save_ideas(manager, updated)
                    prior = {i["id"]: i for i in ideas}
                    for item in updated:
                        before = prior.get(item["id"], {})
                        if item["state"] == "triggered" and item.get("triggered_at") != before.get("triggered_at"):
                            st.toast(f"{item['ticker']}: Einstiegstrigger gespeichert – Details prüfen.")
                        elif item["state"] == "invalidated" and before.get("state") != "invalidated":
                            st.toast(f"{item['ticker']}: Handelsidee invalidiert.")
                ideas = updated
            except Exception as exc:
                st.error("Prüfung/Speicherung fehlgeschlagen: " + str(exc))
        if manager.degraded:
            st.warning("Speicherverbindung eingeschränkt; lokaler Ersatzstand. Geräteübergreifende Synchronisierung nicht bestätigt.")
        if not ideas:
            st.info("Noch kein Chartsetup übernommen. Im Screener oder in der Sofortanalyse „Trade-Idee verfolgen“ öffnen. Ohne erkannten Strukturbeleg gibt es keine Übernahme.")
            return
        if any(i.get("setup_blocked") and not i.get("archived") for i in ideas):
            st.warning("Ältere frei angelegte oder veränderte Ideen bleiben als Verlauf erhalten, werden aber nicht mehr überwacht. Für einen neuen Wächterlauf ist ein neu erkanntes Chartsetup erforderlich.")
        show_archive = st.checkbox("Archiv anzeigen", key="watch_archive_show")
        visible = [i for i in ideas if show_archive or not i.get("archived")]
        if not visible:
            st.info("Alle Ideen sind archiviert.")
            return
        st.dataframe(table_rows(visible), hide_index=True, use_container_width=True)
        selected_id = st.selectbox("Idee / Verlauf", [i["id"] for i in visible],
                                  format_func=lambda value: next(i["ticker"] + " · " + shown_time(i["created_at"]) for i in visible if i["id"] == value), key="watch_detail")
        selected = next(i for i in ideas if i["id"] == selected_id)
        st.write(selected.get("note") or "Keine zusätzliche Begründung hinterlegt.")
        source = selected.get("source") or {}
        package = source.get("watch_setup") or {}
        st.caption(f"Chartsetup-Basis {package.get('basis_date') or 'nicht nachgewiesen'} · Setup-Invalidierung: {package.get('stop_source') or 'nicht aus Chartstruktur abgeleitet'}")
        controls = st.columns(2)
        pause = controls[0].button("Überwachung pausieren", key="watch_pause", disabled=not watch.active(selected))
        archive = controls[1].button("Idee archivieren", key="watch_archive", disabled=bool(selected.get("archived")))
        if pause or archive:
            selected["paused" if pause else "archived"] = True
            watch.event(selected, "pause" if pause else "archive", watch.utc(), "Überwachung pausiert; für einen Neustart neue Idee anlegen." if pause else "Idee archiviert.")
            try:
                watch.save_ideas(manager, ideas)
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
        if selected.get("paused"):
            st.caption("Für eine neue Überwachungsphase diese Idee archivieren und mit bewusst geprüften Levels neu anlegen. Alte Signale bleiben im Verlauf.")
        if selected.get("events"):
            history = [{"Zeit (Berlin)": shown_time(e["at"]), "Ereignis": e["kind"], "Details": e["text"]} for e in reversed(selected["events"])]
            st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
        with st.expander("Gespeicherte Handelsidee / Regeln", expanded=False):
            st.json(selected)
            st.caption("Invalidierung: jede nach Anlage begonnene 1m-Kerze mit Low auf/unter dem Level. Treffen Entry und Invalidierung dieselbe Minute, hat Invalidierung Vorrang. 5-/15m-Bestätigung: vollständig gelieferte Minutenkerzen plus 10 Sekunden Puffer. Historische Signale werden protokolliert, nicht als frischer Einstieg ausgegeben. Eine Überwachungslücke benötigt eine neu angelegte Idee.")

    if hasattr(st, "fragment"):
        st.fragment(run_every=interval if auto else None)(panel)()
    else:
        st.warning("Diese Streamlit-Version unterstützt keine unabhängigen automatischen Prüfungen; manuell prüfen.")
        panel()
