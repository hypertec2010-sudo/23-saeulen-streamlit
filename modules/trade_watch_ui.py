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


def capture_form(st, basis=None, *, key="watch_manual", manager=None):
    """Explicitly review all levels; source snapshot is never changed by monitoring."""
    basis = basis or {}
    source_ticker = str(basis.get("ticker") or "")
    entry = watch.number(basis.get("zone_high")) or watch.number(basis.get("price")) or 0.
    stop = watch.number(basis.get("stop")) or 0.
    target = watch.number(basis.get("target")) or 0.
    stamp = str(basis.get("fingerprint") or "manual")
    prefix = key + "_" + stamp
    st.caption("Long-Idee · US-Aktien und .DE · gespeicherte Levels bleiben unverändert. Erst nach Anlage begonnene Kerzen zählen.")
    if basis:
        st.caption(f"Analysebasis {source_ticker} · Tageskerze {basis.get('data_date') or 'unbekannt'} · Stop: {basis.get('stop_source') or 'nicht belegt'} · Qualität: {basis.get('stop_quality') or 'unbekannt'}")
        st.caption("Der vorgeschlagene Entry ist der obere Zonenrand bzw. Analysepreis. Bitte als konkreten Auslöser prüfen; er ist keine automatisch bestätigte Handelsregel.")
    with st.form(prefix + "_form"):
        ticker = st.text_input("Ticker", value=source_ticker, disabled=bool(source_ticker), key=prefix + "_ticker")
        a, b = st.columns(2)
        e = a.number_input("Entry-Auslöser", min_value=0., value=max(0., entry), step=.01, format="%.4f", key=prefix + "_entry")
        inv = b.number_input("Invalidierung (Low erreicht)", min_value=0., value=max(0., stop), step=.01, format="%.4f", key=prefix + "_stop")
        ceiling = a.number_input("Oberste Entry-Grenze", min_value=0., value=round(max(0., entry) * 1.01, 4), step=.01, format="%.4f", key=prefix + "_ceiling")
        goal = b.number_input("Ziel (optional, 0 = keines)", min_value=0., value=max(0., target), step=.01, format="%.4f", key=prefix + "_goal")
        rule = st.selectbox("Einstiegsbestätigung", list(RULES), key=prefix + "_rule")
        currency = st.text_input("Kurseinheit / Währung", value=str(basis.get("currency") or ""), key=prefix + "_ccy")
        local_now = watch.utc().tz_convert("Europe/Berlin")
        until_day = a.date_input("Gültig bis (Berliner Datum)", value=local_now.date() + timedelta(days=5),
                                 min_value=local_now.date(), max_value=local_now.date() + timedelta(days=29), key=prefix + "_day")
        until_time = b.time_input("Gültig bis (Berliner Uhrzeit)", value=time(22, 0), key=prefix + "_time")
        note = st.text_input("Handelsidee / Begründung", key=prefix + "_note", max_chars=500)
        reviewed = st.checkbox("Entry, obere Grenze und Invalidierung für diese Idee geprüft", key=prefix + "_reviewed")
        save = st.form_submit_button("Trade-Idee verfolgen", type="primary")
    if save:
        try:
            if not reviewed:
                raise ValueError("Bitte die Handelslevels prüfen und bestätigen.")
            expiry = pd.Timestamp(datetime.combine(until_day, until_time), tz="Europe/Berlin")
            idea = watch.make_idea(ticker=ticker, entry=e, invalidation=inv, max_entry=ceiling,
                                   mode=RULES[rule], expires_at=expiry, target=goal, currency=currency,
                                   source=basis, note=note)
            storage = manager or create_storage_manager(st_module=st, app_dir=Path(__file__).resolve().parents[1])
            watch.add_idea(storage, idea)
            st.success(f"{idea['ticker']}: Idee gespeichert. Im Arbeitsbereich Trade-Wächter gezielt prüfen.")
            if storage.degraded:
                st.warning("Nur lokaler Ersatzspeicher verfügbar; geräteübergreifende Speicherung derzeit nicht bestätigt.")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))


def render_capture(st, frame, *, key="watch_screener"):
    """Capture from already analyzed screener rows, including restored snapshots."""
    from modules.risk_calculator import risk_basis_from_row
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Ticker" not in frame:
        return
    with st.expander("👁️ Trade-Idee verfolgen", expanded=False):
        rows = frame.to_dict("records")
        index = st.selectbox("Analysierte Aktie", range(len(rows)),
                             format_func=lambda i: str(rows[i].get("Ticker") or "–") + " · " + str(rows[i].get("Status") or ""), key=key + "_row")
        basis = risk_basis_from_row(rows[index])
        if basis is None:
            st.info("Für diesen gespeicherten Stand fehlt die gemeinsame Analysebasis. Aktie einmal neu scannen.")
            return
        capture_form(st, basis, key=key)


def table_rows(ideas, *, now=None):
    current = watch.utc(now)
    rows = []
    for idea in ideas:
        price = watch.number(idea.get("price"))
        distance = (price / idea["entry"] - 1) * 100 if price is not None else None
        state = "Archiviert" if idea.get("archived") else "Pausiert" if idea.get("paused") else watch.LABELS.get(idea["state"], idea["state"])
        if current >= watch.utc(idea["expires_at"]) and idea["state"] not in watch.TERMINAL:
            state = "Abgelaufen"
        quote = watch.utc(idea["quote_at"]) if idea.get("quote_at") else None
        health = str(idea.get("health") or "Noch nicht geprüft")
        if quote is not None and current - quote > timedelta(minutes=3) and idea["state"] not in watch.TERMINAL:
            health = "Kursdaten veraltet · " + health
        rows.append({"Ticker": idea["ticker"], "Status": state, "Daten": health,
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
    with st.expander("Neue Idee manuell anlegen", expanded=False):
        capture_form(st, key="watch_manual", manager=manager)
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
            st.info("Noch keine Idee gespeichert. Im Screener oder in der Sofortanalyse „Trade-Idee verfolgen“ öffnen oder oben manuell anlegen.")
            return
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
        st.caption(f"Ursprüngliche Analyse · Tageskerze {source.get('data_date') or 'manuell'} · Stopquelle {source.get('stop_source') or 'manuell'} · Stopqualität {source.get('stop_quality') or 'nicht belegt'}")
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
