"""Compact v30.20b Streamlit adapter. Explanations collapsed; results visible.

No global mutable per-user state, no callbacks cached across user sessions.
Storage/FX/entry-context callbacks are supplied by the authenticated app.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Mapping

import pandas as pd
import streamlit as st

from . import trading_package as engine
from . import live_monitor as scan_pipeline
from .live_screener_snapshot import dataframe_from_payload


def _read_positions(storage):
    if hasattr(storage, "load_result"):
        result = storage.load_result("positions")
        if not result.ok:
            raise RuntimeError("Positionsspeicher ist nicht lesbar.")
        if getattr(storage, "remote_enabled", False) and getattr(storage, "degraded", False):
            raise RuntimeError("Datenbank nicht erreichbar; keine Paketfreigabe auf einem lokalen Ersatzbestand.")
        value = result.data if result.found else {}
    else:
        value = storage.load_namespace("positions", default={})
    if not isinstance(value, Mapping):
        raise RuntimeError("Positionsspeicher hat ein unerwartetes Format.")
    return deepcopy(value)


def _save_positions(storage, store):
    ok = bool(storage.save_namespace("positions", engine.safe_json(store)))
    if getattr(storage, "remote_enabled", False) and getattr(storage, "degraded", False):
        return False, "Nur lokaler Ersatzspeicher erreicht; Speicherung in der Datenbank ist NICHT best\u00e4tigt. Verbindung pr\u00fcfen und erneut versuchen."
    if not ok:
        return False, "Vormerkung konnte nicht gespeichert werden. Keine Brokerorder wurde ausgel\u00f6st."
    check = _read_positions(storage)
    if engine.fingerprint(check) != engine.fingerprint(store):
        return False, "Speicherung nicht eindeutig best\u00e4tigt. Positionsspeicher pr\u00fcfen; nicht blind erneut buchen."
    return True, ""


def collect_marks(storage, current_rows):
    """All previously completed user snapshots, selecting the latest row per ticker."""
    candidates = []
    saved = storage.load_namespace("live_screener_snapshots", default={})
    if isinstance(saved, Mapping):
        for snapshot in (saved.get("snapshots") or {}).values():
            if not isinstance(snapshot, Mapping):
                continue
            meta = snapshot.get("scan_meta") or {}
            if not (meta.get("complete") and meta.get("atomic")):
                continue
            frame = dataframe_from_payload(snapshot.get("live_df"))
            candidates.extend(frame.to_dict("records"))
    candidates.extend(current_rows)
    result = {}
    for row in candidates:
        tk = engine.text(row.get("Ticker")).upper()
        ts = engine.timestamp(row.get("__pkg_scan_at") or row.get("Letztes Update"))
        old = result.get(tk, {})
        old_ts = engine.timestamp(old.get("__pkg_scan_at") or old.get("Letztes Update"))
        if tk and ts and (old_ts is None or ts >= old_ts):
            result[tk] = dict(row)
    return result


def _money(value):
    return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")



def _render_snapshot_status(rows):
    """Explain a missing producer/transport contract before budget evaluation.

    Counts transmitted snapshots, not financially eligible candidates. Missing
    stops/currency or stale timestamps are still rejected by the package engine.
    Never synthesize risk fields from another scan or a display-only CRV.
    """
    transmitted = sum(
        bool(engine.text(row.get("__pkg_version")))
        and engine.timestamp(row.get("__pkg_scan_at")) is not None
        for row in rows
    )
    st.markdown(f"**Paket-Snapshots \u00fcbergeben: {transmitted}/{len(rows)}**")
    pipeline_ready = getattr(scan_pipeline, "PACKAGE_SNAPSHOT_FIELDS_PRESERVED", False) is True
    if not pipeline_ready:
        st.error(
            "Paketdaten-Anbindung fehlt: Bitte auch modules/live_monitor.py aus dem "
            "v30.20c-Update ersetzen, die App neu starten und einen neuen vollst\u00e4ndigen "
            "Atomic-Scan ausf\u00fchren. Der geladene Scan-Baustein erh\u00e4lt die Paketfelder noch nicht."
        )
        return False
    if not transmitted:
        st.error(
            "Paketplanung gesperrt: Im geladenen Scan fehlen die Paket-Snapshots. "
            "Nach diesem Update einmal einen NEUEN vollst\u00e4ndigen Atomic-Scan ausf\u00fchren; "
            "ein Seiten-Reload oder erneutes Berechnen allein reicht nicht. "
            "Das ist keine Aussage gegen die Aktien oder dein Budget."
        )
        return False
    if transmitted < len(rows):
        st.warning(
            f"Bei {len(rows)-transmitted} Scan-Zeile(n) fehlen Paket-Snapshots. "
            "Diese Werte bleiben ausgeschlossen. Neuer Vollscan erforderlich; "
            "die \u00fcbrigen Kandidaten werden regul\u00e4r gepr\u00fcft."
        )
    return True


def _render_plan_diagnostics(plan, config):
    if "scan_count" not in plan:
        return
    st.markdown("**Paketpr\u00fcfung \u00b7 woran liegt es?**")
    a, b, c, d = st.columns(4)
    a.metric("Werte im Scan", str(plan["scan_count"]))
    b.metric("Gr\u00fcn / Jetzt pr\u00fcfen", str(plan.get("queue_ready_count", 0)))
    c.metric("Vor Gr\u00f6\u00dfenpr\u00fcfung geeignet", str(plan.get("eligible_count", 0)))
    count = plan.get("single_feasible_count")
    d.metric("Einzeln umsetzbar", str(count) if count is not None else "gesperrt",
             help="Einzelpr\u00fcfung mit dem ganzen verf\u00fcgbaren Budget. Noch keine gemeinsame Paketfreigabe.")
    st.write(
        f"Effektives Kaufbudget: {_money(plan.get('effective_budget', 0))} {config.base} \u00b7 "
        f"Verf\u00fcgbares zus\u00e4tzliches Stop-Risiko: {_money(plan.get('effective_risk', 0))} {config.base} \u00b7 "
        f"Mindest-CRV nach Kosten: {config.min_crv:.2f}"
    )
    summary = plan.get("reason_summary", [])
    if summary:
        st.write("H\u00e4ufigste Ausschlussgr\u00fcnde \u00b7 " + plan.get("reason_summary_scope", "Gesamter Scan"))
        st.dataframe(pd.DataFrame([{k: r[k] for k in ("Grund", "Werte", "Ticker")} for r in summary[:4]]),
                     hide_index=True, use_container_width=True)
    st.info("Cash bleibt frei. Datenl\u00fccken zuerst kl\u00e4ren; die Grenzen werden nicht automatisch gelockert.")


def _render_candidate_details(plan, config):
    diagnostics = plan.get("diagnostics", [])
    if not diagnostics:
        return
    with st.expander("Kandidatenpr\u00fcfung \u00b7 Trigger, CRV und Mindestpositionen", expanded=False):
        st.caption("CRV im Screener kann einen anderen Stop verwenden. F\u00fcr das Paket gelten der gespeicherte Risiko-Stop, das gepufferte Kauflimit und die eingeplanten Kosten. Einzelpr\u00fcfung = gesamtes effektives Budget, nicht Budget geteilt durch Maximalzahl der Positionen.")
        sizing = {row["Ticker"]: row for row in plan.get("sizing_diagnostics", [])}
        rejects = {}
        for row in plan.get("rejected", []):
            rejects[row["Ticker"]] = row["Grund"]
        records = []
        for row in diagnostics:
            size = sizing.get(row["Ticker"], {})
            records.append({
                "Ticker": row["Ticker"], "Queue": row["Queue"], "Trade-State": row["Trade-State"],
                "Confidence": row["Confidence"], "Kursw\u00e4hrung": row["Kursw\u00e4hrung"],
                "Scankurs": row["Scankurs"], "Kauflimit": row["Kauflimit"],
                "Screener-Stop": row["Screener-Stop"], "Ziel": row["Ziel"],
                "CRV im Screener": row["CRV im Screener"],
                "CRV am Scankurs (Paket-Stop)": row["CRV am Scankurs (Paket-Stop)"],
                "CRV am Kauflimit vor Kosten": row["CRV am Kauflimit vor Kosten"],
                "CRV nach Kosten (Einzelpr\u00fcfung)": size.get("net_crv"),
                "Mindest-CRV": row["Mindest-CRV"],
                "Max. St\u00fcck (Einzelpr\u00fcfung)": size.get("max_shares"),
                "Mindestens St\u00fcck": size.get("min_shares"),
                "Mindest-Einsatz ("+config.base+")": size.get("minimum_cost"),
                "Mindest-Risiko ("+config.base+")": size.get("minimum_risk"),
                "Ergebnis / Grund": rejects.get(row["Ticker"], size.get("Grund", row["Grund"])),
            })
        st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
        st.caption("Ein Wert kann mehrere Ausschlussgr\u00fcnde haben. Fehlende Daten bleiben leer und werden nicht als Null oder als Freigabe interpretiert.")


def _input_config(prefix, settings):
    base_options = ["EUR", "USD", "GBP", "CHF"]
    default_base = engine.currency(settings.get("base_currency"))
    default_base = default_base if default_base in base_options else "EUR"
    a, b, c, d = st.columns(4)
    equity = a.number_input("Tradingdepotwert inkl. Cash", min_value=0.0,
                            value=float(engine.number(settings.get("account_size"), 0)), step=500.0, key=prefix+"equity")
    budget = b.number_input("Freies Kaufbudget", min_value=0.0, value=0.0, step=100.0, key=prefix+"budget")
    new_risk = c.number_input("Zus\u00e4tzliches Stop-Risikobudget", min_value=0.0, value=0.0, step=10.0, key=prefix+"risk",
                              help="Geldbetrag in der Basisw\u00e4hrung, einschlie\u00dflich der eingeplanten Kosten. Keine Verlustgarantie.")
    n = d.number_input("Max. neue Positionen", min_value=1, max_value=5, value=3, step=1, key=prefix+"n")
    with st.expander("Grenzen, Kosten & Basisw\u00e4hrung", expanded=False):
        p, q, r = st.columns(3)
        base = p.selectbox("Alle Budgetwerte in", base_options, index=base_options.index(default_base), key=prefix+"base")
        single = q.number_input("Max. Einzelposition (% Tradingdepot)", min_value=0.1, max_value=100.0, value=12.0, step=0.5, key=prefix+"single")
        group = r.number_input("Max. Branchengruppe (% Tradingdepot)", min_value=0.1, max_value=100.0, value=30.0, step=1.0, key=prefix+"group")
        p, q, r = st.columns(3)
        total = p.number_input("Gesamtes Stop-Risikolimit (%)", min_value=0.1, max_value=100.0, value=3.0, step=0.1, key=prefix+"total",
                               help="Bestand + offene Vormerkungen + neues Paket. Beim Bestand vom aktuellen Kurs bis Stop, nicht nur vom Einstand.")
        crv = q.number_input("Mindest-CRV inkl. Kostenpuffer", min_value=0.1, max_value=20.0, value=2.0, step=0.1, key=prefix+"crv")
        order = r.number_input("Mindest-Positionswert", min_value=1.0, value=100.0, step=25.0, key=prefix+"order")
        p, q, r = st.columns(3)
        fee = p.number_input("Fixkosten pro Orderseite", min_value=0.0, value=1.0, step=0.1, key=prefix+"fee")
        variable = q.number_input("Variabler Kosten-/FX-Puffer pro Seite (%)", min_value=0.0, max_value=10.0, value=0.15, step=0.05, key=prefix+"variable")
        buffer = r.number_input("Max. Einstiegspuffer zum Scankurs (%)", min_value=0.0, max_value=5.0, value=0.3, step=0.1, key=prefix+"buffer")
        per_group = st.number_input("Max. neue Positionen je Branchengruppe", min_value=1, max_value=5, value=1, step=1, key=prefix+"per_group")
        age = st.number_input("Maximales Alter des Vollscans (Stunden)", min_value=1.0, max_value=72.0, value=24.0, step=1.0, key=prefix+"age")
        st.caption("Startwerte sind ver\u00e4nderbare Planungsparameter, keine pers\u00f6nliche Risikovorgabe. In dieser Version ganze Aktien; keine automatische Order oder Budgetaussch\u00f6pfung.")
    return engine.PlanConfig(equity=equity, budget=budget, new_risk=new_risk, base=base, max_positions=int(n), max_per_group=int(per_group),
                             max_position_pct=single, max_group_pct=group, max_total_risk_pct=total,
                             min_crv=crv, min_order=order, fixed_fee=fee, variable_fee_pct=variable,
                             entry_buffer_pct=buffer, max_age_hours=age)


def _data_editor(rows, marks, store, prefix):
    pool = {engine.text(r.get("Ticker")).upper(): r for r in rows}
    positions = {}
    for bucket in store.values():
        if isinstance(bucket, Mapping):
            for tk, p in bucket.items():
                if isinstance(p, Mapping) and (engine.number(p.get("shares"), 0) > 0 or p.get("execution_status") == "planned"):
                    positions.setdefault(str(tk).upper(), p)
    records = []
    for tk in sorted(set(pool) | set(positions)):
        row = marks.get(tk) or pool.get(tk) or {}
        pos = positions.get(tk, {})
        group = engine.group_name(row.get("__pkg_group"))
        if group == engine.UNKNOWN:
            group = engine.group_name(pos.get("portfolio_group"))
        records.append({"Ticker": tk, "Bezug": "Bestand/Vormerkung" if pos else "Scan",
                        "Kursw\u00e4hrung": engine.quote_currency(row, pos), "Branchengruppe": group})
    df = pd.DataFrame(records, columns=["Ticker", "Bezug", "Kursw\u00e4hrung", "Branchengruppe"])
    with st.expander("Datenbasis & Ausschl\u00fcsse pr\u00fcfen", expanded=False):
        st.caption("Sektor und Kursw\u00e4hrung stammen aus dem Scan bzw. Bestand. Korrekturen gelten NUR f\u00fcr diese Planung; keine historischen Geldwerte werden umgeschrieben. GBX bedeutet britische Pence.")
        edited = st.data_editor(df, hide_index=True, use_container_width=True, num_rows="fixed",
                                disabled=["Ticker", "Bezug"],
                                column_config={"Kursw\u00e4hrung": st.column_config.SelectboxColumn(options=[""]+sorted(engine.CURRENCIES))},
                                key=prefix+"data_"+engine.fingerprint(records)[:10])
        changed = engine.fingerprint(df.to_dict("records")) != engine.fingerprint(edited.to_dict("records"))
        confirmed = st.checkbox("Meine ge\u00e4nderten Kursw\u00e4hrungen und Branchengruppen sind gepr\u00fcft", key=prefix+"data_confirm") if changed else True
        exclude = st.multiselect("Diese Kandidaten nicht aufnehmen", sorted(pool), key=prefix+"exclude")
    adjusted_rows = deepcopy(rows)
    adjusted_marks = deepcopy(marks)
    if changed and confirmed:
        overrides = {r["Ticker"]: r for r in edited.to_dict("records")}
        for row in adjusted_rows:
            edit = overrides.get(engine.text(row.get("Ticker")).upper(), {})
            row["__pkg_currency"] = engine.currency(edit.get("Kursw\u00e4hrung"))
            row["__pkg_group"] = engine.group_name(edit.get("Branchengruppe"))
        for tk, edit in overrides.items():
            row = adjusted_marks.setdefault(tk, {"Ticker": tk})
            row["__pkg_currency"] = engine.currency(edit.get("Kursw\u00e4hrung"))
            row["__pkg_group"] = engine.group_name(edit.get("Branchengruppe"))
    return adjusted_rows, adjusted_marks, exclude, bool(confirmed)


def _pending_ui(store, storage, prefix):
    pending = []
    for key, bucket in store.items():
        if not isinstance(bucket, Mapping):
            continue
        for tk, pos in bucket.items():
            if not isinstance(pos, Mapping):
                continue
            context = pos.get("entry_context") or {}
            package = context.get("trading_package") or {}
            if pos.get("execution_status") == "planned" and engine.number(pos.get("shares"), 0) == 0 and package.get("id"):
                pending.append((key, tk, package.get("id"), pos.get("planned_shares")))
    if not pending:
        return
    with st.expander(f"Offene Paket-Vormerkungen ({len(pending)})", expanded=False):
        st.dataframe(pd.DataFrame([{"Watchlist": k.replace("v244_open_positions::", ""), "Ticker": t,
                                    "Geplant": qty, "Paket": pid} for k, t, pid, qty in pending]),
                     hide_index=True, use_container_width=True)
        st.caption("Vormerkungen reservieren Planungsbudget, sind aber keine K\u00e4ufe. Nicht gehandelte Vormerkungen hier freigeben.")
        labels = {f"{tk} \u00b7 {key.replace('v244_open_positions::', '')} \u00b7 {pid}": (key, tk, pid) for key, tk, pid, _ in pending}
        selected = st.multiselect("Nicht gehandelte Vormerkungen entfernen", list(labels), key=prefix+"cancel_items")
        confirm = st.checkbox("Diese Vormerkungen wurden nicht ausgef\u00fchrt", key=prefix+"cancel_confirm")
        if st.button("Ausgew\u00e4hlte Vormerkungen freigeben", disabled=not(selected and confirm), key=prefix+"cancel"):
            current = _read_positions(storage)
            for label in selected:
                key, tk, pid = labels[label]
                pos = (current.get(key) or {}).get(tk) or {}
                actual_id = ((pos.get("entry_context") or {}).get("trading_package") or {}).get("id")
                if actual_id != pid or engine.number(pos.get("shares")) != 0 or pos.get("execution_status") != "planned":
                    st.error("Vormerkung wurde inzwischen ver\u00e4ndert/ausgef\u00fchrt. Nicht entfernt.")
                    return
            for label in selected:
                key, tk, _ = labels[label]
                current[key].pop(tk, None)
            ok, error = _save_positions(storage, current)
            if ok:
                st.session_state[prefix+"flash"] = f"{len(selected)} nicht ausgef\u00fchrte Vormerkung(en) freigegeben."
                st.session_state.pop(prefix+"result", None)
                st.rerun()
            else:
                st.error(error)


def render_trading_package(*, watchlist, frame, queue, scan_meta, storage, fx_resolver,
                           capture_context, now_provider):
    """Only called under the existing authenticated Watchlists runtime."""
    prefix = "v3020_" + engine.fingerprint([str(getattr(storage, "user_id", "session")), watchlist])[:16] + "_"
    rows = frame.to_dict("records") if isinstance(frame, pd.DataFrame) else []
    queue_rows = queue.to_dict("records") if isinstance(queue, pd.DataFrame) else []
    with st.container(border=True):
        st.markdown("#### \U0001f4e6 Tradingpaket \u00b7 Budget & Diversifikation")
        flash = st.session_state.get(prefix+"flash")
        if flash:
            st.success(flash)
        st.caption("PLANUNG \u00b7 aktueller Watchlist-Scan + erfasster Tradingbestand aller Watchlists \u00b7 Pies nicht enthalten")
        if not rows:
            st.info("F\u00fcr die Paketplanung zun\u00e4chst einen Vollscan ausf\u00fchren.")
            return
        try:
            store = _read_positions(storage)
            marks = collect_marks(storage, rows)
            settings = storage.load_namespace("portfolio_risk_settings", default={}) or {}
        except Exception:
            st.error("Die Bestandsdaten sind nicht sicher lesbar. Paketplanung bleibt gesperrt; Datenbankverbindung pr\u00fcfen.")
            return
        snapshots_ready = _render_snapshot_status(rows)
        config = _input_config(prefix, settings)
        rows, marks, excluded, data_confirm = _data_editor(rows, marks, store, prefix)
        acknowledged = st.checkbox("Tradingbestand aller Broker erfasst; Budget, Risikogrenzen und Kosten gepr\u00fcft", key=prefix+"scope_confirm")
        # Optional, explicit FX override. Never silently reuse an undated manual rate.
        requested = sorted({engine.quote_currency(r) for r in rows} | {engine.quote_currency(r) for r in marks.values()})
        for bucket in store.values():
            if isinstance(bucket, Mapping):
                requested.extend(engine.quote_currency(p) for p in bucket.values() if isinstance(p, Mapping))
        majors = sorted({{"GBX": "GBP", "ZAC": "ZAR"}.get(c, c) for c in requested if c and c != config.base})
        manual_rates = {}
        with st.expander("FX-Planungskurse bei Bedarf manuell best\u00e4tigen", expanded=False):
            st.caption("Standard: vorhandene ECB-Umrechnung. Nur bei bewusster Best\u00e4tigung ersetzen diese Werte die automatisch ermittelten Referenzkurse.")
            use_manual = st.checkbox("Manuelle Planungskurse verwenden", key=prefix+"manual_fx")
            if use_manual:
                for cur in majors:
                    manual_rates[cur] = st.number_input(f"1 {cur} = wie viel {config.base}?", min_value=0.0, value=0.0, step=0.0001, format="%.6f", key=prefix+"fx_"+cur+config.base)
                manual_confirm = st.checkbox("Diese Umrechnungskurse habe ich f\u00fcr die aktuelle Planung gepr\u00fcft", key=prefix+"fx_confirm")
            else:
                manual_confirm = True
        context_hash = engine.fingerprint({"planner_version": engine.VERSION, "rows": rows, "queue": queue_rows, "store": store, "marks": marks,
                                            "config": config.__dict__, "scan": scan_meta, "exclude": excluded,
                                            "manual": manual_rates, "ack": acknowledged, "dc": data_confirm, "mc": manual_confirm})
        if st.button("Tradingpaket berechnen", type="primary", disabled=not(acknowledged and data_confirm and manual_confirm and snapshots_ready), key=prefix+"compute"):
            now = now_provider()
            if config.errors():
                st.session_state[prefix+"result"] = {"context_hash": context_hash, "ok": False, "errors": config.errors()}
            else:
                fx_info = {}
                try:
                    if use_manual:
                        rates = {config.base: 1.0, **{k: v for k, v in manual_rates.items() if v > 0}}
                        fx_info = {"source": "Manuell best\u00e4tigt", "reference_date": str(now.date()), "rates_to_base": rates}
                    elif not majors:
                        rates = {config.base: 1.0}
                        fx_info = {"source": "Basisw\u00e4hrung", "reference_date": str(now.date()), "rates_to_base": rates}
                    else:
                        fx_info = fx_resolver(majors + [config.base], base_currency=config.base, refresh_token=str(now.date()))
                        rates = dict(fx_info.get("rates_to_base") or {config.base: 1.0})
                except Exception:
                    rates = {config.base: 1.0}
                    fx_info = {"source": "FX nicht verf\u00fcgbar", "reference_date": ""}
                plan = engine.build_plan(rows, queue_rows, store, marks, config, rates, now=now,
                                         scan_id=str((scan_meta or {}).get("run_id") or ""),
                                         scan_complete=bool((scan_meta or {}).get("complete")),
                                         atomic=bool((scan_meta or {}).get("atomic")), excluded=excluded)
                plan["context_hash"] = context_hash
                plan["fx_info"] = engine.safe_json(fx_info)
                st.session_state[prefix+"result"] = plan
                st.session_state.pop(prefix+"flash", None)
        plan = st.session_state.get(prefix+"result")
        if plan and plan.get("context_hash") != context_hash:
            st.info("Eingaben, Scan oder Best\u00e4nde haben sich ge\u00e4ndert. Paket neu berechnen.")
            plan = None
        if plan:
            for error in plan.get("errors", []):
                st.warning(error)
            if not plan.get("ok"):
                _render_plan_diagnostics(plan, config)
            if plan.get("ok"):
                options = list(range(len(plan["alternatives"])))
                chosen_index = st.selectbox("Paket", options, format_func=lambda i: "Bevorzugter Vorschlag" if i == 0 else f"Alternative {i}", key=prefix+"alternative_"+plan["alternatives"][0]["id"])
                chosen = plan["alternatives"][chosen_index]
                p, q, r, s = st.columns(4)
                p.metric("Geplanter Einsatz inkl. Kaufkosten", _money(chosen["cost"])+" "+config.base)
                q.metric("Geplantes Stop-Risiko inkl. Kosten", _money(chosen["risk"])+" "+config.base)
                r.metric("Kaufbudget ungenutzt", _money(chosen["cash_left"])+" "+config.base)
                s.metric("Neue Positionen", str(len(chosen["items"])))
                table = []
                for c in chosen["items"]:
                    table.append({"Ticker": c["ticker"], "Gruppe": c["group"], "St\u00fcck": c["shares"], "Kursw\u00e4hrung": c["currency"],
                                  "Kauflimit (Plan)": round(c["limit"], 4), "Screener-Stop": round(c["stop"], 4),
                                  "Ziel": round(c["target"], 4), "CRV inkl. Kosten": round(c["net_crv"], 2),
                                  "Einsatz ("+config.base+")": round(c["cost"], 2), "Stop-Risiko ("+config.base+")": round(c["risk"], 2),
                                  "Grund": f"Trigger aktiv \u00b7 Score {c['score']:.0f} \u00b7 Confidence {c['confidence']}"})
                st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
                st.success(f"{len(chosen['items'])} Kandidat(en) passen gemeinsam in die gepr\u00fcften Grenzen. Unverplantes Budget bleibt frei.")
                st.caption(f"Scan: {plan['scan_id']} \u00b7 FX: {plan.get('fx_info', {}).get('source', '-')} {plan.get('fx_info', {}).get('reference_date', '')}")
                st.warning("Plan, keine Brokerorder: Kurse/Trigger vor Ausf\u00fchrung pr\u00fcfen. Oberhalb des Kauflimits neu planen. Stop-Risiko ist kein garantierter Maximalverlust.")
                with st.expander("Paketpr\u00fcfung, Stop-Herkunft & Alternative im Detail", expanded=False):
                    st.write(f"Gepr\u00fcfte Kombinationen: {plan.get('combinations_checked', 0)}. Suchraum: {plan.get('searched_count', 0)} von {plan.get('eligible_count', 0)} geeigneten Kandidaten.")
                    st.dataframe(pd.DataFrame([{"Ticker": c["ticker"], "Stop-Quelle": c["stop_source"], "Basis vor ATR": c["stop_basis"],
                                                "Chart-Invalidierung": c["chart_stop"], "Ziel-Quelle": c["target_source"],
                                                "FX in Basisw\u00e4hrung": c["fx"], "Scanzeit": c["scan_at"]} for c in chosen["items"]]), hide_index=True)
                    st.dataframe(pd.DataFrame([{"Gruppe": g, "Wert nach Paket": v, "Anteil Tradingdepot %": 100*v/config.equity}
                                                for g, v in chosen["groups_after"].items()]), hide_index=True)
                    port = plan.get("portfolio", {})
                    st.write(f"Bestandsrisiko: {_money(port.get('risk', 0))} {config.base}; reserviertes Risiko: {_money(port.get('reserved_risk', 0))} {config.base}.")
                confirm = st.checkbox("Dieses Paket bewusst als Screener-Trades vormerken (noch keine Ausf\u00fchrung)", key=prefix+"confirm_"+chosen["id"])
                if st.button("Paket als Screener-Trades vormerken", disabled=not confirm, key=prefix+"save_"+chosen["id"]):
                    try:
                        current = _read_positions(storage)
                        updated, changed = engine.build_intentions(plan, chosen_index, current, watchlist=watchlist,
                                                                    now=now_provider(), capture_context=capture_context)
                        ok, error = _save_positions(storage, updated) if changed else (True, "")
                        if ok:
                            st.session_state[prefix+"flash"] = f"Paket {chosen['id']} vorgemerkt: {len(chosen['items'])} Trades mit St\u00fcckzahl 0. Signal, Stop und geplante Gr\u00f6\u00dfe gespeichert; keine K\u00e4ufe gebucht. Broker-Fills weiterhin in der Importvorschau pr\u00fcfen."
                            st.session_state.pop(prefix+"result", None)
                            st.rerun()
                        else:
                            st.error(error)
                    except (ValueError, RuntimeError) as exc:
                        st.error(str(exc))
            if plan.get("rejected"):
                with st.expander(f"Nicht aufgenommen / noch zu kl\u00e4ren ({len(plan['rejected'])})", expanded=False):
                    st.dataframe(pd.DataFrame(plan["rejected"])[["Ticker", "Grund"]], hide_index=True, use_container_width=True)
            _render_candidate_details(plan, config)
        _pending_ui(store, storage, prefix)
        with st.expander("\u2139\ufe0f Methodik & Grenzen", expanded=False):
            st.markdown(
                "**Auswahl:** nur gr\u00fcne Queue-Kandidaten mit aktivem Trigger, ohne harte Gates und mit mindestens mittlerer Decision-Confidence. "
                "Der endg\u00fcltige Trade-State ist f\u00fcr die Triggerfreigabe ma\u00dfgeblich; Armed/Best\u00e4tigung offen ist nicht aktiv. "
                "Stops und strukturelle Ziele stammen aus demselben Vollscan wie der Kurs. Kein zus\u00e4tzlicher Aktienkursabruf beim Planen. "
                "Die Zahl der \u00fcbergebenen Paket-Snapshots best\u00e4tigt nur die Datenweitergabe, nicht die Vollst\u00e4ndigkeit, "
                "Aktualit\u00e4t oder Eignung zum Kauf; diese Pr\u00fcfungen folgen gesondert.\n\n"
                "**St\u00fcckzahl:** ganze Aktien, gleiches anf\u00e4ngliches Kapital-/Risikobudget je Paketplatz, danach Begrenzung durch Einzelgewicht, Branche, Kosten und Bestand. "
                "Nicht genutzte Kapazit\u00e4t wird nicht zwangsl\u00e4ufig aufgef\u00fcllt. Mindest-CRV gilt am Kauflimit einschlie\u00dflich Kostenpuffer.\n\n"
                "**Vergleich:** maximal 18 Kandidaten und 5 neue Positionen; standardm\u00e4\u00dfig eine neue Position je Branchengruppe. Die besten Gruppenvertreter bleiben im Suchraum. Eine transparente Planungsheuristik gewichtet den bestehenden Live-Score (65 %), "
                "Confidence (20 %) und gedeckeltes CRV (15 %); konkave Kapitalgewichtung und ein Konzentrationsabzug bevorzugen passendere Kombinationen. "
                "Das ist weder eine Gewinnprognose noch ein neuer produktiver Screener-Score.\n\n"
                "**Diversifikation:** Branchengruppen im erfassten Tradingbestand, keine berechnete Kurskorrelation und keine Aussage \u00fcber dein gesamtes Verm\u00f6gen. "
                "Pies bleiben ausgeschlossen. Nicht erfasste Brokerbest\u00e4nde fehlen auch in dieser Pr\u00fcfung.\n\n"
                "**W\u00e4hrung:** explizite Kursw\u00e4hrung; Finanzberichtsw\u00e4hrung und Tickersuffixe werden nicht geraten. "
                "GBX wird als GBP/100 behandelt. FX-Referenzkurse und Kosten sind Planungsannahmen, keine zugesagten Brokerkurse.\n\n"
                "**Risiko:** beim Bestand vom aktuellen Marktkurs bis Stop, inklusive gesch\u00e4tzter Verkaufskosten. "
                "Stops k\u00f6nnen schlechter oder als Stop-Limit gar nicht ausgef\u00fchrt werden. Scan-Frische bedeutet nicht Echtzeitkursgarantie.\n\n"
                "**Vormerken:** Null St\u00fcck, keine Journal-P/L, keine automatische Order. Erst gepr\u00fcfte Broker-Fills oder manuell best\u00e4tigte Ausf\u00fchrungen ergeben reale Trades. "
                "Die bisherige Broker-Zuordnung wird in dieser Version nicht neu entwickelt."
            )
