"""Read-only diagnosis of selection coverage; no ranking or target changes."""
from collections import Counter

import pandas as pd

POLICY = "radar_audit_v30.21ce"


def chart_audit(result, *, now, full_context):
    from modules.watch_setups import detect
    from modules.watch_quality import assess
    try:
        package = detect(result, now=now)
    except Exception as exc:
        return dict(policy=POLICY, full_context=bool(full_context), phase="blocked",
                    reason="Zusatzdiagnose nicht möglich: " + type(exc).__name__, reason_code="audit_error")
    out = {key: package.get(key) for key in ("phase", "label", "reason", "reason_code", "basis_date", "valid_until")}
    out.update(policy=POLICY, full_context=bool(full_context))
    if full_context:
        report = assess(package)
        out.update(rr_entry=report["rr_entry"], rr_cap=report["rr_cap"], rr_reason=report["reason"],
                   rr_allowed=report["allowed"])
    # A light scan lacks earnings and other full-context gates and a verified
    # target. It can never arm an idea or claim CRV/admission approval.
    return out


def tables(payload):
    """Missing audit metadata stays explicitly unknown, especially in old scans."""
    from modules.watch_setups import PHASES
    from modules.watch_quality import MIN_ENTRY_RR
    selected = set(payload.get("preselected_symbols") or [])
    lights = payload.get("preselection") or []
    selection_rows = []
    for rank, row in enumerate(lights, 1):
        audit = row.get("chart_preview") or {}
        verified = audit.get("policy") == POLICY
        selection_rows.append({"Rang Vorscan": rank, "Ticker": row["ticker"],
            "Vollanalyse ausgewählt": row["ticker"] in selected, "Light-Score": row.get("light_score"),
            "OHLCV-Musterphase": PHASES.get(audit.get("phase"), "Nicht geprüft") if verified else "Nicht geprüft",
            "Grund": audit.get("reason") if verified else "Alter Scan ohne Musterdiagnose",
            "Grundcode": audit.get("reason_code") if verified else "audit_missing",
            "Abstand MA20 %": row.get("dist_ma20_pct"), "20T-Rendite %": row.get("ret20")})
    full_rows = []
    for row in payload.get("rows") or []:
        audit = row.get("chart_audit") or {}
        rr = row.get("crv")
        from modules.candidate_radar import number
        rr = number(rr)
        radar_rr = "Nicht bewertbar" if rr is None else f"Unter {MIN_ENTRY_RR:g}R" if rr < MIN_ENTRY_RR else f"Mindestens {MIN_ENTRY_RR:g}R"
        if audit.get("policy") != POLICY:
            category = "Chartdiagnose fehlt"
        elif row.get("gates"):
            category = "Vollanalyse mit Sperre"
        elif audit.get("phase") != "ready":
            category = "Kein bestätigungsbereites Chartsetup"
        elif audit.get("rr_allowed") is not True:
            category = "Chartsetup vorhanden · Setup-CRV unzureichend/unbekannt"
        else:
            category = "Chartsetup und Setup-CRV im Scan prüfenswert"
        price, stop, target = (number(row.get(k)) for k in ("price", "stop", "target"))
        target_pct = (target/price-1)*100 if price is not None and price > 0 and target is not None else None
        stop_pct = (1-stop/price)*100 if price is not None and price > 0 and stop is not None else None
        full_rows.append({"Ticker":row["ticker"], "Diagnose":category,
            "Radar-Status beim Scan":row.get("status"), "Note":row.get("grade"), "Radar-CRV":rr,
            "Analysepreis":price, "Analyse-Stop":stop, "Radar-Ziel":target,
            "Zielabstand %":target_pct, "Stopabstand %":stop_pct,
            "Radar-CRV-Einordnung":radar_rr,
            "Zielquelle Radar":(row.get("target_provenance") or {}).get("label"),
            "Chartphase":PHASES.get(audit.get("phase"),"Nicht geprüft"),
            "Chartgrund":audit.get("reason") or "Neuer Radar-Scan nötig",
            "Setup-CRV oberer Entry":audit.get("rr_cap"), "Setup-CRV-Grund":audit.get("rr_reason"),
            "Sperren":"; ".join(row.get("gates") or [])})
    return pd.DataFrame(selection_rows), pd.DataFrame(full_rows)


def render_audit(st, payload):
    from modules.watch_quality import MIN_ENTRY_RR
    selection, full = tables(payload)
    with st.expander("Warum so wenige Trade-Kandidaten?", expanded=False):
        st.write("Der Light-Score priorisiert Trend/Momentum/Kurslage, nicht das CRV. Die Auswahl und das Limit der Vollanalyse bleiben unverändert.")
        if not selection.empty:
            previews = [r for r in payload.get("preselection", []) if (r.get("chart_preview") or {}).get("policy") == POLICY]
            outside = [r["ticker"] for r in previews if r["ticker"] not in set(payload.get("preselected_symbols") or []) and r["chart_preview"].get("phase") == "ready"]
            st.write(f"Vorscan-Diagnose: {len(previews)}/{payload.get('light_processed', 0)} erfolgreich vorsortierte Werte mit Musterprüfung. Davon {len(outside)} mit OHLCV-Musterphase 'bestätigungsbereit' außerhalb der Vollanalyse.")
            if outside:
                st.write("Für eine gezielte Vollanalyse außerhalb der bisherigen Auswahl: " + ", ".join(outside))
            st.dataframe(selection, hide_index=True, use_container_width=True)
        else:
            st.info("Für diesen Scan fehlt die vollständige Vorscan-Diagnose. Kandidatenliste einmal neu scannen.")
        st.caption("OHLCV-Vorschau: keine vollständige Earnings-, Bestands-, Ziel- oder CRV-Prüfung. Ein Vorschau-Muster ist keine Wächter-Freigabe. Nicht vollständig analysierte Werte sind unbekannt, keine bewiesen verpassten guten Trades. Bei alten Scans kann die Liste auf 30 Werte begrenzt sein.")
        if not full.empty:
            counts = Counter(full["Diagnose"])
            st.write("Vollanalyse-Diagnose: " + " · ".join(f"{v}× {k}" for k,v in counts.items()))
            st.dataframe(full, hide_index=True, use_container_width=True)
        st.caption(f"Radar-CRV und Setup-CRV haben unterschiedliche Plan-Grundlagen. {MIN_ENTRY_RR:g}R dient hier beim Radar als Vergleich, nicht als neue Radar-Sperre. Die bestehende Wächter-Regel wird nur auf Setup-Levels angewandt. Diagnosen sind gespeicherte Scanbeobachtungen und keine aktuelle Einstiegsfreigabe.")
