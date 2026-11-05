"""Explicit handoff of a fresh, sealed full-analysis setup to the watcher."""
from copy import deepcopy


def watch_basis(snapshot, ticker, *, now, holdings_known, held=(), owner=None, max_scan_hours=24):
    from modules import candidate_radar as radar
    from modules.radar_setup_audit import POLICY
    from modules.watch_setups import validate
    from modules.watch_quality import admission
    if not holdings_known:
        return None, "Bestandsabgleich fehlt; Positionen zuerst verlässlich laden."
    if str(ticker).upper() in {str(t).upper() for t in held}:
        return None, "Bereits als Position erfasst; keine neue Einstiegsidee übernehmen."
    if owner is not None and snapshot.get("owner") not in (None, owner):
        return None, "Radar-Scan gehört zu einem anderen Speicherprofil."
    fresh, reason = radar.snapshot_freshness(snapshot, now=now, max_scan_hours=max_scan_hours)
    if not fresh:
        return None, reason
    rows = radar.view_rows(snapshot, now=now, max_scan_hours=max_scan_hours)
    row = next((r for r in rows if r.get("ticker") == str(ticker).upper()), None)
    if row is None:
        return None, "Keine Vollanalyse für diesen Wert im Scan."
    if row.get("gates") or row.get("status") in {radar.BLOCKED, radar.HISTORY}:
        return None, "Radar-Prüfung gesperrt: " + "; ".join(row.get("gates") or [row.get("status", "unbekannt")])
    audit = row.get("chart_audit") or {}
    if audit.get("policy") != POLICY or audit.get("full_context") is not True:
        return None, "Vollständiger Radar-Chartbeleg fehlt; Radar-Liste neu scannen."
    package = audit.get("watch_setup")
    ok, reason = validate(package, now=now, ticker=ticker)
    if not ok:
        return None, reason
    ok, reason = admission(package)
    if not ok:
        return None, "Setup-CRV für neue Übernahme unzureichend: " + reason
    return dict(ticker=row["ticker"], currency=package.get("currency"), watch_setup=deepcopy(package),
        radar=dict(scan_id=snapshot.get("scan_id"), started_at=snapshot.get("started_at"),
                   analyzed_at=row.get("analyzed_at"), grade=row.get("grade"), gates=[])), reason


def render_handoff(st, snapshot, selected, *, storage, holdings_known, held, now, max_scan_hours=24):
    if not selected:
        return
    with st.expander("Geprüftes Radar-Setup im Trade-Wächter verfolgen", expanded=False):
        if st.session_state.get("radar_watch_candidate") not in selected:
            st.session_state.pop("radar_watch_candidate", None)
        ticker = st.selectbox("Ausgewählter Radar-Kandidat", selected, key="radar_watch_candidate")
        basis, reason = watch_basis(snapshot, ticker, now=now, holdings_known=holdings_known, held=held,
            owner=getattr(storage, "user_id", None), max_scan_hours=max_scan_hours)
        if basis is None:
            st.info("Keine Übernahme möglich: " + reason)
            return
        st.caption("Feste Chartregeln aus dieser Vollanalyse. Die Übernahme nutzt den gespeicherten Beleg; sie startet keine neue Vollanalyse und eröffnet keine Position.")
        from modules.trade_watch_ui import capture_form
        capture_form(st, basis, key="radar_watch_" + str(snapshot.get("scan_id")) + "_" + ticker, manager=storage)
