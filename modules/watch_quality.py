"""Explain existing chart evidence and enforce a transparent new-idea RR rule.

No profit prediction, provider requests, new targets or changes to frozen levels.
Existing positions/ideas are never stopped by this admission rule.
"""
from __future__ import annotations

from modules.trade_watch import number

MIN_ENTRY_RR = 1.5


def assess(levels):
    """Use the actual fixed levels, never a source analysis CRV/stop/target."""
    entry, stop, cap, target = (number(levels.get(k)) for k in
                                ("entry", "invalidation", "max_entry", "target"))
    result = dict(allowed=False, rr_entry=None, rr_cap=None, minimum=MIN_ENTRY_RR,
                  risk_pct=None, reason="Gespeicherte Kursregeln unvollständig oder ungültig.")
    if any(v is None for v in (entry, stop, cap)) or not 0 < stop < entry <= cap:
        return result
    result["risk_pct"] = (cap - stop) / cap * 100
    if target is None or target <= 0:
        return {**result, "reason": "Kein technisches Ziel hinterlegt; CRV nicht prüfbar."}
    result.update(rr_entry=(target - entry) / (entry - stop),
                  rr_cap=(target - cap) / (cap - stop))
    if target <= cap:
        return {**result, "reason": "Ziel liegt auf/unter der oberen Entry-Grenze; kein positiver Zielspielraum im gesamten Entryband."}
    if result["rr_cap"] < MIN_ENTRY_RR:
        return {**result, "reason": f"CRV am oberen Entry-Rand unter {MIN_ENTRY_RR:.1f}R."}
    return {**result, "allowed": True,
            "reason": f"CRV im gesamten Entryband mindestens {MIN_ENTRY_RR:.1f}R."}


def admission(levels):
    report = assess(levels)
    return report["allowed"], report["reason"]


def quality_rows(package):
    """Describe only evidence actually present in the fingerprinted package."""
    from modules.watch_setups import evidence_valid, KINDS, PHASES
    verified = evidence_valid(package)
    p = package if verified else {}
    rows = [{"Prüfung": "Chartbeleg", "Befund": "Unveränderter gespeicherter Beleg" if verified else "Nicht nachgewiesen"}]
    if verified:
        rows += [{"Prüfung": "Chartsetup", "Befund": KINDS.get(p.get("kind"), p.get("label") or "Nicht benannt")},
                 {"Prüfung": "Setup-Phase bei Erkennung", "Befund": PHASES.get(p.get("phase"), "Nicht dokumentiert")},
                 {"Prüfung": "Abgeschlossene Chartbasis", "Befund": p.get("basis_date") or "Nicht dokumentiert"}]
        rows += [{"Prüfung": "Strukturbedingung", "Befund": str(check)} for check in p.get("checks", [])]
        evidence = p.get("evidence") or {}
        for field, label in (("width_pct", "Breite der Konsolidierung"),
                             ("pullback_depth_pct", "Tiefe des Rücksetzers")):
            value = number(evidence.get(field))
            if value is not None:
                rows.append({"Prüfung": label, "Befund": f"{value:.2f} %".replace(".", ",")})
    rows += [{"Prüfung": "Ausbruchsvolumen / Liquidität / Spread", "Befund": "Keine umfassende Qualitätsfreigabe vorhanden"},
             {"Prüfung": "Markt- und Branchenstärke", "Befund": "Nicht Bestandteil dieser Qualitätsprüfung"}]
    return rows


def render_quality(st, levels, package):
    import pandas as pd
    st.markdown("**Setup-Qualität und Chancen-Risiko-Prüfung**")
    with st.expander("Begründung der Setup-Qualität", expanded=False):
        st.table(pd.DataFrame(quality_rows(package)))
    report = assess(levels)
    def rr(v):
        return "Nicht bewertbar" if v is None else f"{v:.2f}R".replace(".", ",")
    st.table(pd.DataFrame([
        {"Prüfung": "CRV am Entry-Auslöser", "Befund": rr(report["rr_entry"])},
        {"Prüfung": "CRV am oberen Entry-Rand", "Befund": rr(report["rr_cap"])},
        {"Prüfung": "Mindest-CRV für neue Übernahmen", "Befund": rr(report["minimum"])},
    ]))
    (st.success if report["allowed"] else st.warning)(report["reason"])
    st.caption("Brutto-CRV aus festen Regeln: (Ziel − Einstieg) / (Einstieg − Invalidierung). Kosten, Spread und Slippage fehlen. 1,5R ist eine Startregel, keine empirisch bestätigte optimale Schwelle. Strukturqualität ist keine Gewinnwahrscheinlichkeit.")
    return report


def rule_rows(idea):
    from modules.trade_watch import LABELS
    from modules.trade_watch_overview import _level
    unit = str(idea.get("currency") or "Kurseinheit unbekannt")
    def price(key):
        v = _level(idea.get(key))
        return v if v == "–" else v + " " + unit
    mode = {"close5": "Neuer vollständiger 5-Minuten-Schluss im Entry-Bereich",
            "close15": "Neuer vollständiger 15-Minuten-Schluss im Entry-Bereich",
            "contact": "Kurskontakt in einer 1-Minuten-Momentaufnahme"}
    state = "Archiviert" if idea.get("archived") else "Pausiert" if idea.get("paused") else LABELS.get(idea.get("state"), "Unbekannt")
    return [{"Regel": "Aktie", "Wert": str(idea.get("ticker") or "–")},
            {"Regel": "Gespeicherter Status", "Wert": state},
            {"Regel": "Zuletzt erfasster Kurs", "Wert": price("price")},
            {"Regel": "Entry-Auslöser", "Wert": price("entry")},
            {"Regel": "Oberste Entry-Grenze", "Wert": price("max_entry")},
            {"Regel": "Invalidierung", "Wert": price("invalidation")},
            {"Regel": "Ziel (Referenz)", "Wert": price("target")},
            {"Regel": "Einstiegsbestätigung", "Wert": mode.get(idea.get("mode"), "Nicht dokumentiert")}]


def render_rules(st, idea, *, shown_time):
    import pandas as pd
    from modules.watch_setups import idea_setup
    st.markdown("**Gespeicherte Handelsidee / Regeln**")
    st.table(pd.DataFrame(rule_rows(idea)))
    st.caption("Gültig bis: " + shown_time(idea.get("expires_at")) + " · Kurszeit: " + shown_time(idea.get("quote_at")))
    ok, reason = idea_setup(idea)
    if not ok:
        st.warning("Chartbeleg / Regelzuordnung: " + reason)
    st.caption("Status und Kurs entsprechen dem gespeicherten Stand. Ablauf und Datenzustand in der Übersicht beachten. Ein gespeicherter Trigger ist kein erneutes Einstiegssignal. Das Ziel wird bisher als Referenz angezeigt; kein automatisches Ziel-erreicht-Signal und keine Brokerorder.")
    render_quality(st, idea, (idea.get("source") or {}).get("watch_setup"))
    st.caption("Diese CRV-Prüfung gilt für neue Übernahmen. Bereits gespeicherte Ideen, Levels und deren Überwachung bleiben unverändert.")
