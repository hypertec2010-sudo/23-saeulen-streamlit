"""Descriptive results of manually confirmed, fully closed watcher positions.

No historical price replay, simulated fills, portfolio return or order handling.
"""
import pandas as pd
from modules.trade_watch import number, utc


def trade_results(ideas, *, now=None):
    from modules.watch_setups import idea_setup, KINDS
    now = utc(now)
    records, seen, excluded = [], set(), 0
    for idea in ideas:
        p = idea.get("position") or {}
        if not isinstance(p, dict):
            excluded += 1
            continue
        if p.get("status") != "closed":
            continue
        key = idea.get("id")
        entry, exit = number(p.get("entry_price")), number(p.get("exit_price"))
        try:
            if not p.get("opened_at") or not p.get("closed_at"):
                raise ValueError("Bestätigungszeiten fehlen")
            opened, closed = utc(p.get("opened_at", "")), utc(p.get("closed_at", ""))
            valid_time = opened <= closed <= now
        except (TypeError, ValueError):
            valid_time = False
        if not isinstance(key, str) or not key or key in seen or entry is None or exit is None or min(entry, exit)<=0 or not valid_time:
            excluded += 1
            continue
        seen.add(key)
        source = idea.get("source") or {}
        try:
            verified = idea_setup(idea)[0] if isinstance(source, dict) else False
        except (TypeError, ValueError, KeyError, AttributeError):
            verified = False
        package = (source.get("watch_setup") or {}) if isinstance(source, dict) else {}
        stop = number(idea.get("invalidation"))
        risk = entry-stop if verified and stop is not None and 0 < stop < entry else None
        try:
            checked_until = utc(p["last_bar_end"]) if p.get("last_bar_end") else None
            recent_check = (checked_until is not None and opened.ceil("min") < checked_until <= closed
                            and (closed-checked_until).total_seconds() <= 180)
        except (TypeError, ValueError):
            recent_check = False
        clean = p.get("gap") is False and not p.get("ambiguous_at") and verified and recent_check
        records.append({"Ticker":idea.get("ticker"),
            "Chartsetup":KINDS.get(package.get("kind"), "Muster nicht benannt") if verified else "Chartbeleg nicht verifiziert",
            "Einstieg bestätigt":opened.isoformat(), "Ausstieg bestätigt":closed.isoformat(),
            "Ausführung Einstieg":entry, "Ausführung Ausstieg":exit,
            "Kursrendite %":(exit/entry-1)*100,
            "Ergebnis R":(exit-entry)/risk if risk else None,
            "Tage seit Bestätigung":(closed-opened).total_seconds()/86400,
            "Verlauf für Mustervergleich":clean,
            "Hinweis":"Positionsprüfung vor Abschluss, keine bekannte Lücke" if clean else "Datenlücke, fehlende/alte Positionsprüfung, unklare Reihenfolge oder unvollständiger Beleg"})
    columns=["Ticker","Chartsetup","Einstieg bestätigt","Ausstieg bestätigt","Ausführung Einstieg",
             "Ausführung Ausstieg","Kursrendite %","Ergebnis R","Tage seit Bestätigung",
             "Verlauf für Mustervergleich","Hinweis"]
    return pd.DataFrame(records,columns=columns), excluded


def summarize(frame):
    if frame.empty:
        return {"Trades":0,"Gewinne":0,"Verluste":0,"Neutral":0,"Trefferquote %":None,
                "Ø Kursrendite %":None,"Ø Ergebnis R":None,"R verfügbar":0,"Mit unvollständigem Verlauf":0}
    returns = frame["Kursrendite %"]
    risks = pd.to_numeric(frame["Ergebnis R"], errors="coerce").dropna()
    return {"Trades":len(frame),"Gewinne":int((returns>0).sum()),"Verluste":int((returns<0).sum()),
            "Neutral":int((returns==0).sum()),"Trefferquote %":float((returns>0).mean()*100),
            "Ø Kursrendite %":float(returns.mean()),"Ø Ergebnis R":float(risks.mean()) if len(risks) else None,
            "R verfügbar":len(risks),"Mit unvollständigem Verlauf":int((~frame["Verlauf für Mustervergleich"]).sum())}


def render_results(st, ideas, *, now=None):
    with st.expander("Auswertung abgeschlossener Wächter-Trades", expanded=False):
        frame, excluded = trade_results(ideas, now=now)
        if excluded:
            st.warning(f"{excluded} unvollständige oder doppelte Abschlussdatensätze nicht ausgewertet.")
        if frame.empty:
            st.info("Noch keine auswertbaren manuell bestätigten Abschlüsse. Ein Ziel- oder Stopkontakt allein zählt nicht als ausgeführter Ausstieg.")
            return
        clean_only = st.checkbox("Nur Trades mit geprüftem Verlauf auswerten", value=False, key="watch_results_clean")
        filtered = frame.loc[frame["Verlauf für Mustervergleich"]] if clean_only else frame
        total = summarize(filtered)
        st.table(pd.DataFrame([total]))
        groups = [{"Chartsetup":kind, **summarize(group)} for kind,group in filtered.groupby("Chartsetup",sort=True)]
        if groups:
            st.dataframe(pd.DataFrame(groups),hide_index=True,use_container_width=True)
        st.dataframe(filtered,hide_index=True,use_container_width=True)
        st.caption("Manuell bestätigte tatsächliche Kurse; brutto ohne Gebühren, Steuern oder Währungseffekte. Durchschnitt je Trade, keine Depot- oder Gesamtrendite. R = Kursgewinn/-verlust geteilt durch das anfängliche Risiko zwischen Ausführung und festem, verifiziertem Invalidierungslevel; sonst kein R. Zeitangaben messen den Abstand der Bestätigungen. Archivierte Abschlüsse bleiben enthalten. Der Verlaufsfilter verlangt mindestens eine verarbeitete Positionsminute und eine höchstens drei Minuten vor Ausstiegsbestätigung liegende Prüfung, ohne bekannte Lücke/Mehrdeutigkeit. Das ist kein lückenloser Ticknachweis. Kleine selbst ausgewählte Stichproben belegen keine Profitabilität eines Musters.")
