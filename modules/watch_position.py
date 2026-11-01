"""Manually confirmed long positions; fixed-level notices, never orders.

An open position outlives the original idea's entry expiry. Only complete
post-confirmation minutes are inspected. Missing history is never reconstructed.
"""
from copy import deepcopy
from datetime import timedelta

import pandas as pd

from modules import trade_watch as w


def is_open(idea):
    return (idea.get("position") or {}).get("status") == "open"


def can_open(idea, *, now=None):
    from modules.watch_setups import idea_setup
    current = w.utc(now)
    if idea.get("position"):
        return False, "Für diese Idee wurde bereits eine Position erfasst."
    if idea.get("archived") or idea.get("paused") or idea.get("gap") or idea.get("state") != "triggered":
        return False, "Zuerst ist ein bestätigter Einstiegstrigger ohne Überwachungslücke erforderlich."
    if not str(idea.get("health") or "").startswith("Aktuelle 1m-Daten"):
        return False, "Vor Positionserfassung aktuelle Kurse ohne Abruffehler gezielt prüfen."
    valid, reason = idea_setup(idea, now=current)
    if not valid:
        return False, reason
    try:
        quote, checked = w.utc(idea["quote_at"]), w.utc(idea["last_attempt_at"])
        if not all(t <= current and current-t <= timedelta(minutes=3) for t in (quote, checked)):
            return False, "Vor Positionserfassung aktuelle Kurse gezielt prüfen."
    except (KeyError, ValueError, TypeError):
        return False, "Vor Positionserfassung aktuelle Kurse gezielt prüfen."
    if not w.in_session(current, idea["market"]):
        return False, "Positionserfassung hier nur während des regulären Handelsfensters."
    return True, "Ausführung beim Broker selbst bestätigen; ein Trigger eröffnet keine Position."


def open_position(idea, entry_price, *, now=None):
    now = w.utc(now)
    ok, reason = can_open(idea, now=now)
    if not ok:
        raise ValueError(reason)
    price = w.number(entry_price)
    if price is None or price <= 0:
        raise ValueError("Tatsächlichen positiven Ausführungskurs eintragen.")
    out = deepcopy(idea)
    out["position"] = dict(status="open", entry_price=price, opened_at=now.isoformat(),
        last_bar_end="", gap=False, target_at="", stop_at="", ambiguous_at="",
        health="Positionsüberwachung startet mit der nächsten vollständig nach Bestätigung begonnenen Minute.")
    w.event(out, "position_opened", now, f"Ausführung manuell bestätigt: {price:.6f}. Keine Brokerorder; Überwachung ab jetzt.")
    return out


def render_position(st, idea, *, ideas, manager, shown_time):
    """Record the user's broker execution; all side effects stay explicit."""
    from modules.trade_watch_overview import _level
    p = idea.get("position") or {}
    prefix = "position_" + idea["id"]
    def persist(updated):
        revised = [updated if item["id"] == idea["id"] else item for item in ideas]
        w.save_ideas(manager, revised)
        st.rerun()
    if not p:
        ok, reason = can_open(idea)
        if not ok:
            st.caption("Positionsstatus: kein ausgeführter Einstieg erfasst. " + reason)
            return
        with st.expander("Ausgeführten Einstieg bestätigen", expanded=False):
            st.caption("Nur eine tatsächlich beim Broker ausgeführte Position erfassen. Die Überwachung beginnt ab Bestätigung mit der nächsten vollen Minute. Frühere und innerhalb der Bestätigungsminute liegende Kontakte werden nicht nachträglich zugeordnet.")
            with st.form(prefix + "_open"):
                fill = st.number_input("Tatsächlicher Ausführungskurs", value=None, min_value=0., step=.01, format="%.6f", key=prefix+"_fill")
                confirmed = st.checkbox("Der Einstieg wurde beim Broker tatsächlich ausgeführt", key=prefix+"_executed")
                save = st.form_submit_button("Position als eröffnet bestätigen")
            if save:
                try:
                    if not confirmed:
                        raise ValueError("Tatsächliche Ausführung beim Broker bestätigen.")
                    persist(open_position(idea, fill))
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
        return
    st.markdown("**Positionsstatus: " + status_label(idea) + "**")
    st.caption("Manuell bestätigter Einstieg: " + _level(p.get("entry_price")) + " " + str(idea.get("currency") or "") +
               " · Bestätigt: " + shown_time(p.get("opened_at")))
    if not idea["entry"] <= p["entry_price"] <= idea["max_entry"]:
        st.warning("Erfasster Ausführungskurs liegt außerhalb des geplanten Entry-Bereichs. Gespeicherte Ziel-/Invalidierungslevels bleiben unverändert.")
    if p.get("ambiguous_at"):
        st.warning("Ziel und Invalidierung in derselben Minute erreicht: " + shown_time(p["ambiguous_at"]) + ". Die Reihenfolge ist aus diesen Daten nicht feststellbar.")
    if p.get("stop_at"):
        st.error("Invalidierung erreicht · Nachweiszeit: " + shown_time(p["stop_at"]) + ". Geplanten Ausstieg und Ausführung beim Broker prüfen.")
    if p.get("target_at"):
        st.success("Ziel erreicht · Nachweiszeit: " + shown_time(p["target_at"]) + ". Geplanten Ausstieg und Ausführung beim Broker prüfen.")
    if p.get("gap"):
        st.warning("Positionsüberwachung enthält eine Datenlücke. Weitere vorhandene Kontakte werden erfasst; Kontakte in fehlenden Minuten bleiben unbekannt.")
    st.caption("Kontakt = High/Low einer abgeschlossenen 1m-Kerze mit 10 Sekunden Datenpuffer. Nachweiszeit ist das Minutenende, kein genauer Tickzeitpunkt. Hinweise können verspätet eintreffen und bestätigen keinen Verkauf. Eine offene Position wird auch nach Ablauf der ursprünglichen Einstiegsidee weiter geprüft.")
    if p.get("status") == "closed":
        st.caption("Manuell bestätigter Ausstieg: " + _level(p.get("exit_price")) + " · " + shown_time(p.get("closed_at")) +
                   " · Kursrendite brutto: " + _level(p.get("return_pct")) + " % (ohne Gebühren/Steuern). Keine weitere Positionsabfrage.")
        return
    st.caption("Die Position bleibt bis zur manuellen Ausstiegsbestätigung offen. Vor dem Pausieren oder Archivieren den vollständigen Ausstieg erfassen. Teilverkäufe und ein Handelsjournal-Abgleich sind hier noch nicht enthalten.")
    with st.expander("Ausgeführten Ausstieg bestätigen", expanded=False):
        with st.form(prefix + "_close"):
            fill = st.number_input("Tatsächlicher Ausstiegskurs", value=None, min_value=0., step=.01, format="%.6f", key=prefix+"_exit")
            confirmed = st.checkbox("Die gesamte Position wurde beim Broker tatsächlich geschlossen", key=prefix+"_closed")
            save = st.form_submit_button("Position als geschlossen bestätigen")
        if save:
            try:
                if not confirmed:
                    raise ValueError("Vollständigen Ausstieg beim Broker bestätigen.")
                persist(close_position(idea, fill))
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))


def close_position(idea, exit_price, *, now=None):
    now = w.utc(now)
    if not is_open(idea):
        raise ValueError("Keine offene Position vorhanden.")
    price = w.number(exit_price)
    if price is None or price <= 0:
        raise ValueError("Tatsächlichen positiven Ausstiegskurs eintragen.")
    if now < w.utc(idea["position"]["opened_at"]):
        raise ValueError("Ausstieg liegt vor der Positionsbestätigung.")
    out = deepcopy(idea)
    p = out["position"]
    p.update(status="closed", exit_price=price, closed_at=now.isoformat(),
             return_pct=(price / p["entry_price"] - 1) * 100,
             health="Position manuell als geschlossen bestätigt")
    out["health"] = "Position geschlossen · keine weitere Kursprüfung"
    w.event(out, "position_closed", now, f"Ausstieg manuell bestätigt: {price:.6f}. Keine Brokerorder.")
    return out


def status_label(idea):
    p = idea.get("position") or {}
    if p.get("status") == "closed":
        return "Position geschlossen · manuell"
    if p.get("status") != "open":
        return None
    if p.get("ambiguous_at") and p.get("stop_at") == p.get("target_at"):
        return "Ziel & Invalidierung · Reihenfolge unklar"
    if p.get("stop_at"):
        return "Position offen · Invalidierung erreicht"
    if p.get("target_at"):
        return "Position offen · Ziel erreicht"
    return "Position offen · Überwachungslücke" if p.get("gap") else "Position offen"


def evaluate(idea, frame, *, now=None, fetched_at=None, fetch_error=""):
    out = deepcopy(idea)
    if not is_open(out) or out.get("paused") or out.get("archived"):
        return out
    now = w.utc(now)
    p = out["position"]
    out["last_attempt_at"] = now.isoformat()
    def health(text):
        p["health"] = out["health"] = text
    from modules.watch_setups import idea_setup
    ok, reason = idea_setup(out)  # Integrity required; entry expiry does not end a held position.
    if not ok:
        health("Positionsregeln nicht verifiziert: " + reason)
        return out
    if fetch_error:
        health("Abruffehler bei Positionsüberwachung: " + str(fetch_error)[:240])
        return out
    out["last_fetch_at"] = w.utc(fetched_at if fetched_at is not None else now).isoformat()
    try:
        bars = w.clean_minutes(frame, now, out["market"])
        start = w.utc(p["opened_at"]).ceil("min")
        previous = w.utc(p["last_bar_end"]) if p.get("last_bar_end") else start
    except (ValueError, KeyError, TypeError) as exc:
        health("Positionsdaten nicht verwertbar: " + str(exc))
        return out
    bars = bars.loc[bars.index >= start]
    if bars.empty:
        health("Wartet auf erste vollständig nach Positionsbestätigung begonnene Minute")
        return out
    latest = bars.index[-1]
    # Never roll back a later quote or the already processed cursor.
    if out.get("quote_at") and latest < w.utc(out["quote_at"]):
        health("Provider liefert älteren Positionskursstand")
        return out
    out.update(price=float(bars.iloc[-1].Close), quote_at=latest.isoformat())
    closed = bars.loc[bars.index + pd.Timedelta(minutes=1, seconds=10) <= now]
    expected_end = (now-pd.Timedelta(minutes=1, seconds=10)).floor("min")
    expected = w.expected_minutes(previous, expected_end, out["market"])
    if (expected is None or any(t not in closed.index for t in expected)) and not p["gap"]:
        p["gap"] = True
        w.event(out, "position_gap", now, "Positionsminuten fehlen; unbeobachtete Ziel-/Stopkontakte bleiben unbekannt. Vorhandene Nachweise werden weiter erfasst.")
    for t, row in closed.loc[closed.index >= previous].iterrows():
        end = t + pd.Timedelta(minutes=1)
        stop = float(row.Low) <= out["invalidation"]
        goal = w.number(out.get("target"))
        target = goal is not None and float(row.High) >= goal
        if stop and target and not p["stop_at"] and not p["target_at"]:
            p["ambiguous_at"] = end.isoformat()
            w.event(out, "position_ambiguous", end, "Ziel und Invalidierung in derselben abgeschlossenen Minute; Reihenfolge unbekannt, keine Ausführung behauptet.")
        delayed = now-end > timedelta(minutes=3) or not w.in_session(now, out["market"])
        suffix = " Historischer, verspätet gelieferter Nachweis." if delayed else ""
        if stop and not p["stop_at"]:
            p["stop_at"] = end.isoformat()
            w.event(out, "position_stop", end, "Invalidierung nach Positionsbestätigung erreicht; Ausstieg beim Broker prüfen." + suffix)
        if target and not p["target_at"]:
            p["target_at"] = end.isoformat()
            w.event(out, "position_target", end, "Ziel nach Positionsbestätigung erreicht; geplanten Ausstieg beim Broker prüfen." + suffix)
    if not closed.empty:
        p["last_bar_end"] = max(previous, closed.index[-1]+pd.Timedelta(minutes=1)).isoformat()
    fresh = now-latest <= timedelta(minutes=3) and w.in_session(now, out["market"])
    health("Aktuelle 1m-Daten · Tickzeit/Verzögerung nicht garantiert" if fresh else "Positionsdaten veraltet oder Handelsfenster geschlossen")
    if p["gap"]:
        health("Positionsüberwachung mit Datenlücke · unbeobachtete Kontakte unbekannt")
    return out
