"""Long trade ideas and conservative, provider-free intraday state transitions.

An idea freezes its original analysis. Monitoring never recalculates that
analysis, moves a stop, changes a score, or places an order.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import math
import re
from uuid import uuid4

import pandas as pd

from modules.data_freshness import close_at, session_day

NAMESPACE = "trade_watch_v1"
MAX_ACTIVE = 10
MODES = {"contact": 1, "close5": 5, "close15": 15}
LABELS = {
    "waiting": "Wartet", "near": "Nahe am Einstieg",
    "overextended": "Oberhalb der Entry-Grenze",
    "triggered": "Trigger bestätigt · gespeichert",
    "invalidated": "Invalidiert", "expired": "Abgelaufen",
}
TERMINAL = {"invalidated", "expired"}


def number(value):
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def utc(value=None):
    p = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if pd.isna(p) or p.tzinfo is None:
        raise ValueError("Zeitpunkt muss eine Zeitzone enthalten.")
    return p.tz_convert("UTC")


def market_for(ticker):
    symbol = str(ticker or "").upper()
    if symbol.endswith(".DE"):
        return "XETRA"
    if re.fullmatch(r"[A-Z][A-Z0-9-]{0,14}", symbol) or symbol in {"BRK.B", "BRK.A"}:
        return "US"
    raise ValueError("Version 1 unterstützt US-Aktienticker und Xetra-Ticker (.DE).")


def session_bounds(day, market):
    if market == "US":
        if session_day(day) is not True:
            return None
        return (pd.Timestamp(f"{day} 09:30", tz="America/New_York").tz_convert("UTC"),
                close_at(day).tz_convert("UTC"))
    if market == "XETRA" and day.weekday() < 5:
        # Conservative time window, not a complete Xetra holiday calendar.
        return (pd.Timestamp(f"{day} 09:00", tz="Europe/Berlin").tz_convert("UTC"),
                pd.Timestamp(f"{day} 17:30", tz="Europe/Berlin").tz_convert("UTC"))
    return None


def in_session(timestamp, market):
    t = utc(timestamp)
    local = t.tz_convert("America/New_York" if market == "US" else "Europe/Berlin")
    bounds = session_bounds(local.date(), market)
    return bool(bounds and bounds[0] <= t < bounds[1])


def active(idea):
    from modules.watch_setups import idea_setup
    return (not idea.get("archived") and not idea.get("paused") and idea.get("state") not in TERMINAL
            and idea_setup(idea)[0])


def make_idea(*, ticker, entry, invalidation, mode, expires_at, max_entry,
              target=None, currency="", note="", source=None, now=None):
    now = utc(now)
    symbol = str(ticker or "").strip().upper()
    market = market_for(symbol)
    e, stop, ceiling, goal = map(number, (entry, invalidation, max_entry, target))
    if e is None or stop is None or not 0 < stop < e:
        raise ValueError("Für Long muss gelten: 0 < Invalidierung < Entry.")
    if ceiling is None or ceiling < e:
        raise ValueError("Die obere Entry-Grenze muss mindestens dem Entry entsprechen.")
    if target not in (None, "", 0, 0.0) and (goal is None or goal <= e):
        raise ValueError("Ein angegebenes Long-Ziel muss oberhalb des Entry liegen.")
    if mode not in MODES:
        raise ValueError("Unbekannte Bestätigungsregel.")
    expiry = utc(expires_at)
    if expiry <= now or expiry > now + timedelta(days=30):
        raise ValueError("Gültigkeit muss in der Zukunft und innerhalb von 30 Tagen liegen.")
    idea = dict(id=uuid4().hex, ticker=symbol, market=market, entry=e,
                invalidation=stop, max_entry=ceiling, target=goal if goal and goal > e else None,
                mode=mode, currency=str(currency or "").strip()[:12], note=str(note or "")[:500],
                source=deepcopy(source or {}), created_at=now.isoformat(),
                armed_at=now.isoformat(), expires_at=expiry.isoformat(), state="waiting",
                paused=False, archived=False, health="Noch nicht geprüft", events=[],
                last_bar_end="", last_attempt_at="", last_fetch_at="", quote_at="", price=None,
                gap=False, triggered_at="")
    from modules.watch_setups import idea_setup
    ok, reason = idea_setup(idea, now=now)
    if not ok:
        raise ValueError(reason)
    return idea


def event(idea, kind, at, text):
    events = idea.setdefault("events", [])
    item = dict(kind=kind, at=utc(at).isoformat(), text=text)
    if not events or events[-1] != item:
        events.append(item)
    idea["events"] = events[-80:]


def load_ideas(manager):
    result = manager.load_result(NAMESPACE, force_refresh=True)
    if not result.ok:
        raise RuntimeError("Gespeicherte Trade-Ideen konnten nicht gelesen werden: " + str(result.error))
    if not result.found:
        return []
    if not isinstance(result.data, dict) or result.data.get("version") != 1 or not isinstance(result.data.get("ideas"), list):
        raise RuntimeError("Trade-Wächter-Speicherformat ist ungültig; vorhandene Daten bleiben erhalten.")
    ideas = deepcopy(result.data["ideas"])
    if any(not isinstance(i, dict) or not i.get("id") for i in ideas):
        raise RuntimeError("Ungültige gespeicherte Trade-Idee; vorhandene Daten bleiben erhalten.")
    from modules.watch_setups import idea_setup
    for idea in ideas:
        ok, reason = idea_setup(idea)
        if not ok:
            # Keep old records and their history. They never become evidence of
            # a chart setup merely because this version was installed.
            idea["setup_blocked"] = reason
        else:
            idea.pop("setup_blocked", None)
    return ideas


def save_ideas(manager, ideas):
    result = manager.save_result(NAMESPACE, {"version": 1, "ideas": deepcopy(ideas)})
    if not result.ok:
        raise RuntimeError("Trade-Ideen konnten nicht gespeichert werden: " + str(result.error))


def add_idea(manager, idea):
    from modules.watch_setups import idea_setup
    ok, reason = idea_setup(idea, now=utc())
    if not ok:
        raise ValueError(reason)
    ideas = load_ideas(manager)
    if sum(not i.get("archived") and not i.get("setup_blocked") and i.get("state") not in TERMINAL for i in ideas) >= MAX_ACTIVE:
        raise ValueError(f"Höchstens {MAX_ACTIVE} laufende Ideen; zuerst eine Idee archivieren.")
    fields = ("ticker", "entry", "invalidation", "max_entry", "mode", "target")
    if any(not i.get("archived") and not i.get("setup_blocked") and i.get("state") not in TERMINAL and
           all(i.get(k) == idea.get(k) for k in fields) for i in ideas):
        raise ValueError("Diese Handelsidee wird bereits verfolgt.")
    ideas.append(idea)
    save_ideas(manager, ideas)


def clean_minutes(frame, now, market):
    """Require aware 1m OHLC timestamps; never silently guess UTC for naive bars."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Keine Intraday-Kerzen geliefert.")
    required = ["Open", "High", "Low", "Close"]
    if not all(k in frame for k in required) or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("1m-OHLC-Daten oder eindeutige Kerzen-Zeitzone fehlen.")
    out = frame[required].copy()
    out.index = out.index.tz_convert("UTC")
    if out.index.has_duplicates or any(t.second or t.microsecond for t in out.index):
        raise ValueError("Doppelte oder nicht minutengenaue Kerzenzeitpunkte.")
    for key in required:
        out[key] = pd.to_numeric(out[key], errors="coerce")
    valid = out.apply(lambda col: col.map(lambda x: number(x) is not None and x > 0)).all(axis=1)
    valid &= out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)
    valid &= out["Low"] <= out[["Open", "Close", "High"]].min(axis=1)
    if not valid.all():
        raise ValueError("Ungültige OHLC-Kerzen; keine neue Triggerfreigabe.")
    if any(t > now for t in out.index):
        raise ValueError("Kerzenzeit liegt in der Zukunft.")
    out = out.loc[[in_session(t, market) for t in out.index]].sort_index()
    if out.empty:
        raise ValueError("Keine Kerzen aus dem regulären Handelsfenster.")
    return out


def expected_minutes(start, end, market):
    if end < start:
        return []
    # Only the bounded provider window is traversed; old monitoring gaps fail closed.
    if end - start > timedelta(days=7):
        return None
    return [t for t in pd.date_range(start, end, freq="min") if in_session(t, market)]


def evaluate(idea, frame, *, now=None, fetched_at=None, fetch_error=""):
    """Process post-arming bars chronologically; invalidation always wins ties.

    Closed-bar confirmation requires every 1m constituent and a 10s publication
    buffer. A historical trigger is logged, never emitted as a fresh entry.
    """
    out = deepcopy(idea)
    now = utc(now)
    from modules.watch_setups import idea_setup
    ok, reason = idea_setup(out)
    if not ok:
        out["setup_blocked"] = reason
        out["health"] = "Kein geprüftes Chartsetup · nicht überwacht"
        return out
    if out.get("archived") or out.get("paused") or out["state"] in TERMINAL:
        return out
    out["last_attempt_at"] = now.isoformat()
    if now >= utc(out["expires_at"]):
        out.update(state="expired", health="Gültigkeit beendet")
        event(out, "expired", now, "Idee abgelaufen; keine neue Freigabe.")
        return out
    if fetch_error:
        out["health"] = "Abruffehler: " + str(fetch_error)[:240]
        return out
    out["last_fetch_at"] = utc(fetched_at if fetched_at is not None else now).isoformat()
    try:
        bars = clean_minutes(frame, now, out["market"])
    except ValueError as exc:
        out["health"] = str(exc)
        return out
    arm = utc(out["armed_at"])
    start = arm.ceil("min")
    bars = bars.loc[bars.index >= start]
    if bars.empty:
        out["health"] = "Wartet auf erste vollständig nach Anlage begonnene 1m-Kerze"
        return out
    latest = bars.index[-1]
    previous = utc(out["last_bar_end"]) if out.get("last_bar_end") else start
    if latest < previous and now - latest < timedelta(minutes=3):
        out["health"] = "Provider liefert älteren Stand als die letzte Prüfung; kein neuer Trigger"
        return out
    price = float(bars.iloc[-1]["Close"])
    out.update(price=price, quote_at=latest.isoformat())
    fresh = now - latest <= timedelta(minutes=3) and in_session(now, out["market"])
    out["health"] = "Aktuelle 1m-Daten · Tickzeit/Verzögerung nicht garantiert" if fresh else "Daten veraltet oder Handelsfenster geschlossen"
    closed = bars.loc[bars.index + pd.Timedelta(minutes=1, seconds=10) <= now]
    last_expected = min(latest, (now - pd.Timedelta(minutes=1, seconds=10)).floor("min"))
    expected = expected_minutes(previous, last_expected, out["market"])
    if (expected is None or any(t not in closed.index for t in expected)) and not out.get("gap"):
        out["gap"] = True
        event(out, "gap", now, "Minuten fehlen seit Anlage/letzter Prüfung; neu anlegen für neue Trigger.")

    # Include the running minute for defensive invalidation; only post-arming bars.
    recent = bars.loc[bars.index >= previous]
    # A completed confirmation can be delivered late, so inspect all its constituents
    # while only emitting the first fresh event. Retain a confirmed event thereafter.
    n = MODES[out["mode"]]
    candidates = []
    if n > 1 and not closed.empty:
        # Only the current unfinished confirmation group and new evidence are needed.
        confirmation_bars = closed.loc[closed.index >= previous.floor(f"{n}min")]
        for t, group in confirmation_bars.resample(f"{n}min", origin="start_day"):
            if t < start or len(group) != n or group.index[0] != t or group.index[-1] != t + pd.Timedelta(minutes=n - 1):
                continue
            end = t + pd.Timedelta(minutes=n)
            if end > previous and end <= now - pd.Timedelta(seconds=10):
                candidates.append((end, float(group.iloc[-1]["Close"])))

    # Order actual evidence by time; a lower low inside the trigger bar invalidates first.
    stop_times = [t + pd.Timedelta(minutes=1) for t, row in recent.iterrows()
                  if float(row["Low"]) <= out["invalidation"]]
    first_stop = min(stop_times) if stop_times else None
    for end, close in candidates:
        if first_stop is not None and first_stop <= end:
            break
        if out["entry"] <= close <= out["max_entry"] and out["state"] != "triggered":
            if fresh and now - end <= timedelta(minutes=3) and not out.get("gap"):
                out.update(state="triggered", triggered_at=end.isoformat())
                event(out, "trigger", end, f"{n}m-Schluss {close:.4f} innerhalb der Entry-Grenzen.")
            else:
                event(out, "historical", end, "Vergangenes Schluss-Signal; keine frische Freigabe.")
    if first_stop is not None:
        out.update(state="invalidated", health="Invalidierungsniveau in einer 1m-Kerze erreicht")
        event(out, "invalidated", min(first_stop, now), "Low auf/unter Invalidierung; keine Aussage über eine Broker-Ausführung.")
    elif fresh and not out.get("gap") and out["state"] != "triggered":
        if price > out["max_entry"]:
            out["state"] = "overextended"
        elif out["mode"] == "contact" and price >= out["entry"]:
            out.update(state="triggered", triggered_at=now.isoformat())
            event(out, "trigger", now, "1m-Kursmomentaufnahme innerhalb der Entry-Grenzen; kein Tick-/Schlussnachweis.")
        else:
            out["state"] = "near" if abs(price / out["entry"] - 1) <= .005 else "waiting"
    if not closed.empty:
        out["last_bar_end"] = (closed.index[-1] + pd.Timedelta(minutes=1)).isoformat()
    if out.get("gap") and out["state"] not in TERMINAL:
        out["health"] = "Überwachungslücke · für neue Trigger archivieren und neu anlegen"
    return out


def poll(ideas, provider, *, now=None, force=False):
    """One lightweight 1m history per active ticker; no info, benchmark or analyzer."""
    from modules.provider_manager import history_refresh_scope, MarketDataRateLimitError
    current = utc(now)
    result = deepcopy(ideas)
    responses = {}
    token = "trade-watch-" + uuid4().hex
    rate_error = ""
    for pos, idea in enumerate(result):
        if not active(idea):
            continue
        if current >= utc(idea["expires_at"]):
            result[pos] = evaluate(idea, None, now=current)
            continue
        if not in_session(current, idea["market"]):
            idea["health"] = "Handelsfenster geschlossen · keine automatische Prüfung"
            continue
        last = utc(idea["last_attempt_at"]) if idea.get("last_attempt_at") else None
        if not force and last is not None and (current - last).total_seconds() < 30:
            continue
        ticker = idea["ticker"]
        if ticker not in responses:
            if rate_error:
                responses[ticker] = (None, rate_error)
            else:
                try:
                    with history_refresh_scope(token):
                        frame = provider.get_history_light(ticker, period="5d", interval="1m", prepost=False, auto_adjust=False)
                    responses[ticker] = (frame, "")
                except Exception as exc:
                    responses[ticker] = (None, str(exc))
                    if isinstance(exc, MarketDataRateLimitError):
                        rate_error = "Provider gesperrt/begrenzt; keine weiteren Abfragen in diesem Lauf."
        frame, error = responses[ticker]
        fetched = ((frame.attrs.get("chsm_history") or {}).get("fetched_at")
                   if isinstance(frame, pd.DataFrame) else None)
        # Evaluate with the actual completion time unless a deterministic test time is supplied.
        result[pos] = evaluate(idea, frame, now=current if now is not None else utc(),
                               fetched_at=fetched, fetch_error=error)
    return result
