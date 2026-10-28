"""Provider-free chart candidates, stages and explainable rejection statistics.

These are transparent initial heuristics, not validated profit forecasts. A
forming/old-trigger candidate is informational and can never arm the watcher.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json

import pandas as pd

from modules.trade_watch import utc, number, market_for, session_bounds
from modules.structural_trade_plan import confirmed_support_stop

POLICY = "chart_watch_v30.21bz"
PREVIOUS_POLICY = "chart_watch_v30.21by"
VERSION = 1
PHASES = {"ready": "Bestätigungsbereit", "forming": "Setup entsteht",
          "triggered": "Auslöser bereits erreicht", "missed": "Entry verpasst / überdehnt",
          "invalid": "Struktur ungültig", "blocked": "Prüfung gesperrt", "none": "Kein Chartsetup"}
RANK = {"ready": 7, "triggered": 6, "missed": 5, "invalid": 4, "forming": 3, "none": 1, "blocked": 0}
KINDS = {"range_breakout": "Konsolidierung vor Ausbruch",
         "support_pullback": "Pullback an bestätigter Unterstützung",
         "higher_low_pullback": "Trend-Pullback mit höherem Tief",
         "breakout_retest": "Retest nach bestätigtem Ausbruch"}


def fingerprint(package):
    return sha256(json.dumps({k: v for k, v in package.items() if k != "fingerprint"},
                             sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sealed(package):
    out = deepcopy(package)
    out["fingerprint"] = fingerprint(out)
    return out


def form_id(package):
    # Rerender clocks do not change consent; actual evidence/levels still do.
    return fingerprint({k: v for k, v in package.items() if k not in {"fingerprint", "detected_at"}})[:20]


def evidence_valid(package):
    try:
        return (isinstance(package, dict) and package.get("policy") in {POLICY, PREVIOUS_POLICY}
                and package.get("version") == VERSION and package.get("fingerprint") == fingerprint(package))
    except (ValueError, TypeError, OverflowError):
        return False


def validate(package, *, now=None, ticker=None):
    if not isinstance(package, dict) or package.get("policy") not in {POLICY, PREVIOUS_POLICY} or package.get("version") != VERSION:
        return False, "Kein geprüfter Chartbeleg; Aktie neu analysieren."
    try:
        if package.get("fingerprint") != fingerprint(package):
            return False, "Setupdaten verändert; neue Analyse erforderlich."
        if package.get("eligible") is not True or package.get("phase", "ready") != "ready":
            return False, str(package.get("reason") or "Kein bestätigungsbereites Chartsetup.")
        entry, stop, cap = (number(package.get(k)) for k in ("entry", "invalidation", "max_entry"))
        if not all(x is not None for x in (entry, stop, cap)) or not 0 < stop < entry <= cap:
            return False, "Abgeleitete Setup-Levels sind ungültig."
        if ticker is not None and str(ticker).upper() != package["ticker"]:
            return False, "Chartsetup gehört zu anderem Ticker."
        if now is not None and not utc(package["detected_at"]) <= utc(now) < utc(package["valid_until"]):
            return False, "Chartbeleg abgelaufen oder Zeitstand ungültig; neu scannen."
    except (KeyError, ValueError, TypeError, OverflowError):
        return False, "Chartbeleg unvollständig oder ungültig."
    return True, "Strukturbeleg und feste Kursregeln vorhanden."


def idea_setup(idea, *, now=None):
    package = (idea.get("source") or {}).get("watch_setup")
    ok, reason = validate(package, now=now, ticker=idea.get("ticker"))
    if not ok:
        return False, reason
    if any(idea.get(k) != package.get(k) for k in ("entry", "invalidation", "max_entry", "target", "mode")):
        return False, "Kursregel stimmt nicht mit der erkannten Struktur überein."
    try:
        if utc(idea["expires_at"]) > utc(package["valid_until"]):
            return False, "Idee reicht über die Gültigkeit des Chartbelegs hinaus."
    except (KeyError, TypeError, ValueError):
        return False, "Gültigkeit der Idee fehlt."
    return True, reason


def completed_day(now, market):
    day = now.tz_convert("America/New_York" if market == "US" else "Europe/Berlin").date()
    for n in range(12):
        bounds = session_bounds(day, market)
        if bounds and (n or now >= pd.Timestamp(bounds[1].to_pydatetime() + timedelta(minutes=30))):
            return day
        day -= timedelta(days=1)
    raise ValueError("Vorherige Börsensitzung nicht verifiziert.")


def next_expiry(day, market):
    for n in range(1, 12):
        bounds = session_bounds(day + timedelta(days=n), market)
        if bounds:
            return bounds[1]
    raise ValueError("Nächste Börsensitzung nicht verifiziert.")


def pivots(values, *, high=False):
    points = []
    for i in range(3, len(values) - 3):
        v = float(values.iloc[i])
        sides = list(values.iloc[i - 3:i]) + list(values.iloc[i + 1:i + 4])
        is_pivot = v > max(sides) if high else v < min(sides)
        if is_pivot:
            points.append((i, v))
    return points


def pairs(points, tolerance=.015):
    return [(a, b) for pos, a in enumerate(points) for b in points[pos + 1:]
            if b[0] - a[0] >= 6 and abs(b[1] / a[1] - 1) <= tolerance]


def attempt(kind, code, reason, phase="none", **fields):
    return dict(kind=kind, label=KINDS[kind], phase=phase, reason_code=code, reason=reason, **fields)


def atr_value(data):
    previous = data.Close.shift()
    ranges = pd.concat([data.High-data.Low, (data.High-previous).abs(),
                        (data.Low-previous).abs()], axis=1).max(axis=1)
    return float(ranges.tail(14).mean())


def levels(kind, entry_level, support, atr, price, *, evidence, checks, near=True):
    entry = round(float(entry_level) + max(float(entry_level) * .001, atr * .1), 6)
    stop = round(float(support) - max(float(support) * .006, atr * .25), 6)
    cap = round(entry + min(.5 * atr, .01 * entry), 6)
    common = dict(entry=entry, invalidation=stop, max_entry=cap, evidence=evidence, checks=checks, atr_reference=atr)
    if not 0 < stop < entry or (entry - stop) / entry > .12:
        return attempt(kind, "stop_distance", "Struktur-Invalidierung weiter als 12 % vom Auslöser entfernt.", **common)
    if price <= stop:
        return attempt(kind, "structure_broken", "Kurs hat die Struktur-Invalidierung erreicht.", "invalid", **common)
    if price > cap:
        return attempt(kind, "entry_missed", "Kurs bereits oberhalb der abgeleiteten Entry-Grenze.", "missed", **common)
    if price >= entry:
        return attempt(kind, "trigger_crossed", "Auslöser auf der Chartbasis bereits erreicht; kein neuer Intraday-Trigger behauptet.", "triggered", **common)
    if not near:
        return attempt(kind, "not_near_trigger", "Struktur entsteht; Kurs noch zu weit vom Auslöser entfernt.", "forming", **common)
    return attempt(kind, "ready", "Chartstruktur bestätigt; wartet auf einen neuen 5-Minuten-Schluss im Entryband.", "ready", **common)


def range_candidate(data, length, atr, price):
    kind = "range_breakout"
    if len(data) < length + 1:
        return attempt(kind, "range_history_short", f"{length}T-Basis plus neueste Tageskerze fehlen.", window=length)
    # Freeze the base before the latest completed bar, allowing the latest bar
    # to be classified as waiting / crossed / missed rather than moving the line.
    base = data.iloc[-length - 1:-1]
    tops, lows = pivots(base.High, high=True), pivots(base.Low)
    top_pairs, low_pairs = pairs(tops, .01), pairs(lows)
    if not top_pairs or not low_pairs:
        partial = bool((top_pairs and lows) or (low_pairs and tops))
        return attempt(kind, "range_pivots_incomplete", f"{length}T: Ober-/Unterkante noch nicht jeweils zweimal bestätigt.",
                       "forming" if partial else "none", window=length,
                       evidence={"high_pivots": len(tops), "low_pivots": len(lows)})
    top_pair = max(top_pairs, key=lambda pair: pair[1][0])
    low_pair = max(low_pairs, key=lambda pair: pair[1][0])
    resistance, support = max(p[1] for p in top_pair), min(p[1] for p in low_pair)
    width = (resistance - support) / resistance
    if not .02 <= width <= .12:
        return attempt(kind, "range_width", f"{length}T: Range-Breite außerhalb 2–12 %.", window=length)
    if base.High.max() > resistance * 1.01 or base.Low.min() < support * .985:
        return attempt(kind, "range_boundaries", f"{length}T: Verlauf passt nicht zu den getesteten Grenzen.", window=length)
    base_atr = atr_value(data.iloc[:-1])
    out = levels(kind, resistance, support, base_atr, price,
        near=price >= resistance * .97,
        evidence=dict(window=length, resistance=resistance, support=support, width_pct=width * 100,
            resistance_dates=[base.index[p[0]].date().isoformat() for p in top_pair],
            support_dates=[base.index[p[0]].date().isoformat() for p in low_pair]),
        checks=[f"Bestätigte {length}T-Konsolidierung; Breite 2–12 %",
                "Zwei strikte 3/3-Pivot-Hochs und -Tiefs; mindestens 6 Kerzen Abstand",
                "Oberkante innerhalb 1 %, Unterkante innerhalb 1,5 % getestet"])
    out["window"] = length
    return out


def pullback_candidate(data, atr, price, *, higher=False):
    kind = "higher_low_pullback" if higher else "support_pullback"
    recent = data.tail(60)
    found = pivots(recent.Low)
    candidates = []
    if higher:
        # Compare the latest confirmed pivot to its immediate predecessor.
        # An older lower low cannot conceal an intervening lower-high/lower-low sequence.
        if len(found) >= 2:
            current, prior = found[-1], found[-2]
            if (6 <= current[0]-prior[0] <= 30 and current[1] >= prior[1]*1.005
                    and 3 <= len(recent)-1-current[0] <= 10):
                candidates.append((current[0], current[1], dict(prior_low=prior[1], latest_low=current[1],
                    prior_pivot_date=recent.index[prior[0]].date().isoformat(),
                    latest_pivot_date=recent.index[current[0]].date().isoformat())))
    else:
        structure = confirmed_support_stop(recent, current_price=price, entry_reference=price, atr=atr,
                                            lookback=60, max_pivot_age=10, max_distance=.12)
        if structure.get("quality") == "confirmed_swing":
            candidates.append((int(structure["last_pivot_bar"]), float(structure["support"]),
                               dict(support_touches=int(structure["touches"]))))
    checked = []
    for pivot, support, evidence in candidates:
        before = recent.iloc[max(0, pivot - 15):pivot]
        if before.empty:
            continue
        peak = float(before.High.max())
        depth = (peak - support) / peak
        if not .03 <= depth <= .12:
            checked.append(attempt(kind, "pullback_depth", "Rücksetzer vom vorherigen Hoch außerhalb 3–12 %."))
            continue
        if float(recent.iloc[pivot:].Low.min()) < support * .985:
            checked.append(attempt(kind, "support_broken", "Jüngere Kerzen unterschreiten die bestätigte Unterstützung.", "invalid"))
            continue
        # The three right-hand confirmation bars anchor the trigger. Do not
        # chase later highs by recalculating the trigger from a moving window.
        trigger_level = float(recent.iloc[pivot + 1:pivot + 4].High.max())
        anchored_atr = atr_value(data.iloc[:len(data)-len(recent)+pivot+4])
        out = levels(kind, trigger_level, support, anchored_atr, price,
            near=price <= support + 2 * anchored_atr and trigger_level / price <= 1.05,
            evidence={**evidence, "support": support, "pivot_date": recent.index[pivot].date().isoformat(),
                      "pullback_depth_pct": depth * 100, "confirmation_high": trigger_level},
            checks=["Rücksetzer 3–12 % vom vorherigen lokalen Hoch",
                    "Jüngstes Tief mit drei abgeschlossenen Nachbarkerzen je Seite bestätigt",
                    "Höheres Tief über einem früheren bestätigten Pivot" if higher else
                    "Unterstützung durch zwei getrennte Pivot-Tests innerhalb 1,5 % bestätigt",
                    "Auslöser über Hoch der drei rechten Bestätigungskerzen"])
        checked.append(out)
    if checked:
        return max(checked, key=lambda a: RANK[a["phase"]])
    if higher and found:
        # An uncompleted right flank is visible preparation, never eligible.
        for i in range(max(3, len(recent) - 3), len(recent)):
            v = float(recent.Low.iloc[i])
            older = [p for p in found if p[0] < i]
            prior = older[-1] if older else None
            sides = recent.Low.iloc[i - 3:i]
            following = recent.Low.iloc[i + 1:]
            peak = float(recent.High.iloc[max(0, i - 15):i].max())
            if (prior and 6 <= i-prior[0] <= 30 and v >= prior[1]*1.005
                    and v < sides.min() and (following.empty or v < following.min())
                    and .03 <= (peak - v) / peak <= .12):
                return attempt(kind, "pivot_confirmation_pending", "Mögliches höheres Tief; drei rechte Bestätigungskerzen fehlen.", "forming",
                               evidence=dict(pivot_date=recent.index[i].date().isoformat(), right_bars=len(following)))
    return attempt(kind, "higher_low_missing" if higher else "retested_support_missing",
                   "Kein jüngeres bestätigtes höheres Tief." if higher else "Keine jüngere mehrfach bestätigte Unterstützung.")


def retest_candidate(data, atr, price):
    kind = "breakout_retest"
    checked = []
    for i in range(max(21, len(data) - 15), len(data) - 3):
        base = data.iloc[i - 20:i]
        top_pairs = pairs(pivots(base.High, high=True), .01)
        if not top_pairs:
            continue
        pair = max(top_pairs, key=lambda p: p[1][0])
        resistance = max(p[1] for p in pair)
        width = (resistance - float(base.Low.min())) / resistance
        if not .02 <= width <= .12 or base.High.max() > resistance * 1.01:
            continue
        breakout_atr = atr_value(data.iloc[:i])
        cross = resistance + max(resistance * .001, .1 * breakout_atr)
        if not float(data.Close.iloc[i - 1]) <= cross < float(data.Close.iloc[i]):
            continue
        if float(data.Volume.iloc[i]) < float(base.Volume.mean()):
            checked.append(attempt(kind, "breakout_volume", "Ausbruch ohne mindestens durchschnittliches Basisvolumen.", "forming"))
            continue
        after = data.iloc[i + 1:]
        if (after.Close < resistance * .985).any():
            checked.append(attempt(kind, "failed_breakout", "Tagesschluss nach Ausbruch unter altem Widerstand.", "invalid"))
            continue
        for pivot, support in pivots(data.Low):
            if not i < pivot <= len(data) - 4 or not 3 <= len(data) - 1 - pivot <= 10:
                continue
            if not abs(support / resistance - 1) <= .015:
                continue
            high = float(data.iloc[pivot + 1:pivot + 4].High.max())
            anchored_atr = atr_value(data.iloc[:pivot+4])
            checked.append(levels(kind, high, support, anchored_atr, price,
                near=price <= support + 2 * anchored_atr and high / price <= 1.05,
                evidence=dict(old_resistance=resistance, retest_support=support,
                    breakout_date=data.index[i].date().isoformat(), retest_date=data.index[pivot].date().isoformat()),
                checks=["Alter Widerstand durch zwei getrennte 3/3-Pivot-Hochs bestätigt",
                        "Tagesschluss-Ausbruch mit mindestens durchschnittlichem Basisvolumen",
                        "Rücktest innerhalb 1,5 % des Ausbruchsniveaus, als 3/3-Tief bestätigt",
                        "Kein nachfolgender Tagesschluss mehr als 1,5 % unter altem Widerstand"]))
        if not any(a.get("entry") is not None for a in checked):
            checked.append(attempt(kind, "retest_pending", "Ausbruch erkannt; bestätigter Rücktest fehlt noch.", "forming"))
    return max(checked, key=lambda a: RANK[a["phase"]]) if checked else attempt(kind, "breakout_missing", "Kein bestätigter Ausbruch mit prüfbarem Rücktest.")


def detect(result, *, now=None, hard_gate=False):
    r = result or {}
    current = utc(now)
    ticker = str(r.get("ticker") or r.get("Ticker") or "").upper().strip()
    base = dict(version=VERSION, policy=POLICY, ticker=ticker, detected_at=current.isoformat(), eligible=False,
                phase="blocked", phase_label=PHASES["blocked"], label="Kein überwachbares Chartsetup vorhanden",
                checks=[], attempts=[], basis_date="", valid_until=current.isoformat())
    def blocked(code, reason):
        return sealed({**base, "reason_code": code, "reason": reason})
    try:
        market = market_for(ticker)
        expected = completed_day(current, market)
        df = r.get("df")
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex) or not all(k in df for k in required):
            return blocked("daily_data_missing", "Vollständige Tages-OHLCV-Historie fehlt.")
        if isinstance(df.columns, pd.MultiIndex) or df.index.hasnans or not df.index.is_monotonic_increasing:
            return blocked("daily_index", "Tagesindex/Spaltenformat nicht eindeutig oder ungeordnet.")
        dates = [t.date() for t in df.index]
        if len(set(dates)) != len(dates):
            return blocked("daily_duplicate", "Mehrere Kerzen pro Tagesdatum; keine eindeutige Tageshistorie.")
        today = current.tz_convert("America/New_York" if market == "US" else "Europe/Berlin").date()
        if any(d > today for d in dates):
            return blocked("daily_future", "Tagesdaten liegen in der Zukunft.")
        # 65 bars cover MA50 slope plus the 60-bar base. Do not introduce
        # new rejection gates from older, unused OHLCV when broadening patterns.
        data = df.loc[[d <= expected for d in dates], required].tail(65).copy()
        if len(data) < 60:
            return blocked("daily_short", "Mindestens 60 abgeschlossene Tageskerzen erforderlich.")
        if data.index[-1].date() != expected:
            return blocked("daily_stale", "Letzte abgeschlossene Tageskerze fehlt; neu scannen.")
        data = data.apply(pd.to_numeric, errors="coerce")
        valid = data.apply(lambda col: col.map(lambda v: number(v) is not None and v > 0)).all(axis=1)
        valid &= data.High >= data[["Open", "Close", "Low"]].max(axis=1)
        valid &= data.Low <= data[["Open", "Close", "High"]].min(axis=1)
        if not valid.all():
            return blocked("daily_invalid", "Ungültige OHLCV-Werte (einschließlich fehlendem/Null-Volumen).")
        if any((b - a).days > 5 for a, b in zip(data.index.date[:-1], data.index.date[1:])):
            return blocked("daily_gap", "Größere Lücke in der Tageshistorie.")
        base.update(basis_date=expected.isoformat(), valid_until=next_expiry(expected, market).isoformat())
        if hard_gate:
            return blocked("entry_gate", "Aktives technisches Einstiegsgate.")
        if number(r.get("buy_in")) not in (None, 0):
            return blocked("held_position", "Bestehende Position statt neuer Einstiegsidee.")
        earn = number(r.get("days_earn"))
        if r.get("has_upcoming_earnings") and earn is not None and 0 <= earn < 7:
            return blocked("earnings", "Bekannte Earnings innerhalb von sieben Tagen.")
        close = data.Close
        ma20, ma50 = close.rolling(20).mean(), close.rolling(50).mean()
        if close.iloc[-1] < ma50.iloc[-1] or ma50.iloc[-1] < ma50.iloc[-6]:
            return blocked("trend_missing", "Kurs nicht über/auf einer seit fünf Kerzen stabilen oder steigenden MA50.")
        tr = pd.concat([data.High - data.Low, (data.High - close.shift()).abs(),
                        (data.Low - close.shift()).abs()], axis=1).max(axis=1)
        atr, price = float(tr.tail(14).mean()), float(close.iloc[-1])
        if atr <= 0:
            return blocked("atr_missing", "ATR für Strukturpuffer fehlt.")
        attempts = []
        if ma20.iloc[-1] >= ma50.iloc[-1]:
            attempts.extend(range_candidate(data, n, atr, price) for n in (20, 40, 60))
        else:
            attempts.append(attempt("range_breakout", "range_trend", "MA20 unter MA50; keine Trend-Konsolidierung."))
        if ma20.iloc[-6] >= ma50.iloc[-6]:
            attempts.extend([pullback_candidate(data, atr, price), pullback_candidate(data, atr, price, higher=True)])
        else:
            for kind in ("support_pullback", "higher_low_pullback"):
                attempts.append(attempt(kind, "pullback_trend", "Vor Rücksetzer fehlt MA20 über/auf MA50."))
        attempts.append(retest_candidate(data, atr, price))
        # A newer analysis quote/running daily low is defensive evidence only.
        quote = number(r.get("price"))
        running = df.loc[[expected < d <= today for d in dates]]
        running_low = number(pd.to_numeric(running.Low, errors="coerce").min()) if not running.empty else None
        for pos, candidate in enumerate(attempts):
            stop, cap, entry = (candidate.get(k) for k in ("invalidation", "max_entry", "entry"))
            if stop is None or candidate["phase"] in {"none", "blocked", "invalid"}:
                continue
            if (quote is not None and quote <= stop) or (running_low is not None and running_low <= stop):
                attempts[pos] = {**candidate, "phase": "invalid", "reason_code": "live_invalidation", "reason": "Neuerer Analysepreis/laufendes Tagestief hat Invalidierung erreicht."}
            elif quote is not None and quote > cap:
                attempts[pos] = {**candidate, "phase": "missed", "reason_code": "live_chase", "reason": "Neuerer Analysepreis oberhalb der Entry-Grenze."}
            elif quote is not None and quote >= entry and candidate["phase"] == "ready":
                attempts[pos] = {**candidate, "phase": "triggered", "reason_code": "live_trigger_crossed", "reason": "Auslöser bereits erreicht; 5-Minuten-Bestätigung nicht geprüft."}
        selected = max(attempts, key=lambda a: RANK[a["phase"]])
        stop, cap, entry = (selected.get(k) for k in ("invalidation", "max_entry", "entry"))
        phase = selected["phase"]
        target = number(r.get("technical_target_1"))
        target = target if target is not None and entry is not None and target > entry else None
        return sealed({**base, **selected, "eligible": phase == "ready", "phase_label": PHASES[phase],
            "label": selected["label"] if phase != "none" else "Kein Chartsetup",
            "attempts": attempts, "mode": "close5", "target": target, "atr": round(selected.get("atr_reference", atr), 6), "price_reference": price,
            "currency": str(r.get("ccy") or r.get("currency") or (r.get("info") or {}).get("currency") or ""),
            "stop_source": "Bestätigte Setup-Struktur mit max(0,6 %, 0,25 ATR) Puffer",
            "stop_quality": "confirmed_setup_structure" if stop is not None else "unconfirmed",
            "crv": (target - entry) / (entry - stop) if target is not None and stop is not None and entry > stop else None,
            "chart": [dict(date=idx.date().isoformat(), **{k: float(row[k]) for k in required[:4]})
                      for idx, row in data.tail(65).iterrows()] if phase != "none" else []})
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        return blocked("unreadable", "Chartprüfung nicht möglich: " + str(exc)[:180])


def diagnostic_tables(frame, *, now=None):
    """Main reasons count each stock once; pattern reasons can count it repeatedly."""
    records = frame.to_dict("records") if isinstance(frame, pd.DataFrame) else []
    stock_rows, pattern_rows = [], []
    for row in records:
        p = row.get("__watch_setup")
        if not isinstance(p, dict):
            phase, code, reason = "blocked", "old_scan", "Scan ohne Chartbeleg; neu scannen."
        elif not evidence_valid(p):
            phase, code, reason = "blocked", "invalid_evidence", "Chartbeleg verändert oder unvollständig."
        else:
            phase, code, reason = p.get("phase", "ready" if p.get("eligible") else "none"), p.get("reason_code", "unspecified"), p.get("reason", "")
            if now is not None and p.get("basis_date"):
                try:
                    if utc(now) >= utc(p["valid_until"]):
                        phase, code, reason = "blocked", "setup_expired", "Chartbeleg abgelaufen; neu scannen."
                except (KeyError, ValueError, TypeError):
                    phase, code, reason = "blocked", "invalid_evidence", "Beleg unvollständig."
            for a in p.get("attempts", []):
                pattern_rows.append({"Ticker": str(row.get("Ticker") or "–"), "Muster": a["label"], "Phase": PHASES[a["phase"]], "Grund": a["reason"], "Code": a["reason_code"]})
        stock_rows.append({"Ticker": str(row.get("Ticker") or "–"), "Phase": PHASES.get(phase, phase),
                           "Code": code, "Grund": reason})
    stocks = pd.DataFrame(stock_rows, columns=["Ticker", "Phase", "Code", "Grund"])
    patterns = pd.DataFrame(pattern_rows, columns=["Ticker", "Muster", "Phase", "Grund", "Code"])
    counts = stocks.groupby(["Phase", "Code", "Grund"], dropna=False).size().reset_index(name="Aktien") if not stocks.empty else pd.DataFrame()
    return stocks, counts, patterns
