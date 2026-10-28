"""Explainable chart-only candidates for the trade watcher, not a profit forecast.

Initial, deliberately narrow heuristics on completed daily OHLCV: repeated
range boundaries, or a recent pullback to repeated strict pivot support.
No provider calls, no CRV/market-score gate, no changes to the main analysis.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json

import pandas as pd

from modules.trade_watch import utc, number, market_for, session_bounds
from modules.structural_trade_plan import confirmed_support_stop

POLICY = "chart_watch_v30.21by"
VERSION = 1


def fingerprint(package):
    fields = {k: v for k, v in package.items() if k != "fingerprint"}
    return sha256(json.dumps(fields, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sealed(package):
    out = deepcopy(package)
    out["fingerprint"] = fingerprint(out)
    return out


def form_id(package):
    """A rerender's wall clock must not reset review widgets for the same chart.

    Evidence, levels, chart history and validity still participate: an actual
    structural change resets the confirmation rather than reusing consent.
    """
    return fingerprint({k: v for k, v in package.items() if k not in {"fingerprint", "detected_at"}})[:20]


def validate(package, *, now=None, ticker=None):
    if not isinstance(package, dict) or package.get("policy") != POLICY or package.get("version") != VERSION:
        return False, "Kein geprüftes Chartsetup aus der aktuellen Erkennung; Aktie neu analysieren."
    try:
        if package.get("fingerprint") != fingerprint(package):
            return False, "Setupdaten wurden verändert; neue Analyse erforderlich."
        if package.get("eligible") is not True:
            return False, str(package.get("reason") or "Kein überwachbares Chartsetup vorhanden.")
        e, s, cap = (number(package.get(k)) for k in ("entry", "invalidation", "max_entry"))
        if not all(x is not None for x in (e, s, cap)) or not 0 < s < e <= cap:
            return False, "Abgeleitete Setup-Levels sind ungültig."
        if ticker is not None and str(ticker).upper() != package["ticker"]:
            return False, "Chartsetup gehört zu einem anderen Ticker."
        if now is not None and not utc(package["detected_at"]) <= utc(now) < utc(package["valid_until"]):
            return False, "Chartsetup abgelaufen oder Zeitstand ungültig; Aktie neu analysieren."
    except (KeyError, ValueError, TypeError, OverflowError):
        return False, "Unvollständige oder ungültige Chartsetup-Daten."
    return True, "Chartstruktur und feste Kursregeln vorhanden."


def idea_setup(idea, *, now=None):
    package = (idea.get("source") or {}).get("watch_setup")
    ok, reason = validate(package, now=now, ticker=idea.get("ticker"))
    if not ok:
        return False, reason
    for key in ("entry", "invalidation", "max_entry", "target", "mode"):
        if idea.get(key) != package.get(key):
            return False, "Gespeicherte Kursregel stimmt nicht mit der erkannten Struktur überein."
    try:
        if utc(idea["expires_at"]) > utc(package["valid_until"]):
            return False, "Idee reicht über die Gültigkeit des Chartsetups hinaus."
    except (KeyError, ValueError, TypeError):
        return False, "Gültigkeit der Idee fehlt."
    return True, reason


def completed_day(now, market):
    local = now.tz_convert("America/New_York" if market == "US" else "Europe/Berlin")
    day = local.date()
    for n in range(12):
        bounds = session_bounds(day, market)
        if bounds and (n > 0 or now >= pd.Timestamp(bounds[1].to_pydatetime() + timedelta(minutes=30))):
            return day
        day -= timedelta(days=1)
    raise ValueError("Vorherige Börsensitzung nicht verifiziert.")


def next_expiry(day, market):
    for n in range(1, 12):
        bounds = session_bounds(day + timedelta(days=n), market)
        if bounds:
            # Refresh the daily chart before the following session. A candidate
            # from Friday is still usable Monday, but not indefinitely all week.
            return bounds[1]
    raise ValueError("Nächste Börsensitzung nicht verifiziert.")


def pivots(values, *, high=False):
    found = []
    for i in range(3, len(values) - 3):
        value = float(values.iloc[i])
        sides = list(values.iloc[i - 3:i]) + list(values.iloc[i + 1:i + 4])
        is_pivot = value > max(sides) if high else value < min(sides)
        if is_pivot:
            found.append((i, value))
    return found


def repeated(points, tolerance=.015):
    pairs = []
    for pos, (i, value) in enumerate(points):
        for j, other in points[pos + 1:]:
            if j - i >= 6 and abs(other / value - 1) <= tolerance:
                pairs.append(((i, value), (j, other)))
    return pairs


def detect(result, *, now=None, hard_gate=False):
    r = result or {}
    current = utc(now)
    ticker = str(r.get("ticker") or r.get("Ticker") or "").strip().upper()
    base = dict(version=VERSION, policy=POLICY, eligible=False, ticker=ticker,
                detected_at=current.isoformat(), label="Kein überwachbares Chartsetup vorhanden",
                reason="", checks=[], basis_date="", valid_until=current.isoformat())
    def reject(reason, **details):
        return sealed({**base, "reason": reason, **details})
    try:
        market = market_for(ticker)
        expected = completed_day(current, market)
        df = r.get("df")
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex) or not all(k in df for k in required):
            return reject("Vollständige Tages-OHLCV-Historie fehlt; Aktie neu analysieren.")
        if isinstance(df.columns, pd.MultiIndex):
            return reject("Tagesdaten besitzen kein eindeutiges OHLCV-Format.")
        dates = [x.date() for x in df.index]
        if len(dates) != len(set(dates)) or df.index.hasnans or not df.index.is_monotonic_increasing:
            return reject("Tageshistorie ist doppelt, ungeordnet oder ohne eindeutiges Datum.")
        local_today = current.tz_convert("America/New_York" if market == "US" else "Europe/Berlin").date()
        if any(day > local_today for day in dates):
            return reject("Tageshistorie enthält ein zukünftiges Datum.")
        data = df.loc[[day <= expected for day in dates], required].tail(65).copy()
        if len(data) < 60:
            return reject("Mindestens 60 abgeschlossene Tageskerzen erforderlich.")
        if data.index[-1].date() != expected:
            return reject("Letzte abgeschlossene Tageskerze fehlt; Aktie neu scannen.")
        data = data.apply(pd.to_numeric, errors="coerce")
        good = data.apply(lambda col: col.map(lambda x: number(x) is not None and x > 0)).all(axis=1)
        good &= data["High"] >= data[["Open", "Close", "Low"]].max(axis=1)
        good &= data["Low"] <= data[["Open", "Close", "High"]].min(axis=1)
        if not good.all():
            return reject("Ungültige oder unvollständige Tageskerzen; keine Setup-Freigabe.")
        # Do not mistake weekly/sparse histories for 60 daily sessions.
        if any((b - a).days > 5 for a, b in zip(data.index.date[:-1], data.index.date[1:])):
            return reject("Tageshistorie hat größere Lücken; Struktur nicht verlässlich prüfbar.")
        base.update(basis_date=expected.isoformat(), valid_until=next_expiry(expected, market).isoformat())
        if hard_gate:
            return reject("Aktives technisches Einstiegsgate; kein überwachbares Setup.")
        if number(r.get("buy_in")) not in (None, 0):
            return reject("Bestehende Position: Wächter ist für neue Chart-Einstiege vorgesehen.")
        earnings = number(r.get("days_earn"))
        if r.get("has_upcoming_earnings") and earnings is not None and 0 <= earnings < 7:
            return reject("Bekannter Earnings-Termin innerhalb von sieben Tagen; kein neues Wächter-Setup.")
        closes = data["Close"]
        price = float(closes.iloc[-1])
        ma20 = closes.rolling(20).mean()
        ma50 = closes.rolling(50).mean()
        if not price > ma50.iloc[-1] or ma50.iloc[-1] <= ma50.iloc[-6]:
            return reject("Aufwärtstrend fehlt: Schlusskurs über steigender MA50 erforderlich.")
        true_range = pd.concat([data.High - data.Low, (data.High - closes.shift()).abs(),
                                (data.Low - closes.shift()).abs()], axis=1).max(axis=1)
        atr = float(true_range.tail(14).mean())
        if atr <= 0:
            return reject("ATR für Strukturpuffer fehlt.")

        window = data.tail(20)
        top_pairs = repeated(pivots(window.High, high=True), tolerance=.01)
        low_pairs = repeated(pivots(window.Low))
        selected = None
        # Consolidation: both boundaries repeatedly tested by separate strict pivots.
        if top_pairs and low_pairs and ma20.iloc[-1] > ma50.iloc[-1]:
            top_pair = max(top_pairs, key=lambda pair: pair[1][0])
            low_pair = max(low_pairs, key=lambda pair: pair[1][0])
            resistance = max(x[1] for x in top_pair)
            support = min(x[1] for x in low_pair)
            width = (resistance - support) / resistance
            within = window.High.max() <= resistance * 1.01 and window.Low.min() >= support * .985
            if .02 <= width <= .12 and within and resistance * .97 <= price <= resistance * 1.01:
                selected = dict(kind="range_breakout", label="Ausbruch aus bestätigter Konsolidierung",
                    level=resistance, support=support,
                    trigger= resistance + max(resistance * .001, atr * .1),
                    checks=["Steigende MA50; Kurs darüber, MA20 über MA50",
                            "20 abgeschlossene Tageskerzen; Range-Breite 2–12 %",
                            "Ober- und Unterkante jeweils zweimal als strikte 3/3-Pivots; Abstand mindestens 6 Kerzen",
                            "Kurs höchstens 3 % unter Widerstand"],
                    evidence=dict(resistance=resistance, support=support, range_width_pct=width * 100,
                        resistance_dates=[window.index[x[0]].date().isoformat() for x in top_pair],
                        support_dates=[window.index[x[0]].date().isoformat() for x in low_pair]))
        # Pullback: the close-to-MA label alone never counts as support evidence.
        if selected is None:
            structure = confirmed_support_stop(data, current_price=price, entry_reference=price,
                atr=atr, lookback=60, max_pivot_age=10, max_distance=.12)
            if structure.get("quality") == "confirmed_swing":
                recent = data.tail(60)
                pivot = int(structure["last_pivot_bar"])
                support = float(structure["support"])
                before = recent.iloc[max(0, pivot - 15):pivot]
                peak = float(before.High.max()) if not before.empty else support
                depth = (peak - support) / peak
                trigger = float(recent.tail(3).High.max()) + max(price * .001, atr * .1)
                last_low = float(recent.iloc[pivot:].Low.min())
                if (3 <= len(recent) - 1 - pivot <= 10 and .03 <= depth <= .12
                        and support < price <= support + 2 * atr and last_low >= support * .985
                        and ma20.iloc[-6] > ma50.iloc[-6] and trigger / price <= 1.05):
                    selected = dict(kind="support_pullback", label="Pullback an bestätigter Unterstützung",
                        level=float(recent.tail(3).High.max()), support=support, trigger=trigger,
                        checks=["Kurs über steigender MA50; vor Rücksetzer MA20 über MA50",
                                "Zwei bestätigte 3/3-Pivot-Tiefs innerhalb 1,5 %; mindestens 6 Kerzen Abstand",
                                "Letzter Pivot 3–10 Kerzen alt; Rücksetzer vom vorherigen Hoch 3–12 %",
                                "Stabilisierung höchstens 2 ATR über Unterstützung; Auslöser über letztem 3T-Hoch"],
                        evidence=dict(support=support, support_touches=int(structure["touches"]),
                                      latest_pivot_date=recent.index[pivot].date().isoformat(), pullback_depth_pct=depth * 100))
        if selected is None:
            return reject("Weder bestätigte Konsolidierung vor Ausbruch noch bestätigter Unterstützungs-Pullback erkannt.")
        entry = round(selected["trigger"], 6)
        invalidation = round(selected["support"] - max(.006 * selected["support"], .25 * atr), 6)
        ceiling = round(entry + min(.5 * atr, .01 * entry), 6)
        if not 0 < invalidation < price < entry or (entry - invalidation) / entry > .12:
            return reject("Struktur liegt nicht mehr vor dem Auslöser oder Invalidierung ist über 12 % entfernt.", checks=selected["checks"])
        # Use a newer raw quote only defensively, never to invent structural levels.
        current_price = number(r.get("price"))
        if current_price is not None and (current_price <= invalidation or current_price > ceiling):
            return reject("Aktueller Analysepreis hat die Struktur invalidiert oder die Entry-Grenze bereits überschritten.")
        running = df.loc[[expected < day <= local_today for day in dates]]
        if not running.empty:
            running_low = pd.to_numeric(running["Low"], errors="coerce").min()
            if number(running_low) is not None and running_low <= invalidation:
                return reject("Invalidierungsniveau wurde bereits in der laufenden Tageskerze erreicht.")
        target = number(r.get("technical_target_1"))
        target = target if target is not None and target > entry else None
        return sealed({**base, "eligible": True, "kind": selected["kind"], "label": selected["label"],
            "reason": "Konkrete Chartstruktur erkannt; Einstieg wartet auf neue Intraday-Bestätigung.",
            "checks": selected["checks"], "evidence": selected["evidence"], "entry": entry,
            "invalidation": invalidation, "max_entry": ceiling, "mode": "close5", "target": target,
            "currency": str(r.get("ccy") or r.get("currency") or (r.get("info") or {}).get("currency") or ""),
            "atr": round(atr, 6), "price_reference": price,
            "stop_source": "Unter bestätigter Setup-Unterstützung mit max(0,6 %, 0,25 ATR) Puffer",
            "stop_quality": "confirmed_setup_structure",
            "crv_note": "CRV separat; keine Voraussetzung für die Erkennung.",
            "chart": [dict(date=idx.date().isoformat(), **{key: float(row[key]) for key in required[:4]})
                      for idx, row in data.tail(30).iterrows()],
            "crv": (target - entry) / (entry - invalidation) if target is not None else None})
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        return reject("Chartsetup nicht verlässlich prüfbar: " + str(exc)[:180])
