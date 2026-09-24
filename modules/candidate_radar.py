"""Isolated discovery pipeline (v30.21a).

No positions, orders or REAL journal entries are written here. Existing score and
CRV functions are injected unchanged. Operational display gates are carried
alongside the quality grade, never used to re-calibrate the common engine.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, date, timezone, timedelta
from hashlib import sha256
import json
import math
import re
from typing import Any, Callable
from uuid import uuid4

from .radar_universe import CATALOG_VERSION, normalize_entries, universe_digest

RADAR_VERSION = "v30.21n"
SCHEMA_VERSION = 1
DEFAULT_MAX_SCAN_HOURS = 24
MAX_QUOTE_AGE_DAYS = 7
READY = "Im Live-Screener pr\u00fcfen"
NEAR = "Trigger abwarten"
WATCH = "Beobachten"
NO_PLAN = "Beobachten / noch kein Trade-Plan"
BLOCKED = "Gesperrt / Daten fehlen"
HISTORY = "Historischer Scan"
STYLES = ("Leader", "Charttechnik", "Turnaround", "Ausgewogen")
EMPTY = {"", "-", "--", "n/a", "nan", "none", "null", "unbekannt", "unknown", "nat", "<na>"}


def utcnow():
    return datetime.now(timezone.utc)


def text(value, default=""):
    if value is None:
        return default
    raw = str(value).strip()
    return default if raw.lower() in EMPTY else raw


def number(value):
    """Finite typed quantities only; no guessing localized price strings."""
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
        return num if math.isfinite(num) else None
    except (ValueError, TypeError):
        return None


def truth(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "ja"}
    return number(value) == 1 or value is True


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_safe(v) for v in value]
    try:
        return json_safe(value.item())
    except (AttributeError, TypeError, ValueError):
        return None


def parse_time(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    raw = text(value)
    if not raw:
        return None
    try:
        out = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        out = None
        for fmt in ("%d.%m.%Y", "%d.%m.%Y %H:%M", "%Y-%m-%d"):
            try:
                out = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
    return parse_time(out) if out is not None else None


def scan_key(universe, style, symbols, model_version=RADAR_VERSION):
    value = {"schema": SCHEMA_VERSION, "model": model_version, "radar": RADAR_VERSION,
             "catalog": CATALOG_VERSION, "universe": str(universe), "style": str(style),
             "symbols": universe_digest(symbols)}
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:32]


def resolve_entries(entries, resolver=None):
    """Use the existing name resolver for custom input, not catalogue scans."""
    resolved, notes = [], []
    from .radar_universe import ALIASES, INACTIVE
    for raw in entries:
        raw = str(raw or "").strip()
        if not raw:
            continue
        if len(raw) > 120:
            notes.append({"input": raw[:120], "action": "invalid", "detail": "Eingabe zu lang."})
            continue
        upper = raw.upper()
        direct, direct_notes = normalize_entries([upper])
        if upper in INACTIVE or upper in ALIASES or resolver is None:
            resolved.extend(direct)
            notes.extend(direct_notes)
            continue
        fallback = direct[0] if direct and len(upper) <= 10 and raw == upper else None
        try:
            symbol = resolver(raw, fallback=fallback)
        except Exception:
            symbol = None
        normalized, n = normalize_entries([symbol])
        notes.extend(n)
        if normalized:
            resolved.extend(normalized)
            if upper != normalized[0]:
                notes.append({"input": raw, "action": "resolved", "detail": normalized[0]})
        else:
            notes.append({"input": raw, "action": "invalid", "detail": "Nicht eindeutig aufloesbar; Ticker angeben."})
    symbols, extra = normalize_entries(resolved)
    return symbols, notes + extra


def request_key(universe, style, entries, model_version=RADAR_VERSION):
    # Stable before/after a scan, even for company-name inputs. Actual symbols are
    # additionally recorded and hashed inside the result.
    return scan_key(universe, style, [str(x).strip().upper() for x in entries if str(x).strip()], model_version)


def _first(mapping, *keys):
    for key in keys:
        if text(mapping.get(key)):
            return text(mapping[key])
    return ""


def quote_date(result):
    for key in ("price_asof", "quote_timestamp", "ts", "data_asof"):
        stamp = parse_time(result.get(key))
        if stamp is not None:
            return stamp.date().isoformat()
    frame = result.get("df")
    if frame is not None:
        try:
            stamp = parse_time(frame.index[-1])
            return stamp.date().isoformat() if stamp else None
        except (IndexError, AttributeError, TypeError):
            pass
    return None


def target_provenance(result, rr):
    """Describe existing target, NEVER change it or the computed CRV.

    When a floor cannot be proved from the returned raw fields, call it
    'Herkunft nicht belegt' rather than inventing a technical target.
    """
    used = number(rr.get("tp1"))
    source = text(rr.get("target_source"), "Herkunft nicht belegt")
    original = source
    basis = None
    if source == "TP2 / Hauptziel":
        original = text(result.get("tp2_source"), "Herkunft nicht belegt")
        s = original.lower()
        if "fallback" in s or re.search(r"\b[123](?:[.,]\d)?r\b", s):
            return {"label": "R-Planungsziel", "source": original, "value": used, "base_target": None}
        if "setup" in s or "techn" in s:
            basis = number(result.get("technical_target_1"))
            label = "Technisches Ziel"
        elif "analyst" in s:
            basis = number(result.get("target"))
            label = "Analystenziel"
        elif "52w" in s:
            basis = number(result.get("high52"))
            label = "52W-Hoch"
        else:
            label = "Herkunft nicht belegt"
        if basis is not None and used is not None:
            tolerance = max(0.02, abs(used) * 1e-6)
            if used > basis + tolerance:
                label += " + rechnerische Anhebung"
    else:
        label = source
    return {"label": label, "source": original, "value": used, "base_target": basis}


def gate_reasons(result, decision, rr, asof):
    """Preserve vetoes. A positive grade/trigger never cancels a hard gate."""
    reasons = []
    legacy = text(decision.get("gate_reasons"))
    if legacy and legacy.lower() not in {"keine harten gates", "keine", "none"}:
        reasons.extend(p.strip() for p in legacy.split(";") if p.strip())
    for key in ("hard_gates", "blocking_reasons", "entry_block_reasons"):
        value = result.get(key)
        if isinstance(value, (tuple, list)):
            reasons.extend(text(p) for p in value if text(p))
    days = number(result.get("days_earn"))
    trigger_reason = text(result.get("trigger_reason"))
    earnings_veto = (truth(result.get("earnings_veto")) or
                     (truth(result.get("has_upcoming_earnings")) and days is not None and 0 <= days < 7) or
                     "earnings-veto" in trigger_reason.lower())
    if earnings_veto:
        reasons.append("Earnings-Veto: nach den Zahlen neu pr\u00fcfen")
    for key in ("entry_allowed", "entry_eligible"):
        if key in result and result[key] is not None and not truth(result[key]):
            reasons.append(trigger_reason or "Zentrale Einstiegssperre")
    if truth(decision.get("knockout")):
        reasons.append("Radar-Risikosperre")
    if text(result.get("trigger_status")).lower() == "passiv":
        reasons.append(trigger_reason or "Analyse derzeit passiv")
    # Kurs und Kursalter sind echte Datenvoraussetzungen. Stop, Ziel, CRV und
    # Entry-Zone gehoeren dagegen zum Trade-Plan und werden separat bewertet:
    # Bei einem noch nicht validen Setup ist ihr Fehlen erwartbar und kein
    # Datenfehler.
    price = number(rr.get("price"))
    if price is None or price <= 0:
        reasons.append("Kurs fehlt oder ist unplausibel")
    qdate = quote_date(result)
    qtime = parse_time(qdate)
    if qtime is None:
        reasons.append("Kursdatum unbekannt")
    elif (asof.date() - qtime.date()).days > MAX_QUOTE_AGE_DAYS:
        reasons.append(f"Kursdaten \u00e4lter als {MAX_QUOTE_AGE_DAYS} Kalendertage")
    elif qtime.date() > asof.date():
        reasons.append("Kursdatum liegt in der Zukunft")
    return list(dict.fromkeys(reasons))


def trade_plan_reasons(rr):
    """Missing plan fields, separated from hard/data gates.

    Entry, stop and CRV are intentionally absent for many candidates until the
    central setup is valid. That state is observational, not a data failure.
    """
    reasons = []
    for key, label in (("stop", "Stop-Basis"), ("tp1", "Hauptziel")):
        n = number(rr.get(key))
        if n is None or n <= 0:
            reasons.append(label + " fehlt oder ist unplausibel")
    if number(rr.get("crv")) is None:
        reasons.append("CRV nicht berechenbar")
    if number(rr.get("entry_distance_pct")) is None:
        reasons.append("Entry-Zone nicht belastbar")
    return list(dict.fromkeys(reasons))


def build_candidate(result, *, ticker, style, decide, entry_package, analyzed_at=None):
    if not isinstance(result, Mapping):
        raise ValueError("Keine strukturierte Analyse")
    actual = text(result.get("ticker"), ticker).upper()
    if actual != ticker:
        raise ValueError("Instrumentidentitaet abweichend")
    now = parse_time(analyzed_at) or utcnow()
    decision = decide(result, style)  # Exactly one professional decision per ticker.
    rr = entry_package(result)
    info = result.get("info") if isinstance(result.get("info"), Mapping) else {}
    confidence = result.get("confidence_info") if isinstance(result.get("confidence_info"), Mapping) else {}
    cov = number(confidence.get("coverage"))
    if cov is not None and not 0 <= cov <= 1:
        cov = None
    valid_setup = truth(result.get("valid_trade_setup"))
    plan_reasons = trade_plan_reasons(rr)
    gates = gate_reasons(result, decision, rr, now)
    if cov is None:
        gates.append("Datenabdeckung unbekannt")
    # A missing trade plan is only a blocking inconsistency once the central
    # setup itself is valid. Before that it is the expected observation state.
    if valid_setup:
        gates.extend(plan_reasons)
    gates = list(dict.fromkeys(gates))
    grade = text(decision.get("grade"), "n/a")
    trigger = text(result.get("trigger_status"), "Unbekannt")
    active = trigger.lower() in {"aktiv", "jetzt pr\u00fcfbar", "trigger aktiv"}
    if gates:
        status = BLOCKED
        next_step = "; ".join(gates[:3])
    elif not valid_setup and plan_reasons:
        status = NO_PLAN
        next_step = "Weiter beobachten; zentrales Setup noch nicht valide. Trade-Plan erst bei belastbarem Setup pr\u00fcfen."
    elif active and valid_setup and grade in {"A", "B"} and decision.get("bucket") == "Jetzt pr\u00fcfbar":
        status = READY
        next_step = "Frischen Live-Scan pr\u00fcfen; noch keine Kauf- oder Paketfreigabe."
    elif grade in {"A", "B"} or decision.get("bucket") in {"Nahe am Trigger", "Starke Watchlist"}:
        status = NEAR
        next_step = text(result.get("next_trigger"), "Triggerbest\u00e4tigung abwarten")
        if next_step.lower() in {"aktiv", "jetzt pr\u00fcfbar", "trigger aktiv"}:
            next_step = "Triggerbest\u00e4tigung und aktuellen Entry im Live-Screener pr\u00fcfen"
    else:
        status = WATCH
        next_step = "Als Idee beobachten; Setup und Einstieg sind noch nicht freigegeben."
    mtf = result.get("multi_timeframe_pkg")
    wave = result.get("wave_structure_pkg")
    return json_safe({
        "ticker": ticker, "name": text(result.get("name"), ticker),
        "sector": _first(result, "sector_label", "sector") or _first(info, "sector") or None,
        "industry": _first(result, "industry_label", "industry") or _first(info, "industry") or None,
        "asset_type": _first(result, "asset_type_label", "Asset_Typ") or _first(info, "quoteType") or "Unbekannt",
        "currency": _first(result, "ccy", "currency") or _first(info, "currency") or None,
        "grade": grade, "score": number(decision.get("score")),
        "rank_score": number(decision.get("top_chance_rank")) or number(decision.get("score")) or 0,
        "status": status, "trigger": trigger, "trigger_reason": text(result.get("trigger_reason")),
        "valid_setup": valid_setup, "gates": gates, "trade_plan_reasons": plan_reasons,
        "why": text(decision.get("why_today"), "Keine belastbare Begr\u00fcndung geliefert"),
        "next_step": next_step, "brake": text(decision.get("brake")),
        "crv": number(rr.get("crv")), "price": number(rr.get("price")),
        "stop": number(rr.get("stop")), "target": number(rr.get("tp1")),
        "entry_zone": text(rr.get("entry_zone")), "entry_distance_pct": number(rr.get("entry_distance_pct")),
        "target_provenance": target_provenance(result, rr),
        "coverage": cov, "mtf_available": isinstance(mtf, Mapping) and bool(mtf),
        "wave_available": isinstance(wave, Mapping) and bool(wave),
        "quote_date": quote_date(result), "analyzed_at": now.isoformat(),
        "decision": decision, "rr_package": rr,
        "analysis_note": _first(result, "Analyse_Hinweis", "Analyse_Datenmodus"),
    })


def run_scan(*, universe, style, entries, analyze, decide, entry_package,
             resolver=None, model_version=RADAR_VERSION, progress=None, clock=utcnow,
             source="manual"):
    if style not in STYLES:
        raise ValueError("Unbekannter Radar-Suchstil")
    entries = list(entries)
    symbols, resolution = resolve_entries(entries, resolver if universe == "Eigene Liste" else None)
    if not symbols:
        raise ValueError("Keine g\u00fcltigen Kandidaten. Tickerliste pr\u00fcfen.")
    if len(symbols) > 500:
        raise ValueError("Maximal 500 unterschiedliche Ticker pro bewusster Scan-Anforderung.")
    start = parse_time(clock()) or utcnow()
    candidates, errors = [], []
    for index, ticker in enumerate(symbols):
        try:
            result = analyze(ticker=ticker, horizon="Swing (1-4 Wochen)", depot=10000,
                             risk_pct=1.0, override=0.0, buy_in_override=0.0,
                             smart_money_default=True, strict_mode=True)
            candidate = build_candidate(result, ticker=ticker, style=style, decide=decide,
                                        entry_package=entry_package, analyzed_at=clock())
            candidates.append(candidate)
        except Exception as exc:
            errors.append({"ticker": ticker, "reason": "Nicht auswertbar: " + type(exc).__name__})
        if progress is not None:
            progress(index + 1, len(symbols), ticker)
    end = parse_time(clock()) or utcnow()
    candidates = rank_candidates(candidates)
    for index, row in enumerate(candidates, 1):
        row["rank_at_discovery"] = index
    return {
        "schema_version": SCHEMA_VERSION, "radar_version": RADAR_VERSION,
        "model_version": model_version, "catalog_version": CATALOG_VERSION,
        "key": request_key(universe, style, entries, model_version),
        "scan_id": "radar-" + start.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8],
        "universe": universe, "style": style, "source": source,
        "started_at": start.isoformat(), "completed_at": end.isoformat(),
        "completed": True, "requested": len(symbols), "processed": len(symbols),
        "symbols": symbols, "symbols_hash": universe_digest(symbols),
        "rows": candidates, "errors": errors, "resolution": resolution,
    }


def rank_candidates(rows):
    # Full pool first; presentation limits must never change ranks.
    return sorted((deepcopy(dict(r)) for r in rows), key=lambda r: (
        bool(r.get("gates")), {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(r.get("grade"), 5),
        -(number(r.get("rank_score")) or 0), str(r.get("ticker", ""))))


def snapshot_valid(payload, expected_key=None, owner=None):
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        return False
    if payload.get("radar_version") != RADAR_VERSION or payload.get("completed") is not True:
        return False
    if expected_key is not None and payload.get("key") != expected_key:
        return False
    if owner is not None and payload.get("owner") != owner:
        return False
    rows, errors, symbols = payload.get("rows"), payload.get("errors"), payload.get("symbols")
    if not isinstance(rows, list) or not isinstance(errors, list) or not isinstance(symbols, list):
        return False
    if payload.get("processed") != payload.get("requested") or len(symbols) != payload.get("requested"):
        return False
    if (len(rows) + len(errors) != len(symbols) or parse_time(payload.get("completed_at")) is None
            or parse_time(payload.get("started_at")) is None):
        return False
    required = {"ticker", "grade", "status", "gates", "crv", "entry_distance_pct", "target_provenance", "quote_date", "analyzed_at", "decision", "rr_package"}
    if not all(isinstance(r, Mapping) and required <= r.keys() for r in rows):
        return False
    observed = [r["ticker"] for r in rows] + [e.get("ticker") for e in errors if isinstance(e, Mapping)]
    return len(observed) == len(set(observed)) and set(observed) == set(symbols)


def snapshot_freshness(snapshot, *, now=None, max_scan_hours=DEFAULT_MAX_SCAN_HOURS):
    if not snapshot_valid(snapshot):
        return False, "Scanformat unvollst\u00e4ndig oder aus einer anderen Radar-Version; neu scannen."
    now = parse_time(now) or utcnow()
    # Use START, not save time: a later save cannot make old observations fresh.
    stamp = parse_time(snapshot.get("started_at"))
    if stamp is None or stamp > now + timedelta(minutes=5):
        return False, "Scanzeit fehlt oder liegt in der Zukunft."
    hours = (now - stamp).total_seconds() / 3600
    if hours > max_scan_hours:
        return False, f"Historischer Scan ({hours:.1f} Stunden); f\u00fcr Einstiegspr\u00fcfung neu scannen."
    return True, f"Scan {hours:.1f} Stunden alt; Kursdatenstand steht je Kandidat."


def view_rows(snapshot, *, now=None, max_scan_hours=DEFAULT_MAX_SCAN_HOURS):
    fresh, _ = snapshot_freshness(snapshot, now=now, max_scan_hours=max_scan_hours)
    rows = rank_candidates(snapshot.get("rows") or [])
    now = parse_time(now) or utcnow()
    for r in rows:
        r["scan_status"] = r["status"]
        qtime = parse_time(r.get("quote_date"))
        if not fresh:
            r["status"] = HISTORY
            r["next_step"] = "Nur historische Idee; frischen Radar-/Live-Scan starten."
        elif qtime is None or (now.date() - qtime.date()).days > MAX_QUOTE_AGE_DAYS:
            r["status"] = BLOCKED
            r["gates"] = list(dict.fromkeys(r["gates"] + ["Kursdatenstand nicht mehr aktuell"]))
            r["next_step"] = "Aktuelle Kursdaten erneut abrufen."
    return rows


def _owner(storage):
    user = text(getattr(storage, "user_id", None))
    return user if user and user != "default" else None


def _namespace(key):
    if not re.fullmatch(r"[0-9a-f]{32}", str(key)):
        raise ValueError("Ung\u00fcltiger Radar-Speicherschl\u00fcssel")
    return "candidate_radar_v1_" + key


def save_snapshot(storage, payload):
    owner = _owner(storage)
    if not owner or not snapshot_valid(payload):
        return False, "Radar nicht gespeichert: Nutzerzuordnung oder vollst\u00e4ndiger Scan fehlt."
    copy = json_safe(dict(payload, owner=owner))
    try:
        ok = bool(storage.save_namespace(_namespace(copy["key"]), copy))
        status = storage.status() if hasattr(storage, "status") else {}
    except Exception:
        return False, "Radar-Speicherung fehlgeschlagen; Ergebnis nur in dieser Sitzung."
    if not ok:
        return False, "Radar-Speicherung fehlgeschlagen; Ergebnis nur in dieser Sitzung."
    if status.get("degraded") or str(status.get("last_backend", "")).startswith("local"):
        return True, "Radar nur lokal gesichert; dauerhafte Cloud-Speicherung nicht best\u00e4tigt."
    return True, "Radar \u00fcber den benutzerbezogenen Speicher gesichert."


def load_snapshot(storage, key):
    owner = _owner(storage)
    if not owner:
        return None
    try:
        value = storage.load_namespace(_namespace(key), default=None)
    except Exception:
        return None
    return deepcopy(value) if snapshot_valid(value, key, owner) else None


def save_discoveries(storage, snapshot, watchlist, tickers):
    """Audit context for a WATCHLIST queue request, never proof of execution."""
    if not _owner(storage) or not snapshot_valid(snapshot) or not text(watchlist):
        return False
    selected = set(tickers)
    rows = [r for r in snapshot["rows"] if r["ticker"] in selected]
    if len(rows) != len(selected):
        return False
    key = sha256((snapshot["scan_id"] + "|" + str(watchlist)).encode()).hexdigest()[:32]
    ns = "radar_discovery_v1_" + key
    try:
        old = storage.load_namespace(ns, default={}) or {}
        previous = old.get("candidates", {}) if old.get("owner") == _owner(storage) else {}
        candidates = dict(previous)
        candidates.update({r["ticker"]: r for r in rows})
        return bool(storage.save_namespace(ns, json_safe({
            "owner": _owner(storage), "scan_id": snapshot["scan_id"],
            "universe": snapshot["universe"], "style": snapshot["style"],
            "scan_started_at": snapshot["started_at"], "model_version": snapshot["model_version"],
            "watchlist": watchlist, "action": "watchlist_queue_requested",
            "not_a_trade": True, "updated_at": utcnow().isoformat(), "candidates": candidates,
        })))
    except Exception:
        return False


def known_tickers(watchlist_rows, positions, pending):
    """Labels only. No budget/positions changes and no guessed ticker aliases."""
    watched, held, planned = set(), set(), set()
    for row in watchlist_rows or []:
        if isinstance(row, Mapping):
            t = text(row.get("Ticker")).upper()
            if t:
                watched.add(t)
    for row in pending or []:
        if isinstance(row, Mapping) and text(row.get("Ticker")):
            planned.add(text(row["Ticker"]).upper())
    for value in (positions or {}).values():
        if not isinstance(value, Mapping):
            continue
        for ticker, pos in value.items():
            if isinstance(pos, Mapping):
                qty = number(pos.get("shares", pos.get("quantity")))
                if qty is not None and qty > 0:
                    held.add(str(ticker).upper())
                elif text(pos.get("execution_status") or pos.get("trade_state") or pos.get("status")).lower() in {"planned", "geplant", "vorgemerkt"}:
                    planned.add(str(ticker).upper())
    return watched, held, planned
