"""v30.20c: deterministic, read-only trading-package planner.

No orders, invented returns, FX calls, score changes or learning writes.
Money is converted to one base currency before *any* budget test. Only
explicit quote currencies are accepted (never financialCurrency or a suffix).
The bounded search is a planning heuristic, not a calibrated return model.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
import hashlib
import json
import math
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

VERSION = "v30.20c"
CURRENCIES = frozenset("EUR USD GBP GBX CHF CAD AUD NZD JPY HKD SGD SEK NOK DKK PLN CZK HUF CNY INR KRW ILS ZAR ZAC BRL MXN TRY RON BGN ISK IDR MYR PHP THB".split())
BERLIN = ZoneInfo("Europe/Berlin")
UNKNOWN = "Unbekannt"


def number(value: Any, default=None):
    if isinstance(value, bool) or value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.strip().replace("/100", "").replace("%", "").replace(",", ".")
        out = float(value)
        return out if math.isfinite(out) else default
    except (ValueError, TypeError):
        return default


def text(value: Any) -> str:
    s = str(value).strip() if value is not None else ""
    return "" if s.lower() in {"nan", "nat", "none", "null", "n/a", "na", "-", "<na>"} else s


def currency(value: Any) -> str:
    raw = text(value)
    if raw == "GBp":
        return "GBX"
    if raw == "ZAc":
        return "ZAC"
    return raw.upper() if raw.upper() in CURRENCIES else ""


def quote_currency(*objects: Mapping) -> str:
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        info = obj.get("info") if isinstance(obj.get("info"), Mapping) else {}
        for source, keys in ((obj, ("__pkg_currency", "price_currency", "quoteCurrency")),
                             (info, ("currency", "quoteCurrency")),
                             (obj, ("currency", "W\u00e4hrung", "Waehrung"))):
            for key in keys:
                result = currency(source.get(key))
                if result:
                    return result
    return ""


def timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        out = value
    else:
        s = text(value)
        if not s:
            return None
        try:
            out = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            out = None
            for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d"):
                try:
                    out = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    pass
            if out is None:
                return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=BERLIN)
    return out.astimezone(timezone.utc)


def fresh(value: Any, now: datetime, max_hours: float) -> bool:
    ts = timestamp(value)
    age = ((timestamp(now) - ts).total_seconds() / 3600.0) if ts else None
    return age is not None and -0.083334 <= age <= max_hours


def safe_json(value: Any):
    if isinstance(value, Mapping):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "item"):
        return safe_json(value.item())
    return text(value)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(safe_json(value), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def fx_rate(cur: str, base: str, rates: Mapping[str, Any]) -> float | None:
    cur, base = currency(cur), currency(base)
    if not cur or not base or base in {"GBX", "ZAC"}:
        return None
    if cur == base:
        return 1.0
    # GBX and ZAC are subunits, NOT a one-to-one synonym for GBP/ZAR.
    major, divisor = {"GBX": ("GBP", 100.0), "ZAC": ("ZAR", 100.0)}.get(cur, (cur, 1.0))
    rate = 1.0 if major == base else number(rates.get(major))
    return rate / divisor if rate is not None and rate > 0 else None


_GROUPS = {
    "technology": "Technologie", "technologie": "Technologie", "halbleiter": "Technologie",
    "cloud / cyber / software": "Technologie", "mega-cap tech": "Technologie",
    "healthcare": "Gesundheit", "health care": "Gesundheit", "gesundheit": "Gesundheit",
    "financial services": "Finanzen", "financials": "Finanzen", "finanzen": "Finanzen",
    "industrials": "Industrie", "industrie": "Industrie", "energy": "Energie", "energie": "Energie",
    "consumer cyclical": "Zyklischer Konsum", "consumer defensive": "Basiskonsum", "konsum": "Konsum",
    "basic materials": "Rohstoffe", "utilities": "Versorger", "real estate": "Immobilien",
    "communication services": "Kommunikation", "communication": "Kommunikation",
}


def group_name(value: Any) -> str:
    s = text(value)
    if not s or any(x in s.lower() for x in ("unbekannt", "unknown", "sonstige")):
        return UNKNOWN
    return _GROUPS.get(s.lower(), s)


def make_scan_fields(result: Mapping, risk: Mapping, *, price: Any, now: datetime) -> dict:
    """Called inside the existing full scan, using its existing analysis result."""
    info = result.get("info") if isinstance(result.get("info"), Mapping) else {}
    entry = number(price)
    risk_entry = number(risk.get("entry_default"))
    valid_basis = (entry is not None and entry > 0 and risk_entry is not None and
                   abs(risk_entry / entry - 1) <= 0.001)
    return {
        "__pkg_version": VERSION,
        "__pkg_entry": entry,
        "__pkg_stop": number(risk.get("stop")) if valid_basis else None,
        "__pkg_target": number(risk.get("target")) if valid_basis else None,
        "__pkg_stop_source": text(risk.get("risk_stop_source")),
        "__pkg_stop_basis": text(risk.get("pre_atr_stop_source")),
        "__pkg_chart_stop": number(risk.get("chart_invalidation_stop")),
        "__pkg_target_source": text(risk.get("target_source")),
        "__pkg_currency": quote_currency(result),
        "__pkg_group": group_name(result.get("sector") or info.get("sector")),
        "__pkg_scan_at": timestamp(now).isoformat(),
        "__pkg_data_quality": text(result.get("data_quality")),
    }


@dataclass(frozen=True)
class PlanConfig:
    equity: float
    budget: float
    new_risk: float
    base: str = "EUR"
    max_positions: int = 3
    max_per_group: int = 1
    max_position_pct: float = 12.0
    max_group_pct: float = 30.0
    max_total_risk_pct: float = 3.0
    min_crv: float = 2.0
    min_order: float = 100.0
    fixed_fee: float = 1.0       # per side, base currency; planning input
    variable_fee_pct: float = 0.15
    entry_buffer_pct: float = 0.3
    max_age_hours: float = 24.0
    pool_limit: int = 18

    def errors(self) -> list[str]:
        errors = []
        if currency(self.base) != self.base or self.base in {"GBX", "ZAC"}:
            errors.append("Ung\u00fcltige Basisw\u00e4hrung.")
        for key in ("equity", "budget", "new_risk", "min_crv", "min_order", "max_age_hours"):
            v = number(getattr(self, key))
            if v is None or v <= 0:
                errors.append(f"{key}: Wert muss endlich und gr\u00f6\u00dfer als null sein.")
        for key in ("max_position_pct", "max_group_pct", "max_total_risk_pct"):
            v = number(getattr(self, key))
            if v is None or not 0 < v <= 100:
                errors.append(f"{key}: nur Werte zwischen 0 und 100 Prozent.")
        for key in ("fixed_fee", "variable_fee_pct", "entry_buffer_pct"):
            v = number(getattr(self, key))
            if v is None or not 0 <= v <= (100 if key != "fixed_fee" else 1000000):
                errors.append(f"{key}: ung\u00fcltiger Kosten-/Pufferwert.")
        if not isinstance(self.max_positions, int) or not 1 <= self.max_positions <= 5:
            errors.append("H\u00f6chstens 1 bis 5 neue Positionen erlaubt.")
        if not isinstance(self.max_per_group, int) or not 1 <= self.max_per_group <= 5:
            errors.append("Je Branchengruppe sind 1 bis 5 neue Positionen erlaubt.")
        if not isinstance(self.pool_limit, int) or not 1 <= self.pool_limit <= 18:
            errors.append("Suchraum muss zwischen 1 und 18 Kandidaten liegen.")
        if number(self.budget, 0) > number(self.equity, 0):
            errors.append("Kaufbudget ist gr\u00f6\u00dfer als der Tradingdepotwert.")
        return errors


def _boolean(value: Any) -> bool | None:
    """Only real booleans, including numpy scalar booleans; never truthy text."""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        try:
            scalar = value.item()
            if isinstance(scalar, bool):
                return scalar
        except (ValueError, TypeError):
            pass
    return None


def _active(row: Mapping) -> bool:
    """Respect the final state produced by the live state machine.

    The three old flags describe individual alert paths, not every green trend
    setup. All can be false when the final confirmed state is 'Trigger aktiv'.
    Conversely, an early true flag must not release an armed/weakened final
    state. Only legacy rows without a final state use the raw flags.
    """
    state = text(row.get("Trade-State")).lower()
    if state:
        return state in {"trigger aktiv", "kurzfrist-trigger aktiv", "entry-zone erreicht"}
    keys = ("__wave_active", "__entry_reached", "__bucket_active")
    return any(_boolean(row.get(k)) is True for k in keys)


def _hard_gate(row: Mapping) -> bool:
    if _boolean(row.get("__entry_hard_gate")) is True or _boolean(row.get("__invalidated")) is True:
        return True
    s = text(row.get("Aktive Einstiegsgates")).lower()
    return s not in {"", "keine", "keine harten einstiegsgates aktiv."}


def prepare_candidates(rows: list[dict], queue: list[dict], config: PlanConfig,
                       rates: Mapping, now: datetime, excluded=(),
                       diagnostics: list | None = None) -> tuple[list[dict], list[dict]]:
    qmap = {text(r.get("Ticker")).upper(): r for r in queue}
    accepted, rejected, seen = [], [], set()
    excluded = {str(t).upper() for t in excluded}
    for raw in rows:
        row = dict(raw)
        tk = text(row.get("Ticker")).upper()
        if not tk:
            continue
        reasons, codes = [], []
        def reject(code, message):
            codes.append(code)
            reasons.append(message)
        q = qmap.get(tk, {})
        if tk in seen:
            rejected.append({"Ticker": tk, "Grund": "Doppelte Scan-Zeile; Ticker ausgeschlossen.",
                             "reason_codes": ["duplicate"]})
            accepted = [c for c in accepted if c["ticker"] != tk]
            if diagnostics is not None:
                for diagnostic in diagnostics:
                    if diagnostic.get("Ticker") == tk:
                        diagnostic["eligible"] = False
                        diagnostic["Grund"] = "Doppelte Scan-Zeile; Ticker ausgeschlossen."
            continue
        seen.add(tk)
        ready = text(q.get("Priorit\u00e4t")) == "\U0001f3af Jetzt pr\u00fcfen" and "\U0001f7e2" in text(row.get("Ampel"))
        active = _active(row)
        if tk in excluded:
            reject("excluded", "Von dir ausgeschlossen")
        if not ready:
            reject("queue", "Kein freigegebener gr\u00fcner Queue-Kandidat")
        if _hard_gate(row):
            reject("gate", "Hartes Einstiegsgate / Invalidierung")
        if not active:
            reject("trigger", "Finaler Trigger noch nicht aktiv")
        conf = text(q.get("Decision-Confidence"))
        if conf not in {"Hoch", "Mittel"}:
            reject("confidence", "Decision-Confidence nicht ausreichend")
        if not fresh(row.get("__pkg_scan_at"), now, config.max_age_hours):
            reject("snapshot", "Paket-Snapshot fehlt/ist veraltet; neuen Vollscan starten")
        entry, stop, target = [number(row.get("__pkg_" + k)) for k in ("entry", "stop", "target")]
        price = number(row.get("Kurs"))
        if entry is None or price is None or entry <= 0 or abs(price / entry - 1) > 0.001:
            reject("price", "Kurs und Risiko-Snapshot passen nicht zusammen")
        if stop is None or entry is None or not 0 < stop < entry:
            reject("stop", "Kein g\u00fcltiger unver\u00e4nderter Screener-Stop")
        if target is None or entry is None or target <= entry:
            reject("target", "Kein erreichbares strukturelles Ziel")
        cur = currency(row.get("__pkg_currency"))
        fx = fx_rate(cur, config.base, rates)
        if fx is None:
            reject("fx", "Kursw\u00e4hrung oder FX-Umrechnung fehlt")
        group = group_name(row.get("__pkg_group"))
        if group == UNKNOWN:
            reject("group", "Branche/Gruppe noch nicht zugeordnet")
        score = number(q.get("Live-Score"))
        if score is None or not 0 <= score <= 100:
            reject("score", "G\u00fcltiger Live-Score fehlt")
        limit = entry * (1 + config.entry_buffer_pct / 100) if entry else None
        crv = (target - limit) / (limit - stop) if (limit and stop and target and limit > stop) else None
        scan_crv = (target - entry) / (entry - stop) if (entry and stop and target and entry > stop) else None
        if crv is None:
            reject("crv_missing", "Paket-CRV wegen fehlender/ung\u00fcltiger Kurs-, Stop- oder Zieldaten nicht berechenbar")
        elif crv < config.min_crv:
            reject("crv", f"CRV am Kauflimit vor Kosten {crv:.2f} < Mindest-CRV {config.min_crv:.2f}")
        diagnostic = {
            "Ticker": tk, "Queue": text(q.get("Priorit\u00e4t")),
            "Trade-State": text(row.get("Trade-State")), "Confidence": conf,
            "queue_ready": ready, "active": active, "eligible": not reasons,
            "Kursw\u00e4hrung": cur, "Gruppe": group, "Scankurs": entry,
            "Kauflimit": limit, "Screener-Stop": stop, "Ziel": target,
            "CRV im Screener": number(row.get("CRV")),
            "CRV am Scankurs (Paket-Stop)": scan_crv,
            "CRV am Kauflimit vor Kosten": crv, "Mindest-CRV": config.min_crv,
            "Grund": " \u00b7 ".join(reasons) or "Vorpr\u00fcfung bestanden",
            "reason_codes": codes,
        }
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        if reasons:
            rejected.append({"Ticker": tk, "Grund": diagnostic["Grund"], "reason_codes": codes})
            continue
        merit = 0.65 * (score / 100) + 0.20 * (1.0 if conf == "Hoch" else 0.7) + 0.15 * min(crv, 4) / 4
        accepted.append({
            "ticker": tk, "name": text(row.get("Name")) or tk, "currency": cur,
            "group": group, "entry": entry, "limit": limit, "stop": stop, "target": target,
            "crv": crv, "fx": fx, "score": score, "confidence": conf,
            "merit": merit, "scan_at": row["__pkg_scan_at"],
            "stop_source": text(row.get("__pkg_stop_source")),
            "stop_basis": text(row.get("__pkg_stop_basis")),
            "chart_stop": number(row.get("__pkg_chart_stop")),
            "target_source": text(row.get("__pkg_target_source")),
            "context_row": safe_json(row),
        })
    return sorted(accepted, key=lambda c: (-c["merit"], c["ticker"])), rejected


def portfolio_state(store: Mapping, marks: Mapping[str, Mapping], config: PlanConfig,
                    rates: Mapping, now: datetime) -> dict:
    """Count all user trading watchlists; never silently deduplicate by ticker.

    Exact mirrors with a stable identity may be coalesced. Ambiguous duplicates
    block the plan rather than discarding a second broker position.
    """
    out = {"value": 0.0, "risk": 0.0, "reserved": 0.0, "reserved_risk": 0.0,
           "groups": {}, "weights": {}, "tickers": set(), "errors": [], "rows": [], "pending": []}
    seen: dict[tuple, dict] = {}
    no_identity: set[str] = set()
    for key, bucket in store.items():
        if not isinstance(bucket, Mapping):
            continue
        for tk_key, raw in bucket.items():
            if not isinstance(raw, Mapping):
                continue
            pos = dict(raw)
            tk = text(pos.get("ticker") or tk_key).upper()
            if text(pos.get("strategy_origin")).lower() in {"pie", "external", "extern"}:
                continue
            qty = number(pos.get("shares"))
            if qty is None:
                out["errors"].append(f"{tk}: offene Stueckzahl fehlt/ist ungueltig; Bestand nicht als null behandelt.")
                continue
            if qty < 0:
                out["errors"].append(f"{tk}: Short-Bestand wird vom Long-Planer nicht unterst\u00fctzt.")
                continue
            context = pos.get("entry_context") if isinstance(pos.get("entry_context"), Mapping) else {}
            pkg = context.get("trading_package") if isinstance(context.get("trading_package"), Mapping) else {}
            planned = qty == 0 and text(pos.get("execution_status")) == "planned"
            if qty == 0 and not planned:
                continue
            out["tickers"].add(tk)
            ident = text(pkg.get("id") or pos.get("position_id") or pos.get("opened_at_iso"))
            broker = text(pos.get("broker_source"))
            ident_key = (tk, broker, ident)
            signature = fingerprint({k: pos.get(k) for k in ("shares", "entry", "stop", "planned_shares", "entry_context")})
            if ident_key in seen:
                if ident and seen[ident_key]["signature"] == signature:
                    continue
                out["errors"].append(f"{tk}: uneindeutige doppelte Position \u00fcber mehrere Watchlists.")
                continue
            if not ident and tk in no_identity:
                out["errors"].append(f"{tk}: Positionsidentit\u00e4t fehlt bei Mehrfachbestand.")
                continue
            seen[ident_key] = {"signature": signature}
            if not ident:
                no_identity.add(tk)
            mark = dict(marks.get(tk) or {})
            if planned:
                qty = number(pos.get("planned_shares"), 0)
                price = number(pkg.get("limit"), number(pos.get("planned_entry")))
                cur = quote_currency(pos, pkg, mark)
                age_ok = True  # stale intentions still reserve cash until explicitly removed
                out["pending"].append({"Watchlist": str(key).replace("v244_open_positions::", ""),
                                       "Ticker": tk, "Paket": text(pkg.get("id")), "St\u00fcck geplant": qty})
            else:
                price = number(mark.get("Kurs"))
                age_ok = fresh(mark.get("__pkg_scan_at") or mark.get("Letztes Update"), now, config.max_age_hours)
                if price is None:
                    price = number(pos.get("last_price"))
                    age_ok = fresh(pos.get("last_price_at"), now, config.max_age_hours)
                # Current provider quote currency wins; a conflict is a blocker.
                marked_cur, stored_cur = quote_currency(mark), quote_currency(pos)
                if marked_cur and stored_cur and marked_cur != stored_cur:
                    out["errors"].append(f"{tk}: Kursw\u00e4hrung zwischen Bestand und Scan widerspr\u00fcchlich.")
                cur = marked_cur or stored_cur
            stop = number(pos.get("stop"))
            group = group_name(mark.get("__pkg_group"))
            if group == UNKNOWN:
                group = group_name(pos.get("portfolio_group"))
            fx = fx_rate(cur, config.base, rates)
            if qty <= 0 or price is None or price <= 0 or stop is None or stop <= 0 or fx is None or not age_ok or group == UNKNOWN:
                out["errors"].append(f"{tk}: Bestand/Vormerkung unvollst\u00e4ndig (St\u00fcck, aktueller Kurs, Stop, W\u00e4hrung/FX oder Branche).")
                continue
            if not planned and price <= stop:
                out["errors"].append(f"{tk}: Kurs auf/unter Stop; bestehenden Exit zuerst pr\u00fcfen.")
            value = qty * price * fx
            fee = config.fixed_fee + value * config.variable_fee_pct / 100
            risk = max(0, price - stop) * qty * fx + (2 if planned else 1) * fee
            if planned:
                out["reserved"] += value + fee
                out["reserved_risk"] += risk
            else:
                out["value"] += value
                out["risk"] += risk
            out["groups"][group] = out["groups"].get(group, 0.0) + value
            out["weights"][tk] = out["weights"].get(tk, 0.0) + value
            out["rows"].append({"Ticker": tk, "Gruppe": group, "W\u00e4hrung": cur,
                                "Wert Basisw\u00e4hrung": value, "Risiko Basisw\u00e4hrung": risk,
                                "Vorgemerkt": planned})
    return out


def _size_candidate(c: Mapping, config: PlanConfig, groups: Mapping,
                    cash_cap: float, risk_cap: float) -> tuple[dict | None, dict]:
    """Shared sizing and rejection evidence; no looser diagnostic-only path.

    Money, integer sizing, fees and net CRV match v30.20a. The evidence for
    singleton sizing uses the entire effective budget, not a forced split
    into the maximum number of positions.
    """
    entry_base = c["limit"] * c["fx"]
    risk_unit = (c["limit"] - c["stop"]) * c["fx"]
    fee_unit = entry_base * config.variable_fee_pct / 100
    position_room = config.equity * config.max_position_pct / 100
    group_room = max(0, config.equity * config.max_group_pct / 100 - groups.get(c["group"], 0))
    caps = {
        "budget": (cash_cap - config.fixed_fee) / (entry_base + fee_unit),
        "risk": (risk_cap - 2 * config.fixed_fee) / (risk_unit + 2 * fee_unit),
        "position_limit": position_room / entry_base,
        "group_limit": group_room / entry_base,
    }
    qty = math.floor(max(0, min(caps.values())))
    min_qty = max(1, math.ceil((config.min_order - 1e-8) / entry_base))
    min_value = min_qty * entry_base
    min_cost = min_value + config.fixed_fee + min_qty * fee_unit
    min_risk = min_qty * risk_unit + 2 * (config.fixed_fee + min_qty * fee_unit)
    evidence = {
        "Ticker": c["ticker"], "max_shares": qty, "min_shares": min_qty,
        "minimum_cost": min_cost, "minimum_risk": min_risk,
        "cash_available": cash_cap, "risk_available": risk_cap,
        "position_room": position_room, "group_room": group_room,
        "net_crv": None, "reason_codes": [], "Grund": "",
    }
    reasons = []
    if qty < min_qty:
        checks = (
            ("budget", min_cost, cash_cap, "Kaufbudget inkl. Kosten"),
            ("risk", min_risk, risk_cap, "Stop-Risikobudget inkl. Kosten"),
            ("position_limit", min_value, position_room, "Einzelpositionslimit"),
            ("group_limit", min_value, group_room, "freie Branchenkapazit\u00e4t"),
        )
        for code, needed, available, label in checks:
            if needed > available + 1e-8:
                evidence["reason_codes"].append(code)
                reasons.append(f"{label}: f\u00fcr mindestens {min_qty} Stk. {needed:.2f} {config.base} n\u00f6tig, {max(0, available):.2f} verf\u00fcgbar")
        evidence["Grund"] = " \u00b7 ".join(reasons) or "Ganze Mindestposition passt nicht in die Grenzen"
        if not evidence["reason_codes"]:
            evidence["reason_codes"] = ["minimum_order"]
        return None, evidence
    value = qty * entry_base
    fee = config.fixed_fee + qty * fee_unit
    cost = value + fee
    risk = qty * risk_unit + 2 * fee
    target_exit_fee = config.fixed_fee + qty * c["target"] * c["fx"] * config.variable_fee_pct / 100
    potential = qty * (c["target"] - c["limit"]) * c["fx"] - fee - target_exit_fee
    net_crv = potential / risk if risk > 0 else -1
    evidence["net_crv"] = net_crv
    if net_crv + 1e-9 < config.min_crv:
        evidence["reason_codes"] = ["net_crv"]
        evidence["Grund"] = f"CRV nach Kosten {net_crv:.2f} < Mindest-CRV {config.min_crv:.2f} (bei {qty} Stk.)"
        return None, evidence
    evidence["Grund"] = "Einzeln innerhalb der Grenzen umsetzbar"
    return {**deepcopy(c), "shares": qty, "value": value, "cost": cost,
            "risk": risk, "fee_per_side": fee, "net_crv": net_crv}, evidence


def _allocate(combo: tuple, config: PlanConfig, portfolio: dict, budget: float, risk_budget: float) -> dict | None:
    n = len(combo)
    if any(sum(c["group"] == group for c in combo) > config.max_per_group for group in {c["group"] for c in combo}):
        return None
    cash, remaining_risk = budget, risk_budget
    groups = dict(portfolio["groups"])
    result = []
    for c in sorted(combo, key=lambda c: (-c["merit"], c["ticker"])):
        item, _ = _size_candidate(c, config, groups, min(budget / n, cash), min(risk_budget / n, remaining_risk))
        if item is None:
            return None
        cash -= item["cost"]
        remaining_risk -= item["risk"]
        groups[c["group"]] = groups.get(c["group"], 0) + item["value"]
        if cash < -1e-6 or remaining_risk < -1e-6:
            return None
        result.append(item)
    # Same bounded planning heuristic as v30.20a, not an expected return.
    merit = sum(c["merit"] * math.sqrt(c["value"] / budget) for c in result)
    merit -= 0.2 * sum((v / config.equity) ** 2 for v in groups.values())
    return {"items": result, "cost": budget - cash, "risk": risk_budget - remaining_risk,
            "cash_left": config.budget - (budget - cash), "groups_after": groups, "utility": merit}


_REASON_LABELS = {
    "queue": "Nicht gr\u00fcn / nicht in Jetzt pr\u00fcfen", "trigger": "Finaler Trigger noch nicht aktiv",
    "gate": "Hartes Einstiegsgate / Invalidierung", "confidence": "Decision-Confidence zu niedrig",
    "snapshot": "Paket-Snapshot fehlt oder ist veraltet", "price": "Kursbasis widerspr\u00fcchlich",
    "stop": "G\u00fcltiger Screener-Stop fehlt", "target": "Strukturelles Ziel fehlt",
    "fx": "Kursw\u00e4hrung / FX fehlt", "group": "Branche fehlt", "score": "Live-Score fehlt",
    "crv_missing": "Paket-CRV nicht berechenbar", "crv": "CRV schon vor Kosten unter Mindestwert",
    "net_crv": "CRV nach Kosten unter Mindestwert", "budget": "Kaufbudget reicht nicht f\u00fcr Mindestposition",
    "risk": "Stop-Risikobudget reicht nicht f\u00fcr Mindestposition", "position_limit": "Einzelpositionslimit zu knapp",
    "group_limit": "Branchenkapazit\u00e4t zu knapp", "minimum_order": "Mindestposition nicht darstellbar",
    "held": "Bereits gehalten / vorgemerkt", "excluded": "Von dir ausgeschlossen", "duplicate": "Doppelte Scan-Zeile",
}


def _rejection_summary(rejected: list[dict], ready_tickers: set | None = None) -> list[dict]:
    groups = {}
    for row in rejected:
        tk = row["Ticker"]
        if ready_tickers is not None and tk not in ready_tickers:
            continue
        for code in row.get("reason_codes", []):
            groups.setdefault(code, set()).add(tk)
    return [{"Grund": _REASON_LABELS.get(code, code), "Werte": len(tickers),
             "Ticker": ", ".join(sorted(tickers)), "code": code}
            for code, tickers in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))]


def build_plan(rows: list[dict], queue: list[dict], store: Mapping, marks: Mapping,
               config: PlanConfig, rates: Mapping, *, now: datetime, scan_id: str,
               scan_complete: bool, atomic: bool, excluded=()) -> dict:
    out = {"ok": False, "engine_version": VERSION, "errors": config.errors(), "rejected": [], "alternatives": [],
           "diagnostics": [], "sizing_diagnostics": [], "reason_summary": [],
           "scan_id": scan_id, "created_at": timestamp(now).isoformat(),
           "config": asdict(config), "store_fingerprint": fingerprint(store),
           "rates": dict(rates), "rows_fingerprint": fingerprint(rows)}
    if not scan_complete or not atomic or not text(scan_id):
        out["errors"].append("Nur ein vollst\u00e4ndiger Atomic-Vollscan darf ein Paket freigeben.")
    if out["errors"]:
        return out
    portfolio = portfolio_state(store, marks, config, rates, now)
    out["portfolio"] = safe_json({**portfolio, "tickers": sorted(portfolio["tickers"])})
    out["errors"].extend(portfolio["errors"])
    for tk, value in portfolio["weights"].items():
        if value > config.equity * config.max_position_pct / 100 + 1e-6:
            out["errors"].append(f"{tk}: bestehendes/vorgemerktes Einzelgewicht bereits ueber dem eingestellten Limit.")
    for group, value in portfolio["groups"].items():
        if value > config.equity * config.max_group_pct / 100 + 1e-6:
            out["errors"].append(f"{group}: bestehende/vorgemerkte Branchenkonzentration bereits ueber dem eingestellten Limit.")
    actual_cash = config.equity - portfolio["value"] - portfolio["reserved"]
    budget = min(config.budget, actual_cash)
    risk_room = config.equity * config.max_total_risk_pct / 100 - portfolio["risk"] - portfolio["reserved_risk"]
    risk_budget = min(config.new_risk, risk_room)
    out["effective_budget"] = max(0, budget)
    out["effective_risk"] = max(0, risk_budget)
    if budget <= 0:
        out["errors"].append("Kein freies Budget nach Bestand und offenen Vormerkungen.")
    if risk_budget <= 0:
        out["errors"].append("Gesamtes Stop-Risikolimit durch Bestand/Vormerkungen ausgesch\u00f6pft.")
    candidates, rejected = prepare_candidates(rows, queue, config, rates, now, excluded, out["diagnostics"])
    out["rejected"] = rejected
    pool = []
    for c in candidates:
        if c["ticker"] in portfolio["tickers"]:
            out["rejected"].append({"Ticker": c["ticker"], "Grund": "Bereits gehalten/vorgemerkt; kein automatischer Nachkauf", "reason_codes": ["held"]})
        else:
            pool.append(c)
    out["eligible_count"] = len(pool)
    out["scan_count"] = len({text(row.get("Ticker")).upper() for row in rows if text(row.get("Ticker"))})
    ready_tickers = {d["Ticker"] for d in out["diagnostics"] if d["queue_ready"]}
    out["queue_ready_count"] = len(ready_tickers)
    out["active_ready_count"] = sum(d["queue_ready"] and d["active"] for d in out["diagnostics"])
    # Diagnose every eligible row, also outside the bounded combination pool.
    # Global/portfolio blockers still prohibit all execution and reservations.
    if not out["errors"]:
        for candidate in pool:
            item, diagnostic = _size_candidate(candidate, config, portfolio["groups"], budget, risk_budget)
            diagnostic["feasible"] = item is not None
            out["sizing_diagnostics"].append(diagnostic)
            if item is None:
                out["rejected"].append({"Ticker": candidate["ticker"], "Grund": diagnostic["Grund"],
                                        "reason_codes": diagnostic["reason_codes"]})
        out["single_feasible_count"] = sum(d["feasible"] for d in out["sizing_diagnostics"])
        # A candidate that cannot be sized alone cannot become feasible with
        # smaller per-slot budgets and less group room. Eliminate these BEFORE
        # the bounded search, so expensive high scores cannot displace a
        # feasible lower-ranked candidate in the same group.
        feasible_tickers = {d["Ticker"] for d in out["sizing_diagnostics"] if d["feasible"]}
        pool = [c for c in pool if c["ticker"] in feasible_tickers]
    else:
        out["single_feasible_count"] = None
    out["reason_summary_scope"] = "Jetzt pr\u00fcfen" if ready_tickers else "Gesamter Scan"
    out["reason_summary"] = _rejection_summary(out["rejected"], ready_tickers if ready_tickers else None)
    # Keep the best representative of every available group before filling
    # the bounded pool by quality; 18 technology names must not hide the first
    # eligible non-technology candidate just below that cutoff.
    selected, seen_groups = [], set()
    for candidate in pool:
        if candidate["group"] not in seen_groups and len(selected) < config.pool_limit:
            selected.append(candidate)
            seen_groups.add(candidate["group"])
    selected_names = {c["ticker"] for c in selected}
    for candidate in pool:
        if candidate["ticker"] not in selected_names and len(selected) < config.pool_limit:
            selected.append(candidate)
            selected_names.add(candidate["ticker"])
    pool = sorted(selected, key=lambda c: (-c["merit"], c["ticker"]))
    out["searched_tickers"] = [c["ticker"] for c in pool]
    out["searched_count"] = len(pool)
    if out["errors"]:
        return out
    best = []
    checked = 0
    for n in range(1, min(config.max_positions, len(pool)) + 1):
        for combo in combinations(pool, n):
            checked += 1
            plan = _allocate(combo, config, portfolio, budget, risk_budget)
            if plan:
                best.append(plan)
                if len(best) > 10:
                    best = sorted(best, key=lambda p: (-p["utility"], p["risk"], tuple(c["ticker"] for c in p["items"])))[:3]
    best = sorted(best, key=lambda p: (-p["utility"], p["risk"], tuple(c["ticker"] for c in p["items"])))[:3]
    out["combinations_checked"] = checked
    if not best:
        if out["eligible_count"] == 0:
            out["errors"].append(
                f"Kein Paket: {out['queue_ready_count']} gr\u00fcne Queue-Kandidaten, aber keiner besteht die Daten-, Trigger-, CRV- und Bestandspr\u00fcfung."
            )
        elif not out.get("single_feasible_count"):
            out["errors"].append(
                f"Kein Paket: {out['eligible_count']} Kandidat(en) bestehen die Vorpr\u00fcfung; keiner passt als ganze Mindestposition in Budget, Risiko und Kosten-CRV."
            )
        else:
            out["errors"].append("Im begrenzten Suchraum kein Paket gefunden; Details der Kandidatenpr\u00fcfung beachten.")
        return out
    for plan in best:
        plan["id"] = "PKG-" + fingerprint({"scan": scan_id, "config": asdict(config),
                                              "store": out["store_fingerprint"],
                                              "items": plan["items"]})[:16]
    out["alternatives"] = best
    out["ok"] = True
    return out


def build_intentions(plan: Mapping, alternative: int, store: Mapping, *, watchlist: str,
                     now: datetime, capture_context: Callable[[dict], dict]) -> tuple[dict, bool]:
    """Return a copy to save in ONE positions-namespace write. No journal writes."""
    if not plan.get("ok"):
        raise ValueError("Kein freigegebener Paketplan.")
    if not 0 <= alternative < len(plan.get("alternatives", [])):
        raise ValueError("Paketalternative nicht vorhanden.")
    chosen = plan["alternatives"][alternative]
    key = "v244_open_positions::" + (text(watchlist) or "Standard")
    bucket = dict(store.get(key) or {})
    def existing_id(tk):
        return (((bucket.get(tk) or {}).get("entry_context") or {}).get("trading_package") or {}).get("id")
    if chosen["items"] and all(existing_id(c["ticker"]) == chosen["id"] for c in chosen["items"]):
        return deepcopy(store), False
    if fingerprint(store) != plan["store_fingerprint"]:
        raise ValueError("Bestand/Vormerkungen haben sich ge\u00e4ndert. Paket neu berechnen.")
    conf = PlanConfig(**plan["config"])
    for c in chosen["items"]:
        if not fresh(c.get("scan_at"), now, conf.max_age_hours):
            raise ValueError("Der Scan ist inzwischen veraltet. Neu scannen und Paket neu berechnen.")
        if c["ticker"] in bucket:
            raise ValueError("Eine vorhandene Position darf nicht \u00fcberschrieben werden.")
        context = safe_json(capture_context(dict(c["context_row"])))
        metadata = {"id": chosen["id"], "scan_id": plan["scan_id"], "version": VERSION,
                    "planned_at": timestamp(now).isoformat(), "scan_at": c["scan_at"],
                    "limit": c["limit"], "shares": c["shares"], "currency": c["currency"],
                    "base_currency": conf.base, "fx": c["fx"], "risk_base": c["risk"],
                    "stop": c["stop"], "stop_source": c["stop_source"], "stop_basis": c["stop_basis"],
                    "target": c["target"], "target_source": c["target_source"],
                    "valid_until": (timestamp(c["scan_at"]) + timedelta(hours=conf.max_age_hours)).isoformat()}
        context["trading_package"] = metadata
        context["recommended_stop_initial"] = c["stop"]
        context["risk_per_share_reference"] = c["entry"] - c["stop"]
        bucket[c["ticker"]] = {
            "ticker": c["ticker"], "name": c["name"], "entry": c["entry"],
            "stop": c["stop"], "initial_stop": c["stop"], "target": c["target"],
            "shares": 0, "initial_shares": 0, "realized_shares": 0,
            "realized_pnl": 0.0, "realized_r_weighted": 0.0,
            "stop_history": [], "journal_notes": [], "price_currency": c["currency"],
            "portfolio_group": c["group"], "strategy_origin": "screener",
            "execution_status": "planned", "planned_entry": c["limit"], "planned_shares": c["shares"],
            "entry_context": context, "last_context": deepcopy(context),
            "opened_at_iso": timestamp(now).isoformat(), "created_at": timestamp(now).isoformat(),
            "updated_at": timestamp(now).isoformat(), "last_price": c["entry"],
            "last_price_at": c["scan_at"], "last_price_source": "Atomic Paket-Snapshot",
        }
    result = deepcopy(store)
    result[key] = bucket
    return result, True
