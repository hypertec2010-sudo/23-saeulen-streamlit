"""Central decision adapter introduced in v27.1.

The first migration step is intentionally non-destructive: it converts the
existing analysis result into one stable decision contract without replacing
legacy UI rules yet. Radar, screener, single analysis and position views can
migrate to this contract incrementally.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class Decision:
    version: str
    decision: str
    label: str
    confidence: float
    traffic_light: str
    state: str
    mode: str
    entry: Any
    stop: Any
    target: Any
    reason: tuple[str, ...]
    invalidation: str
    source_action: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["reason"] = list(self.reason)
        return payload


def _first(data: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "-", "none", "nan", "n/a"}:
            continue
        return value
    return default


def _num(value: Any, default: float = 50.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number != number:  # NaN
        return float(default)
    return max(0.0, min(100.0, number))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _nested(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    return value if isinstance(value, Mapping) else {}


def _source_action(result: Mapping[str, Any]) -> str:
    action_pkg = _nested(result, "action_clarity_pkg")
    return _text(_first(
        {**result, "_action_pkg_label": action_pkg.get("label")},
        ["_action_pkg_label", "position_action", "pm_action", "emp", "action", "recommendation"],
        "Beobachten",
    ))


def _classify_action(action: str, position_mode: bool) -> tuple[str, str, str]:
    value = action.lower()
    if any(term in value for term in ("exit", "verkauf", "verkaufen", "reduz", "de-risk", "derisk")):
        return "EXIT", "Exit / Risiko reduzieren", "RED"
    if any(term in value for term in ("teilgewinn", "gewinn sichern")):
        return "PARTIAL_PROFIT", "Teilgewinn prüfen", "ORANGE"
    if any(term in value for term in ("aufstock", "add-on", "add on")):
        return "ADD", "Aufstocken prüfen", "GREEN"
    if any(term in value for term in ("kaufen", "jetzt prüfbar", "jetzt pruefbar", "umsetzen")):
        return "BUY", "Kaufen / aktiv prüfen", "GREEN"
    if any(term in value for term in ("trendfolge", "pullback")):
        return "TREND_FOLLOW", "Trendfolge vorbereiten", "GREEN"
    if any(term in value for term in ("vorbereiten", "reclaim", "trigger", "nahe")):
        return "PREPARE", "Vorbereiten", "ORANGE"
    if any(term in value for term in ("kein trade", "avoid", "abwarten", "warten", "nicht freigegeben")):
        return "NO_TRADE", "Kein Trade", "RED"
    if position_mode:
        return "HOLD", "Position halten / beobachten", "YELLOW"
    return "WATCH", "Beobachten", "YELLOW"


def _derive_state(decision: str, result: Mapping[str, Any]) -> str:
    if decision == "EXIT":
        return "INVALIDATED"
    if decision == "PARTIAL_PROFIT":
        return "WEAKENING"
    if decision in {"BUY", "ADD"}:
        return "TRIGGER_ACTIVE"
    if decision in {"PREPARE", "TREND_FOLLOW"}:
        return "ARMED"
    if decision == "NO_TRADE":
        return "OBSERVE"

    trigger = _text(_first(result, ["trigger_status", "trade_state", "state"], "")).lower()
    if any(term in trigger for term in ("invalid", "gebrochen", "failed")):
        return "INVALIDATED"
    if any(term in trigger for term in ("active", "aktiv", "confirmed", "bestätigt", "bestaetigt")):
        return "TRIGGER_ACTIVE"
    if any(term in trigger for term in ("armed", "bereit", "nahe")):
        return "ARMED"
    return "OBSERVE"


def _derive_confidence(result: Mapping[str, Any], traffic_light: str) -> float:
    timing = _nested(result, "timing_action_confidence_pkg")
    confluence = _nested(result, "trigger_confluence_pkg")
    chart = _nested(result, "charttechnik_setup_pkg")

    candidates = [
        timing.get("score"),
        confluence.get("score"),
        chart.get("score"),
        result.get("tradeability_score"),
        result.get("watchlist_priority_score"),
        result.get("investment"),
        result.get("score"),
    ]
    values = [_num(value, -1.0) for value in candidates if value is not None]
    values = [value for value in values if value >= 0]
    if values:
        confidence = sum(values[:3]) / min(3, len(values))
    else:
        confidence = {"GREEN": 72.0, "ORANGE": 60.0, "YELLOW": 50.0, "RED": 30.0}.get(traffic_light, 50.0)

    fomo = _nested(result, "fomo_smart_money_pkg")
    if _text(fomo.get("label")).lower() == "kritisch":
        confidence = min(confidence, 55.0)
    if result.get("valid_trade_setup") is False:
        confidence = min(confidence, 64.0)
    return round(max(0.0, min(100.0, confidence)), 1)


def _reasons(result: Mapping[str, Any], source_action: str) -> tuple[str, ...]:
    reasons: list[str] = []
    for package_name in (
        "action_clarity_pkg",
        "timing_action_confidence_pkg",
        "trigger_confluence_pkg",
        "charttechnik_setup_pkg",
        "exit_protection_pkg",
    ):
        package = _nested(result, package_name)
        for key in ("summary", "action", "why_text", "reason"):
            value = _text(package.get(key))
            if value and value not in reasons:
                reasons.append(value)
                break
    if not reasons and source_action:
        reasons.append(source_action)
    return tuple(reasons[:4])


def build_decision(result: Optional[Mapping[str, Any]], *, position_mode: Optional[bool] = None) -> Dict[str, Any]:
    """Build the canonical v27.1 decision contract from an analysis result.

    This adapter does not mutate the source mapping and does not perform market
    data access. Missing fields are tolerated so it can be used by all views.
    """
    data: Mapping[str, Any] = result or {}
    inferred_position_mode = bool(position_mode) if position_mode is not None else bool(
        _first(data, ["position_mode", "buy_in_override"], False)
    )
    source_action = _source_action(data)
    decision_code, label, light = _classify_action(source_action, inferred_position_mode)

    exit_score = _num(_first(data, ["tactical_exit_risk", "exit_score"], 0), 0)
    if inferred_position_mode and exit_score >= 80 and decision_code not in {"EXIT", "PARTIAL_PROFIT"}:
        decision_code, label, light = "EXIT", "Exit / Risiko reduzieren", "RED"
    elif inferred_position_mode and exit_score >= 65 and decision_code in {"HOLD", "WATCH"}:
        decision_code, label, light = "PARTIAL_PROFIT", "Teilgewinn / Stop prüfen", "ORANGE"

    state = _derive_state(decision_code, data)
    confidence = _derive_confidence(data, light)
    entry = _first(data, ["suggested_entry_zone", "entry_zone", "entry", "buy_zone"], "-")
    stop = _first(data, ["stop_used", "stop", "tb_stop"], "-")
    target = _first(data, ["tp1", "technical_target_1", "target", "tp2"], "-")

    action_pkg = _nested(data, "action_clarity_pkg")
    chart_pkg = _nested(data, "charttechnik_setup_pkg")
    invalidation = _text(_first(
        {"action": action_pkg.get("invalid"), "chart": chart_pkg.get("invalid"), **data},
        ["action", "chart", "pm_stop_plan", "stop_action"],
        "-",
    ))

    mode = "POSITION" if inferred_position_mode else "WATCHLIST"
    return Decision(
        version="27.1",
        decision=decision_code,
        label=label,
        confidence=confidence,
        traffic_light=light,
        state=state,
        mode=mode,
        entry=entry,
        stop=stop,
        target=target,
        reason=_reasons(data, source_action),
        invalidation=invalidation,
        source_action=source_action,
    ).to_dict()


def attach_decision(result: Optional[Dict[str, Any]], *, position_mode: Optional[bool] = None) -> Dict[str, Any]:
    """Attach the canonical decision without removing legacy fields."""
    payload: Dict[str, Any] = result if isinstance(result, dict) else {}
    payload["decision_engine"] = build_decision(payload, position_mode=position_mode)
    payload["decision_engine_version"] = "27.1"
    return payload
