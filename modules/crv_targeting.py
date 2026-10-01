"""Deterministic helpers for CHSM operational swing-CRV targets.

This module deliberately keeps analyst consensus targets out of the productive
TP2 / swing-CRV path. Analyst targets may still be shown as context elsewhere,
but they must not make the operative CRV jump merely because Yahoo returned or
withheld that optional field on a scan.
"""
from __future__ import annotations

import math
from typing import Any


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def projected_breakout_target(level: Any, price: Any, multiplier: float) -> float:
    """Return a projected target above *price* based on a prior breakout level.

    The old code checked ``level > price`` before multiplying the level. During a
    real breakout that condition is normally false by definition, so the target
    disappeared. We instead validate the *projected* target.
    """
    base = _finite(level)
    current = _finite(price)
    factor = _finite(multiplier)
    if base is None or current is None or factor is None or base <= 0 or factor <= 0:
        return math.nan
    target = base * factor
    return target if target > current else math.nan


def select_operational_tp2(
    *,
    price: Any,
    risk_per_share: Any,
    setup_type: str,
    technical_target: Any = None,
    high52: Any = None,
) -> dict[str, Any]:
    """Select TP2 for the productive swing CRV without analyst targets.

    Existing CHSM behaviour is retained where possible:
    - setup target first,
    - 52-week high second,
    - 2R fallback if no structural target exists,
    - minimum 1.8R planning floor when a structural target lies closer.

    The crucial difference is provenance: when the 1.8R floor wins, the source
    is explicitly marked as synthetic instead of pretending the structural
    target itself produced a 1.8 CRV.
    """
    current = _finite(price)
    risk = _finite(risk_per_share)
    if current is None or risk is None or current <= 0 or risk <= 0:
        return {
            "value": math.nan,
            "source": "CRV-Ziel nicht berechenbar",
            "kind": "missing",
            "base_target": math.nan,
            "base_source": None,
            "floor": math.nan,
            "synthetic": False,
        }

    floor = current + 1.8 * risk
    technical = _finite(technical_target)
    if technical is not None and technical > current:
        if technical + 1e-9 >= floor:
            return {
                "value": round(technical, 2),
                "source": f"Primärziel aus Setup ({setup_type})",
                "kind": "technical",
                "base_target": technical,
                "base_source": f"Setup-Ziel ({setup_type})",
                "floor": floor,
                "synthetic": False,
            }
        return {
            "value": round(floor, 2),
            "source": f"Synthetisches 1,8R-Mindestziel · Setup-Ziel ({setup_type}) lag näher",
            "kind": "synthetic_1_8r_floor",
            "base_target": technical,
            "base_source": f"Setup-Ziel ({setup_type})",
            "floor": floor,
            "synthetic": True,
        }

    yearly_high = _finite(high52)
    if yearly_high is not None and yearly_high > current:
        if yearly_high + 1e-9 >= floor:
            return {
                "value": round(yearly_high, 2),
                "source": "52W-Hoch",
                "kind": "high52",
                "base_target": yearly_high,
                "base_source": "52W-Hoch",
                "floor": floor,
                "synthetic": False,
            }
        return {
            "value": round(floor, 2),
            "source": "Synthetisches 1,8R-Mindestziel · 52W-Hoch lag näher",
            "kind": "synthetic_1_8r_floor",
            "base_target": yearly_high,
            "base_source": "52W-Hoch",
            "floor": floor,
            "synthetic": True,
        }

    value = current + 2.0 * risk
    return {
        "value": round(value, 2),
        "source": "2R-Fallback · kein strukturelles TP2 verfügbar",
        "kind": "synthetic_2r_fallback",
        "base_target": math.nan,
        "base_source": None,
        "floor": floor,
        "synthetic": True,
    }


def structural_crv_from_plan(*, price: Any, risk_per_share: Any, plan: dict[str, Any]) -> dict[str, Any]:
    """Return the measured structural CRV behind an operational TP2 plan.

    Synthetic planning floors are intentionally ignored. The real target is the
    plan's ``base_target`` (setup structure or 52W high). If no such target
    exists, the technical CRV is unavailable rather than fabricated.
    """
    current = _finite(price)
    risk = _finite(risk_per_share)
    base = _finite((plan or {}).get("base_target"))
    source = (plan or {}).get("base_source") or "kein belastbares strukturelles Ziel"
    if current is None or risk is None or base is None or current <= 0 or risk <= 0 or base <= current:
        return {"crv": math.nan, "target": math.nan, "source": "kein belastbares strukturelles Ziel"}
    return {"crv": (base - current) / risk, "target": base, "source": str(source)}
