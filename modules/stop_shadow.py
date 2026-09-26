"""Shadow-only stop models for CHSM.

This module must not alter productive stop, CRV, sizing, or trading decisions.
It exists to compare the current productive risk stop with a structure-first
hybrid proposal before any cutover decision is made.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _num(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _tradeability(distance_pct: Optional[float], *, has_structure: bool) -> str:
    if not has_structure:
        return "kein Strukturstop"
    if distance_pct is None:
        return "n/a"
    if distance_pct <= 8.0:
        return "normal"
    if distance_pct <= 10.0:
        return "erhoeht"
    if distance_pct <= 12.0:
        return "kritisch"
    return "besserer Entry abwarten"


def build_hybrid_stop_shadow(
    structure_stop: Any,
    entry: Any,
    *,
    atr_pct: Any = None,
    atr_abs: Any = None,
    structure_buffer_atr: float = 0.35,
    min_entry_atr: float = 0.80,
    provisional_atr: float = 1.00,
) -> Dict[str, Any]:
    """Build a structure-first hybrid stop for shadow comparison only.

    Long setup logic:
    - A valid chart structure remains the conceptual invalidation anchor.
    - Execution stop gets 0.35 ATR below that structure.
    - The stop is at least 0.80 ATR below entry to avoid normal noise.
    - There is no fixed 3.5% minimum in this shadow model.
    - If no valid structure exists, a provisional 1.0 ATR risk stop is shown,
      but it is explicitly *not* a complete trade plan.
    """
    ref = _num(entry)
    structure = _num(structure_stop)
    apct = _num(atr_pct)
    aabs = _num(atr_abs)

    if ref is None or ref <= 0:
        return {
            "ok": False,
            "hybrid_stop": None,
            "has_structure": False,
            "provisional": False,
            "full_trade_plan": False,
            "reason": "ungueltiger Entry",
        }

    if (aabs is None or aabs <= 0) and apct is not None and apct > 0:
        aabs = ref * apct / 100.0
    if (apct is None or apct <= 0) and aabs is not None and aabs > 0:
        apct = aabs / ref * 100.0

    has_structure = bool(structure is not None and 0 < structure < ref)
    atr_valid = bool(aabs is not None and aabs > 0)

    structure_distance_pct = None
    structure_distance_atr = None
    if has_structure:
        structure_distance_pct = (ref - structure) / ref * 100.0
        if atr_valid:
            structure_distance_atr = (ref - structure) / aabs

    if has_structure and atr_valid:
        structure_buffer_stop = structure - structure_buffer_atr * aabs
        noise_floor_stop = ref - min_entry_atr * aabs
        hybrid = min(structure_buffer_stop, noise_floor_stop)
        if hybrid <= 0 or hybrid >= ref:
            return {
                "ok": False,
                "hybrid_stop": None,
                "has_structure": True,
                "provisional": False,
                "full_trade_plan": False,
                "reason": "Hybrid-Stop numerisch unplausibel",
            }
        hybrid_distance_pct = (ref - hybrid) / ref * 100.0
        hybrid_distance_atr = (ref - hybrid) / aabs
        noise_conflict = bool(structure_distance_atr is not None and structure_distance_atr < min_entry_atr)
        return {
            "ok": True,
            "entry": ref,
            "atr_pct": apct,
            "atr_abs": aabs,
            "structure_stop": structure,
            "structure_distance_pct": structure_distance_pct,
            "structure_distance_atr": structure_distance_atr,
            "structure_buffer_atr": float(structure_buffer_atr),
            "structure_buffer_stop": float(structure_buffer_stop),
            "min_entry_atr": float(min_entry_atr),
            "noise_floor_stop": float(noise_floor_stop),
            "hybrid_stop": float(hybrid),
            "hybrid_distance_pct": float(hybrid_distance_pct),
            "hybrid_distance_atr": float(hybrid_distance_atr),
            "has_structure": True,
            "provisional": False,
            "full_trade_plan": True,
            "noise_conflict": noise_conflict,
            "tradeability": _tradeability(hybrid_distance_pct, has_structure=True),
            "source": f"Struktur + {structure_buffer_atr:.2f} ATR; mind. {min_entry_atr:.2f} ATR vom Entry",
        }

    if has_structure:
        # Without ATR, do not fabricate a volatility buffer. The shadow result is
        # simply the structure and is marked as lacking ATR confirmation.
        hybrid_distance_pct = structure_distance_pct
        return {
            "ok": True,
            "entry": ref,
            "atr_pct": apct,
            "atr_abs": aabs,
            "structure_stop": structure,
            "structure_distance_pct": structure_distance_pct,
            "structure_distance_atr": None,
            "hybrid_stop": structure,
            "hybrid_distance_pct": hybrid_distance_pct,
            "hybrid_distance_atr": None,
            "has_structure": True,
            "provisional": False,
            "full_trade_plan": True,
            "noise_conflict": False,
            "tradeability": _tradeability(hybrid_distance_pct, has_structure=True),
            "source": "Struktur ohne ATR-Puffer (ATR nicht verfuegbar)",
        }

    if atr_valid:
        provisional = ref - provisional_atr * aabs
        if provisional <= 0 or provisional >= ref:
            return {
                "ok": False,
                "hybrid_stop": None,
                "has_structure": False,
                "provisional": True,
                "full_trade_plan": False,
                "reason": "Provisorischer ATR-Stop numerisch unplausibel",
            }
        distance_pct = (ref - provisional) / ref * 100.0
        return {
            "ok": True,
            "entry": ref,
            "atr_pct": apct,
            "atr_abs": aabs,
            "structure_stop": None,
            "structure_distance_pct": None,
            "structure_distance_atr": None,
            "provisional_atr": float(provisional_atr),
            "hybrid_stop": float(provisional),
            "hybrid_distance_pct": float(distance_pct),
            "hybrid_distance_atr": float(provisional_atr),
            "has_structure": False,
            "provisional": True,
            "full_trade_plan": False,
            "noise_conflict": False,
            "tradeability": "kein Strukturstop",
            "source": f"Provisorischer {provisional_atr:.2f}-ATR-Risikostop (keine Struktur)",
        }

    return {
        "ok": False,
        "hybrid_stop": None,
        "has_structure": False,
        "provisional": False,
        "full_trade_plan": False,
        "reason": "Weder Struktur noch ATR belastbar",
    }
