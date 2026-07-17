"""Risk and position sizing helpers extracted in v25.0.

This module contains calculation logic. App-specific analysis callbacks are
registered once by app.py via ``configure_context``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# App callbacks, configured after the analysis engine has been defined.
build_professional_radar_decision_v18 = None
build_radar_entry_rr_package_v182 = None
_v210_alert_price = None
_radar_v182_entry_zone_text = None
_radar_v182_parse_zone = None
_radar_v182_stop_value = None
_radar_v209_main_target_value = None

def configure_context(**callbacks):
    globals().update({k: v for k, v in callbacks.items() if v is not None})

def _v230_safe_float(val, default=None):
    try:
        if val in [None, "", "-", "n/a", "nan"]:
            return default
        f = float(str(val).replace("%", "").replace(",", ".").strip())
        if np.isfinite(f) and not pd.isna(f):
            return f
    except Exception:
        pass
    return default


def _v230_price_text(val, digits=2):
    f = _v230_safe_float(val, default=None)
    if f is None:
        return "n/a"
    try:
        return f"{f:.{int(digits)}f}"
    except Exception:
        return str(round(f, 2))




def _v2410_infer_quote_currency(ticker, row=None, result=None, fallback="USD"):
    """Quote-Waehrung fuer Live-/Risiko-Rechner robust ableiten.

    Der Risiko-Rechner rechnet in der Kurswaehrung des ausgewaehlten Titels.
    US-Ticker sollen daher USD zeigen, europaeische Suffixe typischerweise EUR usw.
    """
    tk = str(ticker or "").strip().upper()
    sources = []
    for obj in (row, result):
        if isinstance(obj, dict):
            sources.append(obj)
            info = obj.get("info")
            if isinstance(info, dict):
                sources.append(info)
    for obj in sources:
        for key in ("currency", "Waehrung", "Währung", "quoteCurrency", "financialCurrency"):
            val = obj.get(key)
            if val is not None and str(val).strip() and str(val).strip().lower() not in {"nan", "none", "-", "n/a"}:
                return str(val).strip().upper()
    # Suffix-basierte Fallbacks. Yahoo-Suffixe sind fuer den Risiko-Rechner
    # ausreichend, falls die Analyse kein currency-Feld liefert.
    if tk.endswith((".DE", ".F", ".BE", ".DU", ".HM", ".HA", ".MU", ".SG", ".MI", ".PA", ".AS", ".MC", ".LS", ".VI", ".BR", ".HE", ".IR", ".AT", ".OL")):
        return "EUR"
    if tk.endswith(".L"):
        return "GBP"
    if tk.endswith(".SW"):
        return "CHF"
    if tk.endswith(".ST"):
        return "SEK"
    if tk.endswith(".CO"):
        return "DKK"
    if tk.endswith(".TO") or tk.endswith(".V"):
        return "CAD"
    if tk.endswith(".AX"):
        return "AUD"
    if tk.endswith(".HK"):
        return "HKD"
    if tk.endswith(".T"):
        return "JPY"
    return str(fallback or "USD").upper()

def _v230_extract_position_inputs(result, style_name="Ausgewogen"):
    """Robuste Entry/Stop/Ziel-Basis fuer den Positionsgroessen-Rechner.

    Der Rechner soll keine neue Kaufempfehlung erzeugen, sondern aus einem bereits
    interessanten Live-/Chart-Setup die Risiko- und Stueckzahlfrage beantworten.
    """
    r = result or {}
    try:
        decision = build_professional_radar_decision_v18(r, style_name)
    except Exception:
        decision = {}
    try:
        rr = build_radar_entry_rr_package_v182(r)
    except Exception:
        rr = {}

    price = _v230_safe_float(rr.get("price"), default=None)
    if price is None:
        try:
            price = _v230_safe_float(_v210_alert_price(r), default=None)
        except Exception:
            price = None
    if price is None:
        price = _v230_safe_float(r.get("price") or r.get("Aktueller_Kurs") or r.get("regularMarketPrice") or r.get("currentPrice"), default=None)

    entry_zone = str(rr.get("entry_zone") or decision.get("entry_zone") or _radar_v182_entry_zone_text(r) or "-").strip()
    try:
        zone_low, zone_high = _radar_v182_parse_zone(entry_zone)
    except Exception:
        zone_low, zone_high = None, None

    stop = _v230_safe_float(rr.get("stop"), default=None)
    if stop is None:
        try:
            stop = _radar_v182_stop_value(r)
        except Exception:
            stop = None
    stop = _v230_safe_float(stop, default=None)

    target = _v230_safe_float(rr.get("tp1"), default=None)
    target_source = str(rr.get("target_source") or "strukturelles Hauptziel").strip()
    if target is None:
        try:
            target, target_source = _radar_v209_main_target_value(r, price=price)
        except Exception:
            target, target_source = None, target_source
    target = _v230_safe_float(target, default=None)

    # Default-Entry: aktueller Kurs, weil Positionsgroesse meist fuer eine jetzt
    # gepruefte Order berechnet wird. Die Entry-Zone bleibt sichtbar und kann
    # manuell uebersteuert werden.
    entry_default = price
    if entry_default is None and zone_high is not None:
        entry_default = zone_high

    status_hint = str(decision.get("bucket") or "-").strip()
    grade = str(decision.get("grade") or "-").strip()
    action = str(decision.get("next_step") or "-").strip()
    crv = _v230_safe_float(rr.get("crv"), default=None)

    return {
        "price": price,
        "entry_default": entry_default,
        "entry_zone": entry_zone,
        "zone_low": _v230_safe_float(zone_low, default=None),
        "zone_high": _v230_safe_float(zone_high, default=None),
        "stop": stop,
        "target": target,
        "target_source": target_source or "Ziel",
        "bucket": status_hint,
        "grade": grade,
        "action": action,
        "crv": crv,
    }


def _v230_calculate_position_size(entry, stop, target, account_size, risk_pct, max_position_pct=None):
    entry = _v230_safe_float(entry, default=None)
    stop = _v230_safe_float(stop, default=None)
    target = _v230_safe_float(target, default=None)
    account_size = _v230_safe_float(account_size, default=None)
    risk_pct = _v230_safe_float(risk_pct, default=None)
    max_position_pct = _v230_safe_float(max_position_pct, default=None)

    if entry is None or stop is None or account_size is None or risk_pct is None:
        return {"ok": False, "error": "Entry, Stop, Depotgroesse oder Risiko fehlen."}
    if entry <= 0 or stop <= 0 or account_size <= 0 or risk_pct <= 0:
        return {"ok": False, "error": "Entry, Stop, Depotgroesse und Risiko muessen groesser als 0 sein."}
    unit_risk = entry - stop
    if unit_risk <= 0:
        return {"ok": False, "error": "Stop/Invalidierung liegt nicht unter dem Entry. Bitte Stop manuell pruefen."}

    risk_amount = account_size * (risk_pct / 100.0)
    shares_raw = risk_amount / unit_risk
    shares_floor = int(max(0, np.floor(shares_raw)))
    position_value = shares_floor * entry
    actual_risk = shares_floor * unit_risk
    actual_risk_pct = (actual_risk / account_size * 100.0) if account_size else None
    stop_distance_pct = unit_risk / entry * 100.0

    max_position_value = None
    max_position_shares = None
    capped = False
    if max_position_pct is not None and max_position_pct > 0:
        max_position_value = account_size * (max_position_pct / 100.0)
        max_position_shares = int(max(0, np.floor(max_position_value / entry)))
        if shares_floor > max_position_shares:
            shares_floor = max_position_shares
            position_value = shares_floor * entry
            actual_risk = shares_floor * unit_risk
            actual_risk_pct = (actual_risk / account_size * 100.0) if account_size else None
            capped = True

    reward = None
    crv = None
    if target is not None and target > entry:
        reward = target - entry
        crv = reward / unit_risk if unit_risk > 0 else None

    return {
        "ok": True,
        "risk_amount": risk_amount,
        "unit_risk": unit_risk,
        "shares_raw": shares_raw,
        "shares": shares_floor,
        "position_value": position_value,
        "actual_risk": actual_risk,
        "actual_risk_pct": actual_risk_pct,
        "stop_distance_pct": stop_distance_pct,
        "target": target,
        "reward": reward,
        "crv": crv,
        "max_position_value": max_position_value,
        "max_position_shares": max_position_shares,
        "capped": capped,
    }


# ---------- v24.5: Positions-/Exit-Monitor mit Persistenz ----------
