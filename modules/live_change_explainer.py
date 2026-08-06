"""Erklaert Live-Screener-Statuswechsel anhand gespeicherter Diagnosedaten.

Die Ampel kann sich auch bei nahezu unveraendertem Kurs aendern, weil Trigger,
Timing, Konfluenz, Radar-Bucket, Freigaben oder Gates neu bewertet werden. Dieses
Modul erzeugt daraus eine kurze, reproduzierbare Begruendung fuer UI und Historie.
"""
from __future__ import annotations

from typing import Any, Mapping
import math


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, "", "-", "n/a", "nan"):
            return default
        text = str(value).strip().replace("%", "").replace(",", ".")
        if "/" in text:
            text = text.split("/", 1)[0]
        number = float(text)
        if not math.isfinite(number):
            return default
        return number
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "ja", "yes", "aktiv"}


def _text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "nan", "-"} else text


def _fmt_number(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}".replace(".", ",")


def _component_changes(previous: Mapping[str, Any], current: Mapping[str, Any]) -> list[tuple[float, str]]:
    labels = {
        "timing_component": "Timing",
        "conf_component": "Konfluenz",
        "chart_component": "Chart",
        "trigger_component": "Trigger",
        "trend_component": "Trend",
        "crv_component": "CRV-Komponente",
    }
    changes: list[tuple[float, str]] = []
    for key, label in labels.items():
        old = _num(previous.get(key))
        new = _num(current.get(key))
        if old is None or new is None:
            continue
        delta = new - old
        if abs(delta) < 4.0:
            continue
        changes.append((delta, f"{label} {_fmt_number(old)}→{_fmt_number(new)}"))
    # Negative Veraenderungen zuerst, danach groesste absolute Veraenderung.
    return sorted(changes, key=lambda item: (item[0] >= 0, -abs(item[0])))


def build_change_explanation(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    change: str,
) -> str:
    """Liefert eine kurze Erklaerung fuer einen finalen Ampel-/Statuswechsel."""
    previous = previous if isinstance(previous, Mapping) else {}
    current = current if isinstance(current, Mapping) else {}
    change = _text(change)

    if not previous or not _text(previous.get("status")):
        return "Erster gespeicherter Vergleichsstand; noch kein vorheriger Status vorhanden."
    if change == "Unverändert":
        return "-"

    old_price = _num(previous.get("price"))
    new_price = _num(current.get("price"))
    price_pct: float | None = None
    if old_price is not None and new_price is not None and old_price > 0:
        price_pct = (new_price / old_price - 1.0) * 100.0

    old_score = _num(previous.get("live_score"))
    new_score = _num(current.get("live_score"))
    score_delta = None if old_score is None or new_score is None else new_score - old_score

    reasons: list[str] = []
    new_ampel = _text(current.get("ampel"))

    # Harte Ursachen zuerst. Diese koennen Rot ausloesen, auch wenn der gerundete
    # Kurs gleich aussieht.
    if _bool(current.get("invalidated")) and not _bool(previous.get("invalidated")):
        reasons.append("Invalidierung wurde neu gebrochen")
    if _bool(current.get("entry_hard_gate")) and not _bool(previous.get("entry_hard_gate")):
        reasons.append("ein hartes Einstiegsgate wurde aktiv")

    old_bucket = _text(previous.get("radar_bucket"))
    new_bucket = _text(current.get("radar_bucket"))
    if old_bucket != new_bucket and new_bucket:
        reasons.append(f"Radar-Bucket {old_bucket or 'n/a'}→{new_bucket}")

    old_release = _bool(previous.get("final_release_ok"))
    new_release = _bool(current.get("final_release_ok"))
    if old_release and not new_release:
        reasons.append("finale Freigabe ist entfallen")
    elif not old_release and new_release:
        reasons.append("finale Freigabe ist neu aktiv")

    trigger_labels = {
        "bucket_active": "Jetzt-prüfbar-Trigger",
        "wave_active": "Wave-Trigger",
        "entry_reached": "Entry-Zone",
        "bucket_near": "Triggernähe",
    }
    for key, label in trigger_labels.items():
        old_flag = _bool(previous.get(key))
        new_flag = _bool(current.get(key))
        if old_flag and not new_flag:
            reasons.append(f"{label} ist entfallen")
        elif not old_flag and new_flag and new_ampel != "🔴":
            reasons.append(f"{label} ist neu aktiv")

    old_grade = _text(previous.get("grade"))
    new_grade = _text(current.get("grade"))
    if old_grade and new_grade and old_grade != new_grade:
        reasons.append(f"Grade {old_grade}→{new_grade}")

    old_crv = _num(previous.get("crv"))
    new_crv = _num(current.get("crv"))
    if old_crv is not None and new_crv is not None and abs(new_crv - old_crv) >= 0.15:
        reasons.append(f"CRV {_fmt_number(old_crv, 2)}→{_fmt_number(new_crv, 2)}")

    if score_delta is not None and abs(score_delta) >= 2.0:
        reasons.append(
            f"Live-Score {_fmt_number(old_score)}→{_fmt_number(new_score)} "
            f"({score_delta:+.0f})".replace(".", ",")
        )

    for _, component_text in _component_changes(previous, current)[:2]:
        reasons.append(component_text)

    # Doppelte oder sehr aehnliche Fragmente vermeiden.
    unique_reasons: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        marker = reason.lower()
        if marker in seen:
            continue
        seen.add(marker)
        unique_reasons.append(reason)

    if not unique_reasons:
        fallback = _text(current.get("reason")) or _text(current.get("status"))
        unique_reasons.append(fallback or "interne Trigger-/Indikatorbewertung hat sich geändert")

    if price_pct is None:
        price_context = "Kursvergleich nicht verfügbar."
    elif abs(price_pct) < 0.10:
        price_context = f"Kurs nahezu unverändert ({price_pct:+.2f} %).".replace(".", ",")
    else:
        price_context = f"Kurs {price_pct:+.2f} %.".replace(".", ",")

    explanation = f"{price_context} Auslöser: " + "; ".join(unique_reasons[:5]) + "."
    return explanation[:520]
