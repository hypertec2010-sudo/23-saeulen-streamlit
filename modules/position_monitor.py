"""Persistent position and exit-monitor helpers. v28.9 adds Exit Engine 2.0."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

_BASE_DIR = Path(__file__).resolve().parent.parent
_event_logger = lambda **kwargs: False
_v230_safe_float = None
_v230_price_text = None
_storage = None
_repository = None

def configure_context(*, base_dir=None, event_logger=None, safe_float=None, price_text=None, storage=None, repository=None):
    global _BASE_DIR, _event_logger, _v230_safe_float, _v230_price_text, _storage, _repository
    if base_dir is not None:
        _BASE_DIR = Path(base_dir)
    if event_logger is not None:
        _event_logger = event_logger
    if safe_float is not None:
        _v230_safe_float = safe_float
    if price_text is not None:
        _v230_price_text = price_text
    if storage is not None:
        _storage = storage
    if repository is not None:
        _repository = repository

def _v244_position_store_key(watchlist_name=""):
    try:
        wl = str(watchlist_name or "Standard").strip() or "Standard"
    except Exception:
        wl = "Standard"
    return f"v244_open_positions::{wl}"


def _v245_positions_store_path():
    """Lokaler Sidecar-Speicher fuer offene Positionen.

    Dieser Speicher entkoppelt den Positions-/Exit-Monitor von der Streamlit-Session.
    Auf lokalem Betrieb oder persistentem App-Speicher bleiben Positionen nach Reload
    erhalten. Auf kurzlebigen Cloud-Dateisystemen dient st.session_state weiterhin als
    Fallback fuer die laufende Session.
    """
    try:
        base_dir = _BASE_DIR
        return base_dir / ".live_monitor_positions_v245.json"
    except Exception:
        return Path("/tmp/.live_monitor_positions_v245.json")


def _v245_safe_json_load(path):
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _v245_load_all_positions():
    storage_store = None
    if _repository is not None:
        try:
            candidate = _repository.load_all()
            if isinstance(candidate, dict):
                storage_store = candidate
        except Exception:
            storage_store = None
    elif _storage is not None:
        try:
            candidate = _storage.load_namespace("positions", default=None)
            if isinstance(candidate, dict):
                storage_store = candidate
        except Exception:
            storage_store = None

    file_store = _v245_safe_json_load(_v245_positions_store_path())
    session_store = {}
    try:
        session_store = st.session_state.get("v245_persistent_positions_store", {})
        if not isinstance(session_store, dict):
            session_store = {}
    except Exception:
        session_store = {}

    # Zentraler Speicher ist die Quelle der Wahrheit. Legacy-Datei und Session
    # bleiben als Rueckwaertskompatibilitaet erhalten, solange noch nicht migriert.
    merged = dict(storage_store if storage_store is not None else file_store)
    if storage_store is None:
        try:
            for k, v in session_store.items():
                if isinstance(v, dict):
                    merged[k] = v
        except Exception:
            pass
    try:
        st.session_state.v245_persistent_positions_store = merged
    except Exception:
        pass
    return merged


def _v245_save_all_positions(store):
    store = store if isinstance(store, dict) else {}
    try:
        st.session_state.v245_persistent_positions_store = store
    except Exception:
        pass
    storage_ok = False
    if _repository is not None:
        try:
            storage_ok = bool(_repository.save_all(store))
        except Exception:
            storage_ok = False
    elif _storage is not None:
        try:
            storage_ok = bool(_storage.save_namespace("positions", store))
        except Exception:
            storage_ok = False
    file_ok = False
    try:
        path = _v245_positions_store_path()
        path.write_text(json.dumps(store, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        file_ok = True
    except Exception:
        file_ok = False
    return bool(storage_ok or file_ok)


def _v244_get_positions(watchlist_name=""):
    key = _v244_position_store_key(watchlist_name)
    # v24.5: zuerst persistenten Gesamtstore laden.
    try:
        all_pos = _v245_load_all_positions()
        data = all_pos.get(key, {})
        if isinstance(data, dict):
            # Session-Key fuer Rueckwaertskompatibilitaet spiegeln.
            try:
                st.session_state[key] = data
            except Exception:
                pass
            return data
    except Exception:
        pass
    # Fallback v24.4: nur Session.
    try:
        data = st.session_state.get(key, {})
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _v244_save_positions(watchlist_name, positions):
    key = _v244_position_store_key(watchlist_name)
    positions = positions if isinstance(positions, dict) else {}
    try:
        st.session_state[key] = positions
    except Exception:
        pass
    try:
        all_pos = _v245_load_all_positions()
        all_pos[key] = positions
        _v245_save_all_positions(all_pos)
    except Exception:
        pass


def _v245_delete_positions_for_watchlist(watchlist_name):
    key = _v244_position_store_key(watchlist_name)
    try:
        st.session_state[key] = {}
    except Exception:
        pass
    try:
        all_pos = _v245_load_all_positions()
        all_pos.pop(key, None)
        _v245_save_all_positions(all_pos)
    except Exception:
        pass


def _v244_row_price(row):
    try:
        return _v230_safe_float(row.get("Kurs"), default=None)
    except Exception:
        return None


def _v244_reference_stop(pos, entry=None, current_stop=None):
    """Return the original risk stop used as the R-multiple denominator.

    A trailing/current stop may legitimately move above the entry. In that case
    it must not replace the original stop for R calculations. Older positions
    can be recovered from the first valid stop-history value.
    """
    pos = pos or {}
    if entry is None:
        entry = _v230_safe_float(pos.get("entry"), default=None)
    if current_stop is None:
        current_stop = _v230_safe_float(pos.get("stop"), default=None)
    if entry is None:
        return None

    candidates = [_v230_safe_float(pos.get("initial_stop"), default=None)]
    try:
        for item in list(pos.get("stop_history") or []):
            if not isinstance(item, dict):
                continue
            candidates.append(_v230_safe_float(item.get("old_stop"), default=None))
            candidates.append(_v230_safe_float(item.get("new_stop"), default=None))
    except Exception:
        pass
    candidates.append(current_stop)

    for candidate in candidates:
        if candidate is not None and candidate > 0 and candidate < entry:
            return candidate
    return None


def _v244_calc_trade_state(pos, live_row=None):
    pos = pos or {}
    live_row = live_row or {}
    entry = _v230_safe_float(pos.get("entry"), default=None)
    stop = _v230_safe_float(pos.get("stop"), default=None)
    target = _v230_safe_float(pos.get("target"), default=None)
    shares = _v230_safe_float(pos.get("shares"), default=0) or 0
    current = _v244_row_price(live_row)
    if current is None:
        current = _v230_safe_float(pos.get("last_price"), default=None)

    reference_stop = _v244_reference_stop(pos, entry=entry, current_stop=stop)
    unit_risk = None
    r_mult = None
    pnl = None
    pnl_pct = None
    if entry is not None and reference_stop is not None and current is not None:
        unit_risk = entry - reference_stop
        r_mult = (current - entry) / unit_risk if unit_risk > 0 else None
    if entry is not None and current is not None:
        pnl = (current - entry) * shares if shares else None
        pnl_pct = (current / entry - 1.0) * 100.0 if entry else None

    ampel = "⚪"
    status = "Unvollständig"
    action = "Entry, Stop und Stückzahl ergänzen."
    stop_hint = "-"
    if current is not None and stop is not None and current <= stop:
        ampel = "🔴"
        status = "Stop / Invalidierung erreicht"
        action = "Sofort prüfen: Stop-Regel, Exit oder These neu bewerten."
        stop_hint = "Stop ausgelöst oder unterschritten."
    elif r_mult is None:
        ampel = "⚪"
        if entry is None:
            status = "Entry fehlt"
            action = "Für die Berechnung wird ein gültiger Entry benötigt."
        elif current is None:
            status = "Kurs fehlt"
            action = "Für die Berechnung wird ein aktueller oder zuletzt gespeicherter Kurs benötigt."
        elif reference_stop is None and stop is not None and stop >= entry:
            status = "Initialrisiko fehlt"
            action = "Der aktuelle Stop liegt auf/über Entry. Für R wird der ursprüngliche Initial-Stop benötigt."
        else:
            status = "Initial-Stop fehlt"
            action = "Für R-Multiple werden Entry und ursprünglicher Initial-Stop benötigt."
    elif r_mult >= 2.0:
        ampel = "🟢"
        status = "2R+ erreicht"
        action = "Teilgewinn/Trailing-Stop prüfen; Restposition laufen lassen, solange Trend hält."
        stop_hint = "Stop mindestens auf Gewinnschutz/Struktur nachziehen prüfen."
    elif r_mult >= 1.0:
        ampel = "🟢"
        status = "1R erreicht"
        action = "Break-even-Stop oder Teilgewinn prüfen; Risiko aus dem Trade nehmen."
        stop_hint = "Stop auf Einstand oder unter kurzfristige Struktur prüfen."
    elif r_mult >= 0.25:
        ampel = "🟡"
        status = "Positiv, noch <1R"
        action = "Laufen lassen; Stop nicht zu früh nachziehen, Trigger/Trend beobachten."
        stop_hint = "Original-Stop oder enger Strukturstop je nach Setup."
    elif r_mult > -0.5:
        ampel = "⚪"
        status = "Nahe Entry"
        action = "Noch keine Management-Aktion; Stop-Regel einhalten."
        stop_hint = "Plan-Stop beibehalten."
    else:
        ampel = "🟡"
        status = "Unter Druck"
        action = "Positionsgröße/Stop-Regel prüfen; kein Nachkaufen ohne neuen Trigger."
        stop_hint = "Stop/Invalidierung eng beobachten."

    if (
        r_mult is not None
        and entry is not None
        and stop is not None
        and stop >= entry
        and current is not None
        and current > stop
    ):
        stop_hint = "Gewinnschutz aktiv: aktueller Stop liegt auf/über Entry; R basiert weiter auf dem Initial-Stop."

    if target is not None and entry is not None and current is not None and current >= target and target > entry:
        status = status + " · Ziel erreicht"
        action = "Ziel/Teilziel erreicht: Teilgewinn oder Trailing-Plan prüfen."

    return {
        "Ampel": ampel,
        "Status": status,
        "Aktion": action,
        "Stop-Hinweis": stop_hint,
        "Aktueller Kurs": current,
        "R-Multiple": r_mult,
        "P/L": pnl,
        "P/L %": pnl_pct,
        "Risiko je Aktie": unit_risk,
        "R-Basis-Stop": reference_stop,
    }



# ---------- v28.9: Positions-/Exit-Engine 2.0 ----------
def _v289_num(value, default=None):
    """Toleranter Parser fuer numerische Live-/Positionswerte."""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"nan", "none", "n/a", "na", "-"}:
                return default
            if "/" in text:
                text = text.split("/", 1)[0]
            text = text.replace("%", "").replace(",", ".").strip()
            value = text
        if _v230_safe_float is not None:
            parsed = _v230_safe_float(value, default=default)
        else:
            parsed = float(value)
        if parsed is None or pd.isna(parsed):
            return default
        return float(parsed)
    except Exception:
        return default


def _v289_text(value, default="-"):
    try:
        text = str(value or "").strip()
    except Exception:
        text = ""
    if not text or text.lower() in {"nan", "none", "n/a", "na"}:
        return default
    return text


def _v289_position_exit_engine(pos, live_row=None):
    """Konservative Management-Engine fuer eine bereits offene Long-Position.

    Die Engine fuehrt keine Orders aus und setzt keine Stops automatisch. Sie
    kombiniert bestehende Exit-/Trend-/Momentum-Signale aus dem aktuellen
    Atomic-Live-Scan mit dem realen Positionspuffer (P/L, R, Stop, Ziel).
    Fehlende technische Felder werden nicht als Nullrisiko interpretiert.
    """
    pos = dict(pos or {})
    live_row = dict(live_row or {})
    calc = _v244_calc_trade_state(pos, live_row)

    entry = _v289_num(pos.get("entry"), None)
    stop = _v289_num(pos.get("stop"), None)
    target = _v289_num(pos.get("target"), None)
    shares = _v289_num(pos.get("shares"), 0.0) or 0.0
    current = _v289_num(calc.get("Aktueller Kurs"), None)
    live_current = _v289_num(live_row.get("Kurs"), None)
    has_current_live = live_current is not None
    pnl_pct = _v289_num(calc.get("P/L %"), None)
    pnl_abs = _v289_num(calc.get("P/L"), None)
    r_mult = _v289_num(calc.get("R-Multiple"), None)
    reference_stop = _v289_num(calc.get("R-Basis-Stop"), None)

    # Technische Exit-Komponenten kommen aus demselben abgeschlossenen
    # Live-Vollscan wie die angezeigte Position. Dadurch entstehen hier keine
    # zusaetzlichen Provider-Abfragen.
    raw = {
        "Exit": _v289_num(live_row.get("Exit-Score"), None),
        "Taktik": _v289_num(live_row.get("Tactical-Exit-Risk"), None),
        "Trendbruch": _v289_num(live_row.get("Trendbruch-Score"), None),
        "Momentum": _v289_num(live_row.get("Momentum-Collapse-Score"), None),
        "Distribution": _v289_num(live_row.get("Distribution-Score"), None),
        "Relative Schwaeche": _v289_num(live_row.get("Relative-Schwäche-Score"), None),
    }
    weights = {
        "Exit": 0.28,
        "Taktik": 0.18,
        "Trendbruch": 0.22,
        "Momentum": 0.12,
        "Distribution": 0.11,
        "Relative Schwaeche": 0.09,
    }
    available = [(name, val) for name, val in raw.items() if val is not None]
    if available:
        wsum = sum(weights[name] for name, _ in available)
        pressure = sum(val * weights[name] for name, val in available) / max(wsum, 1e-9)
    else:
        pressure = 0.0

    market = _v289_text(live_row.get("Marktregime"), "n/a")
    volatility = _v289_text(live_row.get("Volatilitätsregime"), "n/a")
    rs_dyn = _v289_text(live_row.get("RS-Dynamik"), "n/a")
    market_l = market.lower()
    vol_l = volatility.lower()
    rs_l = rs_dyn.lower()

    # Kontext wirkt moderat. Harte technische Signale erhalten spaeter eigene
    # Floors und koennen durch positives Marktumfeld nicht weggebuegelt werden.
    context_adjustment = 0.0
    if "negativ" in market_l or "bear" in market_l:
        context_adjustment += 5.0
    elif "positiv" in market_l or "bull" in market_l:
        context_adjustment -= 2.0
    if "hoch" in vol_l:
        context_adjustment += 5.0
    elif "erhöht" in vol_l or "erhoeht" in vol_l:
        context_adjustment += 2.0
    if "verschlechter" in rs_l:
        context_adjustment += 6.0
    elif "verbessert" in rs_l:
        context_adjustment -= 2.0
    pressure += context_adjustment

    # Einzelne starke Warnungen duerfen in einem gewichteten Mittel nicht
    # verschwinden.
    exit_raw = raw.get("Exit")
    tactical = raw.get("Taktik")
    trend = raw.get("Trendbruch")
    momentum = raw.get("Momentum")
    distribution = raw.get("Distribution")
    relweak = raw.get("Relative Schwaeche")
    accumulation = _v289_num(live_row.get("Akkumulation-Score"), None)
    if exit_raw is not None and exit_raw >= 80:
        pressure = max(pressure, 82.0)
    if trend is not None and trend >= 80:
        pressure = max(pressure, 82.0)
    elif trend is not None and trend >= 65:
        pressure = max(pressure, 68.0)
    if tactical is not None and tactical >= 80:
        pressure = max(pressure, 74.0)
    if momentum is not None and momentum >= 75:
        pressure = max(pressure, 68.0)
    if distribution is not None and distribution >= 75:
        if accumulation is None or distribution >= accumulation + 10:
            pressure = max(pressure, 65.0)
    if relweak is not None and relweak >= 75:
        pressure = max(pressure, 62.0)

    # Stop-/Zielverletzungen werden nur als *aktuell* gewertet, wenn der Ticker
    # im letzten Atomic-Scan wirklich einen Kurs geliefert hat. Ein gespeicherter
    # last_price darf niemals heimlich einen frischen Exit-Alarm vortaeuschen.
    stop_breached = bool(has_current_live and current is not None and stop is not None and stop > 0 and current <= stop)
    target_reached = bool(
        has_current_live and
        current is not None and target is not None and entry is not None
        and target > entry > 0 and current >= target
    )
    stop_protected = bool(stop is not None and entry is not None and stop >= entry > 0)
    if stop_breached:
        pressure = 100.0
    pressure = max(0.0, min(100.0, float(pressure)))

    if not has_current_live:
        confidence = "Kein aktueller Scanwert"
    elif len(available) >= 5:
        confidence = "Hoch"
    elif len(available) >= 3:
        confidence = "Mittel"
    elif len(available) >= 1:
        confidence = "Reduziert"
    else:
        confidence = "Nur Positionsdaten"

    reasons = []
    if not has_current_live:
        reasons.append("Ticker hat im letzten Atomic-Scan keinen aktuellen Kurswert")
    if stop_breached:
        reasons.append("Stop/Invalidierung ist erreicht oder unterschritten")
    ranked = sorted(available, key=lambda item: item[1], reverse=True)
    labels = {
        "Exit": "Exit-Druck",
        "Taktik": "kurzfristiges Ruecksetzerrisiko",
        "Trendbruch": "Trendbruchrisiko",
        "Momentum": "Momentum-Abbau",
        "Distribution": "Distribution",
        "Relative Schwaeche": "relative Schwaeche",
    }
    for name, val in ranked:
        threshold = 55.0 if name in {"Exit", "Taktik", "Trendbruch"} else 60.0
        if val >= threshold and len(reasons) < 4:
            reasons.append(f"{labels.get(name, name)} {val:.0f}/100")
    if ("negativ" in market_l or "bear" in market_l) and len(reasons) < 4:
        reasons.append("negatives Marktregime verstaerkt Positionsrisiko")
    if "hoch" in vol_l and len(reasons) < 4:
        reasons.append("hohes Volatilitaetsregime")
    if "verschlechter" in rs_l and len(reasons) < 4:
        reasons.append("RS-Dynamik verschlechtert sich")
    if not reasons:
        reasons.append("kein dominanter technischer Exit-Treiber")

    # Positionspuffer entscheidet, WIE ein vorhandener Exit-Druck behandelt wird.
    # Ein Gewinner mit Warnsignalen wird eher geschuetzt/teilrealisiert; ein
    # Verlierer mit denselben Warnsignalen wird nicht als 'Gewinnschutz' kaschiert.
    if stop_breached:
        level, ampel, action = "red", "🔴", "Exit prüfen"
    elif pressure >= 82:
        level, ampel = "red", "🔴"
        action = "Exit / deutlichen Risikoabbau prüfen"
    elif pressure >= 68:
        level, ampel = "orange", "🟠"
        if pnl_pct is not None and pnl_pct >= 6:
            action = "Teilgewinn / Risiko reduzieren"
        else:
            action = "Risiko reduzieren"
    elif target_reached:
        level, ampel, action = "orange", "🟠", "Teilgewinn / Zielmanagement"
    elif ((pnl_pct is not None and pnl_pct >= 10 and pressure >= 45)
          or (r_mult is not None and r_mult >= 2.0 and pressure >= 42)):
        level, ampel, action = "orange", "🟠", "Teilgewinn prüfen"
    elif pressure >= 48:
        level, ampel, action = "yellow", "🟡", "Stop enger / eng beobachten"
    elif ((r_mult is not None and r_mult >= 1.0 and not stop_protected)
          or (pnl_pct is not None and pnl_pct >= 6 and "hoch" in vol_l)):
        level, ampel, action = "yellow", "🟡", "Gewinnschutz nachziehen"
    else:
        level, ampel, action = "green", "🟢", "Halten / laufen lassen"

    # Verlierer mit bereits merklichem technischen Druck werden defensiver
    # behandelt als Gewinner mit gleichem Score.
    if pnl_pct is not None and pnl_pct < 0 and pressure >= 58 and not stop_breached:
        level, ampel, action = "orange", "🟠", "Risiko reduzieren / Exit vorbereiten"

    # Missing-data guard fuer Positionsmanagement: kein gruenes Sicherheitslabel
    # aus einem alten last_price oder fehlenden technischen Rohfeldern ableiten.
    if not has_current_live:
        level, ampel, action = "neutral", "⚪", "Aktuellen Scanwert prüfen"
    elif not available and not stop_breached and not target_reached:
        level, ampel, action = "neutral", "⚪", "Technische Exit-Daten prüfen"

    # Wenn die Fuehrungsaktion aus dem realen Positionspuffer entsteht, soll
    # die Begruendung das ebenfalls sichtbar machen und nicht nur sagen, dass
    # technisch kein Exit-Treiber dominiert.
    if reasons == ["kein dominanter technischer Exit-Treiber"]:
        if target_reached:
            reasons = ["Ziel/Teilziel der Position ist erreicht"]
        elif action == "Gewinnschutz nachziehen" and r_mult is not None:
            reasons = [f"Positionspuffer {r_mult:.2f}R erreicht; Gewinnschutz wird pruefbar"]
        elif "Teilgewinn" in action and pnl_pct is not None:
            reasons = [f"Gewinnpuffer {pnl_pct:+.1f}% bei gleichzeitigem Managementdruck"]

    stop_gap_pct = None
    stop_pnl_pct = None
    risk_to_stop = None
    locked_pnl = None
    if current is not None and stop is not None and current > 0 and stop > 0:
        stop_gap_pct = (current / stop - 1.0) * 100.0
        risk_to_stop = max(0.0, (current - stop) * shares) if shares else None
    if entry is not None and stop is not None and entry > 0 and stop > 0:
        stop_pnl_pct = (stop / entry - 1.0) * 100.0
        if shares and stop >= entry:
            locked_pnl = (stop - entry) * shares

    if not has_current_live:
        stop_status = "⚪ Kursbasis nicht aktuell"
        stop_plan = "Erst einen vollständigen aktuellen Scanwert herstellen; Stop nicht auf Basis eines alten last_price verändern."
    elif stop is None or stop <= 0:
        stop_status = "⚪ Stop fehlt"
        stop_plan = "Stop/Invalidierung ergänzen; ohne Stop keine belastbare Exit-Führung."
    elif stop_breached:
        stop_status = "🔴 Stop verletzt"
        stop_plan = "Stop-Regel ausführen bzw. Position/These sofort neu bewerten; Stop nicht nach unten verschieben."
    elif stop_protected:
        stop_status = "🟢 Gewinnstop aktiv"
        stop_plan = "Gewinnschutz nicht wieder lockern; weiteren Trail nur an neuer Struktur ausrichten."
    elif r_mult is not None and r_mult >= 1.0:
        stop_status = "🟡 Gewinnschutz prüfbar"
        stop_plan = "Break-even- oder Strukturstop prüfen; bestehenden Stop nicht unkontrolliert erweitern."
    elif pressure >= 48:
        stop_status = "🟡 Stop eng beobachten"
        stop_plan = "Bestehenden Stop/Invalidierung enger kontrollieren; nur an belastbarer Struktur nachziehen."
    else:
        stop_status = "⚪ Plan-Stop aktiv"
        stop_plan = "Plan-Stop beibehalten; nicht allein wegen normaler Schwankung zu früh nachziehen."

    if target_reached:
        profit_plan = "Ziel/Teilziel erreicht: Teilgewinn und Trailing der Restposition aktiv prüfen."
    elif pnl_pct is not None and pnl_pct >= 15 and pressure >= 45:
        profit_plan = f"Starker Gewinnpuffer ({pnl_pct:+.1f}%): Teilgewinn/Gewinnschutz priorisieren."
    elif pnl_pct is not None and pnl_pct >= 5:
        profit_plan = f"Gewinnpuffer {pnl_pct:+.1f}%: laufen lassen, aber Schutz nicht wieder lockern."
    elif pnl_pct is not None and pnl_pct < 0:
        profit_plan = f"Kein Gewinnpuffer ({pnl_pct:+.1f}%): Verlustbegrenzung statt Teilgewinn-Logik."
    else:
        profit_plan = "Noch kein relevanter Gewinnpuffer; Management folgt Stop und technischer Struktur."

    no_add = bool(
        pressure >= 45
        or "negativ" in market_l or "bear" in market_l
        or "hoch" in vol_l
        or "verschlechter" in rs_l
    )
    if no_add:
        add_plan = "Kein Aufstocken · erst Exit-Druck/RS/Marktumfeld stabilisieren."
    else:
        add_plan = "Aufstocken nur mit frischem Add-on-Trigger; kein blindes Nachkaufen."

    if pnl_pct is None:
        buffer_text = "n/a"
    else:
        buffer_text = f"{pnl_pct:+.1f}%"
        if r_mult is not None:
            buffer_text += f" · {r_mult:.2f}R"

    factor_rows = []
    for name, val in raw.items():
        factor_rows.append({"Faktor": name, "Wert": None if val is None else round(float(val), 1)})
    factor_rows.extend([
        {"Faktor": "Marktregime", "Wert": market},
        {"Faktor": "Volatilitaetsregime", "Wert": volatility},
        {"Faktor": "RS-Dynamik", "Wert": rs_dyn},
        {"Faktor": "Positions-P/L", "Wert": None if pnl_pct is None else round(float(pnl_pct), 2)},
        {"Faktor": "R-Multiple", "Wert": None if r_mult is None else round(float(r_mult), 2)},
    ])

    return {
        "ampel": ampel,
        "level": level,
        "score": round(pressure, 1),
        "action": action,
        "confidence": confidence,
        "why": reasons[:4],
        "why_text": " · ".join(reasons[:4]),
        "stop_status": stop_status,
        "stop_plan": stop_plan,
        "profit_plan": profit_plan,
        "add_plan": add_plan,
        "buffer_text": buffer_text,
        "market": market,
        "volatility": volatility,
        "rs_dynamics": rs_dyn,
        "pnl_pct": pnl_pct,
        "pnl_abs": pnl_abs,
        "r_multiple": r_mult,
        "current": current,
        "entry": entry,
        "stop": stop,
        "target": target,
        "reference_stop": reference_stop,
        "stop_gap_pct": stop_gap_pct,
        "stop_pnl_pct": stop_pnl_pct,
        "risk_to_stop": risk_to_stop,
        "locked_pnl": locked_pnl,
        "target_reached": target_reached,
        "stop_breached": stop_breached,
        "technical_inputs": len(available),
        "has_current_live": has_current_live,
        "factors": factor_rows,
    }

def _v244_positions_dataframe(positions, live_df=None, watchlist_name=""):
    rows = []
    live_map = {}
    try:
        if live_df is not None and not live_df.empty and "Ticker" in live_df.columns:
            for _, row in live_df.iterrows():
                live_map[str(row.get("Ticker") or "").strip().upper()] = row.to_dict()
    except Exception:
        live_map = {}
    for tk, pos in (positions or {}).items():
        ticker = str(tk or pos.get("ticker") or "").strip().upper()
        live_row = live_map.get(ticker, {})
        calc = _v244_calc_trade_state(pos, live_row)
        engine_v289 = _v289_position_exit_engine(pos, live_row)
        # v24.17: Management-Meilensteine dedupliziert protokollieren.
        trade_status = str(calc.get("Status") or "")
        milestone = None
        if "Stop / Invalidierung erreicht" in trade_status:
            milestone = "Stop / Invalidierung erreicht"
        elif "2R+ erreicht" in trade_status:
            milestone = "2R erreicht"
        elif "1R erreicht" in trade_status:
            milestone = "1R erreicht"
        elif "Ziel erreicht" in trade_status:
            milestone = "Ziel erreicht"
        if milestone:
            _event_logger(
                event_type=milestone,
                ticker=ticker,
                watchlist_name=watchlist_name,
                source="Positions-/Exit-Monitor",
                status=trade_status,
                price=calc.get("Aktueller Kurs"),
                trade_state=trade_status,
                details=str(calc.get("Aktion") or ""),
                payload={
                    "Entry": pos.get("entry"), "Stop": pos.get("stop"),
                    "Ziel": pos.get("target"), "Stück": pos.get("shares"),
                    "R-Multiple": calc.get("R-Multiple"), "P/L %": calc.get("P/L %"),
                },
                signature=f"{milestone}|{trade_status}",
            )
        # v28.9: Nur relevante defensive Engine-Zustaende als Event loggen.
        # Die Event-Schicht dedupliziert ueber die stabile Action/Score-Band-Signatur.
        if engine_v289.get("level") in {"orange", "red"}:
            try:
                _score_band_v289 = int(float(engine_v289.get("score") or 0) // 10) * 10
                _event_logger(
                    event_type="Exit Engine 2.0",
                    ticker=ticker,
                    watchlist_name=watchlist_name,
                    source="Positions-/Exit-Monitor",
                    status=str(engine_v289.get("action") or "-"),
                    price=engine_v289.get("current"),
                    trade_state=str(calc.get("Status") or "-"),
                    details=str(engine_v289.get("why_text") or "-"),
                    payload={
                        "Exit-Druck 2.0": engine_v289.get("score"),
                        "Aktion 2.0": engine_v289.get("action"),
                        "P/L %": engine_v289.get("pnl_pct"),
                        "R": engine_v289.get("r_multiple"),
                        "Markt": engine_v289.get("market"),
                        "Volatilitaet": engine_v289.get("volatility"),
                        "RS-Dynamik": engine_v289.get("rs_dynamics"),
                    },
                    signature=f"v289|{engine_v289.get('action')}|{_score_band_v289}",
                )
            except Exception:
                pass
        rows.append({
            "Exit-Ampel 2.0": engine_v289.get("ampel"),
            "Ticker": ticker,
            "Name": pos.get("name") or live_row.get("Name") or ticker,
            "Führung 2.0": engine_v289.get("action"),
            "Exit-Druck 2.0": f"{float(engine_v289.get('score') or 0):.0f}/100",
            "Gewinnpuffer": engine_v289.get("buffer_text"),
            "Stop-Status 2.0": engine_v289.get("stop_status"),
            "Konfidenz 2.0": engine_v289.get("confidence"),
            "Warum 2.0": engine_v289.get("why_text"),
            "Ampel": calc.get("Ampel"),
            "Aktueller Kurs": _v230_price_text(calc.get("Aktueller Kurs")),
            "Entry": _v230_price_text(pos.get("entry")),
            "Stop": _v230_price_text(pos.get("stop")),
            "Initial-Stop (R-Basis)": _v230_price_text(calc.get("R-Basis-Stop")),
            "Stück": int(_v230_safe_float(pos.get("shares"), default=0) or 0),
            "Initial-Stück": int(_v230_safe_float(pos.get("initial_shares"), default=pos.get("shares")) or 0),
            "Realisiert P/L": "n/a" if _v230_safe_float(pos.get("realized_pnl"), default=None) is None else f"{_v230_safe_float(pos.get('realized_pnl'), default=0.0):.2f}",
            "R": "n/a" if calc.get("R-Multiple") is None else f"{calc.get('R-Multiple'):.2f}R",
            "P/L": "n/a" if calc.get("P/L") is None else f"{calc.get('P/L'):.0f}",
            "P/L %": "n/a" if calc.get("P/L %") is None else f"{calc.get('P/L %'):.1f}%",
            "Trade-Status": calc.get("Status"),
            "Aktion": calc.get("Aktion"),
            "Stop-Hinweis": calc.get("Stop-Hinweis"),
            "Erfasst": pos.get("created_at") or "-",
        })
    return pd.DataFrame(rows)


# ---------- v24.17: Persistenter Signal-/Trade-Event-Log ----------
