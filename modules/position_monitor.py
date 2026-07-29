"""Persistent position and exit-monitor helpers extracted in v25.0."""
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
        rows.append({
            "Ampel": calc.get("Ampel"),
            "Ticker": ticker,
            "Name": pos.get("name") or live_row.get("Name") or ticker,
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
