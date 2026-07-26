"""Persistent trade journal helpers introduced in v27.0.

The module deliberately keeps trade-journal persistence separate from the live
screener. It records partial exits, full closes, stop adjustments and notes,
while open positions continue to be managed by ``position_monitor.py``.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

_BASE_DIR = Path(__file__).resolve().parent.parent
_time_provider = datetime.now
_safe_float = None
_event_logger = lambda **kwargs: False
_storage = None
_repository = None


def configure_context(*, base_dir=None, time_provider=None, safe_float=None, event_logger=None, storage=None, repository=None):
    global _BASE_DIR, _time_provider, _safe_float, _event_logger, _storage, _repository
    if base_dir is not None:
        _BASE_DIR = Path(base_dir)
    if time_provider is not None:
        _time_provider = time_provider
    if safe_float is not None:
        _safe_float = safe_float
    if event_logger is not None:
        _event_logger = event_logger
    if storage is not None:
        _storage = storage
    if repository is not None:
        _repository = repository


def _num(value: Any, default=None):
    if _safe_float is not None:
        try:
            return _safe_float(value, default=default)
        except Exception:
            pass
    try:
        out = float(value)
        if pd.isna(out):
            return default
        return out
    except Exception:
        return default


def _now() -> datetime:
    try:
        return _time_provider()
    except Exception:
        return datetime.now()


def _v270_trade_journal_path() -> Path:
    try:
        return _BASE_DIR / ".trade_journal_v270.json"
    except Exception:
        return Path("/tmp/.trade_journal_v270.json")


def _v270_load_trade_journal() -> dict:
    storage_store = None
    if _repository is not None:
        try:
            raw = _repository.load_store()
            if isinstance(raw, dict):
                storage_store = raw
        except Exception:
            storage_store = None
    elif _storage is not None:
        try:
            raw = _storage.load_namespace("trade_journal", default=None)
            if isinstance(raw, dict):
                storage_store = raw
        except Exception:
            storage_store = None
    path = _v270_trade_journal_path()
    file_store = {}
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                file_store = raw
    except Exception:
        file_store = {}
    session_store = {}
    try:
        raw = st.session_state.get("v270_trade_journal_store", {})
        if isinstance(raw, dict):
            session_store = raw
    except Exception:
        session_store = {}

    if storage_store is not None:
        store = storage_store
    else:
        store = session_store if session_store.get("entries") else file_store
    if not isinstance(store, dict):
        store = {}
    store.setdefault("entries", [])
    try:
        st.session_state.v270_trade_journal_store = store
    except Exception:
        pass
    return store


def _v270_save_trade_journal(store: dict) -> bool:
    store = store if isinstance(store, dict) else {"entries": []}
    store["entries"] = list(store.get("entries") or [])[-5000:]
    try:
        st.session_state.v270_trade_journal_store = store
    except Exception:
        pass
    storage_ok = False
    if _repository is not None:
        try:
            storage_ok = bool(_repository.save_store(store))
        except Exception:
            storage_ok = False
    elif _storage is not None:
        try:
            storage_ok = bool(_storage.save_namespace("trade_journal", store))
        except Exception:
            storage_ok = False
    file_ok = False
    try:
        _v270_trade_journal_path().write_text(
            json.dumps(store, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        file_ok = True
    except Exception:
        file_ok = False
    return bool(storage_ok or file_ok)


def _normalise_date(value=None) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    return text or _now().date().isoformat()


def _position_risk(position: dict):
    entry = _num(position.get("entry"), None)
    initial_stop = _num(position.get("initial_stop"), None)
    if initial_stop is None:
        initial_stop = _num(position.get("stop"), None)
    if entry is None or initial_stop is None or entry <= initial_stop:
        return entry, initial_stop, None
    return entry, initial_stop, entry - initial_stop


def _v270_record_journal_entry(
    *,
    watchlist_name: str,
    ticker: str,
    name: str = "",
    action_type: str,
    action_date=None,
    price=None,
    shares=None,
    remaining_shares=None,
    position: dict | None = None,
    realized_pnl=None,
    realized_pct=None,
    realized_r=None,
    total_realized_pnl=None,
    total_realized_r=None,
    old_stop=None,
    new_stop=None,
    note: str = "",
    learning: str = "",
    details: str = "",
) -> dict:
    position = position or {}
    entry_price, initial_stop, _ = _position_risk(position)
    record = {
        "ID": uuid4().hex,
        "Zeit": _now().strftime("%d.%m.%Y %H:%M:%S"),
        "Datum": _normalise_date(action_date),
        "Watchlist": str(watchlist_name or "Standard"),
        "Ticker": str(ticker or "").strip().upper(),
        "Name": str(name or position.get("name") or ticker or "").strip(),
        "Typ": str(action_type or "Journal-Eintrag"),
        "Kurs": _num(price, None),
        "Stück": int(_num(shares, 0) or 0),
        "Verbleibend": int(_num(remaining_shares, 0) or 0),
        "Entry": entry_price,
        "Initial-Stop": initial_stop,
        "Aktueller Stop": _num(position.get("stop"), None),
        "Alter Stop": _num(old_stop, None),
        "Neuer Stop": _num(new_stop, None),
        "Realisiert P/L": _num(realized_pnl, None),
        "Realisiert %": _num(realized_pct, None),
        "Realisiert R": _num(realized_r, None),
        "Gesamt P/L": _num(total_realized_pnl, None),
        "Gesamt R": _num(total_realized_r, None),
        "Notiz": str(note or "").strip(),
        "Erkenntnis": str(learning or "").strip(),
        "Details": str(details or "").strip(),
    }
    store = _v270_load_trade_journal()
    entries = list(store.get("entries") or [])
    entries.append(record)
    store["entries"] = entries[-5000:]
    _v270_save_trade_journal(store)
    return record


def _v270_partial_exit(
    positions: dict,
    *,
    watchlist_name: str,
    ticker: str,
    exit_price,
    exit_shares,
    exit_date=None,
    note: str = "",
    learning: str = "",
) -> dict:
    positions = dict(positions or {})
    ticker = str(ticker or "").strip().upper()
    pos = dict(positions.get(ticker) or {})
    if not pos:
        return {"ok": False, "error": "Offene Position nicht gefunden.", "positions": positions}
    current_shares = int(_num(pos.get("shares"), 0) or 0)
    sold = int(_num(exit_shares, 0) or 0)
    px = _num(exit_price, None)
    if px is None or px <= 0:
        return {"ok": False, "error": "Gültigen Verkaufskurs eingeben.", "positions": positions}
    if sold <= 0 or sold >= current_shares:
        return {"ok": False, "error": "Teilverkauf muss größer 0 und kleiner als die offene Stückzahl sein.", "positions": positions}

    entry, initial_stop, unit_risk = _position_risk(pos)
    pnl = (px - entry) * sold if entry is not None else None
    pnl_pct = (px / entry - 1.0) * 100.0 if entry else None
    r_mult = (px - entry) / unit_risk if unit_risk and entry is not None else None
    remaining = current_shares - sold
    previous_pnl = _num(pos.get("realized_pnl"), 0.0) or 0.0
    previous_weighted_r = _num(pos.get("realized_r_weighted"), 0.0) or 0.0
    total_pnl = previous_pnl + (pnl or 0.0)
    total_weighted_r = previous_weighted_r + ((r_mult or 0.0) * sold)
    initial_shares = int(_num(pos.get("initial_shares"), current_shares) or current_shares)

    pos.setdefault("initial_stop", initial_stop)
    pos.setdefault("initial_shares", initial_shares)
    pos["shares"] = remaining
    pos["realized_pnl"] = total_pnl
    pos["realized_shares"] = int(_num(pos.get("realized_shares"), 0) or 0) + sold
    pos["realized_r_weighted"] = total_weighted_r
    pos["last_exit_price"] = px
    pos["updated_at"] = _now().strftime("%d.%m.%Y %H:%M")
    positions[ticker] = pos

    total_r = total_weighted_r / initial_shares if initial_shares > 0 else None
    journal_entry = _v270_record_journal_entry(
        watchlist_name=watchlist_name,
        ticker=ticker,
        name=pos.get("name") or ticker,
        action_type="Teilverkauf",
        action_date=exit_date,
        price=px,
        shares=sold,
        remaining_shares=remaining,
        position=pos,
        realized_pnl=pnl,
        realized_pct=pnl_pct,
        realized_r=r_mult,
        total_realized_pnl=total_pnl,
        total_realized_r=total_r,
        note=note,
        learning=learning,
        details=f"{sold} Stück verkauft; {remaining} Stück verbleiben.",
    )
    _event_logger(
        event_type="Teilverkauf",
        ticker=ticker,
        watchlist_name=watchlist_name,
        source="Trade-Journal",
        status="Position reduziert",
        price=px,
        details=f"{sold} Stück verkauft; {remaining} verbleiben. Realisiert: {pnl if pnl is not None else 'n/a'}",
        payload={"Stück": sold, "Verbleibend": remaining, "Realisiert P/L": pnl, "Realisiert R": r_mult},
        signature=f"partial|{journal_entry['ID']}",
    )
    return {"ok": True, "positions": positions, "entry": journal_entry, "remaining": remaining}


def _v270_close_position(
    positions: dict,
    *,
    watchlist_name: str,
    ticker: str,
    exit_price,
    exit_date=None,
    reason: str = "Manuell geschlossen",
    note: str = "",
    learning: str = "",
) -> dict:
    positions = dict(positions or {})
    ticker = str(ticker or "").strip().upper()
    pos = dict(positions.get(ticker) or {})
    if not pos:
        return {"ok": False, "error": "Offene Position nicht gefunden.", "positions": positions}
    shares = int(_num(pos.get("shares"), 0) or 0)
    px = _num(exit_price, None)
    if px is None or px <= 0:
        return {"ok": False, "error": "Gültigen Ausstiegskurs eingeben.", "positions": positions}
    if shares <= 0:
        return {"ok": False, "error": "Die Position enthält keine offene Stückzahl.", "positions": positions}

    entry, initial_stop, unit_risk = _position_risk(pos)
    pnl = (px - entry) * shares if entry is not None else None
    pnl_pct = (px / entry - 1.0) * 100.0 if entry else None
    r_mult = (px - entry) / unit_risk if unit_risk and entry is not None else None
    previous_pnl = _num(pos.get("realized_pnl"), 0.0) or 0.0
    previous_weighted_r = _num(pos.get("realized_r_weighted"), 0.0) or 0.0
    initial_shares = int(_num(pos.get("initial_shares"), shares) or shares)
    total_pnl = previous_pnl + (pnl or 0.0)
    total_weighted_r = previous_weighted_r + ((r_mult or 0.0) * shares)
    total_r = total_weighted_r / initial_shares if initial_shares > 0 else None

    journal_entry = _v270_record_journal_entry(
        watchlist_name=watchlist_name,
        ticker=ticker,
        name=pos.get("name") or ticker,
        action_type="Position geschlossen",
        action_date=exit_date,
        price=px,
        shares=shares,
        remaining_shares=0,
        position=pos,
        realized_pnl=pnl,
        realized_pct=pnl_pct,
        realized_r=r_mult,
        total_realized_pnl=total_pnl,
        total_realized_r=total_r,
        note=note,
        learning=learning,
        details=str(reason or "Manuell geschlossen"),
    )
    positions.pop(ticker, None)
    _event_logger(
        event_type="Position geschlossen",
        ticker=ticker,
        watchlist_name=watchlist_name,
        source="Trade-Journal",
        status=str(reason or "Position geschlossen"),
        price=px,
        trade_state="Geschlossen",
        details=f"{shares} Stück geschlossen · Gesamt P/L {total_pnl:.2f} · Gesamt R {total_r if total_r is not None else 'n/a'}",
        payload={"Stück": shares, "Gesamt P/L": total_pnl, "Gesamt R": total_r, "Grund": reason},
        signature=f"closed|{journal_entry['ID']}",
    )
    return {"ok": True, "positions": positions, "entry": journal_entry}


def _v270_adjust_stop(
    positions: dict,
    *,
    watchlist_name: str,
    ticker: str,
    new_stop,
    action_date=None,
    note: str = "",
) -> dict:
    positions = dict(positions or {})
    ticker = str(ticker or "").strip().upper()
    pos = dict(positions.get(ticker) or {})
    if not pos:
        return {"ok": False, "error": "Offene Position nicht gefunden.", "positions": positions}
    old_stop = _num(pos.get("stop"), None)
    new_stop_value = _num(new_stop, None)
    if new_stop_value is None or new_stop_value <= 0:
        return {"ok": False, "error": "Gültigen neuen Stop eingeben.", "positions": positions}
    if old_stop is not None and abs(new_stop_value - old_stop) < 1e-12:
        return {"ok": False, "error": "Der neue Stop entspricht dem bisherigen Stop.", "positions": positions}
    pos.setdefault("initial_stop", old_stop)
    history = list(pos.get("stop_history") or [])
    history.append({
        "date": _normalise_date(action_date),
        "old_stop": old_stop,
        "new_stop": new_stop_value,
        "note": str(note or "").strip(),
    })
    pos["stop_history"] = history[-100:]
    pos["stop"] = new_stop_value
    pos["updated_at"] = _now().strftime("%d.%m.%Y %H:%M")
    positions[ticker] = pos
    journal_entry = _v270_record_journal_entry(
        watchlist_name=watchlist_name,
        ticker=ticker,
        name=pos.get("name") or ticker,
        action_type="Stop angepasst",
        action_date=action_date,
        position=pos,
        remaining_shares=pos.get("shares"),
        old_stop=old_stop,
        new_stop=new_stop_value,
        note=note,
        details=f"Stop von {old_stop if old_stop is not None else 'n/a'} auf {new_stop_value} angepasst.",
    )
    _event_logger(
        event_type="Stop angepasst",
        ticker=ticker,
        watchlist_name=watchlist_name,
        source="Trade-Journal",
        status="Stop aktualisiert",
        details=journal_entry["Details"],
        payload={"Alter Stop": old_stop, "Neuer Stop": new_stop_value},
        signature=f"stop|{journal_entry['ID']}",
    )
    return {"ok": True, "positions": positions, "entry": journal_entry}


def _v270_save_trade_note(
    positions: dict,
    *,
    watchlist_name: str,
    ticker: str,
    note: str,
    learning: str = "",
    action_date=None,
) -> dict:
    positions = dict(positions or {})
    ticker = str(ticker or "").strip().upper()
    pos = dict(positions.get(ticker) or {})
    if not pos:
        return {"ok": False, "error": "Offene Position nicht gefunden.", "positions": positions}
    if not str(note or "").strip() and not str(learning or "").strip():
        return {"ok": False, "error": "Notiz oder Erkenntnis eingeben.", "positions": positions}
    notes = list(pos.get("journal_notes") or [])
    notes.append({
        "date": _normalise_date(action_date),
        "note": str(note or "").strip(),
        "learning": str(learning or "").strip(),
    })
    pos["journal_notes"] = notes[-100:]
    pos["updated_at"] = _now().strftime("%d.%m.%Y %H:%M")
    positions[ticker] = pos
    journal_entry = _v270_record_journal_entry(
        watchlist_name=watchlist_name,
        ticker=ticker,
        name=pos.get("name") or ticker,
        action_type="Trade-Notiz",
        action_date=action_date,
        position=pos,
        remaining_shares=pos.get("shares"),
        note=note,
        learning=learning,
        details="Notiz zur offenen Position gespeichert.",
    )
    return {"ok": True, "positions": positions, "entry": journal_entry}


def _v270_journal_entries_dataframe(watchlist_name=None) -> pd.DataFrame:
    store = _v270_load_trade_journal()
    df = pd.DataFrame(store.get("entries") or [])
    if df.empty:
        return df
    if watchlist_name:
        df = df[df["Watchlist"].astype(str) == str(watchlist_name)]
    return df.iloc[::-1].reset_index(drop=True)


def _v270_journal_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "closed_trades": 0,
            "partial_exits": 0,
            "realized_pnl": 0.0,
            "win_rate": None,
            "avg_r": None,
        }
    work = df.copy()
    exit_rows = work[work["Typ"].isin(["Teilverkauf", "Position geschlossen"])].copy()
    realized_pnl = pd.to_numeric(exit_rows.get("Realisiert P/L"), errors="coerce").fillna(0).sum() if not exit_rows.empty else 0.0
    closed = work[work["Typ"] == "Position geschlossen"].copy()
    total_pnl = pd.to_numeric(closed.get("Gesamt P/L"), errors="coerce") if not closed.empty else pd.Series(dtype=float)
    total_r = pd.to_numeric(closed.get("Gesamt R"), errors="coerce") if not closed.empty else pd.Series(dtype=float)
    valid_pnl = total_pnl.dropna()
    win_rate = float((valid_pnl > 0).mean() * 100.0) if len(valid_pnl) else None
    avg_r = float(total_r.dropna().mean()) if len(total_r.dropna()) else None
    return {
        "closed_trades": int(len(closed)),
        "partial_exits": int((work["Typ"] == "Teilverkauf").sum()),
        "realized_pnl": float(realized_pnl),
        "win_rate": win_rate,
        "avg_r": avg_r,
    }


def _v270_reset_trade_journal(watchlist_name=None) -> None:
    store = _v270_load_trade_journal()
    if not watchlist_name:
        store = {"entries": []}
    else:
        wl = str(watchlist_name)
        store["entries"] = [e for e in (store.get("entries") or []) if str(e.get("Watchlist")) != wl]
    _v270_save_trade_journal(store)
