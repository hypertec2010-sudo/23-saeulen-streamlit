"""Persistent trade journal helpers introduced in v27.0.

The module deliberately keeps trade-journal persistence separate from the live
screener. It records partial exits, full closes, stop adjustments and notes,
while open positions continue to be managed by ``position_monitor.py``.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from copy import deepcopy
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


def _infer_initial_stop(position: dict, entry=None):
    position = position or {}
    if entry is None:
        entry = _num(position.get("entry"), None)
    if entry is None:
        return None

    candidates = [_num(position.get("initial_stop"), None)]
    try:
        for item in list(position.get("stop_history") or []):
            if not isinstance(item, dict):
                continue
            candidates.append(_num(item.get("old_stop"), None))
            candidates.append(_num(item.get("new_stop"), None))
    except Exception:
        pass
    candidates.append(_num(position.get("stop"), None))

    for candidate in candidates:
        if candidate is not None and candidate > 0 and candidate < entry:
            return candidate
    return None


def _position_risk(position: dict):
    entry = _num(position.get("entry"), None)
    initial_stop = _infer_initial_stop(position, entry=entry)
    if entry is None or initial_stop is None:
        return entry, initial_stop, None
    return entry, initial_stop, entry - initial_stop


def _v290_entry_context_columns(position: dict) -> dict:
    """Flatten the v29.0 entry-context snapshot into journal columns.

    Legacy positions simply return empty values. No missing historical context is
    synthesized, which keeps later learning statistics honest.
    """
    position = position or {}
    ctx = position.get("entry_context")
    if not isinstance(ctx, dict):
        ctx = {}
    return {
        "Entry Kontext-Zeit": str(ctx.get("captured_at") or "").strip(),
        "Entry Status": str(ctx.get("status") or "").strip(),
        "Entry Live-Ampel": str(ctx.get("live_ampel") or "").strip(),
        "Entry Shadow-Ampel": str(ctx.get("shadow_ampel") or "").strip(),
        "Entry Live-Score": _num(ctx.get("live_score"), None),
        "Entry Engine-Score": _num(ctx.get("engine_score"), None),
        "Entry Guarded Score": _num(ctx.get("guarded_score"), None),
        "Entry Engine-Empfehlung": str(ctx.get("engine_recommendation") or "").strip(),
        "Entry Guardrail": str(ctx.get("guardrail") or "").strip(),
        "Entry Kontext-Anpassung": _num(ctx.get("context_adjustment"), None),
        "Entry Kontext-Verlässlichkeit": str(ctx.get("context_confidence") or "").strip(),
        "Entry Marktregime": str(ctx.get("market_regime") or "").strip(),
        "Entry Volatilitätsregime": str(ctx.get("volatility_regime") or "").strip(),
        "Entry RS-Dynamik": str(ctx.get("rs_dynamics") or "").strip(),
        "Entry Relative Stärke": str(ctx.get("relative_strength") or "").strip(),
        "Entry Radar-Bucket": str(ctx.get("radar_bucket") or "").strip(),
        "Entry Grade": str(ctx.get("grade") or "").strip(),
        "Entry CRV": _num(ctx.get("crv"), None),
        "Entry Abstand": str(ctx.get("entry_distance") or "").strip(),
        "Entry Setup-Alert": str(ctx.get("setup_alert") or "").strip(),
        "Entry Gates": str(ctx.get("active_gates") or "").strip(),
        "Entry Benchmark": str(ctx.get("benchmark") or "").strip(),
        "Entry Horizont": str(ctx.get("live_horizon") or "").strip(),
    }


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
    position_snapshot: dict | None = None,
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
        # v28.7a: exact pre-close snapshot for lossless undo of future closes.
        "Position vorher": deepcopy(position_snapshot) if isinstance(position_snapshot, dict) else None,
    }
    # v29.0: make entry context self-contained in every journal row. This lets
    # CSV exports and the Learning Engine analyze trades without reaching back
    # into live screener state. Legacy rows remain blank rather than guessed.
    record.update(_v290_entry_context_columns(position))
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
        position_snapshot=pos,
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



def _v287a_same_trade_entry(entry: dict, *, watchlist_name: str, ticker: str) -> bool:
    if not isinstance(entry, dict):
        return False
    return (
        str(entry.get("Watchlist") or "Standard") == str(watchlist_name or "Standard")
        and str(entry.get("Ticker") or "").strip().upper() == str(ticker or "").strip().upper()
    )


def _v287a_legacy_cycle_entries(entries: list[dict], close_index: int, *, watchlist_name: str, ticker: str) -> list[dict]:
    """Entries of the current trade cycle preceding a legacy close record."""
    start = 0
    for idx in range(close_index - 1, -1, -1):
        item = entries[idx]
        if not _v287a_same_trade_entry(item, watchlist_name=watchlist_name, ticker=ticker):
            continue
        if str(item.get("Typ") or "") == "Position geschlossen":
            start = idx + 1
            break
    return [
        dict(item)
        for item in entries[start:close_index]
        if _v287a_same_trade_entry(item, watchlist_name=watchlist_name, ticker=ticker)
    ]


def _v287a_restore_position_from_close_entry(
    entry: dict,
    *,
    prior_entries: list[dict] | None = None,
    fallback_position: dict | None = None,
) -> tuple[dict, bool]:
    """Restore a position from a close journal row.

    v28.7a+ close rows carry an exact ``Position vorher`` snapshot. Older rows
    did not, so the legacy path rebuilds the position from the journal plus an
    optional position/event fallback supplied by the UI.
    """
    entry = dict(entry or {})
    snapshot = entry.get("Position vorher")
    if isinstance(snapshot, dict) and snapshot:
        pos = deepcopy(snapshot)
        ticker = str(entry.get("Ticker") or pos.get("ticker") or "").strip().upper()
        pos["ticker"] = ticker
        pos.setdefault("name", str(entry.get("Name") or ticker).strip())
        pos["updated_at"] = _now().strftime("%d.%m.%Y %H:%M")
        return pos, True

    fallback = deepcopy(fallback_position) if isinstance(fallback_position, dict) else {}
    ticker = str(entry.get("Ticker") or fallback.get("ticker") or "").strip().upper()
    name = str(entry.get("Name") or fallback.get("name") or ticker).strip()
    prior_entries = list(prior_entries or [])

    partial_rows = [x for x in prior_entries if str(x.get("Typ") or "") == "Teilverkauf"]
    stop_rows = [x for x in prior_entries if str(x.get("Typ") or "") == "Stop angepasst"]
    note_rows = [x for x in prior_entries if str(x.get("Typ") or "") == "Trade-Notiz"]

    open_shares = int(_num(entry.get("Stück"), 0) or 0)
    realized_shares = sum(max(0, int(_num(x.get("Stück"), 0) or 0)) for x in partial_rows)
    initial_shares = open_shares + realized_shares
    fallback_initial_shares = int(_num(fallback.get("initial_shares"), 0) or 0)
    if fallback_initial_shares > initial_shares:
        initial_shares = fallback_initial_shares

    previous_pnl = sum((_num(x.get("Realisiert P/L"), 0.0) or 0.0) for x in partial_rows)
    total_pnl = _num(entry.get("Gesamt P/L"), None)
    close_pnl = _num(entry.get("Realisiert P/L"), None)
    if total_pnl is not None and close_pnl is not None:
        previous_pnl = total_pnl - close_pnl

    previous_weighted_r = 0.0
    for row in partial_rows:
        row_r = _num(row.get("Realisiert R"), None)
        row_shares = int(_num(row.get("Stück"), 0) or 0)
        if row_r is not None and row_shares > 0:
            previous_weighted_r += row_r * row_shares

    stop_history = list(fallback.get("stop_history") or [])
    if not stop_history:
        for row in stop_rows:
            stop_history.append({
                "date": _normalise_date(row.get("Datum")),
                "old_stop": _num(row.get("Alter Stop"), None),
                "new_stop": _num(row.get("Neuer Stop"), None),
                "note": str(row.get("Notiz") or "").strip(),
            })

    journal_notes = list(fallback.get("journal_notes") or [])
    if not journal_notes:
        for row in note_rows:
            journal_notes.append({
                "date": _normalise_date(row.get("Datum")),
                "note": str(row.get("Notiz") or "").strip(),
                "learning": str(row.get("Erkenntnis") or "").strip(),
            })

    entry_price = _num(entry.get("Entry"), _num(fallback.get("entry"), None))
    initial_stop = _num(entry.get("Initial-Stop"), _num(fallback.get("initial_stop"), None))
    current_stop = _num(entry.get("Aktueller Stop"), _num(fallback.get("stop"), initial_stop))
    target = _num(fallback.get("target"), 0.0) or 0.0
    last_price = _num(fallback.get("last_price"), None)

    pos = dict(fallback)
    pos.update({
        "ticker": ticker,
        "name": name,
        "entry": entry_price or 0.0,
        "stop": current_stop or 0.0,
        "initial_stop": initial_stop,
        "target": target,
        "shares": open_shares,
        "initial_shares": initial_shares or open_shares,
        "realized_pnl": previous_pnl,
        "realized_shares": realized_shares,
        "realized_r_weighted": previous_weighted_r,
        "stop_history": stop_history[-100:],
        "journal_notes": journal_notes[-100:],
        "created_at": fallback.get("created_at") or f"Wiederhergestellt aus Journal {entry.get('Datum') or ''}".strip(),
        "updated_at": _now().strftime("%d.%m.%Y %H:%M"),
        "last_price": last_price,
    })
    return pos, False


def _v287a_undo_close_position(
    positions: dict,
    *,
    watchlist_name: str,
    journal_id: str,
    fallback_position: dict | None = None,
) -> dict:
    """Undo one full close without creating a synthetic counter-trade.

    The erroneous close is converted into an audit-only journal row so it no
    longer contributes to realized P/L, hit rate or closed-trade statistics.
    """
    positions = dict(positions or {})
    journal_id = str(journal_id or "").strip()
    if not journal_id:
        return {"ok": False, "error": "Journal-ID fehlt.", "positions": positions}

    store = _v270_load_trade_journal()
    entries = list(store.get("entries") or [])
    close_index = None
    close_entry = None
    for idx, item in enumerate(entries):
        if str((item or {}).get("ID") or "") == journal_id:
            close_index = idx
            close_entry = dict(item or {})
            break
    if close_entry is None or close_index is None:
        return {"ok": False, "error": "Geschlossene Position im Journal nicht gefunden.", "positions": positions}
    if str(close_entry.get("Typ") or "") != "Position geschlossen":
        return {"ok": False, "error": "Dieser Journal-Eintrag ist keine aktive Schließung mehr.", "positions": positions}

    ticker = str(close_entry.get("Ticker") or "").strip().upper()
    row_watchlist = str(close_entry.get("Watchlist") or "Standard")
    if row_watchlist != str(watchlist_name or "Standard"):
        return {"ok": False, "error": "Journal-Eintrag gehört zu einer anderen Watchlist.", "positions": positions}
    if not ticker:
        return {"ok": False, "error": "Ticker im Journal-Eintrag fehlt.", "positions": positions}
    if ticker in positions:
        return {"ok": False, "error": f"{ticker} ist bereits als offene Position vorhanden.", "positions": positions}

    cycle_entries = _v287a_legacy_cycle_entries(
        entries,
        close_index,
        watchlist_name=row_watchlist,
        ticker=ticker,
    )
    restored, exact_snapshot = _v287a_restore_position_from_close_entry(
        close_entry,
        prior_entries=cycle_entries,
        fallback_position=fallback_position,
    )
    if int(_num(restored.get("shares"), 0) or 0) <= 0:
        return {"ok": False, "error": "Offene Stückzahl konnte nicht wiederhergestellt werden.", "positions": positions}
    if (_num(restored.get("entry"), 0.0) or 0.0) <= 0:
        return {"ok": False, "error": "Entry konnte nicht wiederhergestellt werden.", "positions": positions}

    wrong_price = _num(close_entry.get("Kurs"), None)
    original_details = str(close_entry.get("Details") or "").strip()
    audit_entry = dict(close_entry)
    audit_entry["Ursprünglicher Typ"] = "Position geschlossen"
    audit_entry["Ursprünglicher Kurs"] = wrong_price
    audit_entry["Ursprünglich Realisiert P/L"] = close_entry.get("Realisiert P/L")
    audit_entry["Ursprünglich Realisiert %"] = close_entry.get("Realisiert %")
    audit_entry["Ursprünglich Realisiert R"] = close_entry.get("Realisiert R")
    audit_entry["Ursprünglich Gesamt P/L"] = close_entry.get("Gesamt P/L")
    audit_entry["Ursprünglich Gesamt R"] = close_entry.get("Gesamt R")
    audit_entry["Typ"] = "Schließung rückgängig"
    audit_entry["Kurs"] = None
    audit_entry["Verbleibend"] = int(_num(restored.get("shares"), 0) or 0)
    audit_entry["Realisiert P/L"] = None
    audit_entry["Realisiert %"] = None
    audit_entry["Realisiert R"] = None
    audit_entry["Gesamt P/L"] = None
    audit_entry["Gesamt R"] = None
    audit_entry["Rückgängig am"] = _now().strftime("%d.%m.%Y %H:%M:%S")
    audit_entry["Details"] = (
        f"Schließung rückgängig gemacht; ursprünglicher Exit {wrong_price if wrong_price is not None else 'n/a'}."
        + (f" Ursprünglicher Grund: {original_details}" if original_details else "")
    )
    entries[close_index] = audit_entry
    store["entries"] = entries[-5000:]
    if not _v270_save_trade_journal(store):
        return {"ok": False, "error": "Trade-Journal konnte nicht aktualisiert werden.", "positions": positions}

    positions[ticker] = restored
    _event_logger(
        event_type="Schließung rückgängig",
        ticker=ticker,
        watchlist_name=watchlist_name,
        source="Trade-Journal",
        status="Position wieder offen",
        price=_num(restored.get("last_price"), _num(restored.get("entry"), None)),
        trade_state="Offen",
        details=f"Versehentliche Schließung rückgängig; {restored.get('shares')} Stück wieder offen.",
        payload={
            "Journal-ID": journal_id,
            "Ursprünglicher Exit": wrong_price,
            "Wiederherstellung": "Exakter Snapshot" if exact_snapshot else "Legacy-Rekonstruktion",
        },
        signature=f"undo-close|{journal_id}",
    )
    return {
        "ok": True,
        "positions": positions,
        "restored_position": restored,
        "journal_entry": audit_entry,
        "exact_snapshot": exact_snapshot,
    }

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
    entry_value = _num(pos.get("entry"), None)
    initial_stop_value = _infer_initial_stop(pos, entry=entry_value)
    if initial_stop_value is None and old_stop is not None and entry_value is not None and old_stop < entry_value:
        initial_stop_value = old_stop
    if initial_stop_value is not None:
        pos["initial_stop"] = initial_stop_value
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
