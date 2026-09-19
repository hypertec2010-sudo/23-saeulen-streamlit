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
        "Realisiert P/L-Währung": str(position.get("price_currency") or "").strip().upper(),
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
            "pnl_by_currency": {},
            "pnl_known_rows": 0,
            "pnl_total_rows": 0,
            "pnl_coverage_pct": 0.0,
        }
    work = df.copy()
    exit_rows = work[work["Typ"].isin(["Teilverkauf", "Position geschlossen"])].copy()
    # Legacy/native P/L remains available for sign/R analytics, but must never be
    # presented as one monetary total across currencies. Build a currency-aware
    # monetary view instead. Prefer the broker's Result when it belongs wholly to
    # the screener execution; for mixed Pie+screener sells use the screener-only
    # native calculation and its price currency.
    pnl_by_currency: dict[str, float] = {}
    known_rows = 0
    for _, row in exit_rows.iterrows():
        details = str(row.get("Details") or "")
        broker_val = pd.to_numeric(pd.Series([row.get("Broker Result")]), errors="coerce").iloc[0]
        def _clean_ccy_v318d(value: Any) -> str:
            # pandas missing values become float NaN; str(np.nan) == "nan" and must
            # never be treated as a real currency code. Keep only plausible ISO-like
            # three-letter currency codes.
            try:
                if pd.isna(value):
                    return ""
            except Exception:
                pass
            text = str(value or "").strip().upper()
            if text in {"", "NAN", "NONE", "NULL", "<NA>"}:
                return ""
            return text if len(text) == 3 and text.isalpha() else ""

        broker_ccy = _clean_ccy_v318d(row.get("Broker Result-Währung"))
        native_val = pd.to_numeric(pd.Series([row.get("Realisiert P/L")]), errors="coerce").iloc[0]
        native_ccy = _clean_ccy_v318d(row.get("Realisiert P/L-Währung")) or _clean_ccy_v318d(row.get("Broker Preis-Währung"))
        amount = None
        ccy = ""
        if pd.notna(broker_val) and broker_ccy and "GEMISCHT" not in details.upper():
            amount, ccy = float(broker_val), broker_ccy
        elif pd.notna(native_val) and native_ccy:
            amount, ccy = float(native_val), native_ccy
        if amount is not None and ccy:
            pnl_by_currency[ccy] = float(pnl_by_currency.get(ccy, 0.0) + amount)
            known_rows += 1

    realized_pnl = pd.to_numeric(exit_rows.get("Realisiert P/L"), errors="coerce").fillna(0).sum() if not exit_rows.empty else 0.0
    closed = work[work["Typ"] == "Position geschlossen"].copy()
    total_pnl = pd.to_numeric(closed.get("Gesamt P/L"), errors="coerce") if not closed.empty else pd.Series(dtype=float)
    total_r = pd.to_numeric(closed.get("Gesamt R"), errors="coerce") if not closed.empty else pd.Series(dtype=float)
    valid_pnl = total_pnl.dropna()
    win_rate = float((valid_pnl > 0).mean() * 100.0) if len(valid_pnl) else None
    avg_r = float(total_r.dropna().mean()) if len(total_r.dropna()) else None
    total_rows = int(len(exit_rows))
    return {
        "closed_trades": int(len(closed)),
        "partial_exits": int((work["Typ"] == "Teilverkauf").sum()),
        "realized_pnl": float(realized_pnl),
        "win_rate": win_rate,
        "avg_r": avg_r,
        "pnl_by_currency": pnl_by_currency,
        "pnl_known_rows": int(known_rows),
        "pnl_total_rows": total_rows,
        "pnl_coverage_pct": float(known_rows / total_rows * 100.0) if total_rows else 0.0,
    }



def _broker_ticker_key(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    # Common broker exports omit exchange suffixes used by the screener (e.g. SAP.DE -> SAP).
    return text.split(".", 1)[0]


def _v3018_currency_backfill_preview(watchlist_name: str, normalized_broker: pd.DataFrame) -> dict:
    """Build a conservative, read-only mapping from old journal exits to broker sells.

    Exact Broker Import IDs win. Legacy journal rows without an ID are matched only when
    ticker (exchange suffix tolerant), execution price and screener quantity identify one
    unique broker SELL. A broker row that would map to more than one journal row is marked
    as a conflict and is never auto-applied. No positions or P/L amounts are changed.
    """
    broker = normalized_broker.copy() if isinstance(normalized_broker, pd.DataFrame) else pd.DataFrame()
    store = _v270_load_trade_journal()
    entries = list(store.get("entries") or [])
    wl = str(watchlist_name or "Standard")
    journal_rows = []
    for idx, raw in enumerate(entries):
        e = dict(raw or {})
        if str(e.get("Watchlist") or "") != wl:
            continue
        if str(e.get("Typ") or "") not in {"Teilverkauf", "Position geschlossen"}:
            continue
        journal_rows.append((idx, e))

    if broker.empty or not journal_rows:
        return {"table": pd.DataFrame(), "updates": [], "summary": {"safe": 0, "conflicts": 0, "unmatched": len(journal_rows), "already_complete": 0}}

    work = broker.copy()
    if "Action-Typ" in work.columns:
        work = work[work["Action-Typ"].astype(str).eq("SELL")].copy()
    if "Import-Status" in work.columns:
        work = work[work["Import-Status"].astype(str).eq("OK")].copy()
    if work.empty:
        return {"table": pd.DataFrame(), "updates": [], "summary": {"safe": 0, "conflicts": 0, "unmatched": len(journal_rows), "already_complete": 0}}

    records = []
    for bidx, ser in work.iterrows():
        r = ser.to_dict()
        r["__idx"] = bidx
        r["__ticker_key"] = _broker_ticker_key(r.get("Ticker"))
        records.append(r)
    by_import_id = {str(r.get("Import-ID") or ""): r for r in records if str(r.get("Import-ID") or "")}

    proposals = []
    already_complete = 0
    for store_idx, e in journal_rows:
        existing_ccy = str(e.get("Broker Result-Währung") or e.get("Realisiert P/L-Währung") or e.get("Broker Preis-Währung") or "").strip()
        existing_result = _num(e.get("Broker Result"), None)
        if existing_ccy and (existing_result is not None or str(e.get("Broker Import ID") or "").strip()):
            already_complete += 1
            proposals.append({"store_idx": store_idx, "entry": e, "status": "BEREITS VOLLSTÄNDIG", "match": None, "match_type": ""})
            continue

        direct_id = str(e.get("Broker Import ID") or "").strip()
        candidates = []
        match_type = ""
        if direct_id and direct_id in by_import_id:
            candidates = [by_import_id[direct_id]]
            match_type = "Broker-ID"
        else:
            tk = _broker_ticker_key(e.get("Ticker"))
            qty = _num(e.get("Stück"), None)
            price = _num(e.get("Kurs"), None)
            if tk and qty is not None and price is not None:
                tol_price = max(0.01, abs(float(price)) * 1e-5)
                exact_qty = []
                larger_qty = []
                for r in records:
                    if r.get("__ticker_key") != tk:
                        continue
                    bqty = _num(r.get("Stück"), None)
                    bprice = _num(r.get("Preis/Aktie"), None)
                    if bqty is None or bprice is None or abs(float(bprice) - float(price)) > tol_price:
                        continue
                    if abs(float(bqty) - float(qty)) <= 1e-6:
                        exact_qty.append(r)
                    elif float(bqty) > float(qty) + 1e-6:
                        larger_qty.append(r)
                if len(exact_qty) == 1:
                    candidates = exact_qty
                    match_type = "Ticker+Preis+Stück"
                elif len(exact_qty) == 0 and len(larger_qty) == 1:
                    candidates = larger_qty
                    match_type = "GEMISCHT · Ticker+Preis"
                elif len(exact_qty) > 1:
                    candidates = exact_qty
                elif len(larger_qty) > 1:
                    candidates = larger_qty

        if len(candidates) == 1:
            proposals.append({"store_idx": store_idx, "entry": e, "status": "VORGESCHLAGEN", "match": candidates[0], "match_type": match_type})
        elif len(candidates) > 1:
            proposals.append({"store_idx": store_idx, "entry": e, "status": "MEHRDEUTIG", "match": None, "match_type": ""})
        else:
            proposals.append({"store_idx": store_idx, "entry": e, "status": "NICHT GEFUNDEN", "match": None, "match_type": ""})

    # One broker execution must never enrich two journal exits. This also exposes duplicate journal closes.
    usage = {}
    for p in proposals:
        r = p.get("match")
        if r is None:
            continue
        bid = str(r.get("Import-ID") or r.get("Broker-ID") or r.get("__idx"))
        usage.setdefault(bid, []).append(p)
    for bid, items in usage.items():
        if len(items) > 1:
            for p in items:
                p["status"] = "KONFLIKT · Brokerzeile mehrfach"

    table_rows = []
    updates = []
    for p in proposals:
        e = p["entry"]
        r = p.get("match") or {}
        status = p["status"]
        mixed = str(p.get("match_type") or "").startswith("GEMISCHT")
        safe = status == "VORGESCHLAGEN"
        if safe:
            updates.append({"store_idx": p["store_idx"], "match": r, "mixed": mixed, "match_type": p.get("match_type") or ""})
        table_rows.append({
            "Journal-ID": str(e.get("ID") or ""),
            "Datum": str(e.get("Datum") or ""),
            "Ticker": str(e.get("Ticker") or ""),
            "Typ": str(e.get("Typ") or ""),
            "Journal Stück": _num(e.get("Stück"), None),
            "Journal Kurs": _num(e.get("Kurs"), None),
            "Status": status,
            "Zuordnung": p.get("match_type") or "",
            "Broker Zeit": str(r.get("Zeit Berlin") or ""),
            "Broker Action": str(r.get("Action") or ""),
            "Broker Stück": _num(r.get("Stück"), None),
            "Broker Kurs": _num(r.get("Preis/Aktie"), None),
            "Preis-Währung": str(r.get("Preis-Währung") or ""),
            "Broker Result": _num(r.get("Result"), None),
            "Result-Währung": str(r.get("Result-Währung") or ""),
        })

    conflicts = sum(1 for p in proposals if str(p.get("status") or "").startswith("KONFLIKT") or p.get("status") == "MEHRDEUTIG")
    unmatched = sum(1 for p in proposals if p.get("status") == "NICHT GEFUNDEN")
    return {
        "table": pd.DataFrame(table_rows),
        "updates": updates,
        "summary": {"safe": len(updates), "conflicts": conflicts, "unmatched": unmatched, "already_complete": already_complete},
    }


def _v3018_apply_currency_backfill(watchlist_name: str, normalized_broker: pd.DataFrame) -> dict:
    """Persist only missing broker/currency metadata for conservatively matched journal exits."""
    preview = _v3018_currency_backfill_preview(watchlist_name, normalized_broker)
    updates = list(preview.get("updates") or [])
    if not updates:
        return {"ok": True, "updated": 0, "preview": preview}
    store = _v270_load_trade_journal()
    entries = list(store.get("entries") or [])
    changed = 0
    stamp = _now().strftime("%d.%m.%Y %H:%M:%S")
    for item in updates:
        idx = int(item.get("store_idx"))
        if idx < 0 or idx >= len(entries):
            continue
        e = dict(entries[idx] or {})
        r = dict(item.get("match") or {})
        if not str(e.get("Broker Import ID") or "").strip():
            e["Broker Import ID"] = str(r.get("Import-ID") or "")
        if not str(e.get("Broker Quelle") or "").strip():
            e["Broker Quelle"] = "Depot-Excel · Backfill"
        if not str(e.get("Broker Preis-Währung") or "").strip():
            e["Broker Preis-Währung"] = str(r.get("Preis-Währung") or "").strip().upper()
        if not str(e.get("Realisiert P/L-Währung") or "").strip():
            e["Realisiert P/L-Währung"] = str(r.get("Preis-Währung") or "").strip().upper()
        if e.get("Broker Result") in (None, "") and _num(r.get("Result"), None) is not None:
            e["Broker Result"] = float(_num(r.get("Result"), 0.0) or 0.0)
        if not str(e.get("Broker Result-Währung") or "").strip():
            e["Broker Result-Währung"] = str(r.get("Result-Währung") or "").strip().upper()
        e["Broker Backfill am"] = stamp
        e["Broker Backfill-Zuordnung"] = str(item.get("match_type") or "")
        if item.get("mixed"):
            details = str(e.get("Details") or "").strip()
            marker = "GEMISCHT: Broker-Ausführung enthält zusätzliche externe/Pie-Stücke; Broker-Result wird nicht vollständig dem Screener zugerechnet."
            if marker not in details:
                e["Details"] = (details + " " + marker).strip()
        entries[idx] = e
        changed += 1
    store["entries"] = entries[-5000:]
    ok = bool(_v270_save_trade_journal(store))
    return {"ok": ok, "updated": changed if ok else 0, "preview": preview}

def _v270_reset_trade_journal(watchlist_name=None) -> None:
    store = _v270_load_trade_journal()
    if not watchlist_name:
        store = {"entries": []}
    else:
        wl = str(watchlist_name)
        store["entries"] = [e for e in (store.get("entries") or []) if str(e.get("Watchlist")) != wl]
    _v270_save_trade_journal(store)
