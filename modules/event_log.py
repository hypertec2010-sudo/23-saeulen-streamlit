"""Persistent signal/trade event log extracted in v25.0."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

_BASE_DIR = Path(__file__).resolve().parent.parent
_time_provider = None
_storage = None
_repository = None

def configure_context(*, base_dir=None, time_provider=None, storage=None, repository=None):
    global _BASE_DIR, _time_provider, _storage, _repository
    if base_dir is not None:
        _BASE_DIR = Path(base_dir)
    if time_provider is not None:
        _time_provider = time_provider
    if storage is not None:
        _storage = storage
    if repository is not None:
        _repository = repository


if _time_provider is None:
    from datetime import datetime
    _time_provider = datetime.now

def _v2416_event_store_path():
    try:
        return _BASE_DIR / ".signal_trade_event_log_v2416.json"
    except Exception:
        return Path("/tmp/.signal_trade_event_log_v2416.json")


def _v2416_load_event_store():
    if _repository is not None:
        try:
            data = _repository.load_store()
            if isinstance(data, dict):
                data.setdefault("events", [])
                data.setdefault("last_signatures", {})
                try:
                    st.session_state.v2416_event_store = data
                except Exception:
                    pass
                return data
        except Exception:
            pass
    elif _storage is not None:
        try:
            data = _storage.load_namespace("event_log", default=None)
            if isinstance(data, dict):
                data.setdefault("events", [])
                data.setdefault("last_signatures", {})
                try:
                    st.session_state.v2416_event_store = data
                except Exception:
                    pass
                return data
        except Exception:
            pass
    path = _v2416_event_store_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("events", [])
                data.setdefault("last_signatures", {})
                return data
    except Exception:
        pass
    try:
        data = st.session_state.get("v2416_event_store", {})
        if isinstance(data, dict):
            data.setdefault("events", [])
            data.setdefault("last_signatures", {})
            return data
    except Exception:
        pass
    return {"events": [], "last_signatures": {}}


def _v2416_save_event_store(store):
    store = store if isinstance(store, dict) else {"events": [], "last_signatures": {}}
    store["events"] = list(store.get("events") or [])[-3000:]
    store["last_signatures"] = dict(store.get("last_signatures") or {})
    try:
        st.session_state.v2416_event_store = store
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
            storage_ok = bool(_storage.save_namespace("event_log", store))
        except Exception:
            storage_ok = False
    file_ok = False
    try:
        _v2416_event_store_path().write_text(
            json.dumps(store, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        file_ok = True
    except Exception:
        file_ok = False
    return bool(storage_ok or file_ok)


def _v2416_log_event(*, event_type, ticker, watchlist_name="", source="", status="", price=None,
                     score=None, trade_state="", details="", payload=None, signature=None):
    """Schreibt ein dedupliziertes Signal-/Trade-Ereignis.

    Gleiche Signaturen werden nicht bei jedem Streamlit-Rerun erneut protokolliert.
    Erst eine echte Status-/Schwellenänderung erzeugt einen neuen Eintrag.
    """
    ticker = str(ticker or "").strip().upper()
    event_type = str(event_type or "").strip()
    if not ticker or not event_type:
        return False
    store = _v2416_load_event_store()
    events = list(store.get("events") or [])
    last = dict(store.get("last_signatures") or {})
    dedupe_key = f"{watchlist_name or 'default'}::{ticker}::{event_type}"
    sig = str(signature if signature is not None else f"{status}|{trade_state}|{price}|{score}|{details}")
    if last.get(dedupe_key) == sig:
        return False
    now = _time_provider().strftime("%d.%m.%Y %H:%M:%S")
    event = {
        "Zeit": now,
        "Watchlist": watchlist_name or "default",
        "Ticker": ticker,
        "Ereignis": event_type,
        "Quelle": source or "-",
        "Status": status or "-",
        "Trade-State": trade_state or "-",
        "Kurs": price,
        "Live-Score": score,
        "Details": details or "-",
    }
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k not in event:
                event[k] = v
    events.append(event)
    last[dedupe_key] = sig
    store["events"] = events[-3000:]
    store["last_signatures"] = last
    _v2416_save_event_store(store)
    return True


def _v2416_events_dataframe(watchlist_name=None):
    store = _v2416_load_event_store()
    df = pd.DataFrame(store.get("events") or [])
    if df.empty:
        return df
    if watchlist_name:
        df = df[df.get("Watchlist", "").astype(str) == str(watchlist_name)]
    return df.iloc[::-1].reset_index(drop=True)


def _v2416_reset_events(watchlist_name=None):
    store = _v2416_load_event_store()
    if not watchlist_name:
        store = {"events": [], "last_signatures": {}}
    else:
        wl = str(watchlist_name)
        store["events"] = [e for e in (store.get("events") or []) if str(e.get("Watchlist")) != wl]
        store["last_signatures"] = {
            k: v for k, v in dict(store.get("last_signatures") or {}).items()
            if not str(k).startswith(f"{wl}::")
        }
    _v2416_save_event_store(store)


# ---------- Main App Flow ----------
logo_path = Path("a_logo_for_the_capital_hill_score_model_is_promi.png")

