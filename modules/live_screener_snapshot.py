# -*- coding: utf-8 -*-
"""Persistent Live-Screener snapshots for reconnect-safe mobile use.

The in-session Streamlit cache remains the fast source. This module mirrors the
last completed scan to the configured StorageManager (Supabase plus local
fallback) and can rebuild the cache after a browser/WebSocket reconnect.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from typing import Any, Mapping

import pandas as pd

SNAPSHOT_NAMESPACE = "live_screener_snapshots"
SNAPSHOT_VERSION = 1


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        scalar = value.item()
    except Exception:
        scalar = value
    if scalar is not value:
        return _json_safe(scalar)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def normalize_cache_key(cache_key: Mapping[str, Any] | None) -> dict[str, Any]:
    source = cache_key if isinstance(cache_key, Mapping) else {}
    return {
        "watchlist": str(source.get("watchlist") or ""),
        "tickers": tuple(
            str(ticker).strip().upper()
            for ticker in (source.get("tickers") or [])
            if str(ticker).strip()
        ),
        "style": str(source.get("style") or ""),
        "horizon": str(source.get("horizon") or ""),
    }


def snapshot_id(cache_key: Mapping[str, Any] | None) -> str:
    normalized = normalize_cache_key(cache_key)
    canonical = json.dumps(
        {
            **normalized,
            "tickers": list(normalized["tickers"]),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def dataframe_to_payload(frame: Any) -> dict[str, Any]:
    df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame or [])
    columns = [str(column) for column in df.columns]
    rows = [
        [_json_safe(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    return {"columns": columns, "rows": rows}


def dataframe_from_payload(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, Mapping):
        return pd.DataFrame()
    columns = [str(column) for column in (payload.get("columns") or [])]
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        return pd.DataFrame(columns=columns)
    try:
        return pd.DataFrame(rows, columns=columns)
    except Exception:
        return pd.DataFrame()


def cache_to_snapshot(
    cache: Mapping[str, Any],
    *,
    ui_state: Mapping[str, Any] | None = None,
    saved_at: str | None = None,
) -> dict[str, Any]:
    key = normalize_cache_key(cache.get("key") if isinstance(cache, Mapping) else {})
    timestamp = str(cache.get("ts") or saved_at or datetime.now().isoformat())
    return {
        "version": SNAPSHOT_VERSION,
        "id": snapshot_id(key),
        "key": {**key, "tickers": list(key["tickers"])},
        "ts": timestamp,
        "saved_at": str(saved_at or datetime.now().isoformat()),
        "live_df": dataframe_to_payload(cache.get("live_df")),
        "live_errors": dataframe_to_payload(cache.get("live_errors")),
        "ui": _json_safe(dict(ui_state or {})),
    }


def snapshot_to_cache(snapshot: Any, expected_key: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(snapshot, Mapping):
        return None
    expected = normalize_cache_key(expected_key)
    actual = normalize_cache_key(snapshot.get("key"))
    if actual != expected:
        return None
    live_df_payload = snapshot.get("live_df")
    if not isinstance(live_df_payload, Mapping):
        return None
    live_df = dataframe_from_payload(live_df_payload)
    return {
        "key": expected,
        "ts": str(snapshot.get("ts") or snapshot.get("saved_at") or ""),
        "live_df": live_df,
        "live_errors": dataframe_from_payload(snapshot.get("live_errors")),
    }


def load_snapshot(storage: Any, expected_key: Mapping[str, Any]) -> dict[str, Any] | None:
    if storage is None:
        return None
    try:
        store = storage.load_namespace(SNAPSHOT_NAMESPACE, default={})
    except Exception:
        return None
    if not isinstance(store, Mapping):
        return None
    snapshots = store.get("snapshots")
    if not isinstance(snapshots, Mapping):
        return None
    entry = snapshots.get(snapshot_id(expected_key))
    cache = snapshot_to_cache(entry, expected_key)
    if cache is None:
        return None
    return {
        "cache": cache,
        "ui": dict(entry.get("ui") or {}) if isinstance(entry, Mapping) else {},
        "saved_at": str(entry.get("saved_at") or cache.get("ts") or "") if isinstance(entry, Mapping) else str(cache.get("ts") or ""),
    }


def save_snapshot(
    storage: Any,
    cache: Mapping[str, Any],
    *,
    ui_state: Mapping[str, Any] | None = None,
    max_snapshots: int = 12,
) -> bool:
    if storage is None or not isinstance(cache, Mapping):
        return False
    snapshot = cache_to_snapshot(cache, ui_state=ui_state)
    try:
        store = storage.load_namespace(SNAPSHOT_NAMESPACE, default={})
    except Exception:
        store = {}
    store = dict(store) if isinstance(store, Mapping) else {}
    snapshots = dict(store.get("snapshots") or {})
    snapshots[snapshot["id"]] = snapshot

    keep = max(1, int(max_snapshots or 12))
    ordered = sorted(
        snapshots.items(),
        key=lambda item: str((item[1] or {}).get("saved_at") or (item[1] or {}).get("ts") or ""),
        reverse=True,
    )[:keep]
    payload = {
        "version": SNAPSHOT_VERSION,
        "updated_at": datetime.now().isoformat(),
        "snapshots": dict(ordered),
    }
    try:
        return bool(storage.save_namespace(SNAPSHOT_NAMESPACE, payload))
    except Exception:
        return False
