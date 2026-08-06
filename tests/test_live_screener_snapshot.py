from __future__ import annotations

import pandas as pd

from modules import live_screener_snapshot as snapshot


class MemoryStorage:
    def __init__(self):
        self.data = {}

    def load_namespace(self, namespace, default=None):
        return self.data.get(namespace, default)

    def save_namespace(self, namespace, payload):
        self.data[namespace] = payload
        return True


def test_snapshot_roundtrip_restores_dataframe_and_cache_key() -> None:
    storage = MemoryStorage()
    key = {
        "watchlist": "Mobil",
        "tickers": ("SAP.DE", "AAPL"),
        "style": "Charttechnik",
        "horizon": "Kurzfrist / Trading",
    }
    cache = {
        "key": key,
        "ts": "2026-08-05T11:30:00",
        "live_df": pd.DataFrame(
            [
                {"Ampel": "🟢", "Ticker": "SAP.DE", "Kurs": 161.34},
                {"Ampel": "🟡", "Ticker": "AAPL", "Kurs": 333.02},
            ]
        ),
        "live_errors": pd.DataFrame([{"Ticker": "BAD", "Fehler": "Keine Daten"}]),
        "scan_meta": {
            "complete": False,
            "selected_count": 3,
            "completed_count": 2,
            "pending_tickers": ["BAD"],
        },
    }

    assert snapshot.save_snapshot(storage, cache, ui_state={"mobile_mode": True})
    restored = snapshot.load_snapshot(storage, key)

    assert restored is not None
    assert restored["cache"]["key"] == key
    assert restored["cache"]["ts"] == cache["ts"]
    assert restored["cache"]["live_df"].to_dict("records") == cache["live_df"].to_dict("records")
    assert restored["cache"]["live_errors"].iloc[0]["Ticker"] == "BAD"
    assert restored["cache"]["scan_meta"]["complete"] is False
    assert restored["cache"]["scan_meta"]["pending_tickers"] == ["BAD"]
    assert restored["ui"]["mobile_mode"] is True


def test_snapshot_rejects_different_watchlist_or_horizon() -> None:
    storage = MemoryStorage()
    key = {"watchlist": "A", "tickers": ("SAP.DE",), "style": "Charttechnik", "horizon": "Swing"}
    cache = {"key": key, "ts": "2026-08-05T10:00:00", "live_df": pd.DataFrame([{"Ticker": "SAP.DE"}]), "live_errors": pd.DataFrame()}
    assert snapshot.save_snapshot(storage, cache)

    wrong = {**key, "watchlist": "B"}
    assert snapshot.load_snapshot(storage, wrong) is None


def test_snapshot_store_is_bounded() -> None:
    storage = MemoryStorage()
    for index in range(5):
        key = {"watchlist": f"W{index}", "tickers": (f"T{index}",), "style": "Charttechnik", "horizon": "Swing"}
        cache = {"key": key, "ts": f"2026-08-05T10:0{index}:00", "live_df": pd.DataFrame([{"Ticker": f"T{index}"}]), "live_errors": pd.DataFrame()}
        assert snapshot.save_snapshot(storage, cache, max_snapshots=3)

    store = storage.data[snapshot.SNAPSHOT_NAMESPACE]
    assert len(store["snapshots"]) == 3
