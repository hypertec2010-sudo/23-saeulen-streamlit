"""Repository for signal and trade events."""
from __future__ import annotations

from typing import Any, Mapping

from modules.domain import SignalEvent
from .base import NamespaceRepository


class EventRepository(NamespaceRepository):
    def __init__(self, storage, namespace: str = "event_log", max_events: int = 3000):
        super().__init__(storage, namespace)
        self.max_events = max(1, int(max_events))

    def load_store(self) -> dict[str, Any]:
        raw = self.load_raw(default={})
        if not isinstance(raw, Mapping):
            raw = {}
        events = [
            SignalEvent.from_legacy_dict(item).to_legacy_dict()
            for item in raw.get("events", [])
            if isinstance(item, Mapping)
        ]
        signatures = dict(raw.get("last_signatures") or {})
        return {"events": events[-self.max_events:], "last_signatures": signatures}

    def save_store(self, store: Mapping[str, Any] | None) -> bool:
        raw = dict(store or {})
        events = [
            SignalEvent.from_legacy_dict(item).to_legacy_dict()
            for item in raw.get("events", [])
            if isinstance(item, Mapping)
        ]
        payload = {
            "events": events[-self.max_events:],
            "last_signatures": dict(raw.get("last_signatures") or {}),
        }
        return self.save_raw(payload)

    def clear(self, watchlist_name: str | None = None) -> bool:
        if not watchlist_name:
            return self.save_store({"events": [], "last_signatures": {}})
        target = str(watchlist_name)
        store = self.load_store()
        store["events"] = [e for e in store["events"] if str(e.get("Watchlist")) != target]
        store["last_signatures"] = {
            k: v for k, v in store["last_signatures"].items()
            if not str(k).startswith(f"{target}::")
        }
        return self.save_store(store)
