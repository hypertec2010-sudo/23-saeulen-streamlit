"""Repository for typed trade-journal persistence."""
from __future__ import annotations

from typing import Any, Mapping

from modules.domain import JournalEntry
from .base import NamespaceRepository


class TradeJournalRepository(NamespaceRepository):
    def __init__(self, storage, namespace: str = "trade_journal", max_entries: int = 5000):
        super().__init__(storage, namespace)
        self.max_entries = max(1, int(max_entries))

    def load_store(self) -> dict[str, list[dict[str, Any]]]:
        raw = self.load_raw(default={})
        entries = raw.get("entries", []) if isinstance(raw, Mapping) else []
        clean = [
            JournalEntry.from_legacy_dict(item).to_legacy_dict()
            for item in entries
            if isinstance(item, Mapping)
        ]
        return {"entries": clean[-self.max_entries:]}

    def save_store(self, store: Mapping[str, Any] | None) -> bool:
        entries = dict(store or {}).get("entries", [])
        clean = [
            JournalEntry.from_legacy_dict(item).to_legacy_dict()
            for item in entries
            if isinstance(item, Mapping)
        ]
        return self.save_raw({"entries": clean[-self.max_entries:]})

    def append(self, entry: JournalEntry | Mapping[str, Any]) -> bool:
        store = self.load_store()
        item = entry.to_legacy_dict() if isinstance(entry, JournalEntry) else JournalEntry.from_legacy_dict(entry).to_legacy_dict()
        store["entries"].append(item)
        return self.save_store(store)

    def clear(self, watchlist_name: str | None = None) -> bool:
        if not watchlist_name:
            return self.save_store({"entries": []})
        target = str(watchlist_name)
        store = self.load_store()
        store["entries"] = [e for e in store["entries"] if str(e.get("Watchlist")) != target]
        return self.save_store(store)
