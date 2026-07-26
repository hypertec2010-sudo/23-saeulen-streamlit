"""Repository for open positions, independent from Streamlit UI state."""
from __future__ import annotations

from typing import Any, Mapping

from modules.domain import Position
from .base import NamespaceRepository


class PositionRepository(NamespaceRepository):
    def __init__(self, storage, namespace: str = "positions"):
        super().__init__(storage, namespace)

    @staticmethod
    def watchlist_key(watchlist_name: str = "") -> str:
        name = str(watchlist_name or "Standard").strip() or "Standard"
        return f"v244_open_positions::{name}"

    @staticmethod
    def _normalise_positions(payload: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for ticker, raw in dict(payload or {}).items():
            if not isinstance(raw, Mapping):
                continue
            model = Position.from_legacy_dict(raw, ticker=str(ticker))
            if not model.ticker:
                continue
            output[model.ticker] = model.to_legacy_dict()
        return output

    def load_all(self) -> dict[str, dict[str, dict[str, Any]]]:
        raw = self.load_raw(default={})
        if not isinstance(raw, Mapping):
            return {}
        output: dict[str, dict[str, dict[str, Any]]] = {}
        for watchlist_key, positions in raw.items():
            if isinstance(positions, Mapping):
                output[str(watchlist_key)] = self._normalise_positions(positions)
        return output

    def save_all(self, store: Mapping[str, Any] | None) -> bool:
        clean: dict[str, dict[str, dict[str, Any]]] = {}
        for watchlist_key, positions in dict(store or {}).items():
            if isinstance(positions, Mapping):
                clean[str(watchlist_key)] = self._normalise_positions(positions)
        return self.save_raw(clean)

    def get_for_watchlist(self, watchlist_name: str = "") -> dict[str, dict[str, Any]]:
        return self.load_all().get(self.watchlist_key(watchlist_name), {})

    def save_for_watchlist(self, watchlist_name: str, positions: Mapping[str, Any] | None) -> bool:
        store = self.load_all()
        store[self.watchlist_key(watchlist_name)] = self._normalise_positions(positions)
        return self.save_all(store)

    def delete_for_watchlist(self, watchlist_name: str) -> bool:
        store = self.load_all()
        store.pop(self.watchlist_key(watchlist_name), None)
        return self.save_all(store)
