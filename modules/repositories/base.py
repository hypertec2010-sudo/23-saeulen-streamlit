"""Repository primitives built on top of the v28 storage manager."""
from __future__ import annotations

from typing import Any


class NamespaceRepository:
    """Small persistence boundary for one storage namespace."""

    def __init__(self, storage, namespace: str):
        if storage is None:
            raise ValueError("storage darf nicht None sein")
        self.storage = storage
        self.namespace = str(namespace or "state").strip() or "state"

    def load_raw(self, default=None):
        return self.storage.load_namespace(self.namespace, default=default)

    def save_raw(self, payload: Any) -> bool:
        return bool(self.storage.save_namespace(self.namespace, payload))

    def delete(self) -> bool:
        return bool(self.storage.delete_namespace(self.namespace))
