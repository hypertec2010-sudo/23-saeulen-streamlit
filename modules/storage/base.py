"""Storage contracts for v28.0.

The application still exchanges plain dictionaries at this migration stage.
A typed repository/domain layer is planned for the following architecture step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class StorageResult:
    ok: bool
    found: bool = False
    data: Any = None
    error: str = ""
    backend: str = ""


class StorageBackend(Protocol):
    name: str

    def load(self, user_id: str, namespace: str) -> StorageResult:
        ...

    def save(self, user_id: str, namespace: str, payload: Any) -> StorageResult:
        ...

    def delete(self, user_id: str, namespace: str) -> StorageResult:
        ...

    def health_check(self) -> StorageResult:
        ...
