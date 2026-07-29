from __future__ import annotations

from modules.storage.base import StorageResult
from modules.storage.local_backend import LocalJsonBackend
from modules.storage.manager import StorageManager


class OfflineBackend:
    name = "offline"

    def load(self, user_id, namespace):
        return StorageResult(ok=False, error="offline", backend=self.name)

    def save(self, user_id, namespace, payload):
        return StorageResult(ok=False, error="offline", backend=self.name)

    def delete(self, user_id, namespace):
        return StorageResult(ok=False, error="offline", backend=self.name)

    def health_check(self):
        return StorageResult(ok=False, error="offline", backend=self.name)


def test_remote_failure_uses_local_mirror(tmp_path) -> None:
    manager = StorageManager(
        user_id="ci-user",
        local_backend=LocalJsonBackend(tmp_path),
        primary_backend=OfflineBackend(),
        requested_backend="supabase",
    )

    payload = {"entries": [{"ticker": "AAPL", "note": "CI"}]}
    assert manager.save_namespace("trade_journal", payload) is True
    assert manager.load_namespace("trade_journal", {}) == payload
    assert manager.status()["degraded"] is True
