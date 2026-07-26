"""Atomic local JSON backend used as fallback and recovery mirror."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .base import StorageResult


def _safe_component(value: str, default: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return (text[:160] or default).strip(".") or default


class LocalJsonBackend:
    name = "local-json"

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str, namespace: str) -> Path:
        user_dir = self.base_dir / _safe_component(user_id, "default")
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / f"{_safe_component(namespace, 'state')}.json"

    def load(self, user_id: str, namespace: str) -> StorageResult:
        path = self._path(user_id, namespace)
        if not path.exists():
            return StorageResult(ok=True, found=False, data=None, backend=self.name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return StorageResult(ok=True, found=True, data=data, backend=self.name)
        except Exception as exc:
            return StorageResult(ok=False, found=False, error=f"{path.name}: {exc}", backend=self.name)

    def save(self, user_id: str, namespace: str, payload: Any) -> StorageResult:
        path = self._path(user_id, namespace)
        tmp_path = None
        try:
            serialised = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent))
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialised)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.replace(path)
            return StorageResult(ok=True, found=True, data=payload, backend=self.name)
        except Exception as exc:
            try:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return StorageResult(ok=False, found=False, error=f"{path.name}: {exc}", backend=self.name)

    def delete(self, user_id: str, namespace: str) -> StorageResult:
        path = self._path(user_id, namespace)
        try:
            existed = path.exists()
            if existed:
                path.unlink()
            return StorageResult(ok=True, found=existed, backend=self.name)
        except Exception as exc:
            return StorageResult(ok=False, found=False, error=f"{path.name}: {exc}", backend=self.name)

    def health_check(self) -> StorageResult:
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            probe = self.base_dir / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return StorageResult(ok=True, found=True, data={"path": str(self.base_dir)}, backend=self.name)
        except Exception as exc:
            return StorageResult(ok=False, error=str(exc), backend=self.name)
