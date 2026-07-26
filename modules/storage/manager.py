"""Configuration, user scoping and resilient primary/fallback orchestration."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .base import StorageResult
from .local_backend import LocalJsonBackend
from .supabase_backend import SupabaseBackend


class StorageManager:
    def __init__(
        self,
        *,
        user_id: str,
        local_backend: LocalJsonBackend,
        primary_backend=None,
        requested_backend: str = "local",
        mirror_local: bool = True,
    ):
        self.user_id = str(user_id or "default")
        self.local = local_backend
        self.primary = primary_backend
        self.requested_backend = str(requested_backend or "local").lower()
        self.mirror_local = bool(mirror_local)
        self.last_error = ""
        self.last_backend = self.local.name
        self.degraded = False

    @property
    def remote_enabled(self) -> bool:
        return self.primary is not None

    def load_result(self, namespace: str) -> StorageResult:
        namespace = str(namespace or "state").strip() or "state"
        if self.primary is not None:
            remote = self.primary.load(self.user_id, namespace)
            if remote.ok and remote.found:
                self.last_backend = remote.backend
                self.degraded = False
                self.last_error = ""
                if self.mirror_local:
                    self.local.save(self.user_id, namespace, remote.data)
                return remote
            if not remote.ok:
                self.degraded = True
                self.last_error = remote.error
        local = self.local.load(self.user_id, namespace)
        self.last_backend = local.backend
        if not local.ok:
            self.last_error = local.error or self.last_error
        return local

    def load_namespace(self, namespace: str, default=None):
        result = self.load_result(namespace)
        if result.ok and result.found:
            return result.data
        return default

    def save_result(self, namespace: str, payload: Any) -> StorageResult:
        namespace = str(namespace or "state").strip() or "state"
        local = self.local.save(self.user_id, namespace, payload)
        remote = None
        if self.primary is not None:
            remote = self.primary.save(self.user_id, namespace, payload)
            if remote.ok:
                self.last_backend = remote.backend
                self.degraded = False
                self.last_error = ""
                return remote
            self.degraded = True
            self.last_error = remote.error
        self.last_backend = local.backend
        return local if local.ok else (remote or local)

    def save_namespace(self, namespace: str, payload: Any) -> bool:
        return bool(self.save_result(namespace, payload).ok)

    def delete_namespace(self, namespace: str) -> bool:
        namespace = str(namespace or "state").strip() or "state"
        local = self.local.delete(self.user_id, namespace)
        if self.primary is not None:
            remote = self.primary.delete(self.user_id, namespace)
            if remote.ok:
                self.last_backend = remote.backend
                self.degraded = False
                self.last_error = ""
                return True
            self.degraded = True
            self.last_error = remote.error
        return bool(local.ok)

    def health_check(self) -> StorageResult:
        if self.primary is not None:
            result = self.primary.health_check()
            self.degraded = not result.ok
            self.last_error = "" if result.ok else result.error
            self.last_backend = result.backend
            return result
        result = self.local.health_check()
        self.degraded = not result.ok
        self.last_error = "" if result.ok else result.error
        self.last_backend = result.backend
        return result

    def status(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.primary.name if self.primary is not None else self.local.name,
            "last_backend": self.last_backend,
            "remote_enabled": self.remote_enabled,
            "degraded": self.degraded,
            "last_error": self.last_error,
            "user_id": self.user_id,
            "local_path": str(self.local.base_dir),
        }


def _plain_mapping(value) -> dict:
    try:
        return {str(k): value[k] for k in value.keys()}
    except Exception:
        return dict(value) if isinstance(value, dict) else {}


def _secret_section(st_module, name: str) -> dict:
    try:
        return _plain_mapping(st_module.secrets.get(name, {}))
    except Exception:
        return {}


def _bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "ja", "on"}


def resolve_user_id(st_module=None, *, mode: str = "email_hash") -> str:
    explicit = str(os.environ.get("APP_USER_ID", "") or "").strip()
    if explicit:
        return explicit
    email = ""
    if st_module is not None:
        try:
            email = str(st_module.user.get("email", "") or "").strip().lower()
        except Exception:
            email = ""
    if not email:
        email = "default"
    if str(mode or "email_hash").lower() in {"email", "plain_email"}:
        return email
    if email == "default":
        return email
    return "user_" + hashlib.sha256(email.encode("utf-8")).hexdigest()[:24]


def create_storage_manager(*, st_module=None, app_dir: str | Path | None = None) -> StorageManager:
    storage_cfg = _secret_section(st_module, "storage") if st_module is not None else {}
    supabase_cfg = _secret_section(st_module, "supabase") if st_module is not None else {}

    requested = str(
        storage_cfg.get("backend")
        or os.environ.get("APP_STORAGE_BACKEND")
        or ("supabase" if (supabase_cfg.get("url") or os.environ.get("SUPABASE_URL")) else "local")
    ).strip().lower()
    user_mode = str(storage_cfg.get("user_scope") or "email_hash")
    user_id = resolve_user_id(st_module, mode=user_mode)
    mirror_local = _bool(storage_cfg.get("mirror_local", True), True)

    root = Path(app_dir or Path.cwd())
    local_dir = storage_cfg.get("local_dir") or os.environ.get("APP_STORAGE_LOCAL_DIR") or ".app_storage"
    local_path = Path(str(local_dir)).expanduser()
    if not local_path.is_absolute():
        local_path = root / local_path
    local = LocalJsonBackend(local_path)

    primary = None
    if requested == "supabase":
        url = str(supabase_cfg.get("url") or os.environ.get("SUPABASE_URL") or "").strip()
        key = str(
            supabase_cfg.get("service_role_key")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or ""
        ).strip()
        table = str(supabase_cfg.get("table") or os.environ.get("SUPABASE_STATE_TABLE") or "app_state")
        timeout = supabase_cfg.get("timeout_seconds") or os.environ.get("SUPABASE_TIMEOUT_SECONDS") or 10
        candidate = SupabaseBackend(url=url, service_role_key=key, table=table, timeout_seconds=float(timeout))
        if candidate.configured:
            primary = candidate

    return StorageManager(
        user_id=user_id,
        local_backend=local,
        primary_backend=primary,
        requested_backend=requested,
        mirror_local=mirror_local,
    )


def should_use_database_watchlists(*, st_module=None, manager: StorageManager | None = None) -> bool:
    cfg = _secret_section(st_module, "storage") if st_module is not None else {}
    # Sobald Supabase bewusst als Ziel gewaehlt wurde, soll die Watchlist-UI auch
    # bei einem voruebergehenden Verbindungsproblem auf den lokalen Spiegel fallen.
    default = bool(manager and (manager.remote_enabled or manager.requested_backend == "supabase"))
    return _bool(cfg.get("use_for_watchlists", default), default)
