"""Configuration, user scoping and resilient primary/fallback orchestration."""
from __future__ import annotations

import copy
import hashlib
import os
import time
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
        # v30.21ae: one StorageManager instance is reused for the Streamlit
        # session so the underlying requests.Session can keep HTTP connections
        # alive. Data itself is only cached inside one script rerun; begin_run()
        # clears it, so another browser/device can still update Supabase between
        # reruns without a long-lived stale application cache.
        self._run_cache: dict[str, StorageResult] = {}
        self._perf = {}
        self.begin_run()

    @property
    def remote_enabled(self) -> bool:
        return self.primary is not None

    @staticmethod
    def _clone_result(result: StorageResult) -> StorageResult:
        try:
            data = copy.deepcopy(result.data)
        except Exception:
            data = result.data
        return StorageResult(
            ok=bool(result.ok),
            found=bool(result.found),
            data=data,
            error=str(result.error or ""),
            backend=str(result.backend or ""),
        )

    def begin_run(self) -> None:
        """Start a new Streamlit rerun without discarding HTTP connections."""
        self._run_cache = {}
        self._perf = {
            "load_calls": 0,
            "backend_loads": 0,
            "cache_hits": 0,
            "backend_ms": 0.0,
            "local_ms": 0.0,
            "mirror_writes": 0,
            "save_calls": 0,
            "save_ms": 0.0,
            "delete_calls": 0,
            "namespaces": {},
        }

    def _record_namespace(self, namespace: str, elapsed_ms: float, *, cached: bool = False) -> None:
        bucket = self._perf.setdefault("namespaces", {}).setdefault(
            namespace, {"calls": 0, "backend_loads": 0, "cache_hits": 0, "ms": 0.0}
        )
        bucket["calls"] += 1
        bucket["ms"] += float(elapsed_ms or 0.0)
        if cached:
            bucket["cache_hits"] += 1
        else:
            bucket["backend_loads"] += 1

    def load_result(self, namespace: str, *, force_refresh: bool = False) -> StorageResult:
        namespace = str(namespace or "state").strip() or "state"
        self._perf["load_calls"] = int(self._perf.get("load_calls", 0)) + 1

        if not force_refresh and namespace in self._run_cache:
            self._perf["cache_hits"] = int(self._perf.get("cache_hits", 0)) + 1
            self._record_namespace(namespace, 0.0, cached=True)
            return self._clone_result(self._run_cache[namespace])

        started = time.perf_counter()
        if self.primary is not None:
            remote_started = time.perf_counter()
            remote = self.primary.load(self.user_id, namespace)
            remote_ms = (time.perf_counter() - remote_started) * 1000.0
            self._perf["backend_loads"] = int(self._perf.get("backend_loads", 0)) + 1
            self._perf["backend_ms"] = float(self._perf.get("backend_ms", 0.0)) + remote_ms
            if remote.ok and remote.found:
                self.last_backend = remote.backend
                self.degraded = False
                self.last_error = ""
                if self.mirror_local:
                    mirror_started = time.perf_counter()
                    self.local.save(self.user_id, namespace, remote.data)
                    self._perf["local_ms"] = float(self._perf.get("local_ms", 0.0)) + (time.perf_counter() - mirror_started) * 1000.0
                    self._perf["mirror_writes"] = int(self._perf.get("mirror_writes", 0)) + 1
                elapsed = (time.perf_counter() - started) * 1000.0
                self._record_namespace(namespace, elapsed)
                self._run_cache[namespace] = self._clone_result(remote)
                return self._clone_result(remote)
            if not remote.ok:
                self.degraded = True
                self.last_error = remote.error

        local_started = time.perf_counter()
        local = self.local.load(self.user_id, namespace)
        self._perf["local_ms"] = float(self._perf.get("local_ms", 0.0)) + (time.perf_counter() - local_started) * 1000.0
        if self.primary is None:
            self._perf["backend_loads"] = int(self._perf.get("backend_loads", 0)) + 1
        self.last_backend = local.backend
        if not local.ok:
            self.last_error = local.error or self.last_error
        elapsed = (time.perf_counter() - started) * 1000.0
        self._record_namespace(namespace, elapsed)
        self._run_cache[namespace] = self._clone_result(local)
        return self._clone_result(local)

    def load_namespace(self, namespace: str, default=None, *, force_refresh: bool = False):
        result = self.load_result(namespace, force_refresh=force_refresh)
        if result.ok and result.found:
            return result.data
        return default

    def save_result(self, namespace: str, payload: Any) -> StorageResult:
        namespace = str(namespace or "state").strip() or "state"
        started = time.perf_counter()
        self._perf["save_calls"] = int(self._perf.get("save_calls", 0)) + 1
        local = self.local.save(self.user_id, namespace, payload)
        remote = None
        result = local
        if self.primary is not None:
            remote = self.primary.save(self.user_id, namespace, payload)
            if remote.ok:
                self.last_backend = remote.backend
                self.degraded = False
                self.last_error = ""
                result = remote
            else:
                self.degraded = True
                self.last_error = remote.error
                self.last_backend = local.backend
                result = local if local.ok else (remote or local)
        else:
            self.last_backend = local.backend
        self._perf["save_ms"] = float(self._perf.get("save_ms", 0.0)) + (time.perf_counter() - started) * 1000.0
        # Make subsequent reads in the same rerun see the just-written state
        # without another network round trip.
        if result.ok:
            self._run_cache[namespace] = StorageResult(
                ok=True, found=True, data=copy.deepcopy(payload), backend=result.backend
            )
        else:
            self._run_cache.pop(namespace, None)
        return self._clone_result(result)

    def save_namespace(self, namespace: str, payload: Any) -> bool:
        return bool(self.save_result(namespace, payload).ok)

    def delete_namespace(self, namespace: str) -> bool:
        namespace = str(namespace or "state").strip() or "state"
        self._perf["delete_calls"] = int(self._perf.get("delete_calls", 0)) + 1
        self._run_cache.pop(namespace, None)
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

    def performance_snapshot(self) -> dict[str, Any]:
        namespaces = []
        for name, stats in (self._perf.get("namespaces") or {}).items():
            namespaces.append({
                "namespace": str(name),
                "calls": int(stats.get("calls", 0)),
                "backend_loads": int(stats.get("backend_loads", 0)),
                "cache_hits": int(stats.get("cache_hits", 0)),
                "ms": round(float(stats.get("ms", 0.0)), 1),
            })
        namespaces.sort(key=lambda row: row["ms"], reverse=True)
        return {
            "load_calls": int(self._perf.get("load_calls", 0)),
            "backend_loads": int(self._perf.get("backend_loads", 0)),
            "cache_hits": int(self._perf.get("cache_hits", 0)),
            "backend_ms": round(float(self._perf.get("backend_ms", 0.0)), 1),
            "local_ms": round(float(self._perf.get("local_ms", 0.0)), 1),
            "mirror_writes": int(self._perf.get("mirror_writes", 0)),
            "save_calls": int(self._perf.get("save_calls", 0)),
            "save_ms": round(float(self._perf.get("save_ms", 0.0)), 1),
            "delete_calls": int(self._perf.get("delete_calls", 0)),
            "slow_namespaces": namespaces[:8],
        }

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

    # v30.21ae: Reuse the manager for one browser session. This preserves the
    # requests.Session/HTTP keep-alive connection across Streamlit reruns while
    # begin_run() guarantees that application data is not cached across reruns.
    fingerprint = "|".join([
        requested, user_id, str(local_path), str(mirror_local),
        str(getattr(primary, "url", "") or ""),
        str(getattr(primary, "table", "") or ""),
        hashlib.sha256(str(getattr(primary, "key", "") or "").encode("utf-8")).hexdigest()[:12],
    ])
    session_key = "_chsm_storage_manager_v3021ae"
    fingerprint_key = "_chsm_storage_manager_fingerprint_v3021ae"
    if st_module is not None:
        try:
            existing = st_module.session_state.get(session_key)
            existing_fp = st_module.session_state.get(fingerprint_key)
            if isinstance(existing, StorageManager) and existing_fp == fingerprint:
                existing.begin_run()
                return existing
        except Exception:
            pass

    manager = StorageManager(
        user_id=user_id,
        local_backend=local,
        primary_backend=primary,
        requested_backend=requested,
        mirror_local=mirror_local,
    )
    if st_module is not None:
        try:
            st_module.session_state[session_key] = manager
            st_module.session_state[fingerprint_key] = fingerprint
        except Exception:
            pass
    return manager


def should_use_database_watchlists(*, st_module=None, manager: StorageManager | None = None) -> bool:
    cfg = _secret_section(st_module, "storage") if st_module is not None else {}
    # Sobald Supabase bewusst als Ziel gewaehlt wurde, soll die Watchlist-UI auch
    # bei einem voruebergehenden Verbindungsproblem auf den lokalen Spiegel fallen.
    default = bool(manager and (manager.remote_enabled or manager.requested_backend == "supabase"))
    return _bool(cfg.get("use_for_watchlists", default), default)
