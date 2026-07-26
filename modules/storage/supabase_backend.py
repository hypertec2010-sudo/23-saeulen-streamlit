"""Minimal Supabase/PostgREST backend without an additional Python dependency.

It expects the SQL schema shipped as ``supabase_schema.sql``. The service-role
key stays server-side in Streamlit Secrets and is never rendered in the UI.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import requests

from .base import StorageResult


class SupabaseBackend:
    name = "supabase"

    def __init__(
        self,
        *,
        url: str,
        service_role_key: str,
        table: str = "app_state",
        timeout_seconds: float = 10.0,
    ):
        self.url = str(url or "").strip().rstrip("/")
        self.key = str(service_role_key or "").strip()
        self.table = str(table or "app_state").strip() or "app_state"
        self.timeout_seconds = max(2.0, float(timeout_seconds or 10.0))
        self.session = requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key)

    def _endpoint(self) -> str:
        return urljoin(self.url + "/", f"rest/v1/{self.table}")

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    @staticmethod
    def _error_text(response: requests.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                return str(payload.get("message") or payload.get("hint") or payload.get("details") or payload)
            return str(payload)
        except Exception:
            return str(response.text or response.reason or f"HTTP {response.status_code}")[:500]

    def _not_configured(self) -> StorageResult:
        return StorageResult(ok=False, error="Supabase URL oder Service-Role-Key fehlt.", backend=self.name)

    def load(self, user_id: str, namespace: str) -> StorageResult:
        if not self.configured:
            return self._not_configured()
        try:
            response = self.session.get(
                self._endpoint(),
                headers=self._headers(),
                params={
                    "select": "payload",
                    "user_id": f"eq.{user_id}",
                    "namespace": f"eq.{namespace}",
                    "limit": "1",
                },
                timeout=self.timeout_seconds,
            )
            if not response.ok:
                return StorageResult(ok=False, error=self._error_text(response), backend=self.name)
            rows = response.json() or []
            if not rows:
                return StorageResult(ok=True, found=False, data=None, backend=self.name)
            payload = rows[0].get("payload") if isinstance(rows[0], dict) else None
            return StorageResult(ok=True, found=True, data=payload, backend=self.name)
        except Exception as exc:
            return StorageResult(ok=False, error=str(exc), backend=self.name)

    def save(self, user_id: str, namespace: str, payload: Any) -> StorageResult:
        if not self.configured:
            return self._not_configured()
        try:
            response = self.session.post(
                self._endpoint(),
                headers=self._headers(prefer="resolution=merge-duplicates,return=minimal"),
                params={"on_conflict": "user_id,namespace"},
                json={"user_id": user_id, "namespace": namespace, "payload": payload},
                timeout=self.timeout_seconds,
            )
            if not response.ok:
                return StorageResult(ok=False, error=self._error_text(response), backend=self.name)
            return StorageResult(ok=True, found=True, data=payload, backend=self.name)
        except Exception as exc:
            return StorageResult(ok=False, error=str(exc), backend=self.name)

    def delete(self, user_id: str, namespace: str) -> StorageResult:
        if not self.configured:
            return self._not_configured()
        try:
            response = self.session.delete(
                self._endpoint(),
                headers=self._headers(prefer="return=minimal"),
                params={"user_id": f"eq.{user_id}", "namespace": f"eq.{namespace}"},
                timeout=self.timeout_seconds,
            )
            if not response.ok:
                return StorageResult(ok=False, error=self._error_text(response), backend=self.name)
            return StorageResult(ok=True, found=True, backend=self.name)
        except Exception as exc:
            return StorageResult(ok=False, error=str(exc), backend=self.name)

    def health_check(self) -> StorageResult:
        if not self.configured:
            return self._not_configured()
        try:
            response = self.session.get(
                self._endpoint(),
                headers=self._headers(),
                params={"select": "namespace", "limit": "1"},
                timeout=self.timeout_seconds,
            )
            if not response.ok:
                return StorageResult(ok=False, error=self._error_text(response), backend=self.name)
            return StorageResult(ok=True, found=True, data={"table": self.table}, backend=self.name)
        except Exception as exc:
            return StorageResult(ok=False, error=str(exc), backend=self.name)
