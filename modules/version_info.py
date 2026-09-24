# -*- coding: utf-8 -*-
"""Single source of truth for the CHSM application version."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION.txt"


def get_app_version(default: str = "unbekannt") -> str:
    """Return the release from VERSION.txt with a safe fallback."""
    try:
        value = VERSION_FILE.read_text(encoding="utf-8").strip()
        return value or default
    except Exception:
        return default


APP_VERSION = get_app_version()
