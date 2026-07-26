"""CLI migration for legacy JSON files.

Environment variables:
  APP_STORAGE_BACKEND=supabase
  SUPABASE_URL=https://...supabase.co
  SUPABASE_SERVICE_ROLE_KEY=...
  APP_USER_ID=user_...

Run: python migrate_storage.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.storage.manager import create_storage_manager
from modules.storage.migration import migrate_legacy_json_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    manager = create_storage_manager(st_module=None, app_dir=base_dir)
    results = migrate_legacy_json_files(base_dir, manager, overwrite=args.overwrite)
    print(json.dumps({"status": manager.status(), "migration": results}, ensure_ascii=False, indent=2))
    return 0 if all(not str(row.get("status", "")).startswith("Fehler") for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
