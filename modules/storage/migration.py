"""One-time migration helpers from legacy sidecar JSON files."""
from __future__ import annotations

import json
from pathlib import Path


LEGACY_NAMESPACES = {
    ".live_monitor_positions_v245.json": "positions",
    ".signal_trade_event_log_v2416.json": "event_log",
    ".trade_journal_v270.json": "trade_journal",
    ".watchlist_start_prices_v2214.json": "watchlist_start_prices",
    ".live_watchlist_status_history_v227.json": "live_history",
}


def migrate_legacy_json_files(base_dir, storage, *, overwrite=False):
    base = Path(base_dir)
    results = []
    for filename, namespace in LEGACY_NAMESPACES.items():
        path = base / filename
        if not path.exists():
            results.append({"file": filename, "namespace": namespace, "status": "nicht vorhanden"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            results.append({"file": filename, "namespace": namespace, "status": f"Lesefehler: {exc}"})
            continue
        existing = storage.load_result(namespace)
        if existing.ok and existing.found and not overwrite:
            results.append({"file": filename, "namespace": namespace, "status": "übersprungen (Ziel vorhanden)"})
            continue
        saved = storage.save_result(namespace, payload)
        results.append({
            "file": filename,
            "namespace": namespace,
            "status": "importiert" if saved.ok else f"Fehler: {saved.error}",
        })
    return results
