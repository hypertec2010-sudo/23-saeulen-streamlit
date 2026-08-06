from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED = [
    "app.py",
    "legacy_app.py",
    "modules/app_shell.py",
    "modules/page_runtime.py",
    "modules/live_refresh_policy.py",
    "modules/live_screener_snapshot.py",
    "modules/live_scan_batches.py",
    "modules/live_change_explainer.py",
    "pages/analysis.py",
    "pages/radar.py",
    "pages/watchlists.py",
    "pages/positions.py",
    "pages/trade_journal.py",
    "modules/risk_calculator.py",
    "modules/position_monitor.py",
    "modules/trade_journal.py",
    "modules/event_log.py",
    "modules/live_monitor.py",
    "modules/storage/manager.py",
    "modules/domain/models.py",
    "modules/repositories/registry.py",
    "supabase_schema.sql",
    ".streamlit/secrets.example.toml",
    ".github/workflows/quality.yml",
    "requirements-ci.txt",
    "pytest.ini",
    "tests/test_streamlit_app.py",
    "tests/test_live_refresh_policy.py",
    "tests/test_live_screener_snapshot.py",
    "tests/test_live_scan_batches.py",
    "tests/test_live_change_explainer.py",
    "tests/test_page_runtime_state.py",
    "tests/test_storage_fallback.py",
    "tests/test_trade_journal_flow.py",
    "tests/test_source_guards.py",
    "regression_check.py",
    "RELEASE_NOTES_v28_4.md",
    "RELEASE_NOTES_v28_4_2.md",
    "RELEASE_NOTES_v28_4_3.md",
    "ARCHITECTURE_V28_4_3.md",
    "RELEASE_NOTES_v28_4_4.md",
    "ARCHITECTURE_V28_4_4.md",
]

missing = [relative for relative in REQUIRED if not (ROOT / relative).exists()]
if missing:
    raise SystemExit(f"Fehlende Dateien: {missing}")

for path in ROOT.rglob("*.py"):
    if "__pycache__" in path.parts:
        continue
    py_compile.compile(str(path), doraise=True)

subprocess.run([sys.executable, str(ROOT / "regression_check.py")], cwd=ROOT, check=True)
print("v28.4.4 Deployment-Struktur, Syntax und deterministische Regressionstests OK")
