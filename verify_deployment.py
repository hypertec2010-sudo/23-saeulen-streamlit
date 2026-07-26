from pathlib import Path
import py_compile
import subprocess
import sys

required = [
    "app.py", "legacy_app.py", "modules/__init__.py",
    "modules/app_shell.py", "modules/page_runtime.py",
    "pages/analysis.py", "pages/radar.py", "pages/watchlists.py",
    "pages/positions.py", "pages/trade_journal.py", "modules/risk_calculator.py",
    "modules/position_monitor.py", "modules/trade_journal.py", "modules/event_log.py", "modules/live_monitor.py",
    "modules/watchlist_storage.py", "modules/chart_overlays.py",
    "modules/radar_view.py", "modules/analysis_view.py", "modules/analysis_engine.py", "modules/cache_layer.py",
    "modules/market_data.py", "modules/ticker_resolver.py", "modules/scoring_engine.py",
    "modules/storage/__init__.py", "modules/storage/base.py", "modules/storage/local_backend.py",
    "modules/storage/supabase_backend.py", "modules/storage/manager.py",
    "modules/storage/watchlist_repository.py", "modules/storage/migration.py",
    "modules/domain/__init__.py", "modules/domain/models.py",
    "modules/repositories/__init__.py", "modules/repositories/base.py",
    "modules/repositories/position_repository.py", "modules/repositories/trade_journal_repository.py",
    "modules/repositories/event_repository.py", "modules/repositories/registry.py",
    "supabase_schema.sql", ".streamlit/secrets.example.toml", "migrate_storage.py", "regression_check.py",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit(f"Fehlende Dateien: {missing}")
for p in required:
    if p.endswith(".py"):
        py_compile.compile(p, doraise=True)
subprocess.run([sys.executable, "regression_check.py"], check=True)
print("Deployment-Struktur, Syntax und Regressionstests OK")
