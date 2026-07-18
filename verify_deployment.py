from pathlib import Path
import py_compile
import subprocess
import sys

required = [
    "app.py", "modules/__init__.py", "modules/risk_calculator.py",
    "modules/position_monitor.py", "modules/event_log.py", "modules/live_monitor.py",
    "modules/watchlist_storage.py", "modules/chart_overlays.py",
    "modules/radar_view.py", "modules/analysis_view.py", "modules/analysis_engine.py", "modules/cache_layer.py",
    "modules/market_data.py", "modules/ticker_resolver.py", "modules/scoring_engine.py", "regression_check.py",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit(f"Fehlende Dateien: {missing}")
for p in required:
    if p.endswith(".py"):
        py_compile.compile(p, doraise=True)
subprocess.run([sys.executable, "regression_check.py"], check=True)
print("Deployment-Struktur, Syntax und Regressionstests OK")
