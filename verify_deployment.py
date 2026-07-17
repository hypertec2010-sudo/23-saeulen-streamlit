from pathlib import Path
import py_compile

root = Path(__file__).resolve().parent
required = [
    root / "app.py",
    root / "modules" / "__init__.py",
    root / "modules" / "risk_calculator.py",
    root / "modules" / "position_monitor.py",
    root / "modules" / "event_log.py",
]
missing = [str(p.relative_to(root)) for p in required if not p.exists()]
if missing:
    raise SystemExit("Fehlende Dateien: " + ", ".join(missing))
for path in required:
    if path.suffix == ".py":
        py_compile.compile(str(path), doraise=True)
print("Deployment-Struktur und Python-Syntax sind in Ordnung.")
