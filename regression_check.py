from __future__ import annotations

import ast
import importlib
import json
import py_compile
import sys
import tempfile
import types
from pathlib import Path

# Minimaler Streamlit-Stub fuer lokale Regressionstests in Umgebungen ohne Streamlit.
# Auf Streamlit Cloud wird das echte Paket verwendet.
try:
    import streamlit  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("streamlit")
    stub.session_state = {}
    sys.modules["streamlit"] = stub

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_FILES = [
    "app.py",
    "modules/__init__.py",
    "modules/risk_calculator.py",
    "modules/position_monitor.py",
    "modules/event_log.py",
    "modules/live_monitor.py",
    "modules/watchlist_storage.py",
    "modules/chart_overlays.py",
    "modules/radar_view.py",
    "modules/analysis_view.py",
]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def compile_and_parse() -> None:
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        check(path.exists(), f"Fehlende Datei: {rel}")
        py_compile.compile(str(path), doraise=True)
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=rel)


def import_modules():
    names = [
        "modules.risk_calculator",
        "modules.position_monitor",
        "modules.event_log",
        "modules.live_monitor",
        "modules.watchlist_storage",
        "modules.chart_overlays",
        "modules.radar_view",
        "modules.analysis_view",
    ]
    return {name: importlib.import_module(name) for name in names}


def test_risk_calculator(mod) -> None:
    result = mod._v230_calculate_position_size(
        entry=100, stop=95, target=115, account_size=50_000,
        risk_pct=0.5, max_position_pct=20,
    )
    check(result["ok"] is True, "Risiko-Rechner liefert kein gueltiges Ergebnis")
    check(result["shares"] == 50, f"Unerwartete Stueckzahl: {result['shares']}")
    check(abs(result["crv"] - 3.0) < 1e-9, "CRV-Berechnung fehlerhaft")
    check(mod._v2410_infer_quote_currency("AAPL") == "USD", "USD-Erkennung fehlerhaft")
    check(mod._v2410_infer_quote_currency("SAP.DE") == "EUR", "EUR-Erkennung fehlerhaft")
    invalid = mod._v230_calculate_position_size(100, 105, 120, 50_000, 0.5)
    check(invalid["ok"] is False, "Stop oberhalb Entry muss abgelehnt werden")


def test_live_monitor(mod) -> None:
    check(mod._v220_live_status_rank("🟢", "Trigger aktiv") < mod._v220_live_status_rank("🟡", "Vorbereiten"), "Ampel-Rangfolge fehlerhaft")
    check(mod._v220_live_change_label("⚪", "Beobachten", "🟡", "Nahe dran") == "Verbessert", "Statuswechsel-Erkennung fehlerhaft")
    first_green = mod._v237_apply_live_signal_hysteresis(
        {"Ampel": "🟢", "Status": "Trendfolge aktiv", "Live-Score": "75/100", "Grund": "-", "Nächste Handlung": "-"},
        {},
    )
    check(first_green["Ampel"] == "🟡", "Knappes erstes Gruensignal muss bestaetigt werden")
    strong_green = mod._v237_apply_live_signal_hysteresis(
        {"Ampel": "🟢", "Status": "Kurzfrist-Trigger aktiv", "Live-Score": "82/100", "Grund": "-", "Nächste Handlung": "-"},
        {},
    )
    check(strong_green["Ampel"] == "🟢", "Sehr starkes Gruensignal wird faelschlich blockiert")
    strong_green["Bestätigungen"] = "2x"
    state, _ = mod._v240_live_trade_state(strong_green)
    check(state in {"Trigger aktiv", "Armed / bereit", "Armed / Bestätigung offen"}, "Trade-State fuer Gruensignal unplausibel")
    red = mod._v237_apply_live_signal_hysteresis(
        {"Ampel": "🔴", "Status": "Invalidiert", "Live-Score": "10/100", "Grund": "Stop gebrochen", "Nächste Handlung": "Kein Kauf"},
        {"ampel": "🟢", "status": "Trigger aktiv", "live_score": "80/100"},
    )
    check(red["Ampel"] == "🔴" and red["Signal-Stabilität"] == "Defensiv", "Invalidierung darf nicht weichgezeichnet werden")


def test_position_monitor(mod) -> None:
    mod.configure_context(
        safe_float=lambda v, default=None: float(v) if v not in (None, "", "n/a", "-") else default,
        price_text=lambda v, digits=2: f"{float(v):.{digits}f}" if v is not None else "n/a",
    )
    one_r = mod._v244_calc_trade_state(
        {"entry": 100, "stop": 95, "target": 115, "shares": 10},
        {"Kurs": 105},
    )
    check("1R erreicht" in one_r["Status"], "1R-Erkennung fehlerhaft")
    stopped = mod._v244_calc_trade_state(
        {"entry": 100, "stop": 95, "target": 115, "shares": 10},
        {"Kurs": 94},
    )
    check(stopped["Ampel"] == "🔴", "Stop-Erkennung fehlerhaft")


def test_event_log(mod) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mod.configure_context(base_dir=tmp)
        mod._v2416_reset_events()
        created = mod._v2416_log_event(
            event_type="Testsignal", ticker="AAPL", watchlist_name="Test",
            source="Regression", status="Gruen", price=100,
            signature="regression-aapl-green",
        )
        check(created is True, "Event konnte nicht gespeichert werden")
        duplicate = mod._v2416_log_event(
            event_type="Testsignal", ticker="AAPL", watchlist_name="Test",
            source="Regression", status="Gruen", price=100,
            signature="regression-aapl-green",
        )
        check(duplicate is False, "Event-Deduplizierung fehlerhaft")
        df = mod._v2416_events_dataframe("Test")
        check(len(df) == 1, "Event-Log enthaelt unerwartete Anzahl Eintraege")
        store_path = Path(tmp) / ".signal_trade_event_log_v2416.json"
        check(store_path.exists(), "Event-Log wurde nicht persistent gespeichert")
        data = json.loads(store_path.read_text(encoding="utf-8"))
        check(len(data.get("events", [])) == 1, "Persistenter Event-Log inkonsistent")


def test_radar_view(mod) -> None:
    check(mod.radar_score_badge(80).startswith("🟢"), "Radar-Score-Badge fehlerhaft")
    check(mod.radar_trigger_badge("Aktiv").startswith("🟢"), "Radar-Trigger-Badge fehlerhaft")


def test_navigation_guards() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    check("v25.5" in source or "v25.6" in source, "Radar-Navigationsfix fehlt in app.py")
    check("query" in source.lower() and "workspace" in source.lower(), "Workspace-Query-State nicht auffindbar")
    check("modules" in source and "radar_view" in source, "Radar-Modul ist nicht integriert")
    check("modules" in source and "live_monitor" in source, "Live-Monitor-Modul ist nicht integriert")


def main() -> None:
    compile_and_parse()
    mods = import_modules()
    test_risk_calculator(mods["modules.risk_calculator"])
    test_live_monitor(mods["modules.live_monitor"])
    test_position_monitor(mods["modules.position_monitor"])
    test_event_log(mods["modules.event_log"])
    test_radar_view(mods["modules.radar_view"])
    test_navigation_guards()
    print("v25.6 Regressionstest: ALLE PRUEFUNGEN ERFOLGREICH")


if __name__ == "__main__":
    main()
