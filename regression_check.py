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
    "modules/trade_journal.py",
    "modules/event_log.py",
    "modules/live_monitor.py",
    "modules/watchlist_storage.py",
    "modules/chart_overlays.py",
    "modules/radar_view.py",
    "modules/analysis_view.py", "modules/analysis_engine.py", "modules/cache_layer.py",
    "modules/market_data.py", "modules/ticker_resolver.py", "modules/scoring_engine.py",
    "modules/storage/__init__.py", "modules/storage/base.py", "modules/storage/local_backend.py",
    "modules/storage/supabase_backend.py", "modules/storage/manager.py",
    "modules/storage/watchlist_repository.py", "modules/storage/migration.py",
    "migrate_storage.py",
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
        "modules.trade_journal",
        "modules.event_log",
        "modules.live_monitor",
        "modules.watchlist_storage",
        "modules.chart_overlays",
        "modules.radar_view",
        "modules.analysis_view",
        "modules.analysis_engine",
        "modules.cache_layer",
        "modules.market_data",
        "modules.ticker_resolver",
        "modules.scoring_engine",
        "modules.storage",
        "modules.storage.base",
        "modules.storage.local_backend",
        "modules.storage.supabase_backend",
        "modules.storage.manager",
        "modules.storage.watchlist_repository",
        "modules.storage.migration",
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


def test_trade_journal(mod) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mod.configure_context(
            base_dir=tmp,
            safe_float=lambda v, default=None: float(v) if v not in (None, "", "n/a", "-") else default,
            event_logger=lambda **kwargs: True,
        )
        mod._v270_reset_trade_journal()
        positions = {
            "AAPL": {
                "ticker": "AAPL", "name": "Apple Inc.", "entry": 100, "stop": 95,
                "initial_stop": 95, "target": 115, "shares": 10, "initial_shares": 10,
            }
        }
        partial = mod._v270_partial_exit(
            positions, watchlist_name="Test", ticker="AAPL", exit_price=105,
            exit_shares=4, exit_date="2026-01-01", note="Teilgewinn",
        )
        check(partial["ok"] is True, "Teilverkäuf konnte nicht gespeichert werden")
        check(partial["positions"]["AAPL"]["shares"] == 6, "Reststueckzahl nach Teilverkauf fehlerhaft")
        closed = mod._v270_close_position(
            partial["positions"], watchlist_name="Test", ticker="AAPL",
            exit_price=110, exit_date="2026-01-02", reason="Ziel erreicht",
        )
        check(closed["ok"] is True, "Position konnte nicht geschlossen werden")
        check("AAPL" not in closed["positions"], "Geschlossene Position wurde nicht entfernt")
        df = mod._v270_journal_entries_dataframe("Test")
        check(len(df) == 2, "Trade-Journal enthaelt unerwartete Anzahl Eintraege")
        summary = mod._v270_journal_summary(df)
        check(summary["closed_trades"] == 1, "Geschlossener Trade wird nicht gezaehlt")
        check(summary["partial_exits"] == 1, "Teilverkäuf wird nicht gezaehlt")
        check(summary["realized_pnl"] > 0, "Realisiertes P/L wurde nicht berechnet")
        check((Path(tmp) / ".trade_journal_v270.json").exists(), "Trade-Journal wurde nicht persistent gespeichert")


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
    check("v28.0" in source, "v28.0 Versionsstand fehlt in app.py")
    check("query" in source.lower() and "workspace" in source.lower(), "Workspace-Query-State nicht auffindbar")
    check("modules" in source and "radar_view" in source, "Radar-Modul ist nicht integriert")
    check("modules" in source and "live_monitor" in source, "Live-Monitor-Modul ist nicht integriert")





def test_storage_layer(mods) -> None:
    local_mod = mods["modules.storage.local_backend"]
    manager_mod = mods["modules.storage.manager"]
    repo_mod = mods["modules.storage.watchlist_repository"]

    with tempfile.TemporaryDirectory() as tmp:
        local = local_mod.LocalJsonBackend(tmp)
        manager = manager_mod.StorageManager(
            user_id="regression-user",
            local_backend=local,
            primary_backend=None,
            requested_backend="local",
        )
        check(manager.save_namespace("probe", {"value": 42}), "Lokaler Storage-Write fehlgeschlagen")
        check(manager.load_namespace("probe", {}).get("value") == 42, "Lokaler Storage-Read fehlgeschlagen")
        check(manager.delete_namespace("probe"), "Lokaler Storage-Delete fehlgeschlagen")
        check(manager.load_namespace("probe", None) is None, "Geloeschter Namespace ist noch vorhanden")

        repo = repo_mod.WatchlistRepository(manager)
        ok, _ = repo.create_watchlist("Testliste", "Watchlist", check_frequency="4x täglich")
        check(ok, "Watchlist konnte im Storage-Repository nicht erstellt werden")
        ok, _ = repo.add_entries_to_watchlist("Testliste", "Watchlist", ["AAPL", "MSFT"], check_frequency="4x täglich")
        check(ok, "Watchlist-Ticker konnten nicht gespeichert werden")
        tickers, err = repo.get_watchlist_tickers("Testliste")
        check(err is None and tickers == ["AAPL", "MSFT"], f"Unerwartete Watchlist-Ticker: {tickers}")
        ok, _ = repo.update_watchlist_alert_mode("Testliste", "Konservativ")
        check(ok and repo.get_watchlist_alert_mode("Testliste") == "Konservativ", "Alert-Modus nicht persistent")
        due, err = repo.get_due_watchlists_for_slot("15:40")
        check(err is None and len(due) == 1, "4x-taegliche Watchlist im 15:40-Slot nicht faellig")
        ok, _ = repo.remove_ticker_from_watchlist("Testliste", "AAPL")
        check(ok, "Ticker konnte nicht entfernt werden")
        tickers, _ = repo.get_watchlist_tickers("Testliste")
        check(tickers == ["MSFT"], "Ticker-Entfernung inkonsistent")

        health = manager.health_check()
        check(health.ok, f"Lokaler Storage-Healthcheck fehlgeschlagen: {health.error}")

        class BrokenRemote:
            name = "broken-remote"
            def load(self, user_id, namespace):
                from modules.storage.base import StorageResult
                return StorageResult(ok=False, error="offline", backend=self.name)
            def save(self, user_id, namespace, payload):
                from modules.storage.base import StorageResult
                return StorageResult(ok=False, error="offline", backend=self.name)
            def delete(self, user_id, namespace):
                from modules.storage.base import StorageResult
                return StorageResult(ok=False, error="offline", backend=self.name)
            def health_check(self):
                from modules.storage.base import StorageResult
                return StorageResult(ok=False, error="offline", backend=self.name)

        fallback_manager = manager_mod.StorageManager(
            user_id="fallback-user",
            local_backend=local_mod.LocalJsonBackend(Path(tmp) / "fallback"),
            primary_backend=BrokenRemote(),
            requested_backend="supabase",
        )
        check(fallback_manager.save_namespace("journal", {"entries": [1]}), "Fallback-Write fehlgeschlagen")
        check(fallback_manager.load_namespace("journal", {}).get("entries") == [1], "Fallback-Read fehlgeschlagen")
        check(fallback_manager.status().get("degraded") is True, "Remote-Ausfall wird nicht als degradiert markiert")


def test_phase4_modules(mods) -> None:
    cache = mods["modules.cache_layer"]
    market = mods["modules.market_data"]
    resolver = mods["modules.ticker_resolver"]
    scoring = mods["modules.scoring_engine"]
    analysis = mods["modules.analysis_engine"]
    check(cache.market_bucket(15) >= 0, "Cache-Bucket fehlerhaft")
    check(market.finite_positive(float("nan")) is None, "NaN-Filter fehlerhaft")
    check(abs(market.atr_percent(2, 100) - 2.0) < 1e-9, "ATR-Prozent fehlerhaft")
    check(resolver.normalize_ticker(" aapl ") == "AAPL", "Ticker-Normalisierung fehlerhaft")
    check("SAP.DE" in resolver.candidate_variants("SAP"), "Ticker-Varianten fehlerhaft")
    check(scoring.clip_score(120) == 100.0, "Score-Clipping fehlerhaft")
    check(abs(scoring.weighted_score([(80, 3), (50, 1)]) - 72.5) < 1e-9, "Gewichteter Score fehlerhaft")
    result = analysis.analyze_stock(
        ticker="AAPL", horizon="Test", depot=10000, risk_pct=1, override=None,
        buy_in_override=None, smart_money_default=None, strict_mode=False,
        core_engine=lambda **kwargs: {"ticker": kwargs["ticker"], "info": {"quoteType": "EQUITY"}},
        legacy_engine=lambda **kwargs: {}, asset_mode="Auto",
    )
    check(result.get("Asset_Typ") == "Aktie", "Analyse-Facade Asset-Typ fehlerhaft")


def main() -> None:
    compile_and_parse()
    mods = import_modules()
    test_risk_calculator(mods["modules.risk_calculator"])
    test_live_monitor(mods["modules.live_monitor"])
    test_position_monitor(mods["modules.position_monitor"])
    test_trade_journal(mods["modules.trade_journal"])
    test_event_log(mods["modules.event_log"])
    test_radar_view(mods["modules.radar_view"])
    test_phase4_modules(mods)
    test_storage_layer(mods)
    test_navigation_guards()
    print("v28.0 Regressionstest: ALLE PRUEFUNGEN ERFOLGREICH")


if __name__ == "__main__":
    main()
