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
    "legacy_app.py",
    "modules/app_shell.py",
    "modules/page_runtime.py",
    "pages/analysis.py",
    "pages/radar.py",
    "pages/watchlists.py",
    "pages/positions.py",
    "pages/trade_journal.py",
    "modules/__init__.py",
    "modules/risk_calculator.py",
    "modules/position_monitor.py",
    "modules/trade_journal.py",
    "modules/event_log.py",
    "modules/live_monitor.py",
    "modules/watchlist_storage.py",
    "modules/chart_overlays.py",
    "modules/radar_view.py",
    "modules/analysis_view.py", "modules/analysis_engine.py", "modules/legacy_analysis_core.py", "modules/cache_layer.py",
    "modules/market_data.py", "modules/ticker_resolver.py", "modules/scoring_engine.py",
    "modules/live_refresh_policy.py",
    "modules/live_screener_snapshot.py",
    "modules/live_change_explainer.py",
    "modules/storage/__init__.py", "modules/storage/base.py", "modules/storage/local_backend.py",
    "modules/storage/supabase_backend.py", "modules/storage/manager.py",
    "modules/storage/watchlist_repository.py", "modules/storage/migration.py",
    "modules/domain/__init__.py", "modules/domain/models.py",
    "modules/repositories/__init__.py", "modules/repositories/base.py",
    "modules/repositories/position_repository.py", "modules/repositories/trade_journal_repository.py",
    "modules/repositories/event_repository.py", "modules/repositories/registry.py",
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
        "modules.page_runtime",
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
        "modules.legacy_analysis_core",
        "modules.cache_layer",
        "modules.market_data",
        "modules.ticker_resolver",
        "modules.scoring_engine",
        "modules.live_refresh_policy",
        "modules.live_screener_snapshot",
        "modules.live_change_explainer",
        "modules.storage",
        "modules.storage.base",
        "modules.storage.local_backend",
        "modules.storage.supabase_backend",
        "modules.storage.manager",
        "modules.storage.watchlist_repository",
        "modules.storage.migration",
        "modules.domain",
        "modules.domain.models",
        "modules.repositories",
        "modules.repositories.base",
        "modules.repositories.position_repository",
        "modules.repositories.trade_journal_repository",
        "modules.repositories.event_repository",
        "modules.repositories.registry",
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


def test_live_change_explainer(mod) -> None:
    text = mod.build_change_explanation(
        {
            "ampel": "🟡", "status": "Nahe am Trigger", "price": 100.0,
            "live_score": "62/100", "radar_bucket": "Nahe am Trigger",
            "entry_hard_gate": False, "final_release_ok": True,
            "timing_component": 68, "conf_component": 64,
        },
        {
            "ampel": "🔴", "status": "Setup blockiert", "price": 100.03,
            "live_score": "38/100", "radar_bucket": "Warnsignale / meiden",
            "entry_hard_gate": True, "final_release_ok": False,
            "timing_component": 41, "conf_component": 46,
        },
        "Verschlechtert",
    )
    check("Kurs nahezu unverändert" in text, "Kurskontext der Statuswechsel-Erklaerung fehlt")
    check("hartes Einstiegsgate" in text, "Harte Gate-Ursache wird nicht erklaert")
    check("Live-Score 62→38" in text, "Score-Differenz wird nicht erklaert")


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
    trailed = mod._v244_calc_trade_state(
        {"entry": 100, "stop": 103, "initial_stop": 95, "shares": 10},
        {"Kurs": 110},
    )
    check(abs(trailed["R-Multiple"] - 2.0) < 1e-9, "R-Multiple nutzt faelschlich den nachgezogenen Stop")
    check(trailed["R-Basis-Stop"] == 95.0, "Initial-Stop wird nicht als R-Basis verwendet")


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
    entry_source = (ROOT / "app.py").read_text(encoding="utf-8")
    shell_source = (ROOT / "modules/app_shell.py").read_text(encoding="utf-8")
    runtime_source = (ROOT / "modules/page_runtime.py").read_text(encoding="utf-8")
    legacy_source = (ROOT / "legacy_app.py").read_text(encoding="utf-8")

    check("render_navigation" in entry_source, "Multipage-Einstieg fehlt in app.py")
    check("st.navigation" in shell_source and "st.Page" in shell_source, "Native Streamlit-Navigation fehlt")
    check("pages/trade_journal.py" in shell_source, "Trade-Journal-Seite ist nicht registriert")
    check("workspace_mode" in runtime_source and "watchlist_cockpit_area_v2413" in runtime_source, "Workspace-Bruecke unvollstaendig")
    check("CAPITAL_HILL_MULTIPAGE" in runtime_source and "CAPITAL_HILL_MULTIPAGE" in legacy_source, "Multipage-Bootstrap-Guard fehlt")
    check('APP_VERSION = "v28.4.3"' in legacy_source, "v28.4.3 Versionsstand fehlt in legacy_app.py")
    check(len(entry_source.splitlines()) < 80, "app.py ist nicht als schlanker Einstiegspunkt umgesetzt")
    check("run_every=_native_refresh_poll_seconds_v2832" in legacy_source, "60-Sekunden-Heartbeat fuer Live-Refresh fehlt")
    check("_live_refresh_policy.evaluate_refresh" in legacy_source, "Testbare Refresh-Policy ist nicht verdrahtet")
    check("_native_refresh_poll_seconds_v2832 = 60" in legacy_source, "Heartbeat-Intervall ist nicht auf 60 Sekunden gesetzt")
    check("v246_live_monitor_cache" in legacy_source and "Nächster Auto-Scan" in legacy_source, "Cache-basierter Refresh-Status fehlt")
    check("_live_screener_snapshot.load_snapshot" in legacy_source, "Persistenter Live-Snapshot ist nicht verdrahtet")
    check("Mobile-Modus" in legacy_source and "v2842-mobile-card" in legacy_source, "Mobile-Screener-Oberfläche fehlt")
    check("Warum geändert?" in legacy_source, "Statuswechsel-Erklaerung fehlt in der Live-Screener-Oberfläche")
    check('st.rerun(scope="app")' not in legacy_source, "Veralteter expliziter App-Scope im Refresh-Fragment vorhanden")
    check("if page_changed:" in runtime_source and "_clear_legacy_workspace_query()" in runtime_source, "Query-Cleanup ist nicht an echte Seitenwechsel gebunden")

    expected_pages = {
        "pages/analysis.py": "Sofortanalyse",
        "pages/radar.py": "Kandidaten-Radar",
        "pages/watchlists.py": "Watchlisten",
        "pages/positions.py": "Positionen",
        "pages/trade_journal.py": "Trade-Journal",
    }
    for rel, marker in expected_pages.items():
        page_source = (ROOT / rel).read_text(encoding="utf-8")
        check("run_workspace_page" in page_source and marker in page_source, f"Seitencontroller unvollstaendig: {rel}")





def test_cockpit_navigation_state(mod) -> None:
    original_st = mod.st
    try:
        fake_st = types.SimpleNamespace(session_state={})
        mod.st = fake_st

        changed = mod._activate_page_context(
            "Watchlisten", "📡 Live-Screener", "Watchlisten"
        )
        check(changed is True, "Erster Seitenaufruf wird nicht als Seitenwechsel erkannt")
        check(
            fake_st.session_state.get("watchlist_cockpit_area_v2413") == "📡 Live-Screener",
            "Startbereich der Watchlisten-Seite wurde nicht gesetzt",
        )

        # Simuliert eine manuelle Cockpit-Auswahl und den darauf folgenden
        # Streamlit-Rerun derselben nativen Seite. Die Auswahl muss erhalten bleiben.
        fake_st.session_state["watchlist_cockpit_area_v2413"] = "📓 Trade-Journal"
        changed = mod._activate_page_context(
            "Watchlisten", "📡 Live-Screener", "Watchlisten"
        )
        check(changed is False, "Widget-Rerun derselben Seite wird faelschlich als Seitenwechsel erkannt")
        check(
            fake_st.session_state.get("watchlist_cockpit_area_v2413") == "📓 Trade-Journal",
            "Cockpit-Auswahl wird beim Widget-Rerun auf den Startbereich zurueckgesetzt",
        )

        # Beim echten Wechsel auf die Positionsseite soll deren Startbereich gelten.
        changed = mod._activate_page_context(
            "Positionen", "📌 Positionen / Exit", "Positionen / Exit"
        )
        check(changed is True, "Echter nativer Seitenwechsel wird nicht erkannt")
        check(
            fake_st.session_state.get("watchlist_cockpit_area_v2413") == "📌 Positionen / Exit",
            "Startbereich der neu geoeffneten Seite wurde nicht gesetzt",
        )
    finally:
        mod.st = original_st


def test_domain_models_and_repositories(mods) -> None:
    models = mods["modules.domain.models"]
    local_mod = mods["modules.storage.local_backend"]
    manager_mod = mods["modules.storage.manager"]
    registry_mod = mods["modules.repositories.registry"]

    position = models.Position.from_legacy_dict({
        "ticker": "aapl", "name": "Apple", "entry": "100", "stop": 95,
        "shares": "10", "custom_flag": "legacy",
    })
    check(position.ticker == "AAPL", "Domain-Modell normalisiert Ticker nicht")
    check(position.unit_risk == 5.0, "Domain-Modell berechnet Positionsrisiko falsch")
    check(position.to_legacy_dict().get("custom_flag") == "legacy", "Unbekannte Legacy-Felder gehen verloren")

    journal = models.JournalEntry.from_legacy_dict({
        "ID": "j1", "Watchlist": "Test", "Ticker": "aapl", "Typ": "Trade-Notiz",
        "Stück": 3, "Notiz": "Test",
    })
    check(journal.ticker == "AAPL" and journal.shares == 3, "Journal-Modell normalisiert Daten nicht")
    check(journal.to_legacy_dict().get("Notiz") == "Test", "Journal-Legacy-Konvertierung fehlerhaft")

    with tempfile.TemporaryDirectory() as tmp:
        manager = manager_mod.StorageManager(
            user_id="repo-user",
            local_backend=local_mod.LocalJsonBackend(tmp),
            primary_backend=None,
            requested_backend="local",
        )
        repos = registry_mod.create_repository_registry(manager)
        check(repos.positions.save_for_watchlist("Test", {"AAPL": position.to_legacy_dict()}), "PositionRepository Write fehlgeschlagen")
        loaded = repos.positions.get_for_watchlist("Test")
        check(loaded.get("AAPL", {}).get("entry") == 100.0, "PositionRepository Read fehlerhaft")

        check(repos.trade_journal.append(journal), "TradeJournalRepository Append fehlgeschlagen")
        journal_store = repos.trade_journal.load_store()
        check(len(journal_store.get("entries", [])) == 1, "TradeJournalRepository inkonsistent")

        event = models.SignalEvent.from_legacy_dict({
            "Zeit": "01.01.2026 10:00:00", "Watchlist": "Test", "Ticker": "aapl",
            "Ereignis": "Testsignal", "Status": "Gruen",
        })
        check(repos.events.save_store({"events": [event.to_legacy_dict()], "last_signatures": {"x": "y"}}), "EventRepository Write fehlgeschlagen")
        event_store = repos.events.load_store()
        check(event_store["events"][0]["Ticker"] == "AAPL", "EventRepository normalisiert Ticker nicht")
        check(event_store["last_signatures"].get("x") == "y", "Event-Signaturen gehen verloren")


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


def test_analysis_core_extraction(mods) -> None:
    legacy_core = mods["modules.legacy_analysis_core"]
    legacy_source = (ROOT / "legacy_app.py").read_text(encoding="utf-8")
    core_source = (ROOT / "modules/legacy_analysis_core.py").read_text(encoding="utf-8")

    check("def _legacy_analyze_stock(" not in legacy_source, "Analyse-Pipeline liegt noch in legacy_app.py")
    check("def _legacy_analyze_stock_impl(" in core_source, "Extrahierte Analyse-Pipeline fehlt")
    check("legacy_analysis_core.legacy_analyze_stock" in legacy_source, "Analyse-Fallback ist nicht mit dem Modul verbunden")
    check(len(legacy_source.splitlines()) < 25_000, "legacy_app.py wurde durch die Core-Extraktion nicht ausreichend verkleinert")

    legacy_core.reset_context()
    status = legacy_core.context_status()
    check(status["required"] >= 90, "Kontextmanifest der Analyse-Pipeline ist unerwartet klein")
    check(status["configured"] == 0, "Analyse-Kontext wurde beim Import unerwartet vorbelegt")
    try:
        legacy_core.legacy_analyze_stock("AAPL", "Test", 10_000, 1, 0, None, None, False)
    except RuntimeError as exc:
        check("Analyse-Kontext ist unvollstaendig" in str(exc), "Fehlender Kontext wird nicht klar gemeldet")
    else:
        raise AssertionError("Legacy-Core muss ohne gebundene Abhaengigkeiten abbrechen")

    legacy_core.configure_context({"np": object(), "pd": object()})
    status = legacy_core.context_status()
    check(status["configured"] == 2, "Explizite Kontextbindung funktioniert nicht")
    legacy_core.reset_context()


def test_live_refresh_policy(mods) -> None:
    from datetime import datetime, timedelta

    policy = mods["modules.live_refresh_policy"]
    expected = policy.build_cache_key("Test", ["aapl"], "Charttechnik", "Swing")
    check(expected["tickers"] == ("AAPL",), "Refresh-Policy normalisiert Ticker nicht")
    schedule = policy.build_schedule_key("Test", ["AAPL"], "Charttechnik", "Swing", 900)
    check(schedule.endswith("|900"), "Refresh-Schedule-Key ist inkonsistent")

    now = datetime(2026, 7, 27, 10, 0, 0)
    waiting = policy.evaluate_refresh(
        now=now,
        cache={"key": expected, "ts": (now - timedelta(minutes=5)).isoformat()},
        expected_cache_key=expected,
        interval_seconds=900,
    )
    check(waiting.due is False and waiting.remaining_seconds >= 599, "Refresh wird zu frueh faellig")
    due = policy.evaluate_refresh(
        now=now,
        cache={"key": expected, "ts": (now - timedelta(minutes=15)).isoformat()},
        expected_cache_key=expected,
        interval_seconds=900,
    )
    check(due.due is True, "Faelliger Refresh wird nicht erkannt")
    check(policy.trigger_is_recent(now=now, last_trigger=(now - timedelta(seconds=30)).isoformat()), "Trigger-Cooldown fehlerhaft")
    check(policy.reconnect_grace_remaining(now=now, restored_at=(now - timedelta(seconds=20)).isoformat()) == 100, "Reconnect-Schutzpause fehlerhaft")


def test_live_screener_snapshot(mods) -> None:
    import pandas as pd

    snapshot = mods["modules.live_screener_snapshot"]

    class MemoryStorage:
        def __init__(self):
            self.data = {}
        def load_namespace(self, namespace, default=None):
            return self.data.get(namespace, default)
        def save_namespace(self, namespace, payload):
            self.data[namespace] = payload
            return True

    storage = MemoryStorage()
    key = {
        "watchlist": "Mobil",
        "tickers": ("SAP.DE", "AAPL"),
        "style": "Charttechnik",
        "horizon": "Kurzfrist / Trading",
    }
    cache = {
        "key": key,
        "ts": "2026-08-05T11:30:00",
        "live_df": pd.DataFrame([{"Ticker": "SAP.DE", "Kurs": 161.34}]),
        "live_errors": pd.DataFrame(),
    }
    check(snapshot.save_snapshot(storage, cache, ui_state={"mobile_mode": True}), "Live-Snapshot konnte nicht gespeichert werden")
    restored = snapshot.load_snapshot(storage, key)
    check(restored is not None, "Live-Snapshot konnte nicht geladen werden")
    check(restored["cache"]["key"] == key, "Live-Snapshot Cache-Key inkonsistent")
    check(restored["cache"]["live_df"].iloc[0]["Ticker"] == "SAP.DE", "Live-Snapshot DataFrame inkonsistent")


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
    test_live_change_explainer(mods["modules.live_change_explainer"])
    test_position_monitor(mods["modules.position_monitor"])
    test_trade_journal(mods["modules.trade_journal"])
    test_event_log(mods["modules.event_log"])
    test_radar_view(mods["modules.radar_view"])
    test_phase4_modules(mods)
    test_live_refresh_policy(mods)
    test_live_screener_snapshot(mods)
    test_analysis_core_extraction(mods)
    test_domain_models_and_repositories(mods)
    test_storage_layer(mods)
    test_navigation_guards()
    test_cockpit_navigation_state(mods["modules.page_runtime"])
    print("v28.4.3 Regressionstest: ALLE PRUEFUNGEN ERFOLGREICH")


if __name__ == "__main__":
    main()
