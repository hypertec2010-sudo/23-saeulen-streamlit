# -*- coding: utf-8 -*-
"""
Headless Auto-Run for Capital Hill Score Modell.

Uses analysis_core.py directly instead of loading app.py, so UI/session-state
changes in Streamlit do not break GitHub auto-runs.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st


REQUIRED_ENV_VARS = [
    "GCP_CREDENTIALS",
    "LOG_SPREADSHEET_NAME",
    "TELEGRAM_TOKEN",
    "TELEGRAM_CHAT_ID",
]


def build_streamlit_secrets() -> dict:
    missing = [key for key in REQUIRED_ENV_VARS if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Fehlende Umgebungsvariablen: {', '.join(missing)}")

    secrets = {
        "GCP_CREDENTIALS": os.environ["GCP_CREDENTIALS"],
        "LOG_SPREADSHEET_NAME": os.environ["LOG_SPREADSHEET_NAME"],
        "TELEGRAM_TOKEN": os.environ["TELEGRAM_TOKEN"],
        "TELEGRAM_CHAT_ID": os.environ["TELEGRAM_CHAT_ID"],
    }

    if os.getenv("SCREENER_EMAIL"):
        secrets["SCREENER_EMAIL"] = os.environ["SCREENER_EMAIL"]
    if os.getenv("SCREENER_APP_PASSWORD"):
        secrets["SCREENER_APP_PASSWORD"] = os.environ["SCREENER_APP_PASSWORD"]
    if os.getenv("SCREENER_RECIPIENT"):
        secrets["SCREENER_RECIPIENT"] = os.environ["SCREENER_RECIPIENT"]

    return secrets


def patch_streamlit_secrets() -> None:
    st.secrets = build_streamlit_secrets()


def default_core_path() -> Path:
    return Path(os.getenv("ANALYSIS_CORE_PATH", "analysis_core.py")).resolve()


def berlin_now() -> datetime:
    return datetime.now(ZoneInfo("Europe/Berlin"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", default="", help="Optional fixed slot like 10:30, 15:40, 18:30, 22:10")
    parser.add_argument("--core-path", default=str(default_core_path()), help="Path to analysis_core.py")
    return parser.parse_args()


def load_analysis_namespace(core_path: Path) -> dict:
    namespace = {"__name__": "analysis_core"}
    source = core_path.read_text(encoding="utf-8")
    exec(compile(source, filename=str(core_path), mode="exec"), namespace, namespace)
    if "analyze_stock" not in namespace:
        available = sorted([k for k, v in namespace.items() if callable(v)])[:80]
        raise RuntimeError(f"analyze_stock wurde in analysis_core.py nicht gefunden. Verfügbare Funktionen: {available}")
    return namespace


def main() -> int:
    args = parse_args()
    patch_streamlit_secrets()

    core_path = Path(args.core_path)
    if not core_path.exists():
        raise FileNotFoundError(f"analysis_core.py nicht gefunden unter: {core_path}")

    namespace = load_analysis_namespace(core_path)

    from logging_utils import (
        append_auto_run_log,
        get_current_berlin_time,
        get_current_schedule_slot,
        get_due_watchlists_for_slot,
        get_watchlist_tickers,
    )
    from telegram_utils import send_watchlist_alerts

    analyze_stock = namespace["analyze_stock"]

    now_berlin = get_current_berlin_time() if "get_current_berlin_time" in locals() else berlin_now()
    slot_label = args.slot.strip() or get_current_schedule_slot(now_berlin)

    if not slot_label:
        print("Kein Slot aktiv. Vor dem ersten Slot des Tages wird nichts ausgeführt.")
        return 0

    due_df, due_err = get_due_watchlists_for_slot(slot_label)
    if due_err:
        raise RuntimeError(f"Due-Watchlists konnten nicht geladen werden: {due_err}")

    if due_df is None or due_df.empty:
        print(f"Für den Slot {slot_label} sind aktuell keine Watchlisten fällig.")
        return 0

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_sent = 0
    log_rows = []

    print(f"Starte Auto-Run für Slot {slot_label} | Berlin-Zeit {now_berlin.strftime('%Y-%m-%d %H:%M')}")

    for _, row in due_df.iterrows():
        wl_name = str(row.get("Watchlist_Name", "")).strip()
        wl_type = str(row.get("Watchlist_Type", "Watchlist")).strip() or "Watchlist"
        wl_alert_mode = str(row.get("Alert_Mode", "Standard")).strip() or "Standard"
        wl_freq = str(row.get("Check_Frequency", "4x täglich")).strip() or "4x täglich"

        tickers, tick_err = get_watchlist_tickers(wl_name)
        if tick_err:
            message = f"Ticker-Laden fehlgeschlagen: {tick_err}"
            print(f"[{wl_name}] {message}")
            log_rows.append({
                "Run_Timestamp": run_ts,
                "Berlin_Time": now_berlin.strftime("%Y-%m-%d %H:%M"),
                "Slot": slot_label,
                "Watchlist_Name": wl_name,
                "Watchlist_Type": wl_type,
                "Alert_Mode": wl_alert_mode,
                "Check_Frequency": wl_freq,
                "Ticker_Count": 0,
                "Analyzed_Count": 0,
                "Sent_Count": 0,
                "Status": "Fehler",
                "Message": message,
            })
            continue

        results = []
        analyze_errors = []
        for ticker in tickers:
            try:
                result = analyze_stock(
                    ticker=ticker,
                    horizon="Swing (1-4 Wochen)",
                    depot=10000,
                    risk_pct=1.0,
                    override=0.0,
                    buy_in_override=0.0,
                    smart_money_default=True,
                    strict_mode=True,
                )
                results.append(result)
            except Exception as exc:
                analyze_errors.append(f"{ticker}: {exc}")

        if results:
            ok, msg, sent_count, alert_diag = send_watchlist_alerts(
                results,
                wl_name,
                wl_type,
                wl_alert_mode,
                return_diagnostics=True,
            )
        else:
            ok, msg, sent_count, alert_diag = (
                False,
                "Keine auswertbaren Ergebnisse",
                0,
                {
                    "matched": 0,
                    "suppressed": 0,
                    "info_sent": 0,
                    "sent": 0,
                    "errors": [],
                    "reason": "Keine auswertbaren Ergebnisse",
                },
            )

        total_sent += int(sent_count or 0)

        full_msg = msg
        if analyze_errors:
            full_msg += " | Analysefehler: " + " ; ".join(analyze_errors[:2])

        diag_summary = (
            f"matched={int(alert_diag.get('matched', 0))}, "
            f"suppressed={int(alert_diag.get('suppressed', 0))}, "
            f"info_sent={int(alert_diag.get('info_sent', 0))}, "
            f"sent={int(alert_diag.get('sent', 0))}, "
            f"reason={alert_diag.get('reason', '-')}"
        )
        print(f"[{wl_name}] {full_msg} | {diag_summary}")

        log_rows.append({
            "Run_Timestamp": run_ts,
            "Berlin_Time": now_berlin.strftime("%Y-%m-%d %H:%M"),
            "Slot": slot_label,
            "Watchlist_Name": wl_name,
            "Watchlist_Type": wl_type,
            "Alert_Mode": wl_alert_mode,
            "Check_Frequency": wl_freq,
            "Ticker_Count": len(tickers),
            "Analyzed_Count": len(results),
            "Matched_Count": int(alert_diag.get("matched", 0)),
            "Suppressed_Count": int(alert_diag.get("suppressed", 0)),
            "Info_Sent_Count": int(alert_diag.get("info_sent", 0)),
            "Sent_Count": int(sent_count or 0),
            "Status": "OK" if ok else "Info",
            "Reason": str(alert_diag.get("reason", "-")),
            "Message": full_msg,
        })

    log_ok, log_msg = append_auto_run_log(log_rows)
    if log_ok:
        print(log_msg)
    else:
        print(f"Auto_Run_Log konnte nicht gespeichert werden: {log_msg}")

    print(f"Fertig. Insgesamt gesendete Telegram-Nachrichten: {total_sent}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        raise
