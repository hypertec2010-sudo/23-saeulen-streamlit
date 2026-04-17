import json
import os
from datetime import datetime

import gspread
import numpy as np
import pandas as pd
import streamlit as st


def build_export_row(result):
    market_info = result.get("market_info", {}) or {}
    confidence_info = result.get("confidence_info", {}) or {}
    run_id = st.session_state.get("current_run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    return {
        "Run-ID": run_id,
        "Export-Zeitpunkt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": result.get("ticker", "-"),
        "Name": result.get("name", "-"),
        "Modus": result.get("mode_label", "-"),
        "Setup-Typ": result.get("setup_type", "-"),
        "Valides Setup": "Ja" if result.get("valid_trade_setup", False) else "Nein",
        "Investment-Attraktivität": result.get("investment_case_score", np.nan),
        "Einstieg jetzt attraktiv?": result.get("trading_case_score", np.nan),
        "Trade-Struktur": result.get("tradeability_score", np.nan),
        "Setup-Confidence": result.get("setup_confidence", np.nan),
        "Entry-Lage": result.get("entry_quality", "-"),
        "Entry-Zone": result.get("suggested_entry_zone", "-"),
        "Marktregime": result.get("market_regime_label", "-") if "market_regime_label" in result else (market_info.get("regime", "UNBEKANNT") if isinstance(market_info, dict) else "-"),
        "Benchmark": result.get("benchmark_label", "-"),
        "Company Quality": result.get("company", np.nan),
        "Setup Quality": result.get("setup_adj", np.nan),
        "Investment Score": result.get("investment", np.nan),
        "Kurzfrist-Timing": result.get("tb_score_100", np.nan),
        "TradingBoard Score": result.get("tb_score", np.nan),
        "Kurs": result.get("price", np.nan),
        "Stop": result.get("stop_used", np.nan),
        "Stop-Herleitung": result.get("stop_source", "-"),
        "TP1": result.get("tp1", np.nan),
        "TP1-Herleitung": result.get("tp1_source", "-"),
        "TP2": result.get("tp2", np.nan),
        "TP2-Herleitung": result.get("tp2_source", "-"),
        "TP3": result.get("tp3", np.nan),
        "TP3-Herleitung": result.get("tp3_source", "-"),
        "Primärziel aus Setup": result.get("technical_target_1", np.nan),
        "Sekundärziel aus Setup": result.get("technical_target_2", np.nan),
        "CRV": result.get("crv", np.nan),
        "Exit-Score": result.get("exit_score", np.nan),
        "Exit-Score-Text": result.get("exit_score_text", "-"),
        "Exit-Aktion": result.get("exit_action", "-"),
        "Exit-Hauptgrund": result.get("exit_reason_top", "-"),
        "Trendbruch-Score": result.get("trend_break_score", np.nan),
        "Momentum-Kollaps-Score": result.get("momentum_collapse_score", np.nan),
        "Relative-Schwäche-Score": result.get("relative_weakness_score", np.nan),
        "Distributions-Score": result.get("distribution_score", np.nan),
        "Exit-Trigger-Score": result.get("exit_trigger_score", np.nan),
        "Positionsgröße": result.get("pos_size", np.nan),
        "Risiko EUR": result.get("risk_eur", np.nan),
        "Handlung": result.get("position_action", result.get("emp", "-")),
        "Trigger-Status": result.get("trigger_status", "-"),
        "Watchlist-Priorität": result.get("watchlist_priority", "-"),
        "Nächster Trigger": result.get("next_trigger", "-"),
        "Trigger-Begründung": result.get("trigger_reason", "-"),
        "Top Red Flag": result.get("top_red_flag", "-"),
        "Kurzfazit": result.get("short_thesis", result.get("decision_summary", "-")),
        "Fundamental-Confidence": round(confidence_info.get("coverage", 0) * 100),
    }


def build_export_df(results):
    rows = [build_export_row(r) for r in results]
    return pd.DataFrame(rows)


def get_secret_or_env(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def get_gsheet_client():
    creds_json_str = get_secret_or_env("GCP_CREDENTIALS")
    if not creds_json_str:
        return None, "GCP_CREDENTIALS fehlt"
    try:
        creds = json.loads(creds_json_str)
        gc = gspread.service_account_from_dict(creds)
        return gc, None
    except Exception as e:
        return None, f"GCP_CREDENTIALS ungueltig: {e}"


def get_or_create_worksheet(spreadsheet, sheet_name, rows=5000, cols=80):
    try:
        return spreadsheet.worksheet(sheet_name)
    except Exception:
        return spreadsheet.add_worksheet(title=sheet_name, rows=str(rows), cols=str(cols))


def append_df_to_gsheet(df, worksheet_name="Analysis_Log"):
    if df is None or df.empty:
        if worksheet_name == "Trigger_Log":
            return False, "Keine Trigger-Kandidaten in diesem Lauf gefunden."
        return False, "Keine Daten zum Schreiben vorhanden."

    client, err = get_gsheet_client()
    if client is None:
        return False, err

    spreadsheet_name = get_secret_or_env("LOG_SPREADSHEET_NAME", "Capital_Hill_Log")
    if not spreadsheet_name:
        return False, "LOG_SPREADSHEET_NAME fehlt"

    try:
        sh = client.open(spreadsheet_name)
        ws = get_or_create_worksheet(sh, worksheet_name, rows=max(5000, len(df) + 20), cols=max(80, len(df.columns) + 5))

        existing_values = ws.get_all_values()
        has_header = len(existing_values) > 0

        if not has_header:
            ws.append_row(df.columns.astype(str).tolist(), value_input_option="USER_ENTERED")

        rows = df.fillna("").astype(str).values.tolist()
        for row in rows:
            ws.append_row(row, value_input_option="USER_ENTERED")

        if worksheet_name == "Analysis_Log":
            if len(rows) == 1:
                return True, "Einzelanalyse erfolgreich in Google Sheets gespeichert."
            return True, f"{len(rows)} Analysen erfolgreich in Google Sheets gespeichert."
        if worksheet_name == "Trigger_Log":
            if len(rows) == 1:
                return True, "1 Trigger-Kandidat erfolgreich in Google Sheets gespeichert."
            return True, f"{len(rows)} Trigger-Kandidaten erfolgreich in Google Sheets gespeichert."
        return True, f"{len(rows)} Zeilen nach {worksheet_name} geschrieben"
    except Exception as e:
        return False, str(e)


def build_trigger_log_df(results):
    rows = []
    for r in results:
        if r.get("mode_label") == "Position":
            continue
        if r.get("trigger_status") in {"Aktiv", "Nahe dran"} or r.get("watchlist_priority") == "Hoch":
            rows.append({
                "Run-ID": st.session_state.get("current_run_id", datetime.now().strftime("%Y%m%d_%H%M%S")),
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Ticker": r.get("ticker", "-"),
                "Name": r.get("name", "-"),
                "Modus": r.get("mode_label", "-"),
                "Handlung": r.get("emp", "-"),
                "Trigger-Status": r.get("trigger_status", "-"),
                "Watchlist-Priorität": r.get("watchlist_priority", "-"),
                "Einstieg jetzt attraktiv?": r.get("trading_case_score", np.nan),
                "Trade-Struktur": r.get("tradeability_score", np.nan),
                "Setup-Confidence": r.get("setup_confidence", np.nan),
                "Entry-Lage": r.get("entry_quality", "-"),
                "Marktregime": market_regime_label((r.get("market_info", {}) or {}).get("regime", "UNBEKANNT")),
                "Top Red Flag": r.get("top_red_flag", "-"),
                "Kurzfazit": r.get("short_thesis", r.get("decision_summary", "-")),
            })
    return pd.DataFrame(rows)
