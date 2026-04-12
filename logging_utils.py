import json
import os
from datetime import datetime

import gspread
import numpy as np
import pandas as pd
import streamlit as st


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
