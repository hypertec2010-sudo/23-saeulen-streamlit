import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread
import numpy as np
import pandas as pd
import streamlit as st



def market_regime_display(regime):
    regime = str(regime or "").upper()
    if regime == "POSITIV":
        return "Positiv"
    if regime == "NEGATIV":
        return "Negativ"
    if regime == "NEUTRAL":
        return "Neutral"
    return "Unbekannt"


def open_log_spreadsheet():
    client, err = get_gsheet_client()
    if client is None:
        return None, err

    spreadsheet_name = get_secret_or_env("LOG_SPREADSHEET_NAME", "Capital_Hill_Log")
    if not spreadsheet_name:
        return None, "LOG_SPREADSHEET_NAME fehlt"

    try:
        sh = client.open(spreadsheet_name)
        return sh, None
    except Exception as e:
        return None, str(e)


def load_watchlists_df():
    sh, err = open_log_spreadsheet()
    if sh is None:
        return pd.DataFrame(columns=["Watchlist_Name", "Watchlist_Type", "Ticker", "Added_At", "Alert_Mode", "Check_Frequency"]), err

    try:
        ws = get_or_create_worksheet(sh, "Watchlists", rows=2000, cols=10)
        values = ws.get_all_values()
        if not values:
            headers = ["Watchlist_Name", "Watchlist_Type", "Ticker", "Added_At", "Alert_Mode", "Check_Frequency"]
            ws.append_row(headers, value_input_option="USER_ENTERED")
            return pd.DataFrame(columns=headers), None

        headers = values[0]
        rows = values[1:]
        if not rows:
            return pd.DataFrame(columns=headers), None

        df = pd.DataFrame(rows, columns=headers)
        for col in ["Watchlist_Name", "Watchlist_Type", "Ticker", "Added_At", "Alert_Mode", "Check_Frequency"]:
            if col not in df.columns:
                df[col] = ""
        return df, None
    except Exception as e:
        return pd.DataFrame(columns=["Watchlist_Name", "Watchlist_Type", "Ticker", "Added_At", "Alert_Mode", "Check_Frequency"]), str(e)


def save_watchlists_df(df):
    sh, err = open_log_spreadsheet()
    if sh is None:
        return False, err

    try:
        ws = get_or_create_worksheet(sh, "Watchlists", rows=max(2000, len(df) + 20), cols=10)
        ws.clear()

        headers = ["Watchlist_Name", "Watchlist_Type", "Ticker", "Added_At", "Alert_Mode", "Check_Frequency"]
        ws.append_row(headers, value_input_option="USER_ENTERED")

        if df is not None and not df.empty:
            data = df[headers].fillna("").astype(str).values.tolist()
            for row in data:
                ws.append_row(row, value_input_option="USER_ENTERED")
        return True, "Watchlists gespeichert"
    except Exception as e:
        return False, str(e)


def create_watchlist(watchlist_name, watchlist_type, check_frequency=None):
    name = str(watchlist_name or "").strip()
    wl_type = str(watchlist_type or "").strip() or "Watchlist"
    freq = str(check_frequency or "").strip() or ("4x täglich" if wl_type == "Watchlist" else "3x täglich")
    if not name:
        return False, "Bitte einen Watchlist-Namen eingeben."

    df, err = load_watchlists_df()
    if err:
        return False, err

    existing = df["Watchlist_Name"].astype(str).str.strip().str.lower().tolist() if not df.empty else []
    if name.lower() in existing:
        return False, "Eine Watchlist mit diesem Namen existiert bereits."

    new_row = pd.DataFrame([{
        "Watchlist_Name": name,
        "Watchlist_Type": wl_type,
        "Ticker": "",
        "Added_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    return save_watchlists_df(df)


def add_entries_to_watchlist(watchlist_name, watchlist_type, entries, check_frequency=None):
    name = str(watchlist_name or "").strip()
    if not name:
        return False, "Keine Watchlist ausgewählt."

    df, err = load_watchlists_df()
    if err:
        return False, err

    if df.empty or name.lower() not in df["Watchlist_Name"].astype(str).str.strip().str.lower().tolist():
        ok, msg = create_watchlist(name, watchlist_type, check_frequency=check_frequency)
        if not ok:
            return False, msg
        df, err = load_watchlists_df()
        if err:
            return False, err

    wl_type = watchlist_type
    freq = str(check_frequency or "").strip() or ("4x täglich" if wl_type == "Watchlist" else "3x täglich")
    existing_rows = df[df["Watchlist_Name"].astype(str).str.strip().str.lower() == name.lower()].copy()
    if not existing_rows.empty:
        non_empty_types = [x for x in existing_rows["Watchlist_Type"].astype(str).tolist() if str(x).strip()]
        if non_empty_types:
            wl_type = non_empty_types[0]
        non_empty_freqs = [x for x in existing_rows["Check_Frequency"].astype(str).tolist() if str(x).strip()] if "Check_Frequency" in existing_rows.columns else []
        if non_empty_freqs:
            freq = non_empty_freqs[0]

    existing_tickers = {
        str(x).strip().upper()
        for x in existing_rows["Ticker"].astype(str).tolist()
        if str(x).strip()
    }

    cleaned = []
    for entry in entries:
        val = str(entry or "").strip().upper()
        if val and val not in existing_tickers:
            cleaned.append(val)

    if not cleaned:
        return False, "Keine neuen Ticker zum Hinzufügen gefunden."

    new_rows = pd.DataFrame([
        {
            "Watchlist_Name": name,
            "Watchlist_Type": wl_type,
            "Ticker": ticker,
            "Added_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Alert_Mode": "Standard",
            "Check_Frequency": "4x täglich",
            "Check_Frequency": freq,
            "Alert_Mode": "Standard",
        }
        for ticker in cleaned
    ])

    df = pd.concat([df, new_rows], ignore_index=True)
    ok, msg = save_watchlists_df(df)
    if ok:
        return True, f"{len(cleaned)} Ticker zu '{name}' hinzugefügt."
    return False, msg


def remove_ticker_from_watchlist(watchlist_name, ticker):
    name = str(watchlist_name or "").strip()
    tkr = str(ticker or "").strip().upper()
    if not name or not tkr:
        return False, "Bitte Watchlist und Ticker auswählen."

    df, err = load_watchlists_df()
    if err:
        return False, err

    mask = ~(
        (df["Watchlist_Name"].astype(str).str.strip().str.lower() == name.lower()) &
        (df["Ticker"].astype(str).str.strip().str.upper() == tkr)
    )
    new_df = df[mask].copy()

    remaining_rows = new_df[new_df["Watchlist_Name"].astype(str).str.strip().str.lower() == name.lower()]
    if remaining_rows.empty:
        # metadata row beibehalten
        new_df = pd.concat([new_df, pd.DataFrame([{
            "Watchlist_Name": name,
            "Watchlist_Type": "Watchlist",
            "Ticker": "",
            "Added_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Alert_Mode": "Standard",
        }])], ignore_index=True)

    ok, msg = save_watchlists_df(new_df)
    if ok:
        return True, f"{tkr} aus '{name}' entfernt."
    return False, msg


def delete_watchlist(watchlist_name):
    name = str(watchlist_name or "").strip()
    if not name:
        return False, "Keine Watchlist ausgewählt."

    df, err = load_watchlists_df()
    if err:
        return False, err

    new_df = df[df["Watchlist_Name"].astype(str).str.strip().str.lower() != name.lower()].copy()
    ok, msg = save_watchlists_df(new_df)
    if ok:
        return True, f"Watchlist '{name}' gelöscht."
    return False, msg





def update_watchlist_check_frequency(watchlist_name, check_frequency):
    name = str(watchlist_name or "").strip()
    freq = str(check_frequency or "").strip() or "4x täglich"
    if not name:
        return False, "Keine Watchlist ausgewählt."

    df, err = load_watchlists_df()
    if err:
        return False, err
    if df.empty:
        return False, "Keine Watchlists vorhanden."

    mask = df["Watchlist_Name"].astype(str).str.strip().str.lower() == name.lower()
    if mask.sum() == 0:
        return False, "Watchlist nicht gefunden."

    df.loc[mask, "Check_Frequency"] = freq
    return save_watchlists_df(df)


def get_watchlist_check_frequency(watchlist_name):
    name = str(watchlist_name or "").strip()
    df, err = load_watchlists_df()
    if err or df.empty or not name:
        return "4x täglich"

    mask = df["Watchlist_Name"].astype(str).str.strip().str.lower() == name.lower()
    subset = df.loc[mask]
    if subset.empty or "Check_Frequency" not in subset.columns:
        return "4x täglich"

    vals = [str(x).strip() for x in subset["Check_Frequency"].tolist() if str(x).strip()]
    if vals:
        return vals[0]

    wl_types = [str(x).strip() for x in subset["Watchlist_Type"].tolist() if str(x).strip()]
    wl_type = wl_types[0] if wl_types else "Watchlist"
    return "4x täglich" if wl_type == "Watchlist" else "3x täglich"


def update_watchlist_alert_mode(watchlist_name, alert_mode):
    name = str(watchlist_name or "").strip()
    mode = str(alert_mode or "").strip() or "Standard"
    if not name:
        return False, "Keine Watchlist ausgewählt."

    df, err = load_watchlists_df()
    if err:
        return False, err
    if df.empty:
        return False, "Keine Watchlists vorhanden."

    mask = df["Watchlist_Name"].astype(str).str.strip().str.lower() == name.lower()
    if mask.sum() == 0:
        return False, "Watchlist nicht gefunden."

    df.loc[mask, "Alert_Mode"] = mode
    return save_watchlists_df(df)


def get_watchlist_alert_mode(watchlist_name):
    name = str(watchlist_name or "").strip()
    df, err = load_watchlists_df()
    if err or df.empty or not name:
        return "Standard"

    mask = df["Watchlist_Name"].astype(str).str.strip().str.lower() == name.lower()
    subset = df.loc[mask]
    if subset.empty:
        return "Standard"

    vals = [str(x).strip() for x in subset["Alert_Mode"].tolist() if str(x).strip()]
    return vals[0] if vals else "Standard"






def get_watchlist_catalog_df():
    cols = ["Watchlist_Name", "Watchlist_Type", "Alert_Mode", "Check_Frequency"]
    df, err = load_watchlists_df()
    if err:
        return pd.DataFrame(columns=cols), err
    if df is None or df.empty:
        return pd.DataFrame(columns=cols), None

    for col in cols:
        if col not in df.columns:
            df[col] = ""

    catalog = (
        df[cols]
        .fillna("")
        .astype(str)
        .query("Watchlist_Name != ''")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return catalog, None


def get_schedule_slots_for_frequency(check_frequency):
    freq = str(check_frequency or "").strip()
    mapping = {
        "Nur manuell": [],
        "2x täglich": ["10:30", "18:30"],
        "3x täglich": ["10:30", "18:30", "22:10"],
        "4x täglich": ["10:30", "15:40", "18:30", "22:10"],
    }
    return mapping.get(freq, [])


def get_current_berlin_time():
    return datetime.now(ZoneInfo("Europe/Berlin"))


def get_current_schedule_slot(now_dt=None):
    now_dt = now_dt or get_current_berlin_time()
    hhmm = now_dt.strftime("%H:%M")
    slots = ["10:30", "15:40", "18:30", "22:10"]
    eligible = [s for s in slots if hhmm >= s]
    return eligible[-1] if eligible else None


def get_due_watchlists_for_slot(slot_label):
    cols = ["Watchlist_Name", "Watchlist_Type", "Alert_Mode", "Check_Frequency", "Due_Now"]
    catalog, err = get_watchlist_catalog_df()
    if err:
        return pd.DataFrame(columns=cols), err
    if catalog is None or catalog.empty:
        return pd.DataFrame(columns=cols), None

    catalog = catalog.copy()
    catalog["Due_Now"] = catalog["Check_Frequency"].apply(lambda x: slot_label in get_schedule_slots_for_frequency(x))
    return catalog[catalog["Due_Now"]].reset_index(drop=True), None


def get_watchlist_tickers(watchlist_name):
    df, err = load_watchlists_df()
    if err:
        return [], err
    if df is None or df.empty:
        return [], None

    mask = df["Watchlist_Name"].astype(str).str.strip().str.lower() == str(watchlist_name).strip().lower()
    subset = df.loc[mask]
    if subset.empty:
        return [], None

    tickers = [str(x).strip().upper() for x in subset["Ticker"].astype(str).tolist() if str(x).strip()]
    seen = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique, None


def append_auto_run_log(rows):
    if not rows:
        return False, "Keine Auto-Run-Log-Daten vorhanden."

    sh, err = open_log_spreadsheet()
    if sh is None:
        return False, err

    try:
        ws = get_or_create_worksheet(sh, "Auto_Run_Log", rows=5000, cols=16)
        existing = ws.get_all_values()
        headers = [
            "Run_Timestamp",
            "Berlin_Time",
            "Slot",
            "Watchlist_Name",
            "Watchlist_Type",
            "Alert_Mode",
            "Check_Frequency",
            "Ticker_Count",
            "Analyzed_Count",
            "Sent_Count",
            "Status",
            "Message",
        ]
        if not existing:
            ws.append_row(headers, value_input_option="USER_ENTERED")

        for row in rows:
            ws.append_row([row.get(h, "") for h in headers], value_input_option="USER_ENTERED")
        return True, f"{len(rows)} Auto-Run-Logzeilen gespeichert."
    except Exception as e:
        return False, str(e)


def load_alert_history_df():
    sh, err = open_log_spreadsheet()
    cols = [
        "Watchlist_Name",
        "Watchlist_Type",
        "Alert_Mode",
        "Ticker",
        "Alert_Type",
        "Alert_Signature",
        "Last_Sent_At",
        "Last_Sent_Date",
    ]
    if sh is None:
        return pd.DataFrame(columns=cols), err

    try:
        ws = get_or_create_worksheet(sh, "Alert_History", rows=5000, cols=12)
        values = ws.get_all_values()
        if not values:
            ws.append_row(cols, value_input_option="USER_ENTERED")
            return pd.DataFrame(columns=cols), None

        headers = values[0]
        rows = values[1:]
        if not rows:
            return pd.DataFrame(columns=headers), None

        df = pd.DataFrame(rows, columns=headers)
        for col in cols:
            if col not in df.columns:
                df[col] = ""
        return df[cols], None
    except Exception as e:
        return pd.DataFrame(columns=cols), str(e)


def save_alert_history_df(df):
    sh, err = open_log_spreadsheet()
    cols = [
        "Watchlist_Name",
        "Watchlist_Type",
        "Alert_Mode",
        "Ticker",
        "Alert_Type",
        "Alert_Signature",
        "Last_Sent_At",
        "Last_Sent_Date",
    ]
    if sh is None:
        return False, err

    try:
        ws = get_or_create_worksheet(sh, "Alert_History", rows=max(5000, len(df) + 20), cols=12)
        ws.clear()
        ws.append_row(cols, value_input_option="USER_ENTERED")
        if df is not None and not df.empty:
            data = df[cols].fillna("").astype(str).values.tolist()
            for row in data:
                ws.append_row(row, value_input_option="USER_ENTERED")
        return True, "Alert-History gespeichert"
    except Exception as e:
        return False, str(e)


def get_alert_history_entry(watchlist_name, ticker, alert_type):
    df, err = load_alert_history_df()
    if err or df.empty:
        return None

    mask = (
        (df["Watchlist_Name"].astype(str).str.strip().str.lower() == str(watchlist_name).strip().lower()) &
        (df["Ticker"].astype(str).str.strip().str.upper() == str(ticker).strip().upper()) &
        (df["Alert_Type"].astype(str).str.strip() == str(alert_type).strip())
    )
    subset = df.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0].to_dict()


def upsert_alert_history_entry(watchlist_name, watchlist_type, alert_mode, ticker, alert_type, alert_signature):
    df, err = load_alert_history_df()
    if err:
        return False, err

    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today = datetime.now().strftime("%Y-%m-%d")

    if df is None or df.empty:
        df = pd.DataFrame(columns=[
            "Watchlist_Name",
            "Watchlist_Type",
            "Alert_Mode",
            "Ticker",
            "Alert_Type",
            "Alert_Signature",
            "Last_Sent_At",
            "Last_Sent_Date",
        ])

    mask = (
        (df["Watchlist_Name"].astype(str).str.strip().str.lower() == str(watchlist_name).strip().lower()) &
        (df["Ticker"].astype(str).str.strip().str.upper() == str(ticker).strip().upper()) &
        (df["Alert_Type"].astype(str).str.strip() == str(alert_type).strip())
    )

    if mask.sum() == 0:
        new_row = pd.DataFrame([{
            "Watchlist_Name": watchlist_name,
            "Watchlist_Type": watchlist_type,
            "Alert_Mode": alert_mode,
            "Ticker": ticker,
            "Alert_Type": alert_type,
            "Alert_Signature": alert_signature,
            "Last_Sent_At": now_ts,
            "Last_Sent_Date": today,
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df.loc[mask, "Watchlist_Type"] = watchlist_type
        df.loc[mask, "Alert_Mode"] = alert_mode
        df.loc[mask, "Alert_Signature"] = alert_signature
        df.loc[mask, "Last_Sent_At"] = now_ts
        df.loc[mask, "Last_Sent_Date"] = today

    return save_alert_history_df(df)


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
        "Leadership": result.get("leadership_score", np.nan),
        "Leadership-Status": result.get("leadership_status", "-"),
        "Sektor-Stärke": result.get("sector_strength_score", np.nan),
        "Industrie-Stärke": result.get("industry_strength_score", np.nan),
        "RS-Benchmark-Score": result.get("rs_benchmark_score", np.nan),
        "RS-Beschleunigung": result.get("rs_acceleration_score", np.nan),
        "Sektor": result.get("sector_label", "-"),
        "Industrie": result.get("industry_label", "-"),
        "Trendqualität": result.get("trend_quality_score", np.nan),
        "Base-Qualität": result.get("base_quality_score", np.nan),
        "Setup-Typ-Qualität": result.get("setup_type_quality_score", np.nan),
        "Setup-Priorität": result.get("setup_priority_score", np.nan),
        "Base-Länge": result.get("base_length_days", np.nan),
        "Korrekturtiefe %": result.get("correction_depth_pct", np.nan),
        "Range-Tightness": result.get("range_tightness_score", np.nan),
        "Volatility Contraction": result.get("volatility_contraction_score", np.nan),
        "Pullback-Qualität": result.get("pullback_quality_score", np.nan),
        "Sektor-ETF": result.get("sector_etf_symbol", "-"),
        "Volumenqualität": result.get("volume_quality_score", np.nan),
        "Akkumulation": result.get("accumulation_score", np.nan),
        "Distribution": result.get("distribution_pressure_score", np.nan),
        "Pullback-Dry-up": result.get("pullback_dryup_score", np.nan),
        "Breakout-Volumen": result.get("breakout_volume_score", np.nan),
        "Up/Down-Vol.-Ratio": result.get("up_down_volume_ratio", np.nan),
        "Volumentrend": result.get("volume_trend_score", np.nan),
        "Akkumulationstage": result.get("accumulation_day_count", np.nan),
        "Distributionstage": result.get("distribution_day_count", np.nan),
        "Pullback-Vol.-Ratio": result.get("recent_pullback_volume_ratio", np.nan),
        "Breakout-Vol.-Ratio": result.get("breakout_day_volume_ratio", np.nan),
        "Katalysator": result.get("catalyst_score", np.nan),
        "Event-Score": result.get("earnings_event_score", np.nan),
        "Revision/Momentum": result.get("revision_momentum_score", np.nan),
        "Event-Risiko": result.get("event_risk_score", np.nan),
        "Event-Phase": result.get("event_phase_label", "-"),
        "Post-Earnings 5d": result.get("earnings_reaction_5d", np.nan),
        "Post-Earnings 10d": result.get("earnings_reaction_10d", np.nan),
        "Institutionelle Qualität": result.get("institutional_quality_score", np.nan),
        "Cashflow-Stabilität": result.get("cashflow_stability_score", np.nan),
        "Margenstabilität": result.get("margin_stability_score", np.nan),
        "Marktregime": result.get("regime_label", "-"),
        "Marktregime-Score": result.get("regime_score", np.nan),
        "Regime-Fit": result.get("regime_fit_score", np.nan),
        "Regime-Anpassung": result.get("regime_adjustment_score", np.nan),
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
                "Marktregime": market_regime_display((r.get("market_info", {}) or {}).get("regime", "UNBEKANNT")),
                "Top Red Flag": r.get("top_red_flag", "-"),
                "Kurzfazit": r.get("short_thesis", r.get("decision_summary", "-")),
            })
    return pd.DataFrame(rows)
