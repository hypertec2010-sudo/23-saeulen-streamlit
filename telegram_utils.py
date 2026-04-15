from datetime import datetime

import requests
import streamlit as st

from logging_utils import get_alert_history_entry, has_any_alert_history_for_ticker, upsert_alert_history_entry


def get_telegram_credentials():
    try:
        token = st.secrets.get("TELEGRAM_TOKEN")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    except Exception:
        token = None
        chat_id = None
    return token, chat_id


def get_alert_type_label(result, watchlist_type):
    if watchlist_type == "Positions-Watchlist":
        position_action = str(result.get("position_action", "")).lower()
        partial_profit = str(result.get("partial_profit_action", "")).lower()
        if "risiko reduzieren" in position_action:
            return "Risiko-Alert"
        if partial_profit.startswith("ja"):
            return "Teilgewinn-Alert"
        return "Positions-Info"


def build_watchlist_telegram_text(result, watchlist_name, watchlist_type, alert_mode="Standard"):
    ticker = result.get("ticker", "-")
    name = result.get("name", "-")

    if watchlist_type == "Positions-Watchlist":
        action = result.get("position_action", "-")
    else:
        action = result.get("emp", "-")

    setup_type = result.get("setup_type", "-")
    trigger_status = result.get("trigger_status", "-")
    priority = result.get("watchlist_priority", "-")
    entry_score = result.get("trading_case_score", "n/a")
    invest_score = result.get("investment_case_score", "n/a")
    entry_zone = result.get("suggested_entry_zone", "-")
    red_flag = result.get("top_red_flag", "-")
    mode = result.get("mode_label", "-")

    alert_type = get_alert_type_label(result, watchlist_type)
    lines = [
        f"Capital Hill | {alert_type}",
        f"Watchlist: {watchlist_name}",
        f"Typ: {watchlist_type}",
        f"Alert-Modus: {alert_mode}",
        f"{ticker} | {name}",
        f"Modus: {mode}",
        f"Handlung: {action}",
        f"Setup: {setup_type}",
    ]

    if watchlist_type == "Positions-Watchlist":
        lines.extend([
            f"Positions-Aktion: {result.get('position_action', '-')}",
            f"Teilgewinn: {result.get('partial_profit_action', '-')}",
            f"Stop: {result.get('stop_action', '-')}",
            f"Risiko-Hinweis: {result.get('risk_note', '-')}",
        ])
    else:
        lines.extend([
            f"Trigger: {trigger_status}",
            f"Priorität: {priority}",
            f"Einstieg: {entry_score}/100",
            f"Investment: {invest_score}/100",
            f"Entry-Zone: {entry_zone}",
        ])

    if red_flag and red_flag != "-":
        lines.append(f"Red Flag: {red_flag}")

    return "\n".join(lines)


def should_alert_for_watchlist_result(result, watchlist_type, alert_mode="Standard"):
    mode = str(alert_mode or "Standard")

    if watchlist_type == "Positions-Watchlist":
        position_action = str(result.get("position_action", "")).lower()
        partial_profit = str(result.get("partial_profit_action", "")).lower()
        risk_note = str(result.get("risk_note", "")).lower()
        setup_conf = float(result.get("setup_confidence", 0) or 0)

        if mode == "Konservativ":
            return (
                "risiko reduzieren" in position_action
                or partial_profit.startswith("ja")
                or "verlustposition" in risk_note
            )
        if mode == "Früh":
            return (
                "risiko reduzieren" in position_action
                or partial_profit.startswith("ja")
                or "erhöht" in risk_note
                or "verlustposition" in risk_note
                or ("eng beobachten" in position_action and setup_conf < 60)
            )
        return (
            "risiko reduzieren" in position_action
            or partial_profit.startswith("ja")
            or "erhöht" in risk_note
            or "verlustposition" in risk_note
            or ("eng beobachten" in position_action and setup_conf < 55)
        )

    trigger_status = str(result.get("trigger_status", ""))
    priority = str(result.get("watchlist_priority", ""))
    entry_score = float(result.get("trading_case_score", 0) or 0)
    invest_score = float(result.get("investment_case_score", 0) or 0)
    setup_conf = float(result.get("setup_confidence", 0) or 0)
    entry_quality = str(result.get("entry_quality", ""))
    emp = str(result.get("emp", ""))

    if mode == "Konservativ":
        return (
            (trigger_status == "Aktiv" and entry_score >= 72 and setup_conf >= 62 and invest_score >= 68)
            or (priority == "Hoch" and entry_score >= 78 and invest_score >= 70 and setup_conf >= 60 and "EINSTIEG PRÜFEN" in emp)
        )
    if mode == "Früh":
        return (
            (trigger_status == "Aktiv" and entry_score >= 64 and setup_conf >= 52)
            or (priority == "Hoch" and entry_score >= 68 and invest_score >= 62 and setup_conf >= 50)
            or (trigger_status == "Nahe dran" and entry_quality in {"gut", "abwarten"} and entry_score >= 70)
        )
    return (
        (trigger_status == "Aktiv" and entry_score >= 68 and setup_conf >= 58)
        or (priority == "Hoch" and entry_score >= 72 and invest_score >= 65 and setup_conf >= 55)
        or (trigger_status == "Nahe dran" and entry_quality in {"gut", "abwarten"} and entry_score >= 74 and "EINSTIEG PRÜFEN" in emp)
    )






def build_new_watchlist_entry_text(result, watchlist_name, watchlist_type, alert_mode="Standard"):
    ticker = result.get("ticker", "-")
    name = result.get("name", "-")
    trigger_status = result.get("trigger_status", "-")
    priority = result.get("watchlist_priority", "-")
    entry_score = result.get("trading_case_score", "n/a")
    invest_score = result.get("investment_case_score", "n/a")
    action = result.get("emp", result.get("position_action", "-"))

    lines = [
        "Capital Hill | Neue Watchlist-Aufnahme",
        f"Watchlist: {watchlist_name}",
        f"Typ: {watchlist_type}",
        f"Alert-Modus: {alert_mode}",
        f"{ticker} | {name}",
        f"Handlung: {action}",
        f"Trigger: {trigger_status}",
        f"Priorität: {priority}",
        f"Einstieg: {entry_score}/100",
        f"Investment: {invest_score}/100",
        "Hinweis: Neuer Wert in der Watchlist, aktuell noch kein harter Trigger-Alert.",
    ]
    return "\n".join(lines)


def build_alert_signature(result, watchlist_type):
    if watchlist_type == "Positions-Watchlist":
        return "|".join([
            str(result.get("position_action", "-")),
            str(result.get("partial_profit_action", "-")),
            str(result.get("stop_action", "-")),
            str(result.get("risk_note", "-")),
            str(result.get("setup_confidence", "-")),
        ])
    return "|".join([
        str(result.get("trigger_status", "-")),
        str(result.get("watchlist_priority", "-")),
        str(result.get("emp", "-")),
        str(result.get("trading_case_score", "-")),
        str(result.get("investment_case_score", "-")),
        str(result.get("entry_quality", "-")),
        str(result.get("setup_confidence", "-")),
    ])


def send_telegram_message(message_text):
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        return False, "TELEGRAM_TOKEN oder TELEGRAM_CHAT_ID fehlt."

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message_text},
            timeout=20,
        )
        if response.status_code == 200:
            return True, "OK"
        return False, f"Telegram API Fehler {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return False, str(e)


def send_watchlist_alerts(results, watchlist_name, watchlist_type, alert_mode="Standard"):
    if not results:
        return False, "Keine Ergebnisse für Telegram-Alerts vorhanden.", 0

    sent = 0
    matched = 0
    suppressed = 0
    info_sent = 0
    errors = []

    for result in results:
        ticker = str(result.get("ticker", "-")).strip().upper()
        has_any_history = has_any_alert_history_for_ticker(watchlist_name, ticker)

        if should_alert_for_watchlist_result(result, watchlist_type, alert_mode):
            matched += 1
            alert_type = get_alert_type_label(result, watchlist_type)
            alert_signature = build_alert_signature(result, watchlist_type)
            history_entry = get_alert_history_entry(watchlist_name, ticker, alert_type)

            same_day = False
            same_signature = False
            if history_entry:
                same_day = str(history_entry.get("Last_Sent_Date", "")) == datetime.now().strftime("%Y-%m-%d")
                same_signature = str(history_entry.get("Alert_Signature", "")) == alert_signature

            if same_day and same_signature:
                suppressed += 1
                continue

            msg = build_watchlist_telegram_text(result, watchlist_name, watchlist_type, alert_mode)
            ok, err = send_telegram_message(msg)
            if ok:
                sent += 1
                hist_ok, hist_err = upsert_alert_history_entry(
                    watchlist_name,
                    watchlist_type,
                    alert_mode,
                    ticker,
                    alert_type,
                    alert_signature,
                )
                if not hist_ok:
                    errors.append(f"History: {hist_err}")
            else:
                errors.append(err)
        else:
            if not has_any_history and watchlist_type != "Positions-Watchlist":
                alert_type = "Neue Watchlist-Aufnahme"
                alert_signature = build_alert_signature(result, watchlist_type)
                history_entry = get_alert_history_entry(watchlist_name, ticker, alert_type)

                already_sent_today = False
                same_signature = False
                if history_entry:
                    already_sent_today = str(history_entry.get("Last_Sent_Date", "")) == datetime.now().strftime("%Y-%m-%d")
                    same_signature = str(history_entry.get("Alert_Signature", "")) == alert_signature

                if already_sent_today and same_signature:
                    suppressed += 1
                    continue

                msg = build_new_watchlist_entry_text(result, watchlist_name, watchlist_type, alert_mode)
                ok, err = send_telegram_message(msg)
                if ok:
                    sent += 1
                    info_sent += 1
                    hist_ok, hist_err = upsert_alert_history_entry(
                        watchlist_name,
                        watchlist_type,
                        alert_mode,
                        ticker,
                        alert_type,
                        alert_signature,
                    )
                    if not hist_ok:
                        errors.append(f"History: {hist_err}")
                else:
                    errors.append(err)

    if matched == 0 and info_sent == 0:
        return False, "Keine alert-relevanten Werte in dieser Watchlist gefunden.", 0
    if sent > 0 and not errors:
        parts = []
        if sent - info_sent > 0:
            parts.append(f"{sent - info_sent} neue Alerts")
        if info_sent > 0:
            parts.append(f"{info_sent} Erst-Checks")
        msg = f"{', '.join(parts)} für '{watchlist_name}' gesendet."
        if suppressed > 0:
            msg += f" {suppressed} doppelte Meldungen wurden unterdrückt."
        return True, msg, sent
    if sent > 0:
        return True, f"{sent} Meldungen gesendet, einzelne Fehler: {' | '.join(errors[:2])}", sent
    if suppressed > 0:
        return False, f"Keine neuen Alerts gesendet. {suppressed} doppelte Meldungen wurden unterdrückt.", 0
    return False, f"Telegram-Versand fehlgeschlagen: {' | '.join(errors[:2])}", 0
