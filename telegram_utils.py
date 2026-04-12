import requests
import streamlit as st


def get_telegram_credentials():
    try:
        token = st.secrets.get("TELEGRAM_TOKEN")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    except Exception:
        token = None
        chat_id = None
    return token, chat_id


def build_watchlist_telegram_text(result, watchlist_name, watchlist_type):
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

    lines = [
        "Capital Hill Alert",
        f"Watchlist: {watchlist_name}",
        f"Typ: {watchlist_type}",
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


def should_alert_for_watchlist_result(result, watchlist_type):
    if watchlist_type == "Positions-Watchlist":
        position_action = str(result.get("position_action", "")).lower()
        partial_profit = str(result.get("partial_profit_action", "")).lower()
        risk_note = str(result.get("risk_note", "")).lower()
        return (
            "risiko reduzieren" in position_action
            or partial_profit.startswith("ja")
            or "erhöht" in risk_note
            or "verlustposition" in risk_note
        )

    trigger_status = str(result.get("trigger_status", ""))
    priority = str(result.get("watchlist_priority", ""))
    entry_score = float(result.get("trading_case_score", 0) or 0)
    entry_quality = str(result.get("entry_quality", ""))
    return (
        trigger_status == "Aktiv"
        or (priority == "Hoch" and entry_score >= 70)
        or (trigger_status == "Nahe dran" and entry_quality in {"gut", "abwarten"})
    )


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


def send_watchlist_alerts(results, watchlist_name, watchlist_type):
    if not results:
        return False, "Keine Ergebnisse für Telegram-Alerts vorhanden.", 0

    sent = 0
    matched = 0
    errors = []

    for result in results:
        if should_alert_for_watchlist_result(result, watchlist_type):
            matched += 1
            msg = build_watchlist_telegram_text(result, watchlist_name, watchlist_type)
            ok, err = send_telegram_message(msg)
            if ok:
                sent += 1
            else:
                errors.append(err)

    if matched == 0:
        return False, "Keine alert-relevanten Werte in dieser Watchlist gefunden.", 0
    if sent > 0 and not errors:
        return True, f"{sent} Telegram-Alerts für '{watchlist_name}' gesendet.", sent
    if sent > 0:
        return True, f"{sent} Alerts gesendet, einzelne Fehler: {' | '.join(errors[:2])}", sent
    return False, f"Telegram-Versand fehlgeschlagen: {' | '.join(errors[:2])}", 0
