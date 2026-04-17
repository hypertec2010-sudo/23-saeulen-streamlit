from datetime import datetime
import html

import requests
import streamlit as st

from logging_utils import get_alert_history_entry, upsert_alert_history_entry

try:
    from logging_utils import has_any_alert_history_for_ticker
except Exception:
    def has_any_alert_history_for_ticker(watchlist_name, ticker):
        return False




def _esc(value):
    return html.escape(str(value if value is not None else "-"))


def _b(label, value):
    return f"<b>{_esc(label)}</b> {_esc(value)}"


def _fmt_trade_value(value, suffix=""):
    if value is None:
        return "-"
    try:
        raw = str(value).strip()
        if raw in {"", "-", "n/a", "None"}:
            return "-"
        num = float(raw)
        return f"{num:.2f}{suffix}"
    except Exception:
        raw = str(value).strip()
        if raw in {"", "-", "n/a", "None"}:
            return "-"
        return f"{raw}{suffix}" if suffix and not raw.endswith(suffix) else raw





def _get_trade_plan_fields(result):
    entry_zone = result.get("suggested_entry_zone", "-")
    stop_value = result.get("stop_used", result.get("stop_current", result.get("stop", result.get("atr_stop", "-"))))
    crv_value = result.get("crv", result.get("rr", result.get("crv_value", "-")))
    return entry_zone, stop_value, crv_value


def _should_show_trade_plan(result, watchlist_type):
    # Show trade-plan details when the message is constructive and actual trade-plan fields exist.
    entry_zone = str(result.get("suggested_entry_zone", "")).strip()
    stop_value = result.get("stop_used", result.get("stop_current", result.get("stop", None)))
    crv_value = result.get("crv", result.get("rr", result.get("crv_value", None)))

    has_trade_fields = (
        entry_zone not in {"", "-", "n/a", "None"}
        or str(stop_value).strip() not in {"", "-", "n/a", "None"}
        or str(crv_value).strip() not in {"", "-", "n/a", "None"}
    )

    if watchlist_type == "Positions-Watchlist":
        position_action = str(result.get("position_action", "")).lower()
        add_on_action = str(result.get("add_on_action", "")).lower()
        positive_terms = ["nachkauf", "aufstock", "aufbauen", "zukauf", "add", "kauf"]
        negative_terms = ["verkauf", "reduzier", "risiko", "stop", "teilgewinn", "abbauen"]

        if any(term in position_action for term in negative_terms):
            return False
        if any(term in add_on_action for term in positive_terms):
            return True
        if any(term in position_action for term in positive_terms):
            return True
        return has_trade_fields and ("beobacht" not in position_action)

    action = str(result.get("emp", "")).lower()
    trigger = str(result.get("trigger_status", "")).lower()
    positive_terms = ["kauf", "aufbau", "einstieg", "nachkauf", "long"]
    negative_terms = ["verkauf", "warn", "stop", "risiko", "reduzier"]

    if any(term in action for term in negative_terms):
        return False
    if any(term in action for term in positive_terms):
        return True
    if any(term in trigger for term in ["aktiv", "bestätigt", "breakout", "retest"]):
        return has_trade_fields
    return False




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
        exit_action = str(result.get("exit_action", "")).lower()
        partial_profit = str(result.get("partial_profit_action", "")).lower()
        if "verkaufen" in exit_action or "verkaufen" in position_action:
            return "Exit-Alert"
        if "risiko reduzieren" in exit_action or "risiko reduzieren" in position_action:
            return "Risiko-Alert"
        if "teilgewinn" in exit_action or partial_profit.startswith("ja"):
            return "Teilgewinn-Alert"
        if "beobachten" in exit_action:
            return "Exit-Warnung"
        return "Positions-Info"

    trigger_status = str(result.get("trigger_status", "")).lower()
    priority = str(result.get("watchlist_priority", "")).lower()
    action = str(result.get("emp", "")).lower()

    if "aktiv" in trigger_status or "bestätigt" in trigger_status or "breakout" in trigger_status:
        return "Trigger-Alert"
    if priority == "hoch":
        return "Prioritäts-Alert"
    if "kauf" in action or "aufbau" in action:
        return "Handlungs-Alert"
    return "Watchlist-Info"


def build_alert_signature(result, watchlist_type):
    if watchlist_type == "Positions-Watchlist":
        return "|".join([
            str(result.get("position_action", "-")),
            str(result.get("exit_action", "-")),
            str(result.get("exit_score", "-")),
            str(result.get("exit_reason_top", "-")),
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


def _parse_signature(signature, watchlist_type):
    parts = str(signature or "").split("|")
    if watchlist_type == "Positions-Watchlist":
        fields = ["position_action", "exit_action", "exit_score", "exit_reason_top", "partial_profit_action", "stop_action", "risk_note", "setup_confidence"]
    else:
        fields = ["trigger_status", "watchlist_priority", "emp", "trading_case_score", "investment_case_score", "entry_quality", "setup_confidence"]

    while len(parts) < len(fields):
        parts.append("-")
    return {field: parts[i] for i, field in enumerate(fields)}


def _format_num_change(old_value, new_value):
    try:
        old_num = float(str(old_value).replace("%", "").replace(",", "."))
        new_num = float(str(new_value).replace("%", "").replace(",", "."))
        if abs(old_num - new_num) < 0.01:
            return None
        return f"{int(round(old_num))} -> {int(round(new_num))}"
    except Exception:
        if str(old_value) == str(new_value):
            return None
        return f"{old_value} -> {new_value}"


def _rank_priority(value):
    mapping = {"niedrig": 1, "mittel": 2, "hoch": 3}
    return mapping.get(str(value).strip().lower(), 0)


def _classify_change(label, old_value, new_value, watchlist_type, numeric=False):
    old_str = str(old_value)
    new_str = str(new_value)

    if numeric:
        try:
            old_num = float(str(old_value).replace("%", "").replace(",", "."))
            new_num = float(str(new_value).replace("%", "").replace(",", "."))
            if abs(old_num - new_num) < 0.01:
                return None, None
            direction = "🟢 verbessert" if new_num > old_num else "🔴 schwächer"
            return direction, f"{label}: {int(round(old_num))} -> {int(round(new_num))}"
        except Exception:
            if old_str == new_str:
                return None, None
            return "🟡 geändert", f"<b>{_esc(label)}</b> {_esc(old_str)} -&gt; {_esc(new_str)}"

    if label == "📌 Priorität":
        old_rank = _rank_priority(old_str)
        new_rank = _rank_priority(new_str)
        if old_rank == new_rank:
            return None, None
        direction = "🟢 verbessert" if new_rank > old_rank else "🔴 schwächer"
        return direction, f"<b>{_esc(label)}</b> {_esc(old_str)} -&gt; {_esc(new_str)}"

    positive_terms = ["aktiv", "bestätigt", "breakout", "hoch", "kaufen", "aufbauen", "beobachten", "ja"]
    negative_terms = ["stop", "verkaufen", "reduzieren", "verlust", "schwach", "nein", "abbauen"]

    if watchlist_type == "Positions-Watchlist":
        if label in ["⚠️ Positions-Aktion", "🚨 Risiko", "🛡️ Stop"]:
            if old_str == new_str:
                return None, None
            if any(x in new_str.lower() for x in ["reduzieren", "verkaufen", "enger", "verlust", "risiko"]):
                return "🔴 verschärft", f"<b>{_esc(label)}</b> {_esc(old_str)} -&gt; {_esc(new_str)}"
            if any(x in new_str.lower() for x in ["halten", "stabil"]):
                return "🟢 entspannt", f"<b>{_esc(label)}</b> {_esc(old_str)} -&gt; {_esc(new_str)}"
            return "🟡 geändert", f"<b>{_esc(label)}</b> {_esc(old_str)} -&gt; {_esc(new_str)}"

    if old_str == new_str:
        return None, None
    if any(x in new_str.lower() for x in positive_terms) and not any(x in new_str.lower() for x in negative_terms):
        return "🟢 verbessert", f"{label}: {old_str} -> {new_str}"
    if any(x in new_str.lower() for x in negative_terms):
        return "🔴 schwächer", f"{label}: {old_str} -> {new_str}"
    return "🟡 geändert", f"<b>{_esc(label)}</b> {_esc(old_str)} -&gt; {_esc(new_str)}"


def _build_change_lines(result, previous_signature, watchlist_type):
    if not previous_signature:
        return []

    prev = _parse_signature(previous_signature, watchlist_type)
    grouped = {"🟢": [], "🔴": [], "🟡": []}

    if watchlist_type == "Positions-Watchlist":
        checks = [
            ("⚠️ Positions-Aktion", prev.get("position_action", "-"), result.get("position_action", "-"), False),
            ("🚪 Exit-Aktion", prev.get("exit_action", "-"), result.get("exit_action", "-"), False),
            ("📊 Exit-Score", prev.get("exit_score", "-"), result.get("exit_score", "-"), True),
            ("🧭 Exit-Grund", prev.get("exit_reason_top", "-"), result.get("exit_reason_top", "-"), False),
            ("💰 Teilgewinn", prev.get("partial_profit_action", "-"), result.get("partial_profit_action", "-"), False),
            ("🛡️ Stop", prev.get("stop_action", "-"), result.get("stop_action", "-"), False),
            ("🚨 Risiko", prev.get("risk_note", "-"), result.get("risk_note", "-"), False),
            ("📊 Setup-Confidence", prev.get("setup_confidence", "-"), result.get("setup_confidence", "-"), True),
        ]
    else:
        checks = [
            ("🔔 Trigger", prev.get("trigger_status", "-"), result.get("trigger_status", "-"), False),
            ("📌 Priorität", prev.get("watchlist_priority", "-"), result.get("watchlist_priority", "-"), False),
            ("⚡ Handlung", prev.get("emp", "-"), result.get("emp", "-"), False),
            ("📈 Einstieg", prev.get("trading_case_score", "-"), result.get("trading_case_score", "-"), True),
            ("🏛️ Investment", prev.get("investment_case_score", "-"), result.get("investment_case_score", "-"), True),
            ("🧭 Lage", prev.get("entry_quality", "-"), result.get("entry_quality", "-"), False),
            ("📊 Setup-Confidence", prev.get("setup_confidence", "-"), result.get("setup_confidence", "-"), True),
        ]

    for label, old_value, new_value, numeric in checks:
        kind, line = _classify_change(label, old_value, new_value, watchlist_type, numeric=numeric)
        if not line:
            continue
        if kind.startswith("🟢"):
            grouped["🟢"].append(f"{kind}: {line}")
        elif kind.startswith("🔴"):
            grouped["🔴"].append(f"{kind}: {line}")
        else:
            grouped["🟡"].append(f"{kind}: {line}")

    ordered = grouped["🟢"] + grouped["🔴"] + grouped["🟡"]
    return ordered


def _headline_lines(result, watchlist_name, watchlist_type, alert_mode, prefix_title):
    ticker = str(result.get("ticker", "-")).strip().upper()
    name = result.get("name", "-")
    alert_type = get_alert_type_label(result, watchlist_type)

    return [
        f"<b>{_esc(prefix_title)}</b>",
        f"📌 <b>WERT:</b> {_esc(ticker)} | {_esc(name)}",
        f"📋 <b>WATCHLIST:</b> {_esc(watchlist_name)} | {_esc(watchlist_type)}",
        f"🚨 <b>ALERT:</b> {_esc(alert_type)} | <b>MODUS:</b> {_esc(alert_mode)}",
    ]


def build_watchlist_telegram_text(result, watchlist_name, watchlist_type, alert_mode="Standard", previous_signature=None):
    setup_type = result.get("setup_type", "-")
    red_flag = result.get("top_red_flag", "-")
    mode = result.get("mode_label", "-")
    lines = _headline_lines(result, watchlist_name, watchlist_type, alert_mode, "🚨 Capital Hill | Alert Update")

    change_lines = _build_change_lines(result, previous_signature, watchlist_type)
    if change_lines:
        lines.extend(["", "<b>🔄 WAS HAT SICH GEÄNDERT</b>"])
        lines.extend(change_lines)

    lines.extend(["", "<b>📊 AKTUELLER STAND</b>", _b("Modus:", mode), _b("Setup:", setup_type)])

    if watchlist_type == "Positions-Watchlist":
        lines.extend([
            _b("⚠️ Positions-Aktion:", result.get('position_action', '-')),
            _b("🚪 Exit-Aktion:", result.get('exit_action', '-')),
            _b("📊 Exit-Score:", f"{result.get('exit_score', '-')}/100"),
            _b("🧭 Exit-Hauptgrund:", result.get('exit_reason_top', '-')),
            _b("💰 Teilgewinn:", result.get('partial_profit_action', '-')),
            _b("🛡️ Stop:", result.get('stop_action', '-')),
            _b("🚨 Risiko-Hinweis:", result.get('risk_note', '-')),
            _b("📊 Setup-Confidence:", f"{result.get('setup_confidence', '-')}/100"),
        ])
    else:
        lines.extend([
            _b("⚡ Handlung:", result.get('emp', '-')),
            _b("🔔 Trigger:", result.get('trigger_status', '-')),
            _b("📌 Priorität:", result.get('watchlist_priority', '-')),
            _b("📈 Einstieg:", f"{result.get('trading_case_score', 'n/a')}/100"),
            _b("🏛️ Investment:", f"{result.get('investment_case_score', 'n/a')}/100"),
        ])
        if _should_show_trade_plan(result, watchlist_type):
            entry_zone, stop_value, crv_value = _get_trade_plan_fields(result)
            lines.extend([
                _b("🎯 Entry-Zone:", entry_zone),
                _b("🛡️ Stop-Loss:", _fmt_trade_value(stop_value, "")),
                _b("⚖️ CRV:", _fmt_trade_value(crv_value, ":1")),
            ])

    if red_flag and red_flag != "-":
        lines.append(_b("⛔ Red Flag:", red_flag))

    return "\n".join(lines)


def build_new_watchlist_entry_text(result, watchlist_name, watchlist_type, alert_mode="Standard"):
    lines = _headline_lines(result, watchlist_name, watchlist_type, alert_mode, "🆕 Capital Hill | Erst-Check")
    lines.extend([
        "",
        "<b>📊 AKTUELLER STAND</b>",
        f"⚡ Handlung: {result.get('emp', result.get('position_action', '-'))}",
        _b("🔔 Trigger:", result.get('trigger_status', '-')),
        _b("📌 Priorität:", result.get('watchlist_priority', '-')),
        _b("📈 Einstieg:", f"{result.get('trading_case_score', 'n/a')}/100"),
        _b("🏛️ Investment:", f"{result.get('investment_case_score', 'n/a')}/100"),
        "Hinweis: Neuer Wert in der Watchlist, aktuell noch kein harter Trigger-Alert.",
    ])
    return "\n".join(lines)


def should_alert_for_watchlist_result(result, watchlist_type, alert_mode="Standard"):
    mode = str(alert_mode or "Standard")

    if watchlist_type == "Positions-Watchlist":
        position_action = str(result.get("position_action", "")).lower()
        exit_action = str(result.get("exit_action", "")).lower()
        partial_profit = str(result.get("partial_profit_action", "")).lower()
        risk_note = str(result.get("risk_note", "")).lower()
        setup_conf = float(result.get("setup_confidence", 0) or 0)
        exit_score = float(result.get("exit_score", 0) or 0)

        if mode == "Konservativ":
            return (
                "verkaufen" in exit_action
                or "risiko reduzieren" in exit_action
                or partial_profit.startswith("ja")
                or exit_score >= 65
            )
        if mode == "Früh":
            return (
                "verkaufen" in exit_action
                or "risiko reduzieren" in exit_action
                or "beobachten" in exit_action
                or partial_profit.startswith("ja")
                or "stop enger" in str(result.get("stop_action", "")).lower()
                or setup_conf < 45
                or exit_score >= 45
            )
        return (
            "verkaufen" in exit_action
            or "risiko reduzieren" in exit_action
            or partial_profit.startswith("ja")
            or "stop enger" in str(result.get("stop_action", "")).lower()
            or exit_score >= 55
            or "risiko reduzieren" in position_action
            or "verkaufen" in position_action
        )

    trigger_status = str(result.get("trigger_status", "")).lower()
    priority = str(result.get("watchlist_priority", "")).lower()
    entry_score = float(result.get("trading_case_score", 0) or 0)
    action = str(result.get("emp", "")).lower()

    if mode == "Konservativ":
        return (
            ("aktiv" in trigger_status or "bestätigt" in trigger_status or "breakout" in trigger_status)
            and priority == "hoch"
            and entry_score >= 70
        )
    if mode == "Früh":
        return (
            priority in ["hoch", "mittel"]
            or entry_score >= 60
            or "kauf" in action
            or "aufbau" in action
            or "aktiv" in trigger_status
        )
    return (
        priority == "hoch"
        or entry_score >= 68
        or "aktiv" in trigger_status
        or "kauf" in action
        or "aufbau" in action
    )


def send_telegram_message(message_text):
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        return False, "TELEGRAM_TOKEN oder TELEGRAM_CHAT_ID fehlt."

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message_text, "parse_mode": "HTML", "disable_web_page_preview": True},
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
        previous_any = has_any_alert_history_for_ticker(watchlist_name, ticker)

        if should_alert_for_watchlist_result(result, watchlist_type, alert_mode):
            matched += 1
            alert_type = get_alert_type_label(result, watchlist_type)
            alert_signature = build_alert_signature(result, watchlist_type)
            history_entry = get_alert_history_entry(watchlist_name, ticker, alert_type)

            same_day = False
            same_signature = False
            previous_signature = None
            if history_entry:
                same_day = str(history_entry.get("Last_Sent_Date", "")) == datetime.now().strftime("%Y-%m-%d")
                same_signature = str(history_entry.get("Alert_Signature", "")) == alert_signature
                previous_signature = history_entry.get("Alert_Signature")

            if same_day and same_signature:
                suppressed += 1
                continue

            msg = build_watchlist_telegram_text(
                result,
                watchlist_name,
                watchlist_type,
                alert_mode,
                previous_signature=previous_signature,
            )
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
            if not previous_any and watchlist_type != "Positions-Watchlist":
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
            parts.append(f"{sent - info_sent} Alerts")
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
