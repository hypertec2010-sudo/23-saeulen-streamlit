from datetime import datetime
import html

import requests
import streamlit as st

from logging_utils import get_alert_history_entry, upsert_alert_history_entry

try:
    from logging_utils import load_alert_history_df as _load_alert_history_df
except Exception:
    _load_alert_history_df = None

try:
    from logging_utils import bulk_upsert_alert_history_entries as _bulk_upsert_alert_history_entries
except Exception:
    _bulk_upsert_alert_history_entries = None

try:
    from logging_utils import has_any_alert_history_for_ticker as _has_any_alert_history_for_ticker
except Exception:
    _has_any_alert_history_for_ticker = None


WATCHLIST_ALERT_TYPES = [
    "Trigger-Alert",
    "Prioritäts-Alert",
    "Handlungs-Alert",
    "Watchlist-Info",
    "Neue Watchlist-Aufnahme",
]

POSITION_ALERT_TYPES = [
    "Exit-Alert",
    "Risiko-Alert",
    "Teilgewinn-Alert",
    "Exit-Warnung",
    "Positions-Info",
]

MAX_TELEGRAM_MESSAGE_LEN = 3800


def _item_separator():
    return "━━━━━━━━━━━━"


def _emphasized_value_line(result):
    ticker = _norm_ticker(result.get("ticker", "-"))
    name = _norm_text(result.get("name", "-"))
    return f"📌 <b>WERT: {_esc(name)} | {_esc(ticker)}</b>"


def _norm_text(value):
    return str(value or "").strip()


def _norm_ticker(value):
    return _norm_text(value).upper()


def _esc(value):
    return html.escape(str(value if value is not None else "-"))


def _b(label, value):
    return f"<b>{_esc(label)}</b> {_esc(value)}"


def _is_non_applicable(value):
    raw = str(value if value is not None else "").strip().lower()
    return raw in {"", "-", "n/a", "none", "nicht anwendbar"}


def _append_detail_if_applicable(lines, label, value):
    if _is_non_applicable(value):
        return
    lines.append(_b(label, value))


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


def _rank_priority(value):
    mapping = {"niedrig": 1, "mittel": 2, "hoch": 3}
    return mapping.get(str(value).strip().lower(), 0)


def _numeric_change_threshold(label):
    label = str(label)
    if label in {"📈 Einstieg", "🏛️ Investment", "📊 Setup-Confidence", "📊 Exit-Score"}:
        return 3.0
    return 1.0


def _classify_change(label, old_value, new_value, watchlist_type, numeric=False):
    old_str = str(old_value)
    new_str = str(new_value)

    if numeric:
        try:
            old_num = float(str(old_value).replace("%", "").replace(",", "."))
            new_num = float(str(new_value).replace("%", "").replace(",", "."))
            if abs(old_num - new_num) < _numeric_change_threshold(label):
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


def _normalize_state_label(value):
    return _norm_text(value).lower()


def _material_trigger_change(old_value, new_value):
    old_s = _normalize_state_label(old_value)
    new_s = _normalize_state_label(new_value)
    if old_s == new_s:
        return False
    stage_rank = {
        "jetzt prüfbar": 5,
        "aktiv": 5,
        "fast prüfbar": 4,
        "nahe dran": 4,
        "früh interessant": 3,
        "frühe beobachtung": 3,
        "weiter beobachten": 2,
        "beobachten": 2,
        "noch warten": 1,
        "warten": 1,
        "aktuell kein fokus": 0,
        "passiv": 0,
    }
    return stage_rank.get(old_s, -99) != stage_rank.get(new_s, -99)


def _material_priority_change(old_value, new_value):
    return _rank_priority(old_value) != _rank_priority(new_value)


def _material_text_change(label, old_value, new_value, watchlist_type):
    old_s = _normalize_state_label(old_value)
    new_s = _normalize_state_label(new_value)
    if old_s == new_s:
        return False
    if label == "trigger_status":
        return _material_trigger_change(old_value, new_value)
    if label == "watchlist_priority":
        return _material_priority_change(old_value, new_value)
    if label == "emp":
        action_rank = {
            "buy / accumulate": 5,
            "buy candidate / timing prüfen": 4,
            "watch / einstieg prüfen": 4,
            "beobachten": 2,
            "avoid / wait": 0,
            "halten / ggf. ausbauen": 5,
            "halten / ausbauen": 5,
            "halten / eng beobachten": 3,
            "halten / risiko prüfen": 2,
            "risiko reduzieren / stopp prüfen": 0,
            "veto - earnings < 7 tage": 0,
        }
        return action_rank.get(old_s, -99) != action_rank.get(new_s, -99)
    if watchlist_type == "Positions-Watchlist" and label in {"position_action", "exit_action", "partial_profit_action", "stop_action", "risk_note", "exit_reason_top"}:
        return True
    return old_s != new_s


def _safe_float(value):
    try:
        return float(str(value).replace('%', '').replace(',', '.'))
    except Exception:
        return None


def _is_material_change(result, previous_signature, watchlist_type):
    if not previous_signature:
        return True
    prev = _parse_signature(previous_signature, watchlist_type)
    if watchlist_type == "Positions-Watchlist":
        checks = [
            ("position_action", prev.get("position_action", "-"), result.get("position_action", "-"), False, 0),
            ("exit_action", prev.get("exit_action", "-"), result.get("exit_action", "-"), False, 0),
            ("exit_reason_top", prev.get("exit_reason_top", "-"), result.get("exit_reason_top", "-"), False, 0),
            ("partial_profit_action", prev.get("partial_profit_action", "-"), result.get("partial_profit_action", "-"), False, 0),
            ("stop_action", prev.get("stop_action", "-"), result.get("stop_action", "-"), False, 0),
            ("risk_note", prev.get("risk_note", "-"), result.get("risk_note", "-"), False, 0),
            ("exit_score", prev.get("exit_score", "-"), result.get("exit_score", "-"), True, 5.0),
            ("setup_confidence", prev.get("setup_confidence", "-"), result.get("setup_confidence", "-"), True, 4.0),
        ]
    else:
        checks = [
            ("trigger_status", prev.get("trigger_status", "-"), result.get("trigger_status", "-"), False, 0),
            ("watchlist_priority", prev.get("watchlist_priority", "-"), result.get("watchlist_priority", "-"), False, 0),
            ("emp", prev.get("emp", "-"), result.get("emp", "-"), False, 0),
            ("entry_quality", prev.get("entry_quality", "-"), result.get("entry_quality", "-"), False, 0),
            ("trading_case_score", prev.get("trading_case_score", "-"), result.get("trading_case_score", "-"), True, 5.0),
            ("investment_case_score", prev.get("investment_case_score", "-"), result.get("investment_case_score", "-"), True, 5.0),
            ("setup_confidence", prev.get("setup_confidence", "-"), result.get("setup_confidence", "-"), True, 4.0),
        ]
    for label, old_value, new_value, numeric, threshold in checks:
        if numeric:
            old_num = _safe_float(old_value)
            new_num = _safe_float(new_value)
            if old_num is None or new_num is None:
                if str(old_value) != str(new_value):
                    return True
            elif abs(new_num - old_num) >= threshold:
                return True
        else:
            if _material_text_change(label, old_value, new_value, watchlist_type):
                return True
    return False


def _build_change_summary(result, previous_signature, watchlist_type, max_items=3):
    change_lines = _build_change_lines(result, previous_signature, watchlist_type)
    if not change_lines:
        return "Keine wesentliche Änderung"
    compact = []
    for raw in change_lines[:max_items]:
        txt = raw.replace('<b>', '').replace('</b>', '')
        txt = txt.replace(' -&gt; ', ' -> ')
        compact.append(txt)
    return " | ".join(compact)


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

    return grouped["🟢"] + grouped["🔴"] + grouped["🟡"]


def _headline_lines(result, watchlist_name, watchlist_type, alert_mode, prefix_title):
    alert_type = get_alert_type_label(result, watchlist_type)

    return [
        f"<b>{_esc(prefix_title)}</b>",
        _emphasized_value_line(result),
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
        lines.extend(["", f"<b>🔄 ÄNDERUNG:</b> {_esc(_build_change_summary(result, previous_signature, watchlist_type))}", "<b>🔍 DETAILS</b>"])
        lines.extend(change_lines)

    lines.extend(["", "<b>📊 AKTUELLER STAND</b>", _b("Modus:", mode), _b("Setup:", setup_type)])

    if watchlist_type == "Positions-Watchlist":
        _append_detail_if_applicable(lines, "⚠️ Positions-Aktion:", result.get('position_action', '-'))
        _append_detail_if_applicable(lines, "🚪 Exit-Aktion:", result.get('exit_action', '-'))
        _append_detail_if_applicable(lines, "📊 Exit-Score:", f"{result.get('exit_score', '-')}/100")
        _append_detail_if_applicable(lines, "🧭 Exit-Hauptgrund:", result.get('exit_reason_top', '-'))
        _append_detail_if_applicable(lines, "💰 Teilgewinn:", result.get('partial_profit_action', '-'))
        _append_detail_if_applicable(lines, "🛡️ Stop:", result.get('stop_action', '-'))
        _append_detail_if_applicable(lines, "🚨 Risiko-Hinweis:", result.get('risk_note', '-'))
        _append_detail_if_applicable(lines, "📊 Setup-Confidence:", f"{result.get('setup_confidence', '-')}/100")
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
            priority_or_trigger(result) or
            float(result.get("trading_case_score", 0) or 0) >= 65
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
            or entry_score >= 62
            or "kauf" in action
            or "aufbau" in action
            or "aktiv" in trigger_status
        )
    return (
        priority == "hoch"
        or entry_score >= 65
        or "aktiv" in trigger_status
        or "kauf" in action
        or "aufbau" in action
    )


def priority_or_trigger(result):
    trigger_status = str(result.get("trigger_status", "")).lower()
    priority = str(result.get("watchlist_priority", "")).lower()
    action = str(result.get("emp", "")).lower()
    return (
        priority == "hoch"
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


def _history_cache_from_sheet():
    if not callable(_load_alert_history_df):
        return None, 'load_alert_history_df nicht verfügbar'
    try:
        df, err = _load_alert_history_df()
    except Exception as e:
        return None, str(e)
    if err:
        return None, str(err)

    by_type = {}
    latest_any = {}
    if df is None or getattr(df, 'empty', True):
        return {"by_type": by_type, "latest_any": latest_any}, None

    for _, row in df.iterrows():
        wl = _norm_text(row.get('Watchlist_Name', '')).lower()
        tk = _norm_ticker(row.get('Ticker', ''))
        at = _norm_text(row.get('Alert_Type', ''))
        date_key = _norm_text(row.get('Last_Sent_At', '')) or _norm_text(row.get('Last_Sent_Date', ''))
        entry = row.to_dict()
        type_key = (wl, tk, at)
        any_key = (wl, tk)
        prev = by_type.get(type_key)
        if prev is None or str(prev.get('Last_Sent_At', '') or prev.get('Last_Sent_Date', '')) <= date_key:
            by_type[type_key] = entry
        prev_any = latest_any.get(any_key)
        if prev_any is None or str(prev_any.get('Last_Sent_At', '') or prev_any.get('Last_Sent_Date', '')) <= date_key:
            latest_any[any_key] = entry
    return {"by_type": by_type, "latest_any": latest_any}, None


def _cache_get_entry(history_cache, watchlist_name, ticker, alert_type=None):
    if not history_cache:
        return None
    wl = _norm_text(watchlist_name).lower()
    tk = _norm_ticker(ticker)
    if alert_type is None:
        return history_cache.get('latest_any', {}).get((wl, tk))
    return history_cache.get('by_type', {}).get((wl, tk, _norm_text(alert_type)))


def _cache_has_any(history_cache, watchlist_name, ticker):
    return _cache_get_entry(history_cache, watchlist_name, ticker, None) is not None


def _cache_upsert(history_cache, watchlist_name, watchlist_type, alert_mode, ticker, alert_type, alert_signature):
    if not history_cache:
        return
    now_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    today = datetime.now().strftime('%Y-%m-%d')
    entry = {
        'Watchlist_Name': watchlist_name,
        'Watchlist_Type': watchlist_type,
        'Alert_Mode': alert_mode,
        'Ticker': _norm_ticker(ticker),
        'Alert_Type': alert_type,
        'Alert_Signature': alert_signature,
        'Last_Sent_At': now_ts,
        'Last_Sent_Date': today,
    }
    wl = _norm_text(watchlist_name).lower()
    tk = _norm_ticker(ticker)
    at = _norm_text(alert_type)
    history_cache.setdefault('by_type', {})[(wl, tk, at)] = entry
    history_cache.setdefault('latest_any', {})[(wl, tk)] = entry


def _is_same_signature(history_entry, signature):
    if not history_entry:
        return False
    return _norm_text(history_entry.get('Alert_Signature', '')) == _norm_text(signature)


def _persist_history_updates(history_updates):
    if not history_updates:
        return True, None
    if callable(_bulk_upsert_alert_history_entries):
        try:
            entries = [
                {
                    'watchlist_name': wl_name,
                    'watchlist_type': wl_type,
                    'alert_mode': al_mode,
                    'ticker': ticker,
                    'alert_type': alert_type,
                    'alert_signature': alert_signature,
                }
                for wl_name, wl_type, al_mode, ticker, alert_type, alert_signature in history_updates
            ]
            return _bulk_upsert_alert_history_entries(entries)
        except Exception as e:
            return False, str(e)
    for wl_name, wl_type, al_mode, ticker, alert_type, alert_signature in history_updates:
        ok, err = upsert_alert_history_entry(wl_name, wl_type, al_mode, ticker, alert_type, alert_signature)
        if not ok:
            return False, err
    return True, None


def _get_previous_history_entry_any(watchlist_name, ticker, watchlist_type):
    watchlist_name = _norm_text(watchlist_name)
    ticker = _norm_ticker(ticker)
    alert_types = POSITION_ALERT_TYPES if watchlist_type == "Positions-Watchlist" else WATCHLIST_ALERT_TYPES
    best_entry = None
    best_key = ""
    for alert_type in alert_types:
        try:
            entry = get_alert_history_entry(watchlist_name, ticker, alert_type)
        except Exception:
            entry = None
        if not entry:
            continue
        date_key = str(entry.get("Last_Sent_Date", ""))
        if date_key >= best_key:
            best_key = date_key
            best_entry = entry
    return best_entry


def _has_prior_history_any(watchlist_name, ticker, watchlist_type):
    watchlist_name = _norm_text(watchlist_name)
    ticker = _norm_ticker(ticker)
    if callable(_has_any_alert_history_for_ticker):
        try:
            return bool(_has_any_alert_history_for_ticker(watchlist_name, ticker))
        except Exception:
            pass
    return _get_previous_history_entry_any(watchlist_name, ticker, watchlist_type) is not None


def _send_chunked_messages(header_lines, item_texts):
    if not item_texts:
        return 0, []
    sent = 0
    errors = []
    current = "\n".join(header_lines)
    for item in item_texts:
        candidate = current + "\n\n" + item if current else item
        if len(candidate) > MAX_TELEGRAM_MESSAGE_LEN and current:
            ok, err = send_telegram_message(current)
            if ok:
                sent += 1
            else:
                errors.append(err)
            current = "\n".join(header_lines) + "\n\n" + item
        else:
            current = candidate
    if current:
        ok, err = send_telegram_message(current)
        if ok:
            sent += 1
        else:
            errors.append(err)
    return sent, errors


def send_watchlist_alerts(results, watchlist_name, watchlist_type, alert_mode="Standard", return_diagnostics=False):
    watchlist_name = _norm_text(watchlist_name)

    if not results:
        diagnostics = {"matched": 0, "suppressed": 0, "info_sent": 0, "sent": 0, "errors": [], "reason": "Keine Ergebnisse vorhanden"}
        if return_diagnostics:
            return False, "Keine Ergebnisse für Telegram-Alerts vorhanden.", 0, diagnostics
        return False, "Keine Ergebnisse für Telegram-Alerts vorhanden.", 0

    matched = 0
    suppressed = 0
    info_sent = 0
    errors = []

    history_cache, history_err = _history_cache_from_sheet()
    history_safe_mode = history_cache is not None and not history_err
    if not history_safe_mode:
        errors.append(f"History-Cache: {history_err or 'nicht verfügbar'}")

    alert_items = []
    first_check_items = []
    history_updates = []
    queued_keys = set()

    for result in results:
        ticker = _norm_ticker(result.get("ticker", "-"))
        alert_type = get_alert_type_label(result, watchlist_type)
        alert_signature = build_alert_signature(result, watchlist_type)
        same_type_entry = _cache_get_entry(history_cache, watchlist_name, ticker, alert_type) if history_safe_mode else None
        any_entry = _cache_get_entry(history_cache, watchlist_name, ticker, None) if history_safe_mode else None
        previous_signature = None
        if same_type_entry:
            previous_signature = same_type_entry.get("Alert_Signature")
        elif any_entry:
            previous_signature = any_entry.get("Alert_Signature")

        alert_key = (ticker, alert_type, alert_signature)
        if should_alert_for_watchlist_result(result, watchlist_type, alert_mode):
            matched += 1

            # Never resend an unchanged signature or immaterial changes.
            if _is_same_signature(same_type_entry, alert_signature) or _is_same_signature(any_entry, alert_signature) or alert_key in queued_keys:
                suppressed += 1
                continue
            if previous_signature and not _is_material_change(result, previous_signature, watchlist_type):
                suppressed += 1
                continue

            alert_items.append(
                build_watchlist_telegram_text(
                    result,
                    watchlist_name,
                    watchlist_type,
                    alert_mode,
                    previous_signature=previous_signature,
                )
            )
            queued_keys.add(alert_key)
            history_updates.append((watchlist_name, watchlist_type, alert_mode, ticker, alert_type, alert_signature))
            _cache_upsert(history_cache, watchlist_name, watchlist_type, alert_mode, ticker, alert_type, alert_signature)
        else:
            # First-checks only when history is available and clean. Otherwise we risk endless duplicates.
            if watchlist_type != "Positions-Watchlist" and history_safe_mode and not _cache_has_any(history_cache, watchlist_name, ticker):
                first_alert_type = "Neue Watchlist-Aufnahme"
                first_signature = alert_signature
                first_key = (ticker, first_alert_type, first_signature)
                first_entry = _cache_get_entry(history_cache, watchlist_name, ticker, first_alert_type)
                if _is_same_signature(first_entry, first_signature) or first_key in queued_keys:
                    suppressed += 1
                    continue
                first_check_items.append(build_new_watchlist_entry_text(result, watchlist_name, watchlist_type, alert_mode))
                queued_keys.add(first_key)
                history_updates.append((watchlist_name, watchlist_type, alert_mode, ticker, first_alert_type, first_signature))
                _cache_upsert(history_cache, watchlist_name, watchlist_type, alert_mode, ticker, first_alert_type, first_signature)
                info_sent += 1

    sent = 0

    combined_items = []
    if alert_items:
        combined_items.append(f"<b>🔔 ALERT-ÄNDERUNGEN</b>")
        for item in alert_items:
            combined_items.append(f"{_item_separator()}\n{item}")
    if first_check_items:
        combined_items.append(f"<b>🆕 NEUE WATCHLIST-WERTE</b>")
        for item in first_check_items:
            combined_items.append(f"{_item_separator()}\n{item}")

    if combined_items:
        header = [
            f"<b>📦 Capital Hill | Watchlist-Sammelupdate</b>",
            f"📋 <b>WATCHLIST:</b> {_esc(watchlist_name)} | {_esc(watchlist_type)}",
            f"🧭 <b>MODUS:</b> {_esc(alert_mode)}",
            f"🔔 <b>ALERTS:</b> {_esc(len(alert_items))} | 🆕 <b>ERST-CHECKS:</b> {_esc(len(first_check_items))}",
        ]
        sent_now, errs = _send_chunked_messages(header, combined_items)
        sent += sent_now
        errors.extend(errs)

    if sent > 0 and history_updates:
        hist_ok, hist_err = _persist_history_updates(history_updates)
        if not hist_ok and hist_err:
            errors.append(f"History: {hist_err}")

    diagnostics = {
        "matched": int(matched or 0),
        "suppressed": int(suppressed or 0),
        "info_sent": int(info_sent or 0),
        "sent": int(sent or 0),
        "errors": list(errors[:5]),
        "reason": "",
    }

    if matched == 0 and info_sent == 0:
        diagnostics["reason"] = "Keine alert-relevanten Werte"
        result_tuple = (False, "Keine alert-relevanten Werte in dieser Watchlist gefunden.", 0)
    elif sent > 0 and not [e for e in errors if e.startswith('Telegram') or e.startswith('History')]:
        parts = []
        if matched > 0:
            parts.append(f"{matched} Alert-Kandidaten")
        if info_sent > 0:
            parts.append(f"{info_sent} Erst-Checks")
        msg = f"{', '.join(parts)} für '{watchlist_name}' gesendet."
        if suppressed > 0:
            msg += f" {suppressed} unveränderte Meldungen wurden unterdrückt."
        if history_err:
            msg += " History war nicht vollständig verfügbar; Erstchecks wurden vorsichtshalber nicht erzeugt."
        diagnostics["reason"] = "Alerts gesendet"
        result_tuple = (True, msg, sent)
    elif sent > 0:
        diagnostics["reason"] = "Teilweise gesendet mit Fehlern"
        result_tuple = (True, f"{sent} Telegram-Nachrichten gesendet, einzelne Fehler: {' | '.join(errors[:2])}", sent)
    elif suppressed > 0:
        diagnostics["reason"] = "Nur Dubletten unterdrückt"
        result_tuple = (False, f"Keine neuen Alerts gesendet. {suppressed} unveränderte Meldungen wurden unterdrückt.", 0)
    else:
        diagnostics["reason"] = "Telegram-Versand fehlgeschlagen" if errors else "Keine alert-relevanten Werte"
        result_tuple = (False, f"Telegram-Versand fehlgeschlagen: {' | '.join(errors[:2])}" if errors else "Keine alert-relevanten Werte in dieser Watchlist gefunden.", 0)

    if return_diagnostics:
        return result_tuple[0], result_tuple[1], result_tuple[2], diagnostics
    return result_tuple
