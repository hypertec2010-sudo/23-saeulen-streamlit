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
        f"Capital Hill Alert",
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

    return "\\n".join(lines)
