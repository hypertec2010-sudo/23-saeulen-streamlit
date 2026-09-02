"""Portfolio & Risk Engine introduced in v29.1.

The engine is deliberately advisory. It aggregates open positions, checks
exposure/concentration/stop coverage and can pre-check a hypothetical new
position. It never changes positions, orders, live/shadow scores or exits.

Cross-currency values are only aggregated when an explicit FX conversion into
the selected portfolio base currency is available. Missing FX or stale market
data therefore reduce confidence instead of being silently guessed.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
import math
import re

import pandas as pd

_position_exit_engine = None
_infer_quote_currency = None
_storage = None


def configure_context(*, position_exit_engine=None, infer_quote_currency=None, storage=None):
    global _position_exit_engine, _infer_quote_currency, _storage
    if position_exit_engine is not None:
        _position_exit_engine = position_exit_engine
    if infer_quote_currency is not None:
        _infer_quote_currency = infer_quote_currency
    if storage is not None:
        _storage = storage


def load_portfolio_settings() -> dict:
    if _storage is not None:
        try:
            data = _storage.load_namespace("portfolio_risk_settings", default={})
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_portfolio_settings(settings: dict | None) -> bool:
    payload = dict(settings or {})
    if _storage is not None:
        try:
            return bool(_storage.save_namespace("portfolio_risk_settings", payload))
        except Exception:
            return False
    return False


def _blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none", "n/a", "na", "-"}


def _num(value: Any, default=None):
    if _blank(value):
        return default
    try:
        if isinstance(value, str):
            value = (
                value.replace("%", "")
                .replace("/100", "")
                .replace("R", "")
                .replace(" ", "")
                .replace(",", ".")
            )
        out = float(value)
        return default if math.isnan(out) else out
    except Exception:
        return default


def _text(value: Any, default="") -> str:
    return default if _blank(value) else str(value).strip()


def _norm_ticker(value: Any) -> str:
    return _text(value, "").upper()


def _parse_watchlist_key(key: Any) -> str:
    text = _text(key, "Standard")
    prefix = "v244_open_positions::"
    return text[len(prefix):] if text.startswith(prefix) else text


def _updated_sort_key(pos: dict) -> tuple:
    for key in ("updated_at", "opened_at_iso", "created_at"):
        raw = _text((pos or {}).get(key), "")
        if not raw:
            continue
        try:
            iso_like = bool(re.match(r"^\d{4}-\d{2}-\d{2}", raw))
            ts = pd.to_datetime(raw, errors="coerce", dayfirst=not iso_like)
            if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
                return (1, ts.value)
        except Exception:
            pass
    return (0, 0)


# Conservative one-primary-cluster fallback. Manual ``portfolio_group`` always
# wins. The map only covers names that can be classified with high confidence;
# everything else is explicitly left as Sonstige/Unbekannt.
_SEMICONDUCTORS = {
    "NVDA", "AMD", "INTC", "AVGO", "MRVL", "MU", "QCOM", "QRVO", "AMAT",
    "LRCX", "KLAC", "TSM", "ASML", "ARM", "ON", "MCHP", "ADI", "TXN",
    "IFX.DE", "STM", "NXPI", "SNDK",
}
_CLOUD_CYBER = {
    "NOW", "NET", "FTNT", "ZS", "CRWD", "PANW", "SNOW", "DDOG", "DT",
    "MDB", "OKTA", "S", "CYBR", "TEAM", "WDAY", "CRM", "ADBE",
}
_MEGA_TECH = {"AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "ORCL", "IBM"}
_FINANCIALS = {"JPM", "BAC", "GS", "MS", "C", "WFC", "HOOD", "V", "MA", "PYPL"}
_INDUSTRIALS = {"CAT", "HON", "UNP", "RTX", "GE", "ETN", "AXON"}
_HEALTHCARE = {"DHR", "LLY", "UNH", "JNJ", "MRK", "PFE", "ABBV", "TMO"}
_CONSUMER = {"NKE", "LULU", "MCD", "SBUX", "COST", "WMT", "HD", "LOW"}


def infer_portfolio_group(ticker: Any, name: Any = "", manual: Any = "") -> tuple[str, str]:
    manual_text = _text(manual, "")
    if manual_text:
        return manual_text, "Manuell"

    tk = _norm_ticker(ticker)
    nm = _text(name, "").lower()
    if tk in _SEMICONDUCTORS or any(x in nm for x in ("semiconductor", "halbleiter", "microchip")):
        return "Halbleiter", "Heuristik"
    if tk in _CLOUD_CYBER or any(x in nm for x in ("cyber", "cloudflare", "snowflake", "datadog")):
        return "Cloud / Cyber / Software", "Heuristik"
    if tk in _MEGA_TECH:
        return "Mega-Cap Tech", "Heuristik"
    if tk in _FINANCIALS or any(x in nm for x in ("bank", "financial", "payments")):
        return "Finanzen", "Heuristik"
    if tk in _INDUSTRIALS or any(x in nm for x in ("railroad", "aerospace", "industrial")):
        return "Industrie", "Heuristik"
    if tk in _HEALTHCARE or any(x in nm for x in ("pharma", "health", "thermo fisher")):
        return "Gesundheit", "Heuristik"
    if tk in _CONSUMER or any(x in nm for x in ("apparel", "retail", "consumer")):
        return "Konsum", "Heuristik"
    if tk.endswith((".DE", ".PA", ".AS", ".BR", ".MI", ".MC", ".SW", ".ST", ".CO", ".OL", ".HE", ".VI", ".LS", ".L")):
        return "Europa · Sonstige/Unbekannt", "Fallback"
    return "Sonstige/Unbekannt", "Fallback"


def _fallback_currency(ticker: str) -> str:
    tk = _norm_ticker(ticker)
    if tk.endswith((".DE", ".PA", ".AS", ".BR", ".MI", ".MC", ".HE", ".VI", ".LS")):
        return "EUR"
    if tk.endswith(".SW"):
        return "CHF"
    if tk.endswith(".ST"):
        return "SEK"
    if tk.endswith(".CO"):
        return "DKK"
    if tk.endswith(".OL"):
        return "NOK"
    if tk.endswith(".L"):
        return "GBP"
    return "USD"


def infer_currency(ticker: Any, live_row: dict | None = None, pos: dict | None = None) -> str:
    tk = _norm_ticker(ticker)
    if _infer_quote_currency is not None:
        try:
            cur = _infer_quote_currency(tk, live_row or {}, pos or {}, fallback=_fallback_currency(tk))
            cur = _text(cur, "").upper()
            if cur:
                return cur
        except TypeError:
            try:
                cur = _infer_quote_currency(tk, live_row or {}, fallback=_fallback_currency(tk))
                cur = _text(cur, "").upper()
                if cur:
                    return cur
            except Exception:
                pass
        except Exception:
            pass
    return _fallback_currency(tk)


def _live_map(live_df: pd.DataFrame | None) -> dict[str, dict]:
    out = {}
    if not isinstance(live_df, pd.DataFrame) or live_df.empty or "Ticker" not in live_df.columns:
        return out
    for _, row in live_df.iterrows():
        rec = row.to_dict()
        tk = _norm_ticker(rec.get("Ticker"))
        if tk:
            out[tk] = rec
    return out


def _flatten_store(all_positions_store: dict | None, scope_watchlist: str | None = None) -> tuple[list[dict], list[str]]:
    raw_rows = []
    for key, positions in (all_positions_store or {}).items():
        if not isinstance(positions, dict):
            continue
        wl = _parse_watchlist_key(key)
        if scope_watchlist and wl != scope_watchlist:
            continue
        for tk, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            ticker = _norm_ticker(tk or pos.get("ticker"))
            if not ticker:
                continue
            raw_rows.append({"ticker": ticker, "watchlist": wl, "position": dict(pos)})

    # A position can be displayed in more than one organizational watchlist.
    # For total-portfolio aggregation we therefore deduplicate identical tickers
    # and keep the most recently updated record rather than double-counting it.
    by_ticker: dict[str, list[dict]] = {}
    for item in raw_rows:
        by_ticker.setdefault(item["ticker"], []).append(item)
    duplicates = sorted([tk for tk, items in by_ticker.items() if len(items) > 1])
    flattened = []
    for tk, items in by_ticker.items():
        chosen = max(items, key=lambda x: _updated_sort_key(x.get("position") or {}))
        flattened.append(chosen)
    flattened.sort(key=lambda x: x["ticker"])
    return flattened, duplicates


def portfolio_currencies(all_positions_store: dict | None, live_df: pd.DataFrame | None = None, scope_watchlist: str | None = None) -> list[str]:
    lmap = _live_map(live_df)
    items, _ = _flatten_store(all_positions_store, scope_watchlist=scope_watchlist)
    currencies = set()
    for item in items:
        tk = item["ticker"]
        currencies.add(infer_currency(tk, lmap.get(tk, {}), item.get("position") or {}))
    return sorted(c for c in currencies if c)


def _fx_rate(currency: str, base_currency: str, fx_rates: dict | None) -> float | None:
    cur = _text(currency, "").upper()
    base = _text(base_currency, "").upper()
    if not cur or not base:
        return None
    if cur == base:
        return 1.0
    raw = None
    if isinstance(fx_rates, dict):
        raw = fx_rates.get(cur)
        if raw is None:
            raw = fx_rates.get(f"{cur}->{base}")
    rate = _num(raw, None)
    return rate if rate is not None and rate > 0 else None


def _portfolio_score(metrics: dict, rows: list[dict]) -> tuple[float, list[str], list[str]]:
    score = 0.0
    drivers: list[str] = []
    actions: list[str] = []

    exposure = metrics.get("exposure_pct")
    if exposure is not None:
        if exposure > 110:
            score += 26; drivers.append(f"Brutto-Exposure {exposure:.0f}% > 110%")
            actions.append("Leverage/Überinvestition abbauen; keine neue Risikoerhöhung.")
        elif exposure > 100:
            score += 20; drivers.append(f"Exposure {exposure:.0f}% über Depotwert")
            actions.append("Vor neuen Käufen zunächst Cash/Exposure normalisieren.")
        elif exposure > 90:
            score += 12; drivers.append(f"Sehr hohe Kapitalbindung {exposure:.0f}%")
        elif exposure > 80:
            score += 6; drivers.append(f"Hohe Kapitalbindung {exposure:.0f}%")

    max_weight = metrics.get("max_position_pct")
    if max_weight is not None:
        if max_weight > 25:
            score += 22; drivers.append(f"Größte Position {max_weight:.1f}% des Depots")
            actions.append("Größte Position auf Klumpenrisiko prüfen bzw. begrenzen.")
        elif max_weight > 18:
            score += 14; drivers.append(f"Große Einzelposition {max_weight:.1f}%")
        elif max_weight > 12:
            score += 7; drivers.append(f"Einzelposition bereits {max_weight:.1f}%")

    top_cluster = metrics.get("top_cluster_invested_pct")
    top_cluster_name = metrics.get("top_cluster")
    if top_cluster is not None:
        if top_cluster > 50:
            score += 22; drivers.append(f"Cluster {top_cluster_name}: {top_cluster:.0f}% des investierten Kapitals")
            actions.append(f"Kein weiteres Risiko im Cluster '{top_cluster_name}' hinzufügen; Diversifikation oder Abbau prüfen.")
        elif top_cluster > 40:
            score += 15; drivers.append(f"Hohe Cluster-Konzentration {top_cluster_name} ({top_cluster:.0f}%)")
        elif top_cluster > 30:
            score += 8; drivers.append(f"Cluster {top_cluster_name} bereits {top_cluster:.0f}%")

    stop_cov = metrics.get("stop_coverage_value_pct")
    if stop_cov is not None:
        if stop_cov < 70:
            score += 18; drivers.append(f"Stop-Abdeckung nur {stop_cov:.0f}% des bewerteten Exposures")
            actions.append("Fehlende Stops/Invalidierungen zuerst ergänzen.")
        elif stop_cov < 90:
            score += 9; drivers.append(f"Stop-Abdeckung unvollständig ({stop_cov:.0f}%)")

    risk_pct = metrics.get("risk_to_stop_pct")
    if risk_pct is not None:
        if risk_pct > 5:
            score += 20; drivers.append(f"Offenes Risiko bis Stop {risk_pct:.1f}% des Depots")
            actions.append("Stop-Risiko/Positionsgrößen reduzieren; Gesamtrisiko ist hoch.")
        elif risk_pct > 3:
            score += 12; drivers.append(f"Risiko bis Stop {risk_pct:.1f}% des Depots")
        elif risk_pct > 2:
            score += 6; drivers.append(f"Risiko bis Stop {risk_pct:.1f}% des Depots")

    breached = int(metrics.get("stop_breached_count") or 0)
    if breached:
        score += min(26.0, 12.0 + 7.0 * max(0, breached - 1))
        drivers.append(f"{breached} Position(en) auf/unter Stop")
        actions.append("Stop-verletzte Positionen vor neuen Käufen priorisiert prüfen.")

    exit_share = metrics.get("defensive_exit_value_pct")
    if exit_share is not None:
        if exit_share > 40:
            score += 14; drivers.append(f"{exit_share:.0f}% des Exposures mit orange/roter Exit Engine")
            actions.append("Neue Käufe pausieren, bis bestehender Exit-Druck abgearbeitet ist.")
        elif exit_share > 20:
            score += 7; drivers.append(f"{exit_share:.0f}% des Exposures mit erhöhtem Exit-Druck")

    # Stale/FX coverage primarily changes confidence. Only very weak coverage
    # contributes a small defensive penalty so the engine cannot look green on
    # a mostly unknown portfolio.
    data_cov = metrics.get("fresh_value_coverage_pct")
    fx_cov = metrics.get("fx_value_coverage_pct")
    if data_cov is not None and data_cov < 60:
        score += 6; drivers.append(f"Aktuelle Kursabdeckung nur {data_cov:.0f}%")
        actions.append("Weitere Positions-Watchlists vollständig scannen, bevor Portfolio-Risiko als aktuell gilt.")
    if fx_cov is not None and fx_cov < 100:
        score += 5; drivers.append(f"FX-Abdeckung {fx_cov:.0f}%")
        actions.append("Fehlende FX-Umrechnung in die Depot-Basiswährung ergänzen.")

    # Stable unique order.
    seen = set()
    actions = [x for x in actions if not (x in seen or seen.add(x))]
    return min(100.0, round(score, 1)), drivers[:6], actions[:6]


def build_portfolio_package(
    all_positions_store: dict | None,
    live_df: pd.DataFrame | None,
    *,
    account_size: float | None,
    base_currency: str = "EUR",
    fx_rates: dict | None = None,
    scope_watchlist: str | None = None,
) -> dict:
    """Build a portfolio-level risk package without new provider calls."""
    base = _text(base_currency, "EUR").upper()
    account = _num(account_size, None)
    if account is not None and account <= 0:
        account = None

    lmap = _live_map(live_df)
    items, duplicates = _flatten_store(all_positions_store, scope_watchlist=scope_watchlist)
    rows: list[dict] = []

    for item in items:
        tk = item["ticker"]
        wl = item["watchlist"]
        pos = item.get("position") or {}
        live = lmap.get(tk, {})
        fresh = bool(live)
        shares = max(0.0, _num(pos.get("shares"), 0.0) or 0.0)
        entry = _num(pos.get("entry"), None)
        stop = _num(pos.get("stop"), None)
        initial_stop = _num(pos.get("initial_stop"), None)
        live_current = _num(live.get("Kurs"), None) if fresh else None
        has_fresh_price = bool(fresh and live_current is not None)
        current = live_current
        if current is None:
            current = _num(pos.get("last_price"), None)
        name = _text(pos.get("name") or live.get("Name"), tk)
        currency = infer_currency(tk, live, pos)
        fx = _fx_rate(currency, base, fx_rates)
        raw_value = (current * shares) if current is not None and shares > 0 else None
        value_base = (raw_value * fx) if raw_value is not None and fx is not None else None
        pnl_raw = ((current - entry) * shares) if current is not None and entry is not None and shares > 0 else None
        pnl_base = (pnl_raw * fx) if pnl_raw is not None and fx is not None else None
        pnl_pct = ((current / entry - 1.0) * 100.0) if current is not None and entry is not None and entry > 0 else None
        stop_valid = bool(stop is not None and stop > 0)
        stop_breached = bool(has_fresh_price and current is not None and stop_valid and current <= stop)
        giveback_raw = None
        if current is not None and stop_valid and shares > 0:
            giveback_raw = max(0.0, current - stop) * shares
        giveback_base = (giveback_raw * fx) if giveback_raw is not None and fx is not None else None

        # Capital-at-risk at the current stop is measured from cost basis, not
        # from the current unrealised profit. A stop above entry therefore locks
        # capital instead of creating artificial portfolio risk. If a fresh price
        # is already below the stop, use that fresh price as the defensive loss
        # reference and flag the breach separately.
        capital_risk_raw = None
        if entry is not None and entry > 0 and stop_valid and shares > 0:
            risk_exit = current if (stop_breached and current is not None) else stop
            capital_risk_raw = max(0.0, entry - risk_exit) * shares
        risk_base = (capital_risk_raw * fx) if capital_risk_raw is not None and fx is not None else None
        initial_risk_raw = None
        if entry is not None and initial_stop is not None and entry > initial_stop > 0 and shares > 0:
            initial_risk_raw = (entry - initial_stop) * shares
        initial_risk_base = (initial_risk_raw * fx) if initial_risk_raw is not None and fx is not None else None

        group, group_source = infer_portfolio_group(tk, name, pos.get("portfolio_group"))
        ctx = pos.get("last_context") if isinstance(pos.get("last_context"), dict) else {}
        market = _text(live.get("Marktregime") or ctx.get("market_regime"), "n/a")
        volatility = _text(live.get("Volatilitätsregime") or ctx.get("volatility_regime"), "n/a")

        exit_pkg = {}
        if _position_exit_engine is not None:
            try:
                exit_pkg = _position_exit_engine(pos, live)
                if not isinstance(exit_pkg, dict):
                    exit_pkg = {}
            except Exception:
                exit_pkg = {}

        rows.append({
            "Ticker": tk,
            "Name": name,
            "Watchlist": wl,
            "Portfolio-Gruppe": group,
            "Gruppenquelle": group_source,
            "Währung": currency,
            "FX": fx,
            "Stück": int(shares) if float(shares).is_integer() else shares,
            "Kurs": current,
            "Kursbasis": "Atomic aktuell" if has_fresh_price else ("Gespeichert / nicht aktuell" if current is not None else "Fehlt"),
            "Aktuell": has_fresh_price,
            "Positionswert Basis": value_base,
            "P/L Basis": pnl_base,
            "P/L %": pnl_pct,
            "Stop": stop,
            "Stop vorhanden": stop_valid,
            "Stop verletzt": stop_breached,
            "Risiko bis Stop Basis": risk_base,
            "Giveback bis Stop Basis": giveback_base,
            "Initialrisiko Basis": initial_risk_base,
            "Exit-Ampel": _text(exit_pkg.get("ampel"), "⚪"),
            "Exit-Level": _text(exit_pkg.get("level"), "neutral"),
            "Exit-Druck": _num(exit_pkg.get("score"), None),
            "Führung": _text(exit_pkg.get("action"), "-"),
            "Marktregime": market,
            "Volatilität": volatility,
        })

    if not rows:
        return {
            "rows": pd.DataFrame(), "clusters": pd.DataFrame(), "metrics": {},
            "score": None, "ampel": "⚪", "status": "Keine offenen Positionen",
            "action": "Offene Positionen erfassen.", "drivers": [], "actions": [],
            "confidence": "Keine Daten", "duplicates": duplicates,
        }

    df = pd.DataFrame(rows)
    valued = df[df["Positionswert Basis"].notna()].copy()
    total_value = float(valued["Positionswert Basis"].sum()) if not valued.empty else 0.0
    all_raw_value_available = df["Kurs"].notna() & (pd.to_numeric(df["Stück"], errors="coerce").fillna(0) > 0)
    raw_count = int(all_raw_value_available.sum())
    fx_valued_count = int(df["Positionswert Basis"].notna().sum())
    fresh_valued = valued[valued["Aktuell"] == True] if not valued.empty else valued
    fresh_value = float(fresh_valued["Positionswert Basis"].sum()) if not fresh_valued.empty else 0.0
    stop_valued = valued[valued["Stop vorhanden"] == True] if not valued.empty else valued
    stop_value = float(stop_valued["Positionswert Basis"].sum()) if not stop_valued.empty else 0.0
    defensive = valued[valued["Exit-Level"].isin(["orange", "red"])] if not valued.empty else valued
    defensive_value = float(defensive["Positionswert Basis"].sum()) if not defensive.empty else 0.0
    total_risk = float(pd.to_numeric(df["Risiko bis Stop Basis"], errors="coerce").fillna(0).sum())
    total_pnl = float(pd.to_numeric(df["P/L Basis"], errors="coerce").fillna(0).sum())

    if account is not None:
        exposure_pct = total_value / account * 100.0
        cash = account - total_value
        cash_pct = cash / account * 100.0
        risk_to_stop_pct = total_risk / account * 100.0
    else:
        exposure_pct = cash = cash_pct = risk_to_stop_pct = None

    if account is not None and total_value > 0:
        df["Gewicht Depot %"] = df["Positionswert Basis"].apply(lambda x: None if pd.isna(x) else float(x) / account * 100.0)
    else:
        df["Gewicht Depot %"] = None
    if total_value > 0:
        df["Gewicht investiert %"] = df["Positionswert Basis"].apply(lambda x: None if pd.isna(x) else float(x) / total_value * 100.0)
    else:
        df["Gewicht investiert %"] = None

    cluster_rows = []
    if not valued.empty:
        grp = valued.groupby("Portfolio-Gruppe", dropna=False)["Positionswert Basis"].agg(["sum", "count"]).reset_index()
        grp = grp.sort_values("sum", ascending=False)
        for _, rec in grp.iterrows():
            val = float(rec["sum"] or 0.0)
            cluster_rows.append({
                "Portfolio-Gruppe": rec["Portfolio-Gruppe"],
                "Positionen": int(rec["count"] or 0),
                f"Wert {base}": val,
                "Anteil investiert %": (val / total_value * 100.0) if total_value > 0 else None,
                "Anteil Depot %": (val / account * 100.0) if account else None,
            })
    cluster_df = pd.DataFrame(cluster_rows)

    max_position_pct = None
    if account is not None and "Gewicht Depot %" in df.columns:
        vals = pd.to_numeric(df["Gewicht Depot %"], errors="coerce").dropna()
        if not vals.empty:
            max_position_pct = float(vals.max())
    top3_invested = None
    inv_vals = pd.to_numeric(df["Gewicht investiert %"], errors="coerce").dropna().sort_values(ascending=False)
    if not inv_vals.empty:
        top3_invested = float(inv_vals.head(3).sum())
    top_cluster = None
    top_cluster_pct = None
    if not cluster_df.empty:
        top_cluster = _text(cluster_df.iloc[0].get("Portfolio-Gruppe"), "n/a")
        top_cluster_pct = _num(cluster_df.iloc[0].get("Anteil investiert %"), None)

    fx_count_cov = (fx_valued_count / raw_count * 100.0) if raw_count > 0 else 0.0
    # Value coverage cannot be measured for rows with missing FX without already
    # knowing their converted value. Count coverage is therefore the honest FX
    # diagnostic; once all rates exist it becomes exactly 100%.
    fx_value_cov = 100.0 if raw_count > 0 and fx_valued_count == raw_count else fx_count_cov
    fresh_value_cov = (fresh_value / total_value * 100.0) if total_value > 0 else 0.0
    stop_value_cov = (stop_value / total_value * 100.0) if total_value > 0 else 0.0
    defensive_pct = (defensive_value / total_value * 100.0) if total_value > 0 else 0.0

    metrics = {
        "positions": len(df),
        "valued_positions": fx_valued_count,
        "total_value": total_value,
        "account_size": account,
        "cash": cash,
        "cash_pct": cash_pct,
        "exposure_pct": exposure_pct,
        "total_pnl": total_pnl,
        "risk_to_stop": total_risk,
        "risk_to_stop_pct": risk_to_stop_pct,
        "max_position_pct": max_position_pct,
        "top3_invested_pct": top3_invested,
        "top_cluster": top_cluster,
        "top_cluster_invested_pct": top_cluster_pct,
        "fresh_value_coverage_pct": fresh_value_cov,
        "fx_value_coverage_pct": fx_value_cov,
        "stop_coverage_value_pct": stop_value_cov,
        "defensive_exit_value_pct": defensive_pct,
        "stop_breached_count": int(df["Stop verletzt"].fillna(False).astype(bool).sum()),
        "missing_fx_count": max(0, raw_count - fx_valued_count),
        "stale_count": int((~df["Aktuell"].fillna(False).astype(bool)).sum()),
        "duplicates_count": len(duplicates),
    }

    score, drivers, actions = _portfolio_score(metrics, rows)
    if score >= 65:
        ampel, status, action = "🔴", "Hohes Portfolio-Risiko", "Gesamtrisiko aktiv reduzieren"
    elif score >= 45:
        ampel, status, action = "🟠", "Erhöhtes Portfolio-Risiko", "Keine neue Risikoerhöhung · Konzentration/Stops prüfen"
    elif score >= 26:
        ampel, status, action = "🟡", "Portfolio-Risiko beobachten", "Neue Trades nur selektiv und portfolio-aware"
    else:
        ampel, status, action = "🟢", "Portfolio-Risiko im Rahmen", "Neue Trades nur bei passender Gesamtallokation"

    fresh_cov = metrics.get("fresh_value_coverage_pct") or 0.0
    fx_cov = metrics.get("fx_value_coverage_pct") or 0.0
    if fx_cov < 100:
        confidence = "Reduziert · FX unvollständig"
    elif fresh_cov < 60:
        confidence = "Reduziert · Kursdaten überwiegend nicht aktuell"
    elif fresh_cov < 95:
        confidence = "Mittel · teilweise gespeicherte Kurse"
    else:
        confidence = "Hoch · aktuelle Datenbasis"

    # Missing-data guard: never show a green portfolio clearance if material
    # valuation or freshness is missing.
    if ampel == "🟢" and (fx_cov < 100 or fresh_cov < 80):
        ampel = "⚪"
        status = "Datenbasis noch nicht vollständig"
        action = "Erst FX-/Kursabdeckung vervollständigen"

    display_df = df.copy()
    sort_cols = [c for c in ["Exit-Druck", "Gewicht Depot %", "Positionswert Basis"] if c in display_df.columns]
    if sort_cols:
        display_df = display_df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")

    return {
        "rows": display_df.reset_index(drop=True),
        "clusters": cluster_df.reset_index(drop=True),
        "metrics": metrics,
        "score": score,
        "ampel": ampel,
        "status": status,
        "action": action,
        "drivers": drivers,
        "actions": actions,
        "confidence": confidence,
        "duplicates": duplicates,
        "base_currency": base,
    }


def assess_candidate(package: dict | None, candidate_row: dict | None, planned_value_base: float, *, group_override: Any = "") -> dict:
    """Assess a hypothetical new position against an existing portfolio package."""
    pkg = package or {}
    metrics = pkg.get("metrics") if isinstance(pkg.get("metrics"), dict) else {}
    rows = pkg.get("rows") if isinstance(pkg.get("rows"), pd.DataFrame) else pd.DataFrame()
    row = dict(candidate_row or {})
    planned = _num(planned_value_base, None)
    account = _num(metrics.get("account_size"), None)
    current_total = _num(metrics.get("total_value"), 0.0) or 0.0
    if planned is None or planned <= 0:
        return {"ok": False, "error": "Geplanten Positionswert > 0 eingeben."}
    if account is None or account <= 0:
        return {"ok": False, "error": "Für den Kandidatencheck wird ein gültiger Gesamtdepotwert benötigt."}

    ticker = _norm_ticker(row.get("Ticker"))
    name = _text(row.get("Name"), ticker)
    group, group_source = infer_portfolio_group(ticker, name, group_override)
    current_cluster = 0.0
    existing_ticker_value = 0.0
    if not rows.empty and "Portfolio-Gruppe" in rows.columns and "Positionswert Basis" in rows.columns:
        mask = rows["Portfolio-Gruppe"].astype(str) == group
        current_cluster = float(pd.to_numeric(rows.loc[mask, "Positionswert Basis"], errors="coerce").fillna(0).sum())
        if "Ticker" in rows.columns:
            tk_mask = rows["Ticker"].astype(str).str.upper() == ticker
            existing_ticker_value = float(pd.to_numeric(rows.loc[tk_mask, "Positionswert Basis"], errors="coerce").fillna(0).sum())
    new_total = current_total + planned
    projected_exposure = new_total / account * 100.0
    candidate_weight = planned / account * 100.0
    projected_ticker_weight = (existing_ticker_value + planned) / account * 100.0
    projected_cluster_invested = ((current_cluster + planned) / new_total * 100.0) if new_total > 0 else 0.0
    projected_cluster_depot = (current_cluster + planned) / account * 100.0

    points = 0
    reasons = []
    if projected_exposure > 100:
        points += 3; reasons.append(f"Exposure würde auf {projected_exposure:.0f}% steigen")
    elif projected_exposure > 90:
        points += 2; reasons.append(f"Exposure danach sehr hoch ({projected_exposure:.0f}%)")
    elif projected_exposure > 80:
        points += 1; reasons.append(f"Exposure danach {projected_exposure:.0f}%")
    if projected_ticker_weight > 25:
        points += 3; reasons.append(f"{ticker} läge nach dem Kauf bei {projected_ticker_weight:.1f}% des Depots")
    elif projected_ticker_weight > 18:
        points += 2; reasons.append(f"{ticker} wäre nach dem Kauf eine große Einzelposition ({projected_ticker_weight:.1f}%)")
    elif projected_ticker_weight > 12:
        points += 1; reasons.append(f"{ticker} läge nach dem Kauf bei {projected_ticker_weight:.1f}%")
    if projected_cluster_invested > 50:
        points += 3; reasons.append(f"Cluster '{group}' würde {projected_cluster_invested:.0f}% des investierten Kapitals ausmachen")
    elif projected_cluster_invested > 40:
        points += 2; reasons.append(f"Cluster '{group}' würde auf {projected_cluster_invested:.0f}% steigen")
    elif projected_cluster_invested > 30:
        points += 1; reasons.append(f"Cluster '{group}' danach bereits {projected_cluster_invested:.0f}%")

    if points >= 6:
        ampel, verdict = "🔴", "Nicht portfolio-konform in dieser Größe"
    elif points >= 3:
        ampel, verdict = "🟠", "Größe reduzieren / anderes Risiko abbauen"
    elif points >= 1:
        ampel, verdict = "🟡", "Selektiv möglich · Konzentration beachten"
    else:
        ampel, verdict = "🟢", "Aus Portfolio-Sicht unauffällig"

    return {
        "ok": True,
        "ampel": ampel,
        "verdict": verdict,
        "ticker": ticker,
        "group": group,
        "group_source": group_source,
        "planned_value": planned,
        "candidate_weight_pct": candidate_weight,
        "existing_ticker_value": existing_ticker_value,
        "projected_ticker_weight_pct": projected_ticker_weight,
        "projected_exposure_pct": projected_exposure,
        "projected_cluster_invested_pct": projected_cluster_invested,
        "projected_cluster_depot_pct": projected_cluster_depot,
        "reasons": reasons or ["Keine dominante Portfolio-Konzentration durch den geplanten Positionswert."],
    }
