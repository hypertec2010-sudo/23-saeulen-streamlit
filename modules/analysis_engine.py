"""Central analysis facade for the Streamlit application.

Keeps core/legacy fallback and asset post-processing outside the UI script.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


COMMODITY_TICKERS = {
    "GC=F": ("Gold Future", "Gold"), "SI=F": ("Silber Future", "Silber"),
    "HG=F": ("Kupfer Future", "Kupfer"), "CL=F": ("WTI Öl Future", "Öl / Energie"),
    "BZ=F": ("Brent Öl Future", "Öl / Energie"), "NG=F": ("Erdgas Future", "Energie"),
    "PL=F": ("Platin Future", "Platin"), "PA=F": ("Palladium Future", "Palladium"),
    "ZC=F": ("Mais Future", "Agrar"), "ZW=F": ("Weizen Future", "Agrar"),
    "ZS=F": ("Sojabohnen Future", "Agrar"),
}


def infer_asset_type(ticker: str, info: Optional[dict] = None, requested: str = "Auto") -> str:
    t = str(ticker or "").strip().upper()
    requested = str(requested or "Auto")
    if requested != "Auto":
        if "Commodity" in requested or "Rohstoff" in requested:
            return "Commodity"
        if "ETF" in requested:
            return "ETF"
        if "Index" in requested:
            return "Index"
        return "Aktie"
    quote_type = str((info or {}).get("quoteType", "")).upper() if isinstance(info, dict) else ""
    if t.endswith("=F") or t in COMMODITY_TICKERS:
        return "Commodity"
    if t.startswith("^") or quote_type == "INDEX":
        return "Index"
    if quote_type in {"ETF", "MUTUALFUND"}:
        return "ETF"
    return "Aktie"


def postprocess_asset_mode(result: Dict[str, Any], ticker: str, requested: str = "Auto") -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    t = str(ticker or result.get("ticker") or result.get("Ticker") or "").strip().upper()
    info = result.get("info", {}) if isinstance(result.get("info", {}), dict) else {}
    asset_type = infer_asset_type(t, info, requested=requested)
    result["asset_type_label"] = asset_type
    result["Asset_Typ"] = asset_type
    if asset_type == "Commodity":
        display, group = COMMODITY_TICKERS.get(t, (t, "Commodity"))
        result["name"] = display
        result["Name"] = display
        result["sector_label"] = "Rohstoffe"
        result["industry_label"] = group
        result["sector"] = "Rohstoffe"
        result["industry"] = group
        result["Asset_Modellhinweis"] = (
            "Commodity: technische Struktur, Volatilität, Trend, S/R und Fibonacci haben Vorrang; "
            "unternehmensbezogene Fundamentaldaten sind nicht maßgeblich."
        )
    return result


def analyze_stock(
    *, ticker: str, horizon: str, depot: float, risk_pct: float, override: Any,
    buy_in_override: Any, smart_money_default: Any, strict_mode: Any,
    core_engine: Callable[..., Dict[str, Any]], legacy_engine: Callable[..., Dict[str, Any]],
    asset_mode: str = "Auto",
) -> Dict[str, Any]:
    kwargs = dict(
        ticker=ticker, horizon=horizon, depot=depot, risk_pct=risk_pct,
        override=override, buy_in_override=buy_in_override,
        smart_money_default=smart_money_default, strict_mode=strict_mode,
    )
    try:
        result = core_engine(**kwargs)
    except ValueError as exc:
        # v28.4.5b4: The historical/core implementation still rejects young
        # listings before the new-listing capable legacy pipeline can run.
        # Route only the specific insufficient-history failure to the
        # reduced-history engine; all other ValueErrors remain visible.
        text = str(exc or "").lower()
        insufficient_history = (
            "nicht genug kursdaten" in text
            or "belastbare analyse" in text
            or "not enough price data" in text
            or "insufficient price data" in text
        )
        if not insufficient_history:
            raise
        result = legacy_engine(**kwargs)
        if isinstance(result, dict):
            result["Analyse_Hinweis"] = (
                "New-Listing-Fallback genutzt: die Standardanalyse verlangt mehr Historie; "
                "die reduzierte Kurzfrist-/Momentum-Analyse wurde verwendet."
            )
            result["Analyse_Datenmodus"] = result.get("Analyse_Datenmodus") or "New-Listing-Fallback"
    except TypeError as exc:
        if "not supported between instances" not in str(exc):
            raise
        result = legacy_engine(**kwargs)
        result["Analyse_Hinweis"] = (
            "Fallback-Analyse genutzt: einzelne Datenfelder hatten ein uneinheitliches Format."
        )
    return postprocess_asset_mode(result, ticker=ticker, requested=asset_mode)
