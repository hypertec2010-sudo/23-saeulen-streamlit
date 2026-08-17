"""Legacy-compatible stock analysis pipeline.

v28.3 moves the large analysis implementation out of ``legacy_app.py`` while
keeping its proven calculations and return schema unchanged.  The function is
isolated from Streamlit and receives its legacy helper dependencies explicitly
through :func:`configure_context`.
"""
from __future__ import annotations

from typing import Any, Mapping

_REQUIRED_CONTEXT = frozenset(['adx14', 'ampel', 'analyst_label', 'bollinger_bands', 'build_chart_structures', 'build_company_summary', 'build_decision_explanation', 'build_market_fomo_package_v1525', 'build_red_flags', 'build_short_thesis', 'build_stock_fomo_package_v1525', 'calc_accumulation_distribution_days', 'calc_accumulation_score', 'calc_base_length_days', 'calc_base_quality_score', 'calc_breakout_day_volume_ratio', 'calc_breakout_volume_score', 'calc_cashflow_stability_score', 'calc_catalyst_score', 'calc_close_near_day_high', 'calc_correction_depth_pct', 'calc_distribution_pressure_score', 'calc_earnings_event_score', 'calc_event_risk_score', 'calc_higher_lows_score', 'calc_industry_strength_score', 'calc_institutional_quality_score', 'calc_leadership_score', 'calc_margin_stability_score', 'calc_post_earnings_reaction', 'calc_post_earnings_reaction_score', 'calc_pullback_dryup_score', 'calc_pullback_quality_score', 'calc_range_tightness_score', 'calc_recent_pullback_volume_ratio', 'calc_return_metrics', 'calc_revision_momentum_score', 'calc_rs_acceleration_score', 'calc_rs_benchmark_score', 'calc_sector_strength_score', 'calc_setup_priority_score', 'calc_setup_type_quality_score', 'calc_slope_pct', 'calc_trend_quality_score', 'calc_up_down_volume_ratio', 'calc_volatility_contraction_score', 'calc_volume_quality_proxy', 'calc_volume_quality_score', 'calc_volume_trend_score', 'catalyst_label', 'clamp', 'classify_event_phase', 'combine_fomo_packages_v1525', 'compute_chart_df', 'date', 'datetime', 'entry_quality_score', 'evaluate_chart_structure_bias', 'evaluate_market_filter', 'event_phase_text', 'fmt_num', 'format_price_zone', 'get_leadership_status', 'get_sector_etf_symbol', 'get_style_sector_adjustment', 'ideal_range_score', 'infer_data_source_flags', 'infer_display_currency', 'infer_market_bucket', 'infer_stock_style_advanced', 'institutional_quality_label', 'investment_case_label', 'is_hard_red_flag_v1604', 'known_ratio', 'linear_score', 'load_benchmark_data', 'load_data', 'load_extended_market_quote', 'load_sector_context', 'normalize_missing', 'normalize_tb_score_100', 'np', 'pd', 'rsi14', 'safe_last', 'sanitize_quality_red_flags_v1601', 'select_benchmark', 'setup_confidence_label', 'soften_growth_red_flag_item_v1604', 'stoch14', 'strength_text', 'tb_signal_label', 'timedelta', 'timezone', 'tradeability_label', 'trading_case_label', 'trading_timing_label', 'true_range', 'williams_r'])
_CONTEXT: dict[str, Any] = {}

# Defensive compatibility default: the legacy pipeline referenced this label
# in one position-management branch without defining it locally.
signal_conflict_label = "-"


def configure_context(context: Mapping[str, Any] | None = None, **dependencies: Any) -> None:
    """Bind helper functions, imports and constants required by the pipeline.

    Only names that are actually referenced by the extracted analysis function
    are copied. This avoids importing the Streamlit UI or mutating unrelated
    module state.
    """
    provided: dict[str, Any] = {}
    if context:
        provided.update({name: context[name] for name in _REQUIRED_CONTEXT if name in context})
    provided.update({name: value for name, value in dependencies.items() if name in _REQUIRED_CONTEXT})
    _CONTEXT.update(provided)
    globals().update(provided)


def reset_context() -> None:
    """Clear injected dependencies; intended for deterministic tests."""
    for name in tuple(_CONTEXT):
        globals().pop(name, None)
    _CONTEXT.clear()


def missing_context() -> tuple[str, ...]:
    """Return unresolved legacy dependencies in deterministic order."""
    return tuple(sorted(name for name in _REQUIRED_CONTEXT if name not in _CONTEXT))


def context_status() -> dict[str, Any]:
    """Small diagnostic payload used by deployment and regression checks."""
    missing = missing_context()
    return {
        "required": len(_REQUIRED_CONTEXT),
        "configured": len(_REQUIRED_CONTEXT) - len(missing),
        "missing": missing,
    }

def _legacy_analyze_stock_impl(
    ticker,
    horizon,
    depot,
    risk_pct,
    override,
    buy_in_override,
    smart_money_default,
    strict_mode
):
    df, info = load_data(ticker)

    # v28.4.5b: New listings are no longer rejected merely because MA200 is
    # unavailable. 20 trading days are enough for a deliberately reduced
    # short-term assessment; unavailable long-term indicators remain NaN and
    # therefore cannot contribute positive long-term evidence.
    history_days = int(len(df))
    if df.empty or history_days < 20:
        raise ValueError(f"Noch zu wenig Kursdaten ({history_days} Handelstage). Mindestens 20 werden für die reduzierte New-Listing-Analyse benötigt. Bei neuen Listings bitte den Smart-Provider-Datenpfad prüfen.")
    if history_days >= 250:
        history_mode = "Vollanalyse"
    elif history_days >= 120:
        history_mode = "Reduzierte Analyse · MA200 noch eingeschränkt"
    elif history_days >= 60:
        history_mode = "Reduzierte Swing-Analyse"
    else:
        history_mode = "New Listing · Kurzfristanalyse"

    benchmark_symbol, benchmark_label = select_benchmark(ticker, info)
    benchmark_df = load_benchmark_data(benchmark_symbol)
    market_info = evaluate_market_filter(benchmark_df)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    price = float(override) if override > 0 else float(close.iloc[-1])
    live_quote = load_extended_market_quote(ticker)
    live_price = live_quote.get("display_price", np.nan) if isinstance(live_quote, dict) else np.nan
    live_price_source = live_quote.get("source", "Schlusskurs") if isinstance(live_quote, dict) else "Schlusskurs"
    live_price_diff_pct = live_quote.get("diff_pct", np.nan) if isinstance(live_quote, dict) else np.nan
    live_price_note = live_quote.get("note", "") if isinstance(live_quote, dict) else ""

    # v16.1.2: In der Kursbasis-Karte niemals "Aktuell: n/a" anzeigen.
    # Wenn Yahoo/yfinance keinen belastbaren Live-/Pre-/After-Hours-Kurs liefert,
    # ist der aktuelle Vergleichswert schlicht die reguläre Analysebasis.
    if not pd.notna(live_price):
        live_price = float(price) if pd.notna(price) else np.nan
        live_price_source = "Schlusskurs / Analysebasis"
        live_price_diff_pct = 0.0 if pd.notna(live_price) else np.nan
        live_price_note = "Kein separater Live-/Vor-/Nachbörsenkurs verfügbar; angezeigt wird die reguläre Analysebasis."

    name = info.get("longName") or info.get("shortName") or info.get("displayName") or info.get("quoteSourceName") or ticker
    raw_ccy = info.get("currency", "USD")
    ccy = infer_display_currency(ticker, info, raw_ccy)
    exch = info.get("exchange", "-")
    ts = df.index[-1].strftime("%d.%m.%Y")
    sector = info.get("sector", "-")
    industry = info.get("industry", "-")

    company_summary = build_company_summary(info, ticker)

    confidence_info = infer_data_source_flags(info)

    # ---------- Technicals ----------
    ma10_series = close.rolling(10).mean()
    ma20_series = close.rolling(20).mean()
    ma50_series = close.rolling(50).mean()
    ma150_series = close.rolling(150).mean()
    ma200_series = close.rolling(200).mean()

    ma10 = safe_last(ma10_series)
    ma20 = safe_last(ma20_series)
    ma50 = safe_last(ma50_series)
    ma150 = safe_last(ma150_series)
    ma200 = safe_last(ma200_series)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_v = safe_last(macd)
    signal_v = safe_last(signal)
    macd_up = macd_v > signal_v

    rsi = safe_last(rsi14(close))
    adx = safe_last(adx14(high, low, close))
    tr = true_range(high, low, close)
    atr = safe_last(tr.rolling(14).mean())
    atr_pct = atr / price * 100 if price else 0

    ret_metrics = calc_return_metrics(close)
    ret21 = ret_metrics["ret21"]
    ret63 = ret_metrics["ret63"]
    ret126 = ret_metrics["ret126"]
    roc20 = safe_last(close.pct_change(20) * 100)
    roc60 = safe_last(close.pct_change(60) * 100)
    ret5 = safe_last(close.pct_change(5) * 100, 0)
    ret20 = safe_last(close.pct_change(20) * 100, 0)

    # v15.26.2: 10-Tage-Linie als kurzfristiger Timing-/Pullback-Kontext
    ma10_dist_pct = ((price / ma10 - 1) * 100) if pd.notna(price) and pd.notna(ma10) and ma10 else np.nan
    ma10_slope = calc_slope_pct(ma10_series, lookback=5)
    if pd.isna(ma10_dist_pct):
        ma10_timing_label = "n/a"
        ma10_timing_text = "10-Tage-Linie noch nicht belastbar."
    elif price >= ma10 and abs(ma10_dist_pct) <= 2.0 and pd.notna(ma10_slope) and ma10_slope >= 0:
        ma10_timing_label = "konstruktiv"
        ma10_timing_text = "Kurs hält die 10-Tage-Linie; kurzfristiges Timing wirkt konstruktiv."
    elif price >= ma10 and ma10_dist_pct > 5.0:
        ma10_timing_label = "gedehnt"
        ma10_timing_text = "Kurs liegt deutlich über der 10-Tage-Linie; nicht hinterherlaufen, Rücksetzer/Bestätigung bevorzugen."
    elif price < ma10 and ma10_dist_pct >= -2.5:
        ma10_timing_label = "prüfen"
        ma10_timing_text = "Kurs testet die 10-Tage-Linie; Reaktion daran entscheidet über kurzfristige Stärke."
    elif price < ma10:
        ma10_timing_label = "angeschlagen"
        ma10_timing_text = "Kurs liegt unter der 10-Tage-Linie; kurzfristiges Momentum ist angeschlagen."
    else:
        ma10_timing_label = "neutral"
        ma10_timing_text = "10-Tage-Linie liefert aktuell kein klares Zusatzsignal."

    vol20 = safe_last(vol.rolling(20).mean(), 1)
    vol5 = safe_last(vol.rolling(5).mean(), 1)
    vol_ratio = vol5 / vol20 if vol20 else 1

    high52 = safe_last(close.rolling(252).max(), float(close.max()))
    low52 = safe_last(close.rolling(252).min(), float(close.min()))
    dist52 = price / high52 * 100 if high52 else 50

    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    obv_trend = "steigend" if float(obv.iloc[-1]) > float(obv.iloc[-20]) else "fallend"

    stoch_k, stoch_d = stoch14(high, low, close)
    stoch_k_v = safe_last(stoch_k, 50)
    stoch_d_v = safe_last(stoch_d, 50)
    willr_v = safe_last(williams_r(high, low, close), -50)

    bb_mid, bb_upper_s, bb_lower_s, bb_width_s = bollinger_bands(close)
    bb_upper = safe_last(bb_upper_s, np.nan)
    bb_lower = safe_last(bb_lower_s, np.nan)
    bb_width = safe_last(bb_width_s, np.nan)
    bb_width_thresh = safe_last(bb_width_s.rolling(60).quantile(0.2), np.nan)
    bb_squeeze = pd.notna(bb_width) and pd.notna(bb_width_thresh) and bb_width <= bb_width_thresh
    prev20_high = safe_last(close.shift(1).rolling(20).max(), np.nan)
    prev20_low = safe_last(close.shift(1).rolling(20).min(), np.nan)

    macd_hist_series = macd - signal
    macd_hist_current = safe_last(macd_hist_series, 0)
    macd_hist_prev = safe_last(macd_hist_series.shift(1), 0)
    macd_bull_cross = macd_v > signal_v and macd_hist_current > 0 and macd_hist_prev < 0

    # ---------- Trendqualität / Base ----------
    ma20_slope = calc_slope_pct(ma20_series, lookback=20)
    ma50_slope = calc_slope_pct(ma50_series, lookback=20)
    ma200_slope = calc_slope_pct(ma200_series, lookback=20)
    higher_lows_score = calc_higher_lows_score(close, low)
    trend_quality_score = calc_trend_quality_score(
        price, ma20, ma50, ma200, ma20_slope, ma50_slope, ma200_slope, higher_lows_score
    )
    base_length_days = calc_base_length_days(close, ma20)
    correction_depth_pct = calc_correction_depth_pct(close)
    range_tightness_score = calc_range_tightness_score(close)
    atr_pct_series = (tr.rolling(14).mean() / close) * 100
    volatility_contraction_score = calc_volatility_contraction_score(atr_pct_series, bb_width_s)
    pullback_quality_score = calc_pullback_quality_score(price, ma20, ma50, rsi, atr_pct, ret20)
    base_quality_score = calc_base_quality_score(
        base_length_days,
        correction_depth_pct,
        range_tightness_score,
        volatility_contraction_score,
        pullback_quality_score
    )
    volume_quality_proxy = calc_volume_quality_proxy(vol_ratio, obv_trend)

    # ---------- Volumenqualität / Akkumulation ----------
    up_down_volume_ratio = calc_up_down_volume_ratio(close, vol, lookback=20)
    accumulation_day_count, distribution_day_count = calc_accumulation_distribution_days(close, vol, lookback=20)
    volume_trend_score = calc_volume_trend_score(vol, close)
    recent_pullback_volume_ratio = calc_recent_pullback_volume_ratio(close, vol, lookback=10)
    breakout_day_volume_ratio = calc_breakout_day_volume_ratio(close, vol, lookback=20)
    close_near_day_high = calc_close_near_day_high(close, high, low)

    ret1_series = close.pct_change(1) * 100
    ret1_last = float(ret1_series.iloc[-1]) if pd.notna(ret1_series.iloc[-1]) else np.nan
    vol_ma20_last = float(vol.rolling(20).mean().iloc[-1]) if pd.notna(vol.rolling(20).mean().iloc[-1]) else np.nan
    vol_last = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else np.nan

    strong_up_volume_day = (
        pd.notna(ret1_last) and ret1_last >= 1.5
        and pd.notna(vol_last) and pd.notna(vol_ma20_last) and vol_last > 1.2 * vol_ma20_last
    )
    down_volume_heavy = (
        pd.notna(ret1_last) and ret1_last <= -1.5
        and pd.notna(vol_last) and pd.notna(vol_ma20_last) and vol_last > 1.2 * vol_ma20_last
    )
    weak_rebound_on_volume = (
        pd.notna(ret1_last) and 0 < ret1_last < 0.8
        and pd.notna(vol_last) and pd.notna(vol_ma20_last) and vol_last > 1.15 * vol_ma20_last
    )

    pullback_active = bool(pd.notna(ret20) and ret20 < 0 and ((pd.notna(ma50) and pd.notna(price) and price >= ma50 * 0.95) or not pd.notna(ma50)))
    prev20_high_local = float(close.shift(1).rolling(20).max().iloc[-1]) if pd.notna(close.shift(1).rolling(20).max().iloc[-1]) else np.nan
    breakout_context = bool(pd.notna(prev20_high_local) and pd.notna(price) and price >= prev20_high_local * 0.995)

    accumulation_score = calc_accumulation_score(up_down_volume_ratio, obv_trend, accumulation_day_count, strong_up_volume_day)
    distribution_pressure_score = calc_distribution_pressure_score(distribution_day_count, obv_trend, down_volume_heavy, weak_rebound_on_volume)
    pullback_dryup_score = calc_pullback_dryup_score(pullback_active, recent_pullback_volume_ratio, pullback_quality_score, volatility_contraction_score)
    breakout_volume_score = calc_breakout_volume_score(breakout_context, breakout_day_volume_ratio, 50, close_near_day_high, breakout_failure_risk_low=True)
    volume_quality_score = calc_volume_quality_score(accumulation_score, distribution_pressure_score, pullback_dryup_score, breakout_volume_score, volume_trend_score)

    # ---------- Fundamentals ----------
    target = info.get("targetMeanPrice", np.nan)
    upside = ((target / price - 1) * 100) if pd.notna(target) and price else np.nan
    pe = info.get("forwardPE", np.nan)
    peg = info.get("pegRatio", np.nan)
    ps = info.get("priceToSalesTrailing12Months", np.nan)
    pb = info.get("priceToBook", np.nan)
    beta = info.get("beta", np.nan)
    market_cap = info.get("marketCap", np.nan)

    profit_margin = info.get("profitMargins", np.nan)
    oper_margin = info.get("operatingMargins", np.nan)
    gross_margin = info.get("grossMargins", np.nan)
    roe = info.get("returnOnEquity", np.nan)
    roa = info.get("returnOnAssets", np.nan)
    revenue_growth = info.get("revenueGrowth", np.nan)
    earnings_growth = info.get("earningsGrowth", np.nan)
    current_ratio = info.get("currentRatio", np.nan)
    quick_ratio = info.get("quickRatio", np.nan)
    debt_to_equity = info.get("debtToEquity", np.nan)
    fcf = info.get("freeCashflow", np.nan)
    op_cf = info.get("operatingCashflow", np.nan)
    short_pct = info.get("shortPercentOfFloat", np.nan)

    # ---------- Benchmark / Market ----------
    bench_ret21 = market_info["ret21"]
    bench_ret63 = market_info["ret63"]
    bench_ret126 = market_info["ret126"]

    rs_vs_benchmark_21 = ret21 - bench_ret21 if pd.notna(bench_ret21) else np.nan
    rs_vs_benchmark_63 = ret63 - bench_ret63 if pd.notna(bench_ret63) else np.nan
    rs_vs_benchmark_126 = ret126 - bench_ret126 if pd.notna(bench_ret126) else np.nan

    rs_terms = []
    if pd.notna(rs_vs_benchmark_21):
        rs_terms.append(rs_vs_benchmark_21 * 0.25)
    if pd.notna(rs_vs_benchmark_63):
        rs_terms.append(rs_vs_benchmark_63 * 0.45)
    if pd.notna(rs_vs_benchmark_126):
        rs_terms.append(rs_vs_benchmark_126 * 0.30)
    rs_composite = sum(rs_terms) if rs_terms else np.nan

    # ---------- Stock Style ----------
    stock_style = infer_stock_style_advanced(
        revenue_growth, earnings_growth, pe, pb, beta, debt_to_equity, roe, profit_margin, sector
    )
    style_adj = get_style_sector_adjustment(stock_style, sector)

    # ---------- Horizon ----------
    if "1-7" in horizon:
        hd, ws, wc = 7, 0.82, 0.18
    elif "1-4" in horizon:
        hd, ws, wc = 21, 0.68, 0.32
    elif "1-3" in horizon:
        hd, ws, wc = 60, 0.52, 0.48
    elif "1-2" in horizon:
        hd, ws, wc = 365, 0.30, 0.70
    else:
        hd, ws, wc = 730, 0.15, 0.85

    # ---------- Earnings ----------
    earnings_ts = normalize_missing(info.get("earningsTimestamp"))
    if pd.notna(earnings_ts):
        days_earn = (float(earnings_ts) - datetime.now(timezone.utc).timestamp()) / 86400
    else:
        days_earn = np.nan

    has_upcoming_earnings = pd.notna(days_earn) and days_earn >= 0
    has_past_earnings = pd.notna(days_earn) and days_earn < 0

    if has_upcoming_earnings:
        sg_earn = "🟢" if days_earn > 30 else ("🟡" if days_earn > 7 else "🔴")
    elif has_past_earnings:
        sg_earn = "🟡"
    else:
        sg_earn = "⚪"

    if pd.notna(earnings_ts):
        earnings_dt = datetime.fromtimestamp(float(earnings_ts), tz=timezone.utc)
        if has_upcoming_earnings:
            sg_earn_txt = earnings_dt.strftime("%d.%m.%Y")
        else:
            sg_earn_txt = f"Letzte Earnings: {earnings_dt.strftime('%d.%m.%Y')}"
    else:
        sg_earn_txt = "kein Datum"

    earnings_warning = has_upcoming_earnings and days_earn <= 7

    # ---------- Technical Scores ----------
    if price > ma50 > ma150 > ma200:
        regime, reg_amp = "UPTREND", "🟢"
    elif price < ma50 < ma150 < ma200:
        regime, reg_amp = "DOWNTREND", "🔴"
    else:
        regime, reg_amp = "SIDEWAYS", "🟡"

    s3 = 100 if price > ma20 > ma50 > ma150 else (15 if price < ma20 < ma50 < ma150 else 52)
    s3a = ampel(s3)
    s3t = "Trend-Stack sauber" if s3 >= 80 else ("Trend gemischt" if s3 >= 45 else "Trend schwach")

    rsi_s = 100 if 55 <= rsi <= 72 else (70 if 48 <= rsi < 55 or 72 < rsi <= 78 else 25)
    macd_s = 100 if (macd_v > 0 and macd_up) else (68 if macd_up else 22)
    adx_s = 100 if adx > 25 else (65 if adx > 18 else 30)
    roc_s = 100 if roc20 > 4 else (72 if roc20 > 0 else (45 if roc20 > -4 else 20))
    s4 = round(rsi_s * 0.25 + macd_s * 0.30 + adx_s * 0.20 + roc_s * 0.25)
    s4a = ampel(s4)
    s4t = f"RSI {rsi:.1f} - MACD {'up' if macd_up else 'dn'} - ADX {adx:.1f} - ROC20 {roc20:.1f}%"

    if ret5 > 0 and vol_ratio > 1.12 and obv_trend == "steigend":
        s5, s5a, s5t = 100, "🟢", f"Vol {vol_ratio:.2f}x - OBV steigend"
    elif ret20 > 0 and obv_trend == "steigend":
        s5, s5a, s5t = 68, "🟡", f"Vol {vol_ratio:.2f}x - Nachfrage ok"
    elif ret20 > 0:
        s5, s5a, s5t = 52, "🟡", f"Momentum ok - OBV {obv_trend}"
    else:
        s5, s5a, s5t = 28, "🔴", f"Momentum/Volumen schwach - OBV {obv_trend}"

    if atr_pct < 2.8:
        s6, s6a, s6t = 92, "🟢", f"ATR {atr_pct:.1f}% niedrig"
    elif atr_pct < 5.5:
        s6, s6a, s6t = 66, "🟡", f"ATR {atr_pct:.1f}% normal"
    elif atr_pct < 8.0:
        s6, s6a, s6t = 44, "🟡", f"ATR {atr_pct:.1f}% erhoeht"
    else:
        s6, s6a, s6t = 20, "🔴", f"ATR {atr_pct:.1f}% hoch"

    w52 = 100 if 80 <= dist52 <= 98 else (72 if 70 <= dist52 < 80 else (55 if 98 < dist52 <= 101 else (35 if dist52 >= 55 else 15)))

    if pd.notna(rs_composite):
        if rs_composite > 8:
            rs_score = 100
        elif rs_composite > 3:
            rs_score = 78
        elif rs_composite > -3:
            rs_score = 55
        else:
            rs_score = 22
    else:
        rs_score = 100 if ret63 > 12 else (78 if ret63 > 4 else (55 if ret63 > -5 else 22))

    kb = sum([s3 >= 65, s4 >= 65, s5 >= 65, s6 >= 65])

    setup_raw = (
        s3 * (0.22 * style_adj["trend"])
        + s4 * (0.24 * style_adj["momentum"])
        + s5 * 0.18
        + s6 * 0.10
        + rs_score * 0.16
        + w52 * 0.10
    ) / (
        0.22 * style_adj["trend"] + 0.24 * style_adj["momentum"] + 0.18 + 0.10 + 0.16 + 0.10
    )

    if strict_mode:
        if kb < 2:
            setup_raw = min(setup_raw, 44)
        elif kb == 2:
            setup_raw = min(setup_raw, 58)
    setup = round(clamp(setup_raw))
    setup_adj = round(clamp(setup * 0.88 + market_info["score"] * 0.12))

    # ---------- Fundamental Scores ----------
    fundamental_fields = [
        profit_margin, oper_margin, gross_margin, roe, roa,
        revenue_growth, earnings_growth, current_ratio, quick_ratio,
        debt_to_equity, fcf, op_cf, pe, peg, ps, pb,
        beta, short_pct, info.get("recommendationMean", np.nan),
        info.get("numberOfAnalystOpinions", np.nan), target
    ]
    fund_cov = known_ratio(fundamental_fields)
    fund_fields_loaded = int(info.get("_fund_fields_loaded", 0) or 0)
    fund_data_warning = fund_cov < 0.35

    quality_parts = []
    quality_parts.append(90 if pd.notna(profit_margin) and profit_margin > 0.20 else (75 if pd.notna(profit_margin) and profit_margin > 0.10 else (55 if pd.notna(profit_margin) and profit_margin > 0 else 40)))
    quality_parts.append(90 if pd.notna(oper_margin) and oper_margin > 0.25 else (75 if pd.notna(oper_margin) and oper_margin > 0.15 else (55 if pd.notna(oper_margin) and oper_margin > 0.08 else 40)))
    quality_parts.append(92 if pd.notna(roe) and roe > 0.25 else (78 if pd.notna(roe) and roe > 0.15 else (58 if pd.notna(roe) and roe > 0.08 else 42)))
    quality_parts.append(85 if pd.notna(fcf) and fcf > 0 else (60 if pd.notna(fcf) else 45))
    quality_score = round(np.mean(quality_parts))

    growth_parts = []
    growth_parts.append(90 if pd.notna(revenue_growth) and revenue_growth > 0.15 else (75 if pd.notna(revenue_growth) and revenue_growth > 0.05 else (55 if pd.notna(revenue_growth) and revenue_growth > 0 else 35)))
    growth_parts.append(92 if pd.notna(earnings_growth) and earnings_growth > 0.20 else (76 if pd.notna(earnings_growth) and earnings_growth > 0.08 else (56 if pd.notna(earnings_growth) and earnings_growth > 0 else 34)))
    growth_parts.append(88 if ret126 > 20 else (72 if ret126 > 5 else (55 if ret126 > -8 else 35)))
    growth_score = round(np.mean(growth_parts))

    growth_quality = 50
    if pd.notna(revenue_growth) and revenue_growth > 0.08:
        growth_quality += 10
    if pd.notna(earnings_growth) and pd.notna(revenue_growth) and earnings_growth > revenue_growth:
        growth_quality += 10
    if pd.notna(fcf) and fcf > 0:
        growth_quality += 10
    if pd.notna(profit_margin) and profit_margin > 0.10:
        growth_quality += 10
    if pd.notna(oper_margin) and oper_margin > 0.12:
        growth_quality += 10
    growth_quality = round(clamp(growth_quality))

    valuation_parts = []
    if pd.notna(pe):
        valuation_parts.append(86 if 0 < pe < 20 else (72 if pe < 28 else (58 if pe < 38 else 42)))
    if pd.notna(peg):
        valuation_parts.append(84 if 0 < peg < 1.5 else (70 if peg < 2.2 else (55 if peg < 3.0 else 42)))
    if pd.notna(ps):
        if pd.notna(revenue_growth) and revenue_growth > 0.15:
            valuation_parts.append(78 if ps < 8 else (62 if ps < 12 else 45))
        else:
            valuation_parts.append(82 if ps < 4 else (68 if ps < 8 else 42))
    if pd.notna(pb):
        valuation_parts.append(80 if pb < 3 else (65 if pb < 6 else 45))
    valuation_parts.append(82 if pd.notna(upside) and upside > 20 else (70 if pd.notna(upside) and upside > 10 else (55 if pd.notna(upside) and upside > 0 else 40)))
    valuation_score = round(np.mean(valuation_parts)) if valuation_parts else 50

    balance_parts = []
    balance_parts.append(88 if pd.notna(current_ratio) and current_ratio >= 1.5 else (72 if pd.notna(current_ratio) and current_ratio >= 1.1 else 48))
    balance_parts.append(88 if pd.notna(quick_ratio) and quick_ratio >= 1.0 else (70 if pd.notna(quick_ratio) and quick_ratio >= 0.8 else 48))
    balance_parts.append(90 if pd.notna(debt_to_equity) and debt_to_equity < 60 else (72 if pd.notna(debt_to_equity) and debt_to_equity < 120 else 45))
    balance_score = round(np.mean(balance_parts))

    rec = info.get("recommendationKey", "hold")
    rec_label = analyst_label(rec)
    rec_mean = info.get("recommendationMean", np.nan)
    analysts = info.get("numberOfAnalystOpinions", np.nan)

    sentiment_parts = []
    sentiment_parts.append(88 if rec in ["strong_buy", "buy"] else (65 if rec in ["hold"] else 50))
    sentiment_parts.append(84 if pd.notna(analysts) and analysts >= 20 else (72 if pd.notna(analysts) and analysts >= 10 else (58 if pd.notna(analysts) and analysts >= 5 else (52 if pd.notna(target) else 48))))
    sentiment_parts.append(84 if pd.notna(rec_mean) and rec_mean <= 2.0 else (68 if pd.notna(rec_mean) and rec_mean <= 2.5 else (55 if pd.notna(rec_mean) and rec_mean <= 3.0 else (50 if pd.notna(target) else 42))))
    sentiment_score = round(np.mean(sentiment_parts))

    risk_parts = []
    risk_parts.append(80 if pd.notna(beta) and beta < 1.2 else (62 if pd.notna(beta) and beta < 1.6 else 45))
    risk_parts.append(78 if pd.notna(short_pct) and short_pct < 0.03 else (62 if pd.notna(short_pct) and short_pct < 0.07 else 45))
    risk_parts.append(82 if atr_pct < 3.5 else (65 if atr_pct < 6 else 45))
    risk_score = round(np.mean(risk_parts))

    base_company = round(
        quality_score * (0.24 * style_adj["quality"])
        + growth_score * (0.18 * style_adj["growth"])
        + growth_quality * 0.12
        + valuation_score * (0.18 * style_adj["valuation"])
        + balance_score * (0.16 * style_adj["balance"])
        + sentiment_score * 0.06
        + risk_score * 0.06
    )
    base_company = round(base_company / (
        0.24 * style_adj["quality"]
        + 0.18 * style_adj["growth"]
        + 0.12
        + 0.18 * style_adj["valuation"]
        + 0.16 * style_adj["balance"]
        + 0.06
        + 0.06
    ))

    red_flag_items, red_flag_penalty_total = build_red_flags(
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        profit_margin=profit_margin,
        fcf=fcf,
        op_cf=op_cf,
        debt_to_equity=debt_to_equity,
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        has_upcoming_earnings=has_upcoming_earnings,
        days_earn=days_earn
    )
    red_flag_items, red_flag_penalty_total = sanitize_quality_red_flags_v1601(red_flag_items)
    # v16.0.4: finale Klassifizierung, damit Growth-only-Hinweise nie als harte Red Flags durchrutschen.
    red_flag_items = [soften_growth_red_flag_item_v1604(x) for x in (red_flag_items or [])]
    red_flag_penalty_total = sum(
        float(x.get("Penalty", 0) or 0)
        for x in red_flag_items
        if is_hard_red_flag_v1604(x)
    )
    hard_red_flag_items = [x for x in red_flag_items if is_hard_red_flag_v1604(x)]
    soft_red_flag_items = [x for x in red_flag_items if not is_hard_red_flag_v1604(x)]
    red_flag_notes = [f"{x['Kategorie']}: {x['Detail']}" for x in hard_red_flag_items]
    red_flag_hint_notes = [f"{x['Kategorie']}: {x['Detail']}" for x in soft_red_flag_items]

    coverage_penalty = 0
    if fund_cov < 0.35:
        coverage_penalty = 12
    elif fund_cov < 0.55:
        coverage_penalty = 6

    base_company = max(25, round(base_company - red_flag_penalty_total - coverage_penalty))

    if hd < 30:
        company = round(base_company * 0.55 + 50 * 0.45)
    else:
        company = base_company
    company = int(clamp(company))

    # ---------- Institutionelle Qualität ----------
    cashflow_stability_score = calc_cashflow_stability_score(
        fcf, op_cf, market_cap, revenue_growth, earnings_growth
    )
    margin_stability_score = calc_margin_stability_score(
        profit_margin, oper_margin, gross_margin, roe, roa
    )
    institutional_quality_score = calc_institutional_quality_score(
        cashflow_stability_score,
        margin_stability_score,
        balance_score,
        quality_score,
        risk_score
    )
    institutional_quality_text = institutional_quality_label(institutional_quality_score)

    company = round(clamp(company * 0.90 + institutional_quality_score * 0.10))
    investment = round(clamp(setup_adj * ws + company * wc))

    if bb_squeeze and pd.notna(prev20_high) and price > prev20_high and vol_ratio >= 1.0:
        setup_type = "Range-Breakout"
        preferred_entry = "Ausbruch über Range-Oberkante"
    elif pd.notna(prev20_high) and price > prev20_high and vol_ratio >= 1.05 and rsi < 78:
        setup_type = "Breakout"
        preferred_entry = "Breakout über 20T-Hoch"
    elif (
        pd.notna(prev20_high)
        and price > ma20 > ma50
        and abs(price - prev20_high) / prev20_high <= 0.02
        and -3 <= ret5 <= 4
    ):
        setup_type = "Breakout-Retest"
        preferred_entry = "Retest des Ausbruchsniveaus"
    elif price > ma50 and ma20 > ma50 and pd.notna(ma20) and abs(price - ma20) / price <= 0.025:
        setup_type = "Pullback an MA20"
        preferred_entry = "Pullback nahe MA20"
    elif price > ma200 and pd.notna(ma50) and abs(price - ma50) / price <= 0.03:
        setup_type = "Pullback an MA50"
        preferred_entry = "Pullback nahe MA50"
    elif price > ma200 and rsi < 42 and macd_hist_current > macd_hist_prev:
        setup_type = "Rebound"
        preferred_entry = "Rebound nach Stabilisierung"
    elif price > ma50 and price > ma200 and rs_score >= 55:
        setup_type = "Trendfolge"
        preferred_entry = "Trendfolge bei Rücksetzer"
    else:
        setup_type = "Kein sauberes Setup"
        preferred_entry = "Aktuell kein sauberer Einstieg"

    setup_base_score = {
        "Breakout": 88,
        "Breakout-Retest": 86,
        "Pullback an MA20": 84,
        "Pullback an MA50": 78,
        "Trendfolge": 76,
        "Rebound": 68,
        "Range-Breakout": 82,
        "Kein sauberes Setup": 35,
    }.get(setup_type, 40)

    setup_confidence = round(clamp(
        setup_base_score * 0.38
        + s3 * 0.22
        + s4 * 0.22
        + min(kb / 4 * 100, 100) * 0.10
        + (85 if market_info["regime"] == "POSITIV" else 60 if market_info["regime"] == "NEUTRAL" else 35) * 0.08
    ))
    setup_confidence_text = setup_confidence_label(setup_confidence)

    valid_trade_setup = (
        investment >= 60
        and setup_adj >= 55
        and kb >= 2
        and setup_type != "Kein sauberes Setup"
        and market_info["regime"] != "NEGATIV"
        and not (has_upcoming_earnings and pd.notna(days_earn) and days_earn < 7)
    )

    # ---------- Trade Setup ----------
    if valid_trade_setup:
        # Einstieg je Setup-Typ
        if setup_type == "Breakout":
            anchor_price = prev20_high if pd.notna(prev20_high) else price
            entry_low = max(anchor_price, price * 0.995)
            entry_high = max(anchor_price * 1.015, price * 1.005)
            entry_source = "Breakout-Zone über dem Ausbruchsniveau"
        elif setup_type == "Breakout-Retest":
            anchor_price = prev20_high if pd.notna(prev20_high) else price
            entry_low = anchor_price * 0.99 if pd.notna(anchor_price) else price * 0.98
            entry_high = anchor_price * 1.01 if pd.notna(anchor_price) else price * 1.00
            entry_source = "Retest-Zone am früheren Breakout-Level"
        elif setup_type == "Pullback an MA20":
            anchor_price = ma20 if pd.notna(ma20) else ma50
            entry_low = anchor_price * 0.99 if pd.notna(anchor_price) else price * 0.98
            entry_high = anchor_price * 1.01 if pd.notna(anchor_price) else price
            entry_source = "Pullback-Zone nahe MA20"
        elif setup_type == "Pullback an MA50":
            anchor_price = ma50 if pd.notna(ma50) else ma20
            entry_low = anchor_price * 0.985 if pd.notna(anchor_price) else price * 0.97
            entry_high = anchor_price * 1.015 if pd.notna(anchor_price) else price
            entry_source = "Pullback-Zone nahe MA50"
        elif setup_type == "Rebound":
            anchor_price = prev20_low if pd.notna(prev20_low) else ma20
            entry_low = anchor_price * 1.00 if pd.notna(anchor_price) else price * 0.98
            entry_high = anchor_price * 1.03 if pd.notna(anchor_price) else price
            entry_source = "Rebound-Zone nach Stabilisierung"
        elif setup_type == "Range-Breakout":
            anchor_price = prev20_high if pd.notna(prev20_high) else price
            entry_low = anchor_price * 1.000 if pd.notna(anchor_price) else price * 0.995
            entry_high = anchor_price * 1.012 if pd.notna(anchor_price) else price * 1.005
            entry_source = "Ausbruchszone über der Range"
        elif setup_type == "Trendfolge":
            anchor_price = ma20 if pd.notna(ma20) else price
            entry_low = anchor_price * 0.995 if pd.notna(anchor_price) else price * 0.99
            entry_high = anchor_price * 1.015 if pd.notna(anchor_price) else price * 1.01
            entry_source = "Trendfolge-Zone bei Rücksetzer"
        else:
            entry_low = price
            entry_high = price
            entry_source = "Aktueller Kurs / kein sauberes Setup"

        suggested_entry_zone = format_price_zone(entry_low, entry_high, ccy)
        entry_quality = (
            "gut" if pd.notna(entry_low) and pd.notna(entry_high) and entry_low <= price <= entry_high
            else ("abwarten" if pd.notna(entry_high) and price > entry_high else "früh")
        )

        # Stop-Logik je Setup-Typ
        generic_atr_stop = price - 1.8 * atr if pd.notna(atr) else np.nan
        generic_struct_stop = ma50 * 0.965 if pd.notna(ma50) else np.nan
        setup_stop_candidates = []
        stop_source = "Standard-Stop"

        if setup_type == "Breakout":
            breakout_level = prev20_high if pd.notna(prev20_high) else price
            setup_stop_candidates = [
                breakout_level * 0.975 if pd.notna(breakout_level) else np.nan,
                price - 1.6 * atr if pd.notna(atr) else np.nan,
                ma20 * 0.985 if pd.notna(ma20) else np.nan,
            ]
            stop_source = "Unter Breakout-Level / ATR"
        elif setup_type == "Breakout-Retest":
            retest_level = prev20_high if pd.notna(prev20_high) else price
            setup_stop_candidates = [
                retest_level * 0.985 if pd.notna(retest_level) else np.nan,
                prev20_low * 0.995 if pd.notna(prev20_low) else np.nan,
                price - 1.4 * atr if pd.notna(atr) else np.nan,
            ]
            stop_source = "Unter Retest-Niveau"
        elif setup_type == "Pullback an MA20":
            setup_stop_candidates = [
                ma20 * 0.985 if pd.notna(ma20) else np.nan,
                prev20_low * 0.995 if pd.notna(prev20_low) else np.nan,
                price - 1.4 * atr if pd.notna(atr) else np.nan,
            ]
            stop_source = "Unter MA20 / Pullback-Tief"
        elif setup_type == "Pullback an MA50":
            setup_stop_candidates = [
                ma50 * 0.985 if pd.notna(ma50) else np.nan,
                prev20_low * 0.99 if pd.notna(prev20_low) else np.nan,
                price - 1.5 * atr if pd.notna(atr) else np.nan,
            ]
            stop_source = "Unter MA50 / Pullback-Tief"
        elif setup_type == "Rebound":
            setup_stop_candidates = [
                prev20_low * 0.99 if pd.notna(prev20_low) else np.nan,
                price - 1.3 * atr if pd.notna(atr) else np.nan,
                ma20 * 0.98 if pd.notna(ma20) else np.nan,
            ]
            stop_source = "Unter Rebound-Tief"
        elif setup_type == "Range-Breakout":
            range_top = prev20_high if pd.notna(prev20_high) else price
            setup_stop_candidates = [
                range_top * 0.985 if pd.notna(range_top) else np.nan,
                price - 1.5 * atr if pd.notna(atr) else np.nan,
                ma20 * 0.985 if pd.notna(ma20) else np.nan,
            ]
            stop_source = "Unter Range-Oberkante"
        elif setup_type == "Trendfolge":
            setup_stop_candidates = [
                ma20 * 0.985 if pd.notna(ma20) else np.nan,
                ma50 * 0.985 if pd.notna(ma50) else np.nan,
                price - 1.8 * atr if pd.notna(atr) else np.nan,
            ]
            stop_source = "Unter Trendzone / Higher Low"

        stop_candidates = [
            x for x in setup_stop_candidates + [generic_atr_stop, generic_struct_stop]
            if pd.notna(x) and x > 0 and x < price
        ]

        if stop_candidates:
            stop_used = round(max(stop_candidates), 2)
        else:
            stop_used = round(price - max(price * 0.08, (atr * 1.8 if pd.notna(atr) else price * 0.06)), 2)
            stop_source = "Fallback-Stop"

        atr_stop = round(generic_atr_stop, 2) if pd.notna(generic_atr_stop) else np.nan
        stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0
        if stop_used <= 0 or stop_used >= price:
            stop_used = round(price - max(price * 0.08, (atr * 1.8 if pd.notna(atr) else price * 0.06)), 2)
            stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0
            stop_source = "Fallback-Stop"

        practical_min_stop_dist_pct = 3.5
        if price > 0 and stop_dist < practical_min_stop_dist_pct:
            stop_used = round(price * (1 - practical_min_stop_dist_pct / 100), 2)
            stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0
            stop_source = "Praxis-Mindestabstand"

        risk_per_share = price - stop_used

        tp1 = round(price + 1 * risk_per_share, 2)
        tp1_source = "1R vom Stop"

        # Setup-spezifische Ziel-Logik
        technical_target_1 = np.nan
        technical_target_2 = np.nan

        if setup_type in {"Breakout", "Range-Breakout"}:
            technical_target_1 = prev20_high * 1.03 if pd.notna(prev20_high) and prev20_high > price else np.nan
            technical_target_2 = high52 if pd.notna(high52) and high52 > price else np.nan
        elif setup_type == "Breakout-Retest":
            technical_target_1 = prev20_high * 1.02 if pd.notna(prev20_high) and prev20_high > price else np.nan
            technical_target_2 = high52 if pd.notna(high52) and high52 > price else np.nan
        elif setup_type == "Pullback an MA20":
            technical_target_1 = prev20_high if pd.notna(prev20_high) and prev20_high > price else np.nan
            technical_target_2 = high52 if pd.notna(high52) and high52 > price else np.nan
        elif setup_type == "Pullback an MA50":
            technical_target_1 = ma20 * 1.03 if pd.notna(ma20) and ma20 > price else prev20_high if pd.notna(prev20_high) and prev20_high > price else np.nan
            technical_target_2 = high52 if pd.notna(high52) and high52 > price else np.nan
        elif setup_type == "Rebound":
            technical_target_1 = ma50 if pd.notna(ma50) and ma50 > price else ma20 if pd.notna(ma20) and ma20 > price else np.nan
            technical_target_2 = prev20_high if pd.notna(prev20_high) and prev20_high > price else high52 if pd.notna(high52) and high52 > price else np.nan
        elif setup_type == "Trendfolge":
            technical_target_1 = prev20_high if pd.notna(prev20_high) and prev20_high > price else np.nan
            technical_target_2 = high52 if pd.notna(high52) and high52 > price else np.nan

        tp2_floor = price + 1.8 * risk_per_share
        if pd.notna(technical_target_1) and technical_target_1 > price:
            tp2 = round(max(float(technical_target_1), tp2_floor), 2)
            tp2_source = f"Primärziel aus Setup ({setup_type})"
        elif pd.notna(target) and target > price:
            tp2 = round(max(float(target), tp2_floor), 2)
            tp2_source = "Analysten-Target"
        elif pd.notna(high52) and high52 > price:
            tp2 = round(max(float(high52), tp2_floor), 2)
            tp2_source = "52W-Hoch"
        else:
            tp2 = round(price + 2 * risk_per_share, 2)
            tp2_source = "2R-Fallback"

        tp3_floor = max(price + 2.8 * risk_per_share, tp2 + 0.8 * risk_per_share)
        if pd.notna(technical_target_2) and technical_target_2 > tp2:
            tp3 = round(max(float(technical_target_2), tp3_floor), 2)
            tp3_source = f"Sekundärziel aus Setup ({setup_type})"
        elif pd.notna(target) and target > tp2:
            tp3 = round(max(float(target), tp3_floor), 2)
            tp3_source = "Analysten-Target"
        elif pd.notna(high52) and high52 > tp2:
            tp3 = round(max(float(high52), tp3_floor), 2)
            tp3_source = "52W-Hoch"
        else:
            tp3 = round(max(price + 3 * risk_per_share, tp2 + risk_per_share), 2)
            tp3_source = "3R-Ziel"

        crv = (tp2 - price) / (price - stop_used) if (price - stop_used) > 0 else 0
        timing_trade_score = round(clamp(s4 * 0.45 + s5 * 0.25 + rs_score * 0.20 + s6 * 0.10))
        stop_score = ideal_range_score(stop_dist, ideal_low=3.0, ideal_high=7.5, hard_low=1.0, hard_high=14.1)
        crv_score = linear_score(crv, low=0.9, high=3.0, floor=15, ceiling=95)
        market_trade_score = 85 if market_info["regime"] == "POSITIV" else (60 if market_info["regime"] == "NEUTRAL" else 25)
        entry_score = entry_quality_score(entry_quality, price, entry_low, entry_high)

        tradeability_score = round(clamp(
            crv_score * 0.34
            + stop_score * 0.18
            + timing_trade_score * 0.22
            + market_trade_score * 0.10
            + entry_score * 0.16
        ))
        tradeability_text = tradeability_label(tradeability_score)

        setup_confidence = round(clamp(
            (88 if setup_type in {"Breakout", "Pullback im Aufwärtstrend", "Trendfolge"} else 72 if setup_type in {"Rebound im Aufwärtstrend"} else 35) * 0.35
            + s3 * 0.20
            + s4 * 0.20
            + min(kb / 4 * 100, 100) * 0.15
            + (100 if entry_quality == "gut" else 60 if entry_quality == "abwarten" else 45) * 0.10
        ))
        setup_confidence_text = setup_confidence_label(setup_confidence)

        confidence_numeric = round(confidence_info.get("coverage", 0) * 100) if isinstance(confidence_info, dict) else 50
        market_long_score = 85 if market_info["regime"] == "POSITIV" else (60 if market_info["regime"] == "NEUTRAL" else 30)
        red_flag_adjustment = clamp(100 - min(red_flag_penalty_total * 4, 55), 35, 100)

        investment_case_score = round(clamp(
            company * 0.35
            + investment * 0.35
            + confidence_numeric * 0.10
            + market_long_score * 0.10
            + red_flag_adjustment * 0.10
        ))
        investment_case_text = investment_case_label(investment_case_score)

        entry_location_score = 90 if entry_quality == "gut" else (58 if entry_quality == "abwarten" else 44)
        trading_case_score = round(clamp(
            tradeability_score * 0.35
            + timing_trade_score * 0.25
            + setup_adj * 0.20
            + entry_location_score * 0.10
            + market_trade_score * 0.10
        ))

        # Konsistenz-Deckel: ein Einstiegs-Case darf nicht sehr hoch werden,
        # wenn Setup-Typ, Timing oder Setup-Confidence dagegen sprechen.
        if setup_type == "Kein sauberes Setup":
            trading_case_score = min(trading_case_score, 55)
        if pd.notna(timing_trade_score) and timing_trade_score < 50:
            trading_case_score = min(trading_case_score, 52)
        if pd.notna(setup_confidence) and setup_confidence < 60:
            trading_case_score = min(trading_case_score, 58)
        if entry_quality == "abwarten":
            trading_case_score = min(trading_case_score, 60)
        elif entry_quality == "früh":
            trading_case_score = min(trading_case_score, 56)

        trading_case_score = round(clamp(trading_case_score))
        trading_case_text = trading_case_label(trading_case_score)

        risk_eur = depot * (risk_pct / 100)
        pos_size = int(risk_eur / risk_per_share) if risk_per_share > 0 else 0
        time_stop = (date.today() + timedelta(days=hd)).strftime("%d.%m.%Y")
    else:
        atr_stop = np.nan
        stop_used = np.nan
        stop_dist = np.nan
        tp1 = np.nan
        tp2 = np.nan
        tp3 = np.nan
        tp1_source = "-"
        tp2_source = "-"
        tp3_source = "-"
        technical_target_1 = np.nan
        technical_target_2 = np.nan
        stop_source = "-"
        suggested_entry_zone = "-"
        entry_source = "-"
        entry_quality = "-"
        crv = np.nan
        timing_trade_score = round(clamp(s4 * 0.45 + s5 * 0.25 + rs_score * 0.20 + s6 * 0.10))
        market_trade_score = 85 if market_info["regime"] == "POSITIV" else (60 if market_info["regime"] == "NEUTRAL" else 25)
        tradeability_score = round(clamp(
            20 * 0.34
            + 30 * 0.18
            + timing_trade_score * 0.22
            + market_trade_score * 0.10
            + 35 * 0.16
        ))
        tradeability_text = tradeability_label(tradeability_score)
        crv_score = np.nan
        stop_score = np.nan
        entry_score = np.nan
        confidence_numeric = round(confidence_info.get("coverage", 0) * 100) if isinstance(confidence_info, dict) else 50
        market_long_score = 85 if market_info["regime"] == "POSITIV" else (60 if market_info["regime"] == "NEUTRAL" else 30)
        red_flag_adjustment = clamp(100 - min(red_flag_penalty_total * 4, 55), 35, 100)
        investment_case_score = round(clamp(
            company * 0.35
            + investment * 0.35
            + confidence_numeric * 0.10
            + market_long_score * 0.10
            + red_flag_adjustment * 0.10
        ))
        investment_case_text = investment_case_label(investment_case_score)
        trading_case_score = round(clamp(
            (tradeability_score if pd.notna(tradeability_score) else 20) * 0.35
            + timing_trade_score * 0.25
            + setup_adj * 0.20
            + 40 * 0.10
            + (85 if market_info["regime"] == "POSITIV" else 60 if market_info["regime"] == "NEUTRAL" else 25) * 0.10
        ))
        if setup_type == "Kein sauberes Setup":
            trading_case_score = min(trading_case_score, 55)
        if pd.notna(timing_trade_score) and timing_trade_score < 50:
            trading_case_score = min(trading_case_score, 52)
        if pd.notna(setup_confidence) and setup_confidence < 60:
            trading_case_score = min(trading_case_score, 58)
        trading_case_score = round(clamp(trading_case_score))
        trading_case_text = trading_case_label(trading_case_score)
        risk_eur = depot * (risk_pct / 100)
        pos_size = 0
        time_stop = "-"

    short_term_raw = round(clamp(s4 * 0.45 + s5 * 0.28 + s6 * 0.17 + rs_score * 0.10))
    swing_raw = round(clamp(s3 * 0.26 + s4 * 0.28 + s5 * 0.16 + s6 * 0.10 + rs_score * 0.12 + w52 * 0.08))
    mid_term_raw = round(clamp(setup_adj * 0.55 + company * 0.45))
    long_term_raw = round(clamp(company * 0.50 + growth_score * 0.15 + growth_quality * 0.10 + quality_score * 0.15 + valuation_score * 0.10))
    very_long_term_raw = round(clamp(company * 0.40 + quality_score * 0.20 + growth_score * 0.15 + growth_quality * 0.10 + valuation_score * 0.15))

    market_hmap_adj = 6 if market_info["regime"] == "POSITIV" else (-8 if market_info["regime"] == "NEGATIV" else 0)
    red_flag_hmap_penalty = min(red_flag_penalty_total * 2, 18)

    short_term_score = round(clamp(
        short_term_raw * 0.55
        + trading_case_score * 0.35
        + setup_confidence * 0.10
        + market_hmap_adj
        - red_flag_hmap_penalty * 0.35
    ))
    swing_score = round(clamp(
        swing_raw * 0.45
        + trading_case_score * 0.35
        + investment_case_score * 0.10
        + setup_confidence * 0.10
        + market_hmap_adj
        - red_flag_hmap_penalty * 0.30
    ))
    mid_term_score = round(clamp(
        mid_term_raw * 0.40
        + investment_case_score * 0.35
        + trading_case_score * 0.15
        + company * 0.10
        + market_hmap_adj * 0.7
        - red_flag_hmap_penalty * 0.45
    ))
    long_term_score = round(clamp(
        long_term_raw * 0.45
        + investment_case_score * 0.35
        + company * 0.10
        + quality_score * 0.10
        + market_hmap_adj * 0.35
        - red_flag_hmap_penalty * 0.55
    ))
    very_long_term_score = round(clamp(
        very_long_term_raw * 0.45
        + investment_case_score * 0.30
        + company * 0.15
        + quality_score * 0.10
        + market_hmap_adj * 0.20
        - red_flag_hmap_penalty * 0.60
    ))

    # Zeithorizonte sollen nicht deutlich positiver wirken als die Gesamtsicht
    short_term_score = min(short_term_score, max(trading_case_score + 8, 0))
    swing_score = min(swing_score, max(trading_case_score + 10, 0))
    mid_term_score = min(mid_term_score, max(investment_case_score + 10, 0))
    long_term_score = min(long_term_score, max(investment_case_score + 8, 0))
    very_long_term_score = min(very_long_term_score, max(investment_case_score + 6, 0))

    hmap = {
        "Kurzfrist": short_term_score,
        "Swing": swing_score,
        "Mittelfrist": mid_term_score,
        "Langfrist": long_term_score,
        "Sehr langfristig": very_long_term_score,
    }

    # ---------- Katalysatoren / Event-Kontext ----------
    event_phase_label = classify_event_phase(days_earn)
    earnings_reaction_5d, earnings_reaction_10d = calc_post_earnings_reaction(close, days_earn)

    event_risk_score = calc_event_risk_score(
        days_earn,
        has_upcoming_earnings,
        atr_pct,
        breakout_context if 'breakout_context' in locals() else False
    )

    post_earnings_reaction_score = calc_post_earnings_reaction_score(
        earnings_reaction_5d,
        earnings_reaction_10d,
        rs_score if pd.notna(rs_score) else 50
    )

    revision_momentum_score = calc_revision_momentum_score(
        upside,
        revenue_growth,
        earnings_growth,
        ret21,
        rs_score if pd.notna(rs_score) else 50
    )

    earnings_event_score = calc_earnings_event_score(
        event_phase_label,
        event_risk_score,
        post_earnings_reaction_score
    )

    catalyst_score = calc_catalyst_score(
        earnings_event_score,
        revision_momentum_score,
        post_earnings_reaction_score,
        event_phase_label
    )

    catalyst_text = catalyst_label(catalyst_score)
    post_earnings_text = event_phase_text(event_phase_label)

    # ---------- Leadership / Marktbreite ----------
    sector_label = sector if pd.notna(sector) and sector not in ["", "-", None] else "Unbekannt"
    industry_label = industry if pd.notna(industry) and industry not in ["", "-", None] else "Unbekannt"
    sector_etf_symbol = get_sector_etf_symbol(sector_label)
    sector_ctx = load_sector_context(sector_etf_symbol) if sector_etf_symbol else None

    sector_strength_score = calc_sector_strength_score(sector_ctx)
    sector_strength_available = pd.notna(sector_strength_score)
    rs_benchmark_score = calc_rs_benchmark_score(rs_vs_benchmark_21, rs_vs_benchmark_63, rs_vs_benchmark_126)
    rs_acceleration_score = calc_rs_acceleration_score(rs_vs_benchmark_21, rs_vs_benchmark_63, rs_vs_benchmark_126)
    industry_strength_score = calc_industry_strength_score(
        sector_strength_score if pd.notna(sector_strength_score) else 50,
        rs_score if pd.notna(rs_score) else 50,
        company if pd.notna(company) else 50,
    )
    leadership_score = calc_leadership_score(
        sector_strength_score if pd.notna(sector_strength_score) else 50,
        industry_strength_score,
        rs_benchmark_score,
        rs_acceleration_score,
        rs_score if pd.notna(rs_score) else 50,
    )
    leadership_status = get_leadership_status(leadership_score, rs_acceleration_score)
    sector_trend_text = strength_text(sector_strength_score)
    industry_trend_text = strength_text(industry_strength_score)

    setup_type_quality_score = calc_setup_type_quality_score(
        setup_type,
        base_quality_score,
        volume_quality_proxy,
        rs_score if pd.notna(rs_score) else 50,
        trend_quality_score,
        setup_confidence if pd.notna(setup_confidence) else 50,
        pullback_quality_score
    )
    setup_priority_score = calc_setup_priority_score(
        setup_type_quality_score,
        leadership_score,
        trend_quality_score,
        base_quality_score,
        trading_case_score
    )

    trading_case_score = round(clamp(trading_case_score * 0.78 + trend_quality_score * 0.08 + base_quality_score * 0.07 + setup_type_quality_score * 0.07))
    investment_case_score = round(clamp(investment_case_score * 0.90 + leadership_score * 0.10))
    tradeability_score = round(clamp(tradeability_score * 0.82 + trend_quality_score * 0.08 + base_quality_score * 0.05 + setup_priority_score * 0.05))
    trading_case_score = round(clamp(
        trading_case_score * 0.82
        + volume_quality_score * 0.10
        + breakout_volume_score * 0.04
        + pullback_dryup_score * 0.04
    ))
    setup_type_quality_score = round(clamp(
        setup_type_quality_score * 0.82
        + volume_quality_score * 0.10
        + breakout_volume_score * 0.04
        + pullback_dryup_score * 0.04
    ))
    setup_priority_score = round(clamp(
        setup_priority_score * 0.88
        + volume_quality_score * 0.12
    ))

    investment_case_score = round(clamp(
        investment_case_score * 0.80
        + catalyst_score * 0.10
        + institutional_quality_score * 0.10
    ))
    setup_priority_score = round(clamp(
        setup_priority_score * 0.92
        + catalyst_score * 0.08
    ))
    tradeability_score = round(clamp(
        tradeability_score * 0.96
        + earnings_event_score * 0.04
    ))

    trading_case_text = trading_case_label(trading_case_score)
    investment_case_text = investment_case_label(investment_case_score)
    tradeability_text = tradeability_label(tradeability_score)

    position_mode = buy_in_override > 0

    # ---------- Recommendations ----------
    if has_upcoming_earnings and days_earn < 7:
        emp, conv = ("VETO - Earnings < 7 Tage", "-")
    elif position_mode:
        if investment >= 78 and kb >= 3 and market_info["regime"] == "POSITIV":
            emp, conv = ("HALTEN / AUSBAUEN", "HIGH")
        elif investment >= 65 and market_info["regime"] != "NEGATIV":
            emp, conv = ("HALTEN / ENGE BEOBACHTUNG", "MEDIUM")
        elif investment >= 52:
            emp, conv = ("HALTEN / RISIKO PRÜFEN", "LOW-MEDIUM")
        else:
            emp, conv = ("RISIKO REDUZIEREN / STOPP PRÜFEN", "LOW")
    else:
        if (
            investment >= 78 and kb >= 3 and market_info["regime"] == "POSITIV"
            and valid_trade_setup and entry_quality == "gut"
            and trading_case_score >= 68 and setup_confidence >= 60
        ):
            emp, conv = ("BUY / ACCUMULATE", "HIGH")
        elif investment >= 78 and kb >= 3 and market_info["regime"] == "POSITIV":
            emp, conv = ("BUY CANDIDATE / TIMING PRÜFEN", "HIGH")
        elif investment >= 68 and market_info["regime"] != "NEGATIV":
            emp, conv = ("WATCH / EINSTIEG PRÜFEN", "MEDIUM")
        elif investment >= 52:
            emp, conv = ("BEOBACHTEN", "LOW-MEDIUM")
        else:
            emp, conv = ("AVOID / WAIT", "LOW")

    # ---------- Trading Board ----------
    tb_score = 0
    tb_details = []
    tb_context = []

    tb_buy = buy_in_override if buy_in_override > 0 else 0.0
    tb_basispreis = tb_buy if tb_buy > 0 else price
    tb_perf = ((price - tb_buy) / tb_buy) * 100 if tb_buy > 0 else 0.0
    tb_stop = price - (2.5 * atr)
    tb_tp1 = tb_basispreis + (2.5 * atr)
    tb_tp2 = target if pd.notna(target) and target > tb_tp1 else tb_basispreis + (5.0 * atr)

    tb_details.append(f"S0: {price:.2f} {ccy}")

    if pd.notna(earnings_ts):
        if has_past_earnings:
            tb_details.append(f"S1 Earnings: letzte Earnings am {earnings_dt.strftime('%d.%m.%Y')}")
        else:
            tb_details.append(f"S1 Earnings: in {int(days_earn)}d ({sg_earn_txt})")
    else:
        tb_details.append("S1 Earnings: kein Datum")

    if price > ma200:
        tb_score += 1
        tb_details.append(f"S2: Über MA200 (Kurs {price:.2f} / MA200 {ma200:.2f}) ✓")
    else:
        tb_details.append(f"S2: Unter MA200 (Kurs {price:.2f} / MA200 {ma200:.2f}) ❌")

    if price > ma50:
        tb_score += 1
        tb_details.append(f"S3: Über MA50 (+1) (Kurs {price:.2f} / MA50 {ma50:.2f}) ✓")
    else:
        tb_score -= 1
        tb_details.append(f"S3: Unter MA50 (-1) (Kurs {price:.2f} / MA50 {ma50:.2f}) ❌")

    if ma50 > ma200:
        tb_score += 1
        tb_details.append(f"S4: Golden Cross (MA50 {ma50:.2f} > MA200 {ma200:.2f}) ✓")
    else:
        tb_details.append(f"S4: Trendstruktur schwach (MA50 {ma50:.2f} / MA200 {ma200:.2f}) ❌")

    if 40 < rsi < 60 or rsi < 30:
        tb_score += 1
        tb_details.append(f"S5: RSI {rsi:.1f} konstruktiv ✓")
    else:
        tb_details.append(f"S5: RSI {rsi:.1f} hoch/niedrig ❌")

    if position_mode:
        if tb_perf > 5:
            tb_score += 1
            tb_details.append(f"S6: +{tb_perf:.1f}% seit Einstieg ✓")
        else:
            tb_details.append(f"S6: {tb_perf:.1f}% seit Einstieg ❌")
    else:
        tb_details.append("S6: Watchlist-Modus (neutral)")

    if macd_hist_current > macd_hist_prev:
        tb_score += 1
        tb_details.append("S7: Momentum steigt ✓")
    else:
        tb_details.append("S7: Momentum fällt ❌")

    if earnings_warning:
        tb_score -= 3
        tb_details.insert(0, "⚠️ EARNINGS IN <7 TAGEN (Vorsicht!)")

    if 20 < rsi < 80:
        tb_context.append(f"S8: Vola ok (RSI {rsi:.1f}) ✓")

    if macd_bull_cross:
        tb_context.append(f"S9: MACD Bull-Cross (MACD {macd_v:.3f} / Signal {signal_v:.3f}) 🚀")

    if smart_money_default:
        tb_context.append(f"S10: Smart Money sammelt ein (Akkumulation {accumulation_score:.0f} / Distribution {distribution_pressure_score:.0f}) ✓")
    else:
        tb_context.append(f"S10: Smart Money schwach (Akkumulation {accumulation_score:.0f} / Distribution {distribution_pressure_score:.0f}) ❌")

    if adx > 25:
        tb_context.append(f"S11: ADX>25 starker Trend (ADX {adx:.1f}) ✓")
    else:
        tb_context.append(f"S11: ADX<25 Seitwärts (ADX {adx:.1f}) ❌")

    if stoch_k_v < 20 and stoch_d_v < 20 and stoch_k_v > stoch_d_v:
        tb_context.append(f"S12: Stoch Oversold Cross (K {stoch_k_v:.1f} / D {stoch_d_v:.1f}) ✓")
    elif stoch_k_v > 80:
        tb_context.append(f"S12: Stoch überkauft (K {stoch_k_v:.1f}) ❌")
    else:
        tb_context.append(f"S12: Stoch neutral (K {stoch_k_v:.1f} / D {stoch_d_v:.1f}) ❌")

    if willr_v < -80:
        tb_context.append(f"S13: Williams%R extrem Oversold ({willr_v:.1f}) ✓")
    elif willr_v > -20:
        tb_context.append(f"S13: Williams%R überkauft ({willr_v:.1f}) ❌")
    else:
        tb_context.append(f"S13: Williams%R neutral ({willr_v:.1f}) ❌")

    if obv_trend == "steigend" and vol_ratio >= 1.0:
        tb_context.append(f"S14: OBV/Volumen bestätigt (Vol.-Ratio {vol_ratio:.2f}) ✓")
    else:
        tb_context.append(f"S14: OBV/Volumen schwach (Vol.-Ratio {vol_ratio:.2f}) ❌")

    if pd.notna(prev20_high) and price > prev20_high:
        tb_context.append(f"S15: 20D Breakout (Kurs {price:.2f} > Hoch {prev20_high:.2f}) ✓")
    elif pd.notna(prev20_low) and price < prev20_low:
        tb_context.append(f"S15: 20D Breakdown (Kurs {price:.2f} < Tief {prev20_low:.2f}) ❌")
    else:
        tb_context.append(f"S15: Range intakt (20D {prev20_low:.2f}-{prev20_high:.2f}) ❌")

    if pd.notna(bb_upper) and price > bb_upper:
        tb_context.append(f"S16: BB Breakout UP (Kurs {price:.2f} > BB oben {bb_upper:.2f}) ✓")
    elif bb_squeeze:
        tb_context.append("S16: BB Squeeze Achtung ✓")
    elif pd.notna(bb_lower) and price < bb_lower:
        tb_context.append(f"S16: BB Breakout DOWN (Kurs {price:.2f} < BB unten {bb_lower:.2f}) ❌")
    else:
        tb_context.append(f"S16: BB neutral (BB {bb_lower:.2f}-{bb_upper:.2f}) ❌")

    if pd.notna(target) and target > 0 and price > 0:
        tb_potenzial = ((target - price) / price) * 100
        if tb_potenzial > 15:
            tb_context.append(f"S17: Target +{tb_potenzial:.1f}% ✓")
        elif tb_potenzial < 0:
            tb_context.append(f"S17: Target -{abs(tb_potenzial):.1f}% ❌")
        else:
            tb_context.append(f"S17: Target +{tb_potenzial:.1f}% neutral ❌")
    else:
        tb_context.append("S17: Kein valides Target ❌")

    current_month = datetime.now().month
    if current_month in [8, 9]:
        tb_context.append("S18: Seasonality schlecht (-1) ❌")
    elif current_month in [11, 12, 1]:
        tb_context.append("S18: Seasonality stark (+1) ✓")
    else:
        tb_context.append("S18: Seasonality neutral ❌")

    if crv >= 2.0:
        tb_context.append("S19: CRV >= 2.0 ✓")
    elif crv < 1.5:
        tb_context.append("S19: CRV schwach ❌")
    else:
        tb_context.append("S19: CRV ok/neutral ❌")

    short_squeeze = pd.notna(short_pct) and short_pct > 0.12 and ret5 > 0 and vol_ratio > 1.2
    if short_squeeze:
        tb_context.append("S20: 🚀 SHORT SQUEEZE POTENZIAL ✓")
    else:
        tb_context.append(f"S20: kein Short-Squeeze-Signal (Short {fmt_num(short_pct*100 if pd.notna(short_pct) else np.nan,1,'%')}) ❌")

    if pd.notna(pe) and 0 < pe < 15:
        tb_context.append(f"S21: 🟢 VALUE KGV ({pe:.1f}) ✓")
    elif pd.notna(pe) and pe > 50:
        tb_context.append(f"S21: 🔴 TEUER KGV>50 ({pe:.1f}) ❌")
    else:
        tb_context.append(f"S21: Value neutral ({fmt_num(pe,1)}) ❌")

    if market_info["regime"] == "POSITIV":
        tb_context.append(f"S22: Marktfilter positiv ({benchmark_label}) ✓")
    elif market_info["regime"] == "NEGATIV":
        tb_context.append(f"S22: Marktfilter negativ ({benchmark_label}) ❌")
    else:
        tb_context.append(f"S22: Marktfilter neutral ({benchmark_label}) ❌")

    if pd.notna(rs_vs_benchmark_63):
        if rs_vs_benchmark_63 > 0:
            tb_context.append(f"S23: Outperformance vs {benchmark_label} +{rs_vs_benchmark_63:.1f}% ✓")
        else:
            tb_context.append(f"S23: Underperformance vs {benchmark_label} {rs_vs_benchmark_63:.1f}% ❌")
    else:
        tb_context.append("S23: Benchmark-Vergleich n/a ❌")

    tb_signal, tb_empf = tb_signal_label(tb_score)
    tb_score_100 = normalize_tb_score_100(tb_score)
    tb_timing_text = trading_timing_label(tb_score)

    # ---------- Positionsmanagement 2.0 ----------
    if position_mode:
        add_on_action = "Nein"
        partial_profit_action = "Nein"
        stop_action = "Beibehalten"
        risk_note = "Keine Auffälligkeit"
        position_action = "Halten"
        legacy_position_action = "Halten"

        if has_upcoming_earnings and pd.notna(days_earn) and days_earn < 7:
            risk_note = "Earnings-Risiko kurzfristig erhöht"

        if pd.notna(tb_perf) and tb_perf >= 12 and pd.notna(tp1) and price >= tp1 * 0.98:
            partial_profit_action = "Ja, Teilgewinn prüfen"

        if (
            investment_case_score >= 74
            and trading_case_score >= 70
            and valid_trade_setup
            and setup_confidence >= 62
            and market_info["regime"] == "POSITIV"
        ):
            add_on_action = "Ja, selektiv möglich"

        if trading_case_score < 48 or market_info["regime"] == "NEGATIV" or setup_confidence < 45:
            legacy_position_action = "Risiko reduzieren"
        elif trading_case_score < 60 or setup_confidence < 60 or entry_quality == "früh":
            legacy_position_action = "Halten / eng beobachten"
        elif investment_case_score >= 75 and trading_case_score >= 68 and setup_confidence >= 60:
            legacy_position_action = "Halten / ggf. ausbauen"
        else:
            legacy_position_action = "Halten"

        if pd.notna(stop_used) and price > 0:
            if pd.notna(tb_perf) and tb_perf >= 15 and pd.notna(tp1) and price >= tp1:
                stop_action = f"Stop auf {max(stop_used, tb_basispreis):.2f} {ccy} anheben"
            elif pd.notna(tb_perf) and tb_perf >= 8:
                stop_action = f"Stop enger nachziehen auf {stop_used:.2f} {ccy}"
            else:
                stop_action = f"Stop aktuell bei {stop_used:.2f} {ccy}"

        if pd.notna(tb_perf) and tb_perf < -6 and trading_case_score < 55:
            risk_note = "Verlustposition mit schwächerem Setup"
        elif pd.notna(tb_perf) and tb_perf > 18:
            risk_note = "Gewinnposition, aktives Management sinnvoll"
    else:
        position_action = "Nicht anwendbar"
        add_on_action = "Nicht anwendbar"
        partial_profit_action = "Nicht anwendbar"
        stop_action = "Nicht anwendbar"
        risk_note = "Watchlist-Modus"

    # ---------- Watchlist / Trigger-Logik ----------
    if not position_mode:
        if has_upcoming_earnings and pd.notna(days_earn) and days_earn < 7:
            trigger_status = "Warten"
            watchlist_priority = "Niedrig"
            watchlist_priority_score = 30
            next_trigger = "Nach den Zahlen neu prüfen"
            trigger_reason = "Earnings-Veto kurzfristig"
        elif market_info["regime"] == "NEGATIV":
            trigger_status = "Passiv"
            watchlist_priority = "Niedrig"
            watchlist_priority_score = 28
            next_trigger = "Auf besseres Marktumfeld warten"
            trigger_reason = "Marktumfeld aktuell zu schwach"
        elif valid_trade_setup and entry_quality == "gut" and trading_case_score >= 68 and setup_confidence >= 60:
            trigger_status = "Aktiv"
            watchlist_priority = "Hoch"
            watchlist_priority_score = 85
            next_trigger = "Einstieg in Entry-Zone prüfen; gültig solange Trigger-/Support-Zone hält"
            trigger_reason = "Setup valide, Timing stimmig und Kurs in sinnvoller Entry-Zone"
        elif valid_trade_setup and entry_quality == "abwarten" and trading_case_score >= 60 and setup_confidence >= 55:
            trigger_status = "Nahe dran"
            watchlist_priority = "Hoch"
            watchlist_priority_score = 76
            next_trigger = "Rücksetzer in Entry-Zone abwarten"
            trigger_reason = "Setup valide, aber Kurs aktuell über Entry-Zone"
        elif setup_type != "Kein sauberes Setup" and entry_quality == "früh" and trading_case_score >= 55 and setup_confidence >= 50:
            trigger_status = "Frühe Beobachtung"
            watchlist_priority = "Mittel"
            watchlist_priority_score = 60
            next_trigger = "Setup-Confirmation oder bessere Entry-Lage"
            trigger_reason = "Setup vorhanden, aber noch zu früh für einen sauberen Einstieg"
        elif investment_case_score >= 70:
            trigger_status = "Beobachten"
            watchlist_priority = "Mittel"
            watchlist_priority_score = 52
            next_trigger = "Trading-Case verbessern"
            trigger_reason = "Guter Investment-Case, aber noch kein sauberer Trigger"
        else:
            trigger_status = "Passiv"
            watchlist_priority = "Niedrig"
            watchlist_priority_score = 35
            next_trigger = "Auf klareres Setup warten"
            trigger_reason = "Noch kein priorisierter Watchlist-Kandidat"

        if setup_type == "Kein sauberes Setup":
            trigger_status = "Passiv"
            watchlist_priority = "Niedrig" if investment_case_score < 75 else "Mittel"
            watchlist_priority_score = 38 if investment_case_score < 75 else 48
            next_trigger = "Auf neues Setup warten"
            trigger_reason = "Ohne sauberes Setup kein aktiver Trigger"
    else:
        trigger_status = "Nicht anwendbar"
        watchlist_priority = "Nicht anwendbar"
        watchlist_priority_score = np.nan
        next_trigger = "Nicht anwendbar"
        trigger_reason = "Positionsmodus"

    # ---------- Finale Konsistenzregeln ----------
    if not position_mode:
        if setup_type == "Kein sauberes Setup":
            emp = "BEOBACHTEN" if investment_case_score >= 70 else "AVOID / WAIT"
            conv = "LOW-MEDIUM" if investment_case_score >= 70 else "LOW"
        elif trigger_status == "Aktiv" and trading_case_score >= 68:
            if investment >= 78 and kb >= 3 and market_info["regime"] == "POSITIV":
                emp = "BUY / ACCUMULATE"
                conv = "HIGH"
            else:
                emp = "WATCH / EINSTIEG PRÜFEN"
                conv = "MEDIUM" if conv == "-" else conv
        elif trigger_status in {"Nahe dran", "Frühe Beobachtung"}:
            if emp == "BUY / ACCUMULATE":
                emp = "BUY CANDIDATE / TIMING PRÜFEN"
            elif emp not in {"BUY CANDIDATE / TIMING PRÜFEN", "WATCH / EINSTIEG PRÜFEN"}:
                emp = "WATCH / EINSTIEG PRÜFEN"
        elif trigger_status in {"Passiv", "Warten"}:
            emp = "AVOID / WAIT" if investment_case_score < 70 else "BEOBACHTEN"

    if position_mode and position_action == "Risiko reduzieren":
        emp = "RISIKO REDUZIEREN / STOPP PRÜFEN"
        conv = "LOW"

    # ---------- Short-term helper board ----------
    stb_score = 0
    stb_items = []

    if price > ma50:
        stb_score += 2
        stb_items.append("MA50 +2")
    elif price > ma200:
        stb_score += 1
        stb_items.append("MA200 +1")
    else:
        stb_score -= 1
        stb_items.append("Trend -1")

    if 40 < rsi < 60 or rsi < 30:
        stb_score += 1
        stb_items.append("RSI +1")

    if 20 < rsi < 80:
        stb_score += 1
        stb_items.append("Vola +1")

    if macd_hist_current > macd_hist_prev:
        stb_score += 1
        stb_items.append("Momentum +1")

    if macd_bull_cross:
        stb_score += 1
        stb_items.append("Bull-Cross +1")

    if smart_money_default:
        stb_score += 1
        stb_items.append("Smart Money +1")

    if adx > 25:
        stb_score += 1
        stb_items.append("ADX +1")

    if stoch_k_v < 20 and stoch_d_v < 20 and stoch_k_v > stoch_d_v:
        stb_score += 1
        stb_items.append("Stoch +1")

    if willr_v < -80:
        stb_score += 1
        stb_items.append("Williams +1")

    stb_signal, stb_empf = tb_signal_label(stb_score)
    stb_text = ", ".join(stb_items) if stb_items else "keine positiven Kurzfrist-Signale"

    chart_structures_analysis = None
    chart_bias_info = {
        "bias": 0,
        "setup_bias": 0,
        "tradeability_bias": 0,
        "notes_pos": [],
        "notes_neg": [],
        "summary": [],
    }
    try:
        chart_analysis_df = compute_chart_df(df, "1 Jahr")
        chart_structures_analysis = build_chart_structures(chart_analysis_df)
        chart_bias_info = evaluate_chart_structure_bias(chart_analysis_df, chart_structures_analysis)
    except Exception:
        chart_structures_analysis = None
        chart_bias_info = {
            "bias": 0,
            "setup_bias": 0,
            "tradeability_bias": 0,
            "notes_pos": [],
            "notes_neg": [],
            "summary": [],
        }

    if chart_bias_info.get("bias", 0) or chart_bias_info.get("setup_bias", 0) or chart_bias_info.get("tradeability_bias", 0):
        trading_case_score = round(clamp(trading_case_score + chart_bias_info.get("bias", 0)))
        setup_confidence = round(clamp(setup_confidence + chart_bias_info.get("setup_bias", 0)))
        tradeability_score = round(clamp(tradeability_score + chart_bias_info.get("tradeability_bias", 0)))
        trading_case_text = trading_case_label(trading_case_score)
        tradeability_text = tradeability_label(tradeability_score)
        setup_confidence_text = setup_confidence_label(setup_confidence)


    # ---------- Explanations ----------
    strengths, weaknesses, decision_summary = build_decision_explanation(
        setup=setup_adj,
        company=company,
        investment=investment,
        market_regime=market_info["regime"],
        rs_vs_benchmark_63=rs_vs_benchmark_63,
        quality_score=quality_score,
        growth_score=growth_score,
        valuation_score=valuation_score,
        balance_score=balance_score,
        red_flag_items=red_flag_items,
        earnings_warning=earnings_warning,
        kb=kb,
        position_mode=position_mode
    )

    chart_pos_notes = chart_bias_info.get("notes_pos", [])[:2]
    chart_neg_notes = chart_bias_info.get("notes_neg", [])[:2]
    for _note in chart_pos_notes:
        if _note not in strengths:
            strengths.append(f"Chart: {_note}.")
    for _note in chart_neg_notes:
        if _note not in weaknesses:
            weaknesses.append(f"Chart: {_note}.")
    strengths = strengths[:5]
    weaknesses = weaknesses[:5]
    if chart_bias_info.get("summary"):
        decision_summary = f"{decision_summary} Chart-Kontext: {' '.join(chart_bias_info.get('summary', [])[:2])}"


    rows = []
    for line in tb_details:
        if ": " in line:
            k, v = line.split(": ", 1)
            rows.append({"Punkt": k, "Detail": v})
        else:
            rows.append({"Punkt": "Info", "Detail": line})
    tb_df = pd.DataFrame(rows)

    # v15.26.8: Score-relevante TradingBoard-Punkte mit konkreten Messwerten anzeigen.
    # Hintergrund: In der Score-Tabelle stand bei RSI/Trend teils nur "hoch/niedrig"
    # mit Haken/Kreuz, ohne direkt sichtbaren Messwert. Die Werte werden hier als
    # eigene Spalte ergänzt und fehlende Klammerwerte in der Detailspalte nachgezogen.
    def _tb_val_num(value, fmt="{:.1f}", suffix=""):
        try:
            if pd.isna(value):
                return "n/a"
            return fmt.format(float(value)) + suffix
        except Exception:
            return "n/a"

    tb_value_map = {
        "S0": f"Kurs {_tb_val_num(price, '{:.2f}')} {ccy}",
        "S1 Earnings": earnings_dt.strftime('%d.%m.%Y') if pd.notna(earnings_ts) else "kein Datum",
        "S2": f"Kurs {_tb_val_num(price, '{:.2f}')} / MA200 {_tb_val_num(ma200, '{:.2f}')}",
        "S3": f"Kurs {_tb_val_num(price, '{:.2f}')} / MA50 {_tb_val_num(ma50, '{:.2f}')}",
        "S4": f"MA50 {_tb_val_num(ma50, '{:.2f}')} / MA200 {_tb_val_num(ma200, '{:.2f}')}",
        "S5": f"RSI {_tb_val_num(rsi, '{:.1f}')}",
        "S6": f"Performance {_tb_val_num(tb_perf, '{:.1f}', '%')}" if position_mode else "Watchlist-Modus",
        "S7": f"MACD-Hist. aktuell {_tb_val_num(macd_hist_current, '{:.4f}')} / vorher {_tb_val_num(macd_hist_prev, '{:.4f}')}",
        "Info": "Hinweis",
    }

    if not tb_df.empty and "Punkt" in tb_df.columns:
        tb_df["Wert"] = tb_df["Punkt"].astype(str).map(lambda p: tb_value_map.get(p, ""))
        # RSI-Zeile zusätzlich in der Detailspalte absichern, falls alte Texte ohne Wert auftauchen.
        if "Detail" in tb_df.columns:
            rsi_mask = tb_df["Punkt"].astype(str).eq("S5")
            tb_df.loc[rsi_mask, "Detail"] = tb_df.loc[rsi_mask, "Detail"].astype(str).map(
                lambda d: d if "RSI" in d and any(ch.isdigit() for ch in d) else f"{d} (RSI {_tb_val_num(rsi, '{:.1f}')})"
            )
        # Bessere Spaltenreihenfolge: Punkt, Wert, Detail.
        preferred_cols = [c for c in ["Punkt", "Wert", "Detail"] if c in tb_df.columns]
        tb_df = tb_df[preferred_cols + [c for c in tb_df.columns if c not in preferred_cols]]

    context_rows = []
    for line in tb_context:
        if ": " in line:
            k, v = line.split(": ", 1)
            context_rows.append({"Punkt": k, "Detail": v})
        else:
            context_rows.append({"Punkt": "Info", "Detail": line})
    tb_context_df = pd.DataFrame(context_rows)

    # v15.29: Auch Kontext-/Symboltabellen bekommen konkrete Messwerte.
    # Ziel: Keine Blackbox-Zeilen mit nur Haken/Kreuz oder Ampeltexten.
    def _tb_value_safe(value, fmt="{:.1f}", suffix=""):
        try:
            if pd.isna(value):
                return "n/a"
            return fmt.format(float(value)) + suffix
        except Exception:
            return "n/a"

    tb_context_value_map = {
        "RSI / Vola-Kontext": f"RSI {_tb_value_safe(rsi, '{:.1f}')}",
        "MACD": f"MACD {_tb_value_safe(macd_v, '{:.3f}')} / Signal {_tb_value_safe(signal_v, '{:.3f}')}",
        "Smart Money": f"Akkumulation {_tb_value_safe(accumulation_score, '{:.0f}')} / Distribution {_tb_value_safe(distribution_pressure_score, '{:.0f}')}",
        "ADX / Trendstärke": f"ADX {_tb_value_safe(adx, '{:.1f}')}",
        "Stochastik": f"%K {_tb_value_safe(stoch_k_v, '{:.1f}')} / %D {_tb_value_safe(stoch_d_v, '{:.1f}')}",
        "Williams %R": f"Williams %R {_tb_value_safe(willr_v, '{:.1f}')}",
        "OBV / Volumen": f"Vol.-Ratio {_tb_value_safe(vol_ratio, '{:.2f}')}",
        "20D-Range": f"{_tb_value_safe(prev20_low, '{:.2f}')} - {_tb_value_safe(prev20_high, '{:.2f}')} / Kurs {_tb_value_safe(price, '{:.2f}')}",
        "Bollinger-Band": f"{_tb_value_safe(bb_lower, '{:.2f}')} - {_tb_value_safe(bb_upper, '{:.2f}')} / Kurs {_tb_value_safe(price, '{:.2f}')}",
    }
    if not tb_context_df.empty and "Punkt" in tb_context_df.columns:
        tb_context_df["Wert"] = tb_context_df["Punkt"].astype(str).map(lambda p: tb_context_value_map.get(p, ""))
        preferred_cols = [c for c in ["Punkt", "Wert", "Detail"] if c in tb_context_df.columns]
        tb_context_df = tb_context_df[preferred_cols + [c for c in tb_context_df.columns if c not in preferred_cols]]

    red_flags_df = pd.DataFrame(red_flag_items) if red_flag_items else pd.DataFrame(
        [{"Kategorie": "-", "Status": "🟢", "Detail": "Keine relevanten Red Flags erkannt", "Penalty": 0}]
    )

    top_red_flag = hard_red_flag_items[0]["Detail"] if hard_red_flag_items else "-"
    short_thesis = build_short_thesis(investment, tb_score, market_info["regime"], top_red_flag, position_mode)


    # ---------- Exit / Verkaufssystem ----------
    avg_cost = normalize_missing(override) if "override" in locals() else np.nan
    if not pd.notna(avg_cost) or avg_cost <= 0:
        avg_cost = np.nan

    position_pnl_pct = ((price / avg_cost) - 1) * 100 if pd.notna(avg_cost) and avg_cost > 0 and pd.notna(price) else np.nan

    # Kontext für Gewinner / Verlierer / Korrektur
    if pd.notna(position_pnl_pct):
        if position_pnl_pct >= 20:
            pnl_bucket = "starker Gewinner"
        elif position_pnl_pct >= 8:
            pnl_bucket = "Gewinner"
        elif position_pnl_pct <= -10:
            pnl_bucket = "klarer Verlierer"
        elif position_pnl_pct < 0:
            pnl_bucket = "leichter Verlierer"
        else:
            pnl_bucket = "nahe Einstand"
    else:
        pnl_bucket = "ohne Einstandsdaten"
    horizon_label = str(horizon or "").strip() or "unbekannt"

    healthy_trend_context = (
        pd.notna(price) and pd.notna(ma50) and price >= ma50
        and market_info["regime"] == "POSITIV"
        and pd.notna(setup_confidence) and setup_confidence >= 55
        and pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 >= 0
    )

    winner_context = pd.notna(position_pnl_pct) and position_pnl_pct >= 8
    strong_winner_context = pd.notna(position_pnl_pct) and position_pnl_pct >= 18
    loser_context = pd.notna(position_pnl_pct) and position_pnl_pct <= -5

    trend_break_score = 0
    if pd.notna(price) and pd.notna(ma20) and price < ma20:
        trend_break_score += 5 if healthy_trend_context else 8
    if pd.notna(price) and pd.notna(ma50) and price < ma50:
        trend_break_score += 16
    if pd.notna(price) and pd.notna(ma200) and price < ma200:
        trend_break_score += 22
    if pd.notna(ma20) and pd.notna(ma50) and ma20 < ma50:
        trend_break_score += 10
    if pd.notna(ma50) and pd.notna(ma200) and ma50 < ma200:
        trend_break_score += 12
    swing_low_20 = safe_last(close.shift(1).rolling(20).min(), np.nan)
    if pd.notna(price) and pd.notna(swing_low_20) and price < swing_low_20:
        trend_break_score += 10
    if healthy_trend_context and pd.notna(price) and pd.notna(ma50) and price >= ma50:
        trend_break_score = max(0, trend_break_score - 6)
    trend_break_score = min(100, trend_break_score)

    momentum_collapse_score = 0
    if pd.notna(rsi) and rsi < 50:
        momentum_collapse_score += 5
    if pd.notna(rsi) and rsi < 45:
        momentum_collapse_score += 9
    if pd.notna(rsi) and rsi < 40:
        momentum_collapse_score += 14
    if pd.notna(macd_v) and pd.notna(signal_v) and macd_v < signal_v:
        momentum_collapse_score += 8
    if pd.notna(macd_hist_current) and macd_hist_current < 0:
        momentum_collapse_score += 6
    if pd.notna(roc20) and roc20 < 0:
        momentum_collapse_score += 7
    if pd.notna(roc20) and roc20 < -5:
        momentum_collapse_score += 12
    if pd.notna(adx) and adx < 18 and pd.notna(roc20) and roc20 < 0:
        momentum_collapse_score += 5
    if healthy_trend_context and pd.notna(rsi) and rsi >= 45:
        momentum_collapse_score = max(0, momentum_collapse_score - 5)
    momentum_collapse_score = min(100, momentum_collapse_score)

    risky_title_context = (
        (pd.notna(market_cap) and market_cap < 1_000_000_000)
        or (pd.notna(atr_pct) and atr_pct >= 6)
        or (pd.notna(event_risk_score) and event_risk_score >= 60)
    )

    relative_weakness_score = 0
    if risky_title_context:
        relative_weakness_score = 12
    if pd.notna(rs_score) and rs_score < 55:
        relative_weakness_score += 5
    if pd.notna(rs_score) and rs_score < 50:
        relative_weakness_score += 7
    if pd.notna(rs_score) and rs_score < 40:
        relative_weakness_score += 11
    if pd.notna(rs_vs_benchmark_21) and rs_vs_benchmark_21 < 0:
        relative_weakness_score += 6
    if pd.notna(rs_vs_benchmark_21) and rs_vs_benchmark_21 < -4:
        relative_weakness_score += 5
    if pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 < 0:
        relative_weakness_score += 10
    if pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 < -6:
        relative_weakness_score += 6
    if pd.notna(rs_vs_benchmark_126) and rs_vs_benchmark_126 < 0:
        relative_weakness_score += 7
    if pd.notna(rs_composite) and rs_composite < 50:
        relative_weakness_score += 6
    if pd.notna(rs_composite) and rs_composite < 45:
        relative_weakness_score += 10
    if pd.notna(ret5) and ret5 < 0:
        relative_weakness_score += 4 if risky_title_context else 2
    if pd.notna(ret21) and ret21 < 0:
        relative_weakness_score += 5 if risky_title_context else 3
    if healthy_trend_context and pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 > 3:
        relative_weakness_score = max(0, relative_weakness_score - 6)
    relative_weakness_score = min(100, relative_weakness_score)

    vol_ma20 = safe_last(vol.rolling(20).mean(), np.nan)
    prev_close = safe_last(close.shift(1), np.nan)
    ret1 = safe_last(close.pct_change(1) * 100, np.nan)
    ret2 = safe_last(close.shift(1).pct_change(1) * 100, np.nan)
    vol_now = safe_last(vol, np.nan)
    vol_prev = safe_last(vol.shift(1), np.nan)
    down_day = pd.notna(price) and pd.notna(prev_close) and price < prev_close
    high_volume = pd.notna(vol_now) and pd.notna(vol_ma20) and vol_now > 1.3 * vol_ma20

    distribution_score = 0
    if risky_title_context:
        distribution_score = 10
    if down_day and high_volume:
        distribution_score += 10
    dist_day_1 = (
        pd.notna(ret1) and ret1 < -1.5 and
        pd.notna(vol_now) and pd.notna(vol_ma20) and vol_now > 1.2 * vol_ma20
    )
    dist_day_prev = (
        pd.notna(ret2) and ret2 < -1.5 and
        pd.notna(vol_prev) and pd.notna(vol_ma20) and vol_prev > 1.2 * vol_ma20
    )
    if dist_day_1:
        distribution_score += 8
    if dist_day_1 and dist_day_prev:
        distribution_score += 12
    if pd.notna(ret21) and ret21 < 0 and down_day and high_volume:
        distribution_score += 8
    if risky_title_context and pd.notna(ret1) and ret1 < 0:
        distribution_score += 4
    if risky_title_context and pd.notna(ret5) and ret5 < 0:
        distribution_score += 5
    if risky_title_context and pd.notna(vol_ratio) and vol_ratio > 1.05 and pd.notna(ret5) and ret5 < 0:
        distribution_score += 6
    if risky_title_context and pd.notna(price) and pd.notna(ma20) and price < ma20:
        distribution_score += 5
    if healthy_trend_context and not dist_day_1:
        distribution_score = max(0, distribution_score - 4)
    distribution_score = min(100, distribution_score)

    exit_trigger_score = 0
    stop_broken = pd.notna(stop_used) and pd.notna(price) and price < stop_used
    if stop_broken:
        exit_trigger_score += 32
    if pd.notna(days_earn) and days_earn <= 7 and pd.notna(trading_case_score) and trading_case_score < 55:
        exit_trigger_score += 10
    if pd.notna(setup_confidence) and setup_confidence < 40:
        exit_trigger_score += 8
    if red_flag_penalty_total >= 12:
        exit_trigger_score += 8
    gap_down_pct = ((price / prev_close) - 1) * 100 if pd.notna(price) and pd.notna(prev_close) and prev_close != 0 else np.nan
    if pd.notna(gap_down_pct) and gap_down_pct <= -4:
        exit_trigger_score += 15
    if stop_broken and market_info["regime"] == "NEGATIV":
        exit_trigger_score += 8
    exit_trigger_score = min(100, exit_trigger_score)

    # ---------- Kurzfristiger Exit- / De-Risking-Layer ----------
    # bewusst weich kalibriert: neutral startet nicht bei 0, damit fruehe Warnstufen sichtbar werden
    momentum_rollover_score = 24
    if pd.notna(rsi) and rsi < 62:
        momentum_rollover_score += 6
    if pd.notna(rsi) and rsi < 58:
        momentum_rollover_score += 8
    if pd.notna(rsi) and rsi < 54:
        momentum_rollover_score += 10
    if pd.notna(macd_hist_current) and pd.notna(macd_hist_prev) and macd_hist_current < macd_hist_prev:
        momentum_rollover_score += 8
    if pd.notna(macd_v) and pd.notna(signal_v) and macd_v < signal_v:
        momentum_rollover_score += 10
    if pd.notna(price) and pd.notna(ma20) and price < ma20:
        momentum_rollover_score += 14
    if pd.notna(ret5) and ret5 < -0.5:
        momentum_rollover_score += 8
    if pd.notna(ret5) and ret5 < -2:
        momentum_rollover_score += 10
    if pd.notna(ret1) and pd.notna(ret2) and ret1 < 0 and ret2 < 0:
        momentum_rollover_score += 8
    if healthy_trend_context and pd.notna(rsi) and rsi >= 58 and pd.notna(price) and pd.notna(ma20) and price >= ma20:
        momentum_rollover_score = max(14, momentum_rollover_score - 8)
    momentum_rollover_score = min(100, momentum_rollover_score)

    dist_to_ma20_pct = ((price / ma20) - 1) * 100 if pd.notna(price) and pd.notna(ma20) and ma20 != 0 else np.nan
    upper_bb_stretch = pd.notna(bb_upper) and pd.notna(price) and price >= bb_upper * 0.995
    stretch_risk_score = 18
    if pd.notna(dist_to_ma20_pct) and dist_to_ma20_pct >= 2.5:
        stretch_risk_score += 8
    if pd.notna(dist_to_ma20_pct) and dist_to_ma20_pct >= 4:
        stretch_risk_score += 10
    if pd.notna(dist_to_ma20_pct) and dist_to_ma20_pct >= 6.5:
        stretch_risk_score += 14
    if pd.notna(rsi) and rsi >= 64:
        stretch_risk_score += 6
    if pd.notna(rsi) and rsi >= 69:
        stretch_risk_score += 8
    if pd.notna(rsi) and rsi >= 75:
        stretch_risk_score += 12
    if upper_bb_stretch:
        stretch_risk_score += 10
    if strong_winner_context:
        stretch_risk_score += 8
    stretch_risk_score = min(100, stretch_risk_score)

    near_high52 = pd.notna(high52) and pd.notna(price) and high52 > 0 and price >= high52 * 0.985
    near_prev20_high = pd.notna(prev20_high) and pd.notna(price) and prev20_high > 0 and price >= prev20_high * 0.985
    resistance_rejection_score = 18 if (near_high52 or near_prev20_high) else 14
    if down_day and (near_high52 or near_prev20_high):
        resistance_rejection_score += 12
    if down_day and high_volume and (near_high52 or near_prev20_high):
        resistance_rejection_score += 16
    if upper_bb_stretch and pd.notna(ret1) and ret1 < 0:
        resistance_rejection_score += 10
    if pd.notna(price) and pd.notna(ma20) and pd.notna(dist_to_ma20_pct) and dist_to_ma20_pct >= 4 and pd.notna(ret1) and ret1 < 0:
        resistance_rejection_score += 8
    if not (near_high52 or near_prev20_high) and healthy_trend_context:
        resistance_rejection_score = max(10, resistance_rejection_score - 6)
    resistance_rejection_score = min(100, resistance_rejection_score)

    short_term_pressure_score = 18 if (down_day and pd.notna(ret1) and ret1 < 0) else 14
    if down_day and high_volume:
        short_term_pressure_score += 14
    if dist_day_1:
        short_term_pressure_score += 16
    if dist_day_prev:
        short_term_pressure_score += 10
    if pd.notna(ret5) and ret5 < -1.0:
        short_term_pressure_score += 8
    if pd.notna(ret5) and ret5 < -2.5:
        short_term_pressure_score += 10
    if pd.notna(vol_ratio) and vol_ratio > 1.08 and pd.notna(ret5) and ret5 < 0:
        short_term_pressure_score += 8
    if pd.notna(price) and pd.notna(ma20) and price < ma20:
        short_term_pressure_score += 8
    if healthy_trend_context and not dist_day_1 and not high_volume:
        short_term_pressure_score = max(12, short_term_pressure_score - 8)
    short_term_pressure_score = min(100, short_term_pressure_score)

    failed_breakout_score = 18
    failed_breakout = False
    if pd.notna(prev20_high) and pd.notna(prev_close) and pd.notna(price):
        failed_breakout = prev_close > prev20_high * 1.002 and price < prev20_high * 0.998
    if failed_breakout:
        failed_breakout_score += 24
    if failed_breakout and high_volume:
        failed_breakout_score += 12
    if pd.notna(ret5) and ret5 < 0 and (near_prev20_high or breakout_context):
        failed_breakout_score += 10
    if breakout_context and pd.notna(ret1) and ret1 < -0.8:
        failed_breakout_score += 8
    if not breakout_context and not near_prev20_high:
        failed_breakout_score = max(16, failed_breakout_score - 4)
    failed_breakout_score = min(100, failed_breakout_score)

    instrument_volatility_risk_score = 24
    if pd.notna(atr_pct):
        if atr_pct >= 10:
            instrument_volatility_risk_score += 26
        elif atr_pct >= 7:
            instrument_volatility_risk_score += 18
        elif atr_pct >= 5:
            instrument_volatility_risk_score += 10
        elif atr_pct >= 3.5:
            instrument_volatility_risk_score += 4
    if pd.notna(event_risk_score):
        if event_risk_score >= 75:
            instrument_volatility_risk_score += 18
        elif event_risk_score >= 60:
            instrument_volatility_risk_score += 12
        elif event_risk_score >= 45:
            instrument_volatility_risk_score += 6
    if pd.notna(market_cap):
        if market_cap < 300_000_000:
            instrument_volatility_risk_score += 18
        elif market_cap < 1_000_000_000:
            instrument_volatility_risk_score += 10
        elif market_cap < 3_000_000_000:
            instrument_volatility_risk_score += 4
    if pd.notna(ret1) and abs(ret1) >= 8:
        instrument_volatility_risk_score += 12
    elif pd.notna(ret1) and abs(ret1) >= 5:
        instrument_volatility_risk_score += 7
    if pd.notna(ret5) and abs(ret5) >= 18:
        instrument_volatility_risk_score += 10
    elif pd.notna(ret5) and abs(ret5) >= 12:
        instrument_volatility_risk_score += 6
    instrument_volatility_risk_score = min(100, instrument_volatility_risk_score)

    chart_event_risk = clamp(
        momentum_rollover_score * 0.30
        + resistance_rejection_score * 0.20
        + short_term_pressure_score * 0.20
        + stretch_risk_score * 0.15
        + failed_breakout_score * 0.15
    )

    tactical_exit_risk = round(clamp(
        chart_event_risk * 0.68
        + instrument_volatility_risk_score * 0.32
    ))

    if healthy_trend_context and not down_day and not dist_day_1 and pd.notna(rsi) and rsi >= 58:
        tactical_exit_risk = max(12, tactical_exit_risk - 6)
    if strong_winner_context and tactical_exit_risk >= 35:
        tactical_exit_risk = min(100, tactical_exit_risk + 6)
    if instrument_volatility_risk_score >= 55:
        tactical_exit_risk = max(tactical_exit_risk, 28)
    elif instrument_volatility_risk_score >= 42:
        tactical_exit_risk = max(tactical_exit_risk, 22)

    if tactical_exit_risk >= 78:
        tactical_exit_text = "akute Ruecksetzergefahr"
    elif tactical_exit_risk >= 60:
        tactical_exit_text = "de-risking sinnvoll"
    elif tactical_exit_risk >= 42:
        tactical_exit_text = "Stop enger / Teilgewinn"
    elif tactical_exit_risk >= 25:
        tactical_exit_text = "fruehe Warnung"
    else:
        tactical_exit_text = "ruhig, aber beobachten"

    if tactical_exit_risk >= 78:
        tactical_exit_action = "De-Risking / Teilverkauf"
    elif tactical_exit_risk >= 60:
        tactical_exit_action = "Stop enger ziehen"
    elif tactical_exit_risk >= 42:
        tactical_exit_action = "Teilgewinn pruefen" if winner_context else "Stop enger ziehen"
    elif tactical_exit_risk >= 25:
        tactical_exit_action = "Kurzfristig vorsichtiger"
    else:
        tactical_exit_action = "Weiter halten / beobachten"

    tactical_exit_reasons = []
    if instrument_volatility_risk_score >= 60:
        tactical_exit_reasons.append("Titel mit hoher Kurzfrist- und Gap-Gefahr")
    elif instrument_volatility_risk_score >= 45:
        tactical_exit_reasons.append("Titel bleibt kurzfristig ruecksetzeranfaellig")
    if failed_breakout:
        tactical_exit_reasons.append("Fehlausbruch am 20T-Hoch")
    if resistance_rejection_score >= 24:
        tactical_exit_reasons.append("Rejection nahe Widerstand / Hoch")
    if pd.notna(dist_to_ma20_pct) and dist_to_ma20_pct >= 8:
        tactical_exit_reasons.append("Stark ueber MA20 gedehnt")
    if dist_day_1:
        tactical_exit_reasons.append("Distributionstag im kurzfristigen Fenster")
    if pd.notna(price) and pd.notna(ma20) and price < ma20:
        tactical_exit_reasons.append("Kurs unter MA20")
    if pd.notna(rsi) and rsi >= 76 and pd.notna(ret1) and ret1 < 0:
        tactical_exit_reasons.append("ueberdehnte Rally kippt")
    if instrument_volatility_risk_score >= 55 and pd.notna(atr_pct):
        tactical_exit_reasons.append(f"ATR {atr_pct:.1f}% zeigt erhoehte Kurzfrist-Volatilitaet")
    tactical_exit_reason_top = tactical_exit_reasons[0] if tactical_exit_reasons else "leichte taktische Beobachtung ohne akuten Ruecksetzerhinweis"

    tactical_critical_signals = []
    tactical_watch_signals = []
    tactical_ok_signals = []

    if failed_breakout or resistance_rejection_score >= 48:
        tactical_critical_signals.append("Rejection / Fehlausbruch an Widerstand")
    elif resistance_rejection_score >= 24:
        tactical_watch_signals.append("Widerstandsnaehe mit erster Rejection")
    else:
        tactical_ok_signals.append("Keine klare Rejection an Widerstand")

    if momentum_rollover_score >= 50:
        tactical_critical_signals.append("Momentum kippt kurzfristig")
    elif momentum_rollover_score >= 28:
        tactical_watch_signals.append("Momentum rollt sichtbar ab")
    else:
        tactical_ok_signals.append("Momentum bleibt stabil")

    if short_term_pressure_score >= 44:
        tactical_critical_signals.append("Kurzfristiger Abgabedruck / Distribution")
    elif short_term_pressure_score >= 24:
        tactical_watch_signals.append("Erste Drucksignale im kurzfristigen Fenster")
    else:
        tactical_ok_signals.append("Kein markanter kurzfristiger Volumendruck")

    if stretch_risk_score >= 48:
        tactical_watch_signals.append("Rally ueberdehnt, Ruecksetzer anfaellig")
    elif stretch_risk_score <= 28:
        tactical_ok_signals.append("Keine starke Ueberdehnung")

    if instrument_volatility_risk_score >= 60:
        tactical_critical_signals.append("Titel selbst bleibt sehr ruecksetzeranfaellig")
    elif instrument_volatility_risk_score >= 42:
        tactical_watch_signals.append("Erhoehte Titel-Volatilitaet / Gap-Risiko")
    else:
        tactical_ok_signals.append("Titel-Risiko kurzfristig beherrschbar")

    tactical_signal_summary = tactical_critical_signals[:2] + tactical_watch_signals[:2]

    exit_score = round(clamp(
        trend_break_score * 0.31
        + momentum_collapse_score * 0.20
        + relative_weakness_score * 0.18
        + distribution_score * 0.13
        + exit_trigger_score * 0.18
    ))

    if strong_winner_context and healthy_trend_context:
        exit_score = max(0, exit_score - 10)
    elif winner_context and healthy_trend_context:
        exit_score = max(0, exit_score - 6)
    elif loser_context:
        exit_score = min(100, exit_score + 8)

    if horizon == "Swing (1-4 Wochen)":
        exit_score = min(100, exit_score + 5)
    elif horizon == "Langfristig (6-24 Monate)":
        exit_score = max(0, exit_score - 4)

    # Mindest-Score, wenn bereits echte Exit-Gründe vorliegen
    structural_exit_reasons = 0
    if pd.notna(ma20) and pd.notna(ma50) and ma20 < ma50:
        structural_exit_reasons += 1
    if pd.notna(rs_vs_benchmark_21) and rs_vs_benchmark_21 < 0:
        structural_exit_reasons += 1
    if pd.notna(price) and pd.notna(ma50) and price < ma50:
        structural_exit_reasons += 1
    if pd.notna(rsi) and rsi < 45:
        structural_exit_reasons += 1
    if pd.notna(macd_v) and pd.notna(signal_v) and macd_v < signal_v:
        structural_exit_reasons += 1
    if stop_broken:
        structural_exit_reasons += 2

    if structural_exit_reasons >= 3:
        exit_score = max(exit_score, 35)
    elif structural_exit_reasons == 2:
        exit_score = max(exit_score, 24)
    elif structural_exit_reasons == 1:
        exit_score = max(exit_score, 12)

    near_tp1 = pd.notna(tp1) and pd.notna(price) and price >= tp1 * 0.98
    near_tp2 = pd.notna(tp2) and pd.notna(price) and price >= tp2 * 0.96
    de_risk_gain_zone = winner_context and (near_tp1 or near_tp2)

    if exit_score >= 80:
        exit_score_text = "klarer Exit-Druck"
    elif exit_score >= 65:
        exit_score_text = "Verkaufsdruck erhöht"
    elif exit_score >= 45:
        exit_score_text = "Gewinne absichern"
    elif exit_score >= 25:
        exit_score_text = "erste Schwäche"
    else:
        exit_score_text = "stabil"

    if stop_broken:
        exit_action = "Verkaufen"
    elif exit_score >= 80:
        exit_action = "Verkaufen"
    elif exit_score >= 65:
        exit_action = "Risiko reduzieren"
    elif exit_score >= 45:
        if winner_context or de_risk_gain_zone:
            exit_action = "Teilgewinn prüfen"
        else:
            exit_action = "Risiko reduzieren"
    elif exit_score >= 25:
        exit_action = "Beobachten"
    else:
        exit_action = "Halten"

    if position_mode:
        legacy_action_for_merge = legacy_position_action if "legacy_position_action" in locals() else position_action
        if exit_action in {"Verkaufen", "Risiko reduzieren"}:
            position_action = exit_action
        elif tactical_exit_risk >= 78:
            position_action = "De-Risking / Teilverkauf"
        elif exit_action == "Teilgewinn prüfen" or tactical_exit_action == "Teilgewinn pruefen":
            position_action = "Teilgewinn prüfen"
        elif tactical_exit_action == "Stop enger ziehen":
            position_action = "Halten / Stop enger"
        elif exit_action == "Beobachten" or tactical_exit_action == "Kurzfristig vorsichtiger":
            position_action = "Halten / eng beobachten"
        elif str(add_on_action).lower().startswith("ja") and exit_score < 25 and tactical_exit_risk < 25:
            position_action = "Halten / ggf. ausbauen"
        elif str(partial_profit_action).lower().startswith("ja") and winner_context:
            position_action = "Teilgewinn prüfen"
        else:
            position_action = legacy_action_for_merge

        if de_risk_gain_zone and exit_action in {"Teilgewinn prüfen", "Beobachten"}:
            partial_profit_action = "Ja, Teilgewinn prüfen"
        elif tactical_exit_risk >= 78:
            partial_profit_action = "Ja, Teilgewinn prüfen"
            add_on_action = "Nein"
            risk_note = f"Taktischer Exit: akute Ruecksetzergefahr - {tactical_exit_reason_top} - {pnl_bucket}"
        elif exit_action == "Verkaufen":
            partial_profit_action = "Nein"
            add_on_action = "Nein"
            risk_note = f"Exit-Modell: klarer Verkaufsdruck - {pnl_bucket}"
        elif exit_action == "Risiko reduzieren":
            add_on_action = "Nein"
            risk_note = f"Exit-Modell: Risikoabbau sinnvoll - {pnl_bucket}"
        elif exit_action == "Teilgewinn prüfen" or tactical_exit_action == "Teilgewinn pruefen":
            partial_profit_action = "Ja, Teilgewinn prüfen"
            risk_note = f"Gewinnsicherung sinnvoll - {tactical_exit_reason_top if tactical_exit_risk >= 42 else pnl_bucket}"
        elif tactical_exit_action == "Stop enger ziehen":
            stop_action = f"Stop enger ziehen - {stop_action}" if str(stop_action).strip() not in {"", "-", "Nicht anwendbar"} else "Stop enger ziehen"
            risk_note = f"Kurzfristige Ruecksetzergefahr - {tactical_exit_reason_top}"
        elif exit_action == "Beobachten" or tactical_exit_action == "Kurzfristig vorsichtiger":
            risk_note = f"Fruehe Exit-Schwäche - {tactical_exit_reason_top if tactical_exit_risk >= 25 else pnl_bucket}"
        elif str(add_on_action).lower().startswith("ja"):
            risk_note = f"Konstruktive Lage trotz Positionsmodus - {pnl_bucket}"

        if pd.notna(days_earn) and days_earn <= 7 and max(exit_score, tactical_exit_risk) >= 45:
            risk_note = f"Earnings-Risiko bei erhoehter Exit-Schwäche - {pnl_bucket}"

        # ---------- v15.22: professioneller Positionsmanagement-Layer ----------
        # Ziel: die Post-Entry-Ausgabe trennt Fuehrungsaktion, Stop/Absicherung,
        # Gewinnschutz, Ausbauverbot und Exit-Druck klarer als die alte Sammellogik.
        _max_exit_pressure = max(
            float(exit_score) if pd.notna(exit_score) else 0.0,
            float(tactical_exit_risk) if pd.notna(tactical_exit_risk) else 0.0,
        )

        if stop_broken or exit_action == "Verkaufen" or _max_exit_pressure >= 82:
            pm_action = "Exit prüfen"
            pm_action_reason = "Stop-/Exit-Signal ist kritisch genug, um die Position nicht mehr nur passiv zu halten."
            pm_exit_pressure = "hoch"
        elif exit_action == "Risiko reduzieren" or _max_exit_pressure >= 65:
            pm_action = "Reduzieren / absichern"
            pm_action_reason = "Exit- und Taktiksignale sprechen für Risikoabbau statt Nachkauf."
            pm_exit_pressure = "erhöht"
        elif str(partial_profit_action).lower().startswith("ja") or (winner_context and _max_exit_pressure >= 42):
            pm_action = "Teilgewinn prüfen"
            pm_action_reason = "Die Position liegt im Gewinn und erste Gegen-/Risikofaktoren nehmen zu."
            pm_exit_pressure = "mittel"
        elif tactical_exit_action == "Stop enger ziehen" or _max_exit_pressure >= 35:
            pm_action = "Halten, Stop enger"
            pm_action_reason = "Grundsetup bleibt haltbar, kurzfristige Schwäche verlangt aber engere Führung."
            pm_exit_pressure = "mittel"
        elif (
            str(add_on_action).lower().startswith("ja")
            and _max_exit_pressure < 18
            and winner_context
            and market_info["regime"] == "POSITIV"
            and setup_confidence >= 75
            and trading_case_score >= 72
            and signal_conflict_label.lower() in {"konsistent", "-", ""}
        ):
            pm_action = "Aufstocken prüfen"
            pm_action_reason = "Position ist im Gewinn, Setup/Timing sind hochwertig, Marktregime unterstützt und Exit-Druck bleibt sehr niedrig."
            pm_exit_pressure = "niedrig"
        elif str(add_on_action).lower().startswith("ja") and _max_exit_pressure < 25 and winner_context and market_info["regime"] == "POSITIV":
            pm_action = "Halten / selektiv aufstocken"
            pm_action_reason = "Position ist konstruktiv, Marktregime unterstützt und Exit-Druck bleibt niedrig; Ausbau nur bei sauberem Trigger."
            pm_exit_pressure = "niedrig"
        elif loser_context and trading_case_score < 58:
            pm_action = "Nicht nachkaufen"
            pm_action_reason = "Verlustposition ohne klaren frischen Vorteil sollte nicht verbilligt werden."
            pm_exit_pressure = "mittel"
        else:
            pm_action = "Halten"
            pm_action_reason = "Keine ausreichend starken Signale für Ausbau, Teilgewinn oder Exit."
            pm_exit_pressure = "niedrig" if _max_exit_pressure < 25 else "mittel"

        # Stop-/Absicherungsplan verständlicher formulieren.
        if stop_broken:
            pm_stop_plan = "Stop/Invalidation verletzt - Exit sofort prüfen"
        elif pd.notna(stop_used) and pd.notna(price) and float(stop_used) > 0:
            if winner_context and pd.notna(tb_basispreis) and float(tb_basispreis) > 0:
                _breakeven_stop = max(float(stop_used), float(tb_basispreis))
                pm_stop_plan = f"Stop mindestens Richtung Einstand/Support führen: ca. {_breakeven_stop:.2f} {ccy}"
            elif tactical_exit_action in {"Stop enger ziehen", "Kurzfristig vorsichtiger"} or _max_exit_pressure >= 35:
                pm_stop_plan = f"Stop enger kontrollieren: aktuell ca. {float(stop_used):.2f} {ccy}"
            else:
                pm_stop_plan = f"Stop beibehalten: ca. {float(stop_used):.2f} {ccy}"
        else:
            pm_stop_plan = "Kein belastbares Stopniveau ableitbar"

        if str(partial_profit_action).lower().startswith("ja") or pm_action == "Teilgewinn prüfen":
            pm_profit_plan = "Teilgewinn prüfen, Restposition nur mit sauberem Stop führen"
        elif winner_context and _max_exit_pressure < 35:
            pm_profit_plan = "Gewinne laufen lassen, aber Stop systematisch nachziehen"
        elif loser_context:
            pm_profit_plan = "Kein Gewinnschutz - Fokus auf Verlustbegrenzung und Setup-Validität"
        else:
            pm_profit_plan = "Noch kein separater Gewinnschutz nötig"

        if pm_action == "Aufstocken prüfen":
            pm_add_plan = "Aufstocken aktiv prüfen - nur mit sauberem Add-on-Trigger und begrenzter Positionsgröße"
        elif pm_action == "Halten / selektiv aufstocken":
            pm_add_plan = "Nur selektiv aufstocken, nicht in Überdehnung oder vor Event-Risiko"
        elif _max_exit_pressure >= 35 or loser_context or market_info["regime"] == "NEGATIV":
            pm_add_plan = "Nicht nachkaufen"
        else:
            pm_add_plan = "Aufstocken nur bei frischem Trigger"

        pm_no_add_if = "Nicht nachkaufen, wenn Exit-Druck steigt, der Stop fällt oder der Kurs ohne neuen Trigger überdehnt."
        if market_info["regime"] == "NEGATIV":
            pm_no_add_if = "Nicht nachkaufen im schwachen Marktregime; zuerst Stabilisierung und relative Stärke abwarten."
        elif pd.notna(days_earn) and days_earn <= 7:
            pm_no_add_if = "Nicht vor nahen Earnings aufstocken; Event-Risiko zuerst abwarten."
        elif loser_context:
            pm_no_add_if = "Nicht in eine Verlustposition hinein verbilligen, solange der Trading-Case nicht klar dreht."

        # Hauptaktion mit dem neuen Positionsmanagement synchronisieren.
        position_action = pm_action
        risk_note = pm_action_reason

    if not position_mode:
        pm_action = "Nicht anwendbar"
        pm_action_reason = "Pre-Entry-Modus"
        pm_stop_plan = "Nicht anwendbar"
        pm_profit_plan = "Nicht anwendbar"
        pm_add_plan = "Nicht anwendbar"
        pm_exit_pressure = "Nicht anwendbar"
        pm_no_add_if = "Nicht anwendbar"

    exit_reason_list = []
    if pd.notna(price) and pd.notna(ma50) and price < ma50:
        exit_reason_list.append("Kurs unter MA50")
    if pd.notna(price) and pd.notna(ma200) and price < ma200:
        exit_reason_list.append("Kurs unter MA200")
    if pd.notna(ma20) and pd.notna(ma50) and ma20 < ma50:
        exit_reason_list.append("MA20 unter MA50")
    if pd.notna(rsi) and rsi < 45:
        exit_reason_list.append("RSI unter 45")
    if pd.notna(macd_v) and pd.notna(signal_v) and macd_v < signal_v:
        exit_reason_list.append("MACD unter Signal")
    if pd.notna(roc20) and roc20 < 0:
        exit_reason_list.append("ROC20 negativ")
    if pd.notna(rs_vs_benchmark_21) and rs_vs_benchmark_21 < 0:
        exit_reason_list.append("Relative Schwäche vs Benchmark")
    if stop_broken:
        exit_reason_list.append("Stop unterschritten")
    if pd.notna(gap_down_pct) and gap_down_pct <= -4:
        exit_reason_list.append("deutlicher Gap-down")
    if dist_day_1:
        exit_reason_list.append("Distributionstag")
    if de_risk_gain_zone and not exit_reason_list:
        exit_reason_list.append("Gewinnzone erreicht, Teilgewinn sinnvoll")
    if tactical_exit_risk >= 42:
        exit_reason_list.insert(0, tactical_exit_reason_top)

    # Doppelte Exit-Gründe entfernen, Reihenfolge aber beibehalten
    deduped_exit_reason_list = []
    seen_exit_reasons = set()
    for reason in exit_reason_list:
        reason_key = str(reason).strip()
        if reason_key and reason_key not in seen_exit_reasons:
            deduped_exit_reason_list.append(reason_key)
            seen_exit_reasons.add(reason_key)
    exit_reason_list = deduped_exit_reason_list

    exit_reason_top = exit_reason_list[0] if exit_reason_list else "kein akuter Exit-Grund"

    if position_mode:
        if exit_score >= 80:
            hmap["Kurzfrist"] = min(hmap["Kurzfrist"], 25)
            hmap["Swing"] = min(hmap["Swing"], 22)
            hmap["Mittelfrist"] = min(hmap["Mittelfrist"], 28)
        elif exit_score >= 65:
            hmap["Kurzfrist"] = min(hmap["Kurzfrist"], 35)
            hmap["Swing"] = min(hmap["Swing"], 32)
            hmap["Mittelfrist"] = min(hmap["Mittelfrist"], 40)
        elif exit_score >= 45:
            hmap["Kurzfrist"] = min(hmap["Kurzfrist"], 48)
            hmap["Swing"] = min(hmap["Swing"], 50)

    stock_fomo_pkg = build_stock_fomo_package_v1525({
        "price": price,
        "high52": high52,
        "dist52": dist52,
        "ret21": ret21,
        "ret63": ret63,
        "stretch_risk_score": stretch_risk_score,
        "volume_quality_score": volume_quality_score,
        "accumulation_score": accumulation_score,
        "distribution_pressure_score": distribution_pressure_score,
        "distribution_score": distribution_score,
        "breakout_volume_score": breakout_volume_score,
        "leadership_score": leadership_score,
        "rs_acceleration_score": rs_acceleration_score,
        "entry_quality": entry_quality,
    })
    market_fomo_pkg = build_market_fomo_package_v1525(market_info)
    fomo_smart_money_pkg = combine_fomo_packages_v1525(stock_fomo_pkg, market_fomo_pkg)

    return {
        "ticker": ticker,
        "df": df,
        "info": info,
        "name": name,
        "ccy": ccy,
        "exch": exch,
        "ts": ts,
        "sector": sector,
        "industry": industry,
        "company_summary": company_summary,
        "confidence_info": confidence_info,
        "market_info": market_info,
        "stock_fomo_pkg": stock_fomo_pkg,
        "market_fomo_pkg": market_fomo_pkg,
        "fomo_smart_money_pkg": fomo_smart_money_pkg,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_label": benchmark_label,
        "price": price,
        "analysis_price": price,
        "live_quote": live_quote,
        "live_price": live_price,
        "live_price_source": live_price_source,
        "live_price_diff_pct": live_price_diff_pct,
        "live_price_note": live_price_note,
        "target": target,
        "upside": upside,
        "regime": regime,
        "reg_amp": reg_amp,
        "sg_earn": sg_earn,
        "sg_earn_txt": sg_earn_txt,
        "days_earn": days_earn,
        "has_upcoming_earnings": has_upcoming_earnings,
        "has_past_earnings": has_past_earnings,
        "fund_cov": fund_cov,
        "fund_fields_loaded": fund_fields_loaded,
        "fund_data_warning": fund_data_warning,
        "red_flag_items": red_flag_items,
        "red_flags_df": red_flags_df,
        "red_flag_notes": red_flag_notes,
        "red_flag_penalty_total": red_flag_penalty_total,
        "top_red_flag": top_red_flag,
        "quality_score": quality_score,
        "growth_score": growth_score,
        "growth_quality": growth_quality,
        "valuation_score": valuation_score,
        "balance_score": balance_score,
        "sentiment_score": sentiment_score,
        "risk_score": risk_score,
        "company": company,
        "setup": setup,
        "setup_adj": setup_adj,
        "investment": investment,
        "tb_score": tb_score,
        "tb_score_100": tb_score_100,
        "tb_timing_text": tb_timing_text,
        "position_action": position_action,
        "exit_score": exit_score,
        "exit_score_text": exit_score_text,
        "tactical_exit_risk": tactical_exit_risk,
        "tactical_exit_text": tactical_exit_text,
        "tactical_exit_action": tactical_exit_action,
        "tactical_exit_reason_top": tactical_exit_reason_top,
        "instrument_volatility_risk_score": instrument_volatility_risk_score,
        "tactical_critical_signals": tactical_critical_signals,
        "tactical_watch_signals": tactical_watch_signals,
        "tactical_ok_signals": tactical_ok_signals,
        "tactical_signal_summary": tactical_signal_summary,
        "momentum_rollover_score": momentum_rollover_score,
        "resistance_rejection_score": resistance_rejection_score,
        "short_term_pressure_score": short_term_pressure_score,
        "stretch_risk_score": stretch_risk_score,
        "failed_breakout_score": failed_breakout_score,
        "trend_break_score": trend_break_score,
        "momentum_collapse_score": momentum_collapse_score,
        "relative_weakness_score": relative_weakness_score,
        "distribution_score": distribution_score,
        "exit_trigger_score": exit_trigger_score,
        "exit_action": exit_action,
        "exit_reason_top": exit_reason_top,
        "exit_reason_list": exit_reason_list,
        "position_pnl_pct": position_pnl_pct,
        "pnl_bucket": pnl_bucket,
        "horizon_label": horizon_label,
        "add_on_action": add_on_action,
        "partial_profit_action": partial_profit_action,
        "stop_action": stop_action,
        "risk_note": risk_note,
        "pm_action": pm_action,
        "pm_action_reason": pm_action_reason,
        "pm_stop_plan": pm_stop_plan,
        "pm_profit_plan": pm_profit_plan,
        "pm_add_plan": pm_add_plan,
        "pm_exit_pressure": pm_exit_pressure,
        "pm_no_add_if": pm_no_add_if,
        "trigger_status": trigger_status,
        "watchlist_priority": watchlist_priority,
        "watchlist_priority_score": watchlist_priority_score,
        "sector_strength_score": sector_strength_score,
        "industry_strength_score": industry_strength_score,
        "rs_benchmark_score": rs_benchmark_score,
        "rs_acceleration_score": rs_acceleration_score,
        "leadership_score": leadership_score,
        "leadership_status": leadership_status,
        "sector_label": sector_label,
        "industry_label": industry_label,
        "sector_trend_text": sector_trend_text,
        "industry_trend_text": industry_trend_text,
        "trend_quality_score": trend_quality_score,
        "ma20_slope": ma20_slope,
        "ma50_slope": ma50_slope,
        "ma200_slope": ma200_slope,
        "higher_lows_score": higher_lows_score,
        "base_quality_score": base_quality_score,
        "base_length_days": base_length_days,
        "correction_depth_pct": correction_depth_pct,
        "range_tightness_score": range_tightness_score,
        "volatility_contraction_score": volatility_contraction_score,
        "pullback_quality_score": pullback_quality_score,
        "volume_quality_proxy": volume_quality_proxy,
        "setup_type_quality_score": setup_type_quality_score,
        "setup_priority_score": setup_priority_score,
        "sector_strength_available": sector_strength_available,
        "volume_quality_score": volume_quality_score,
        "accumulation_score": accumulation_score,
        "distribution_pressure_score": distribution_pressure_score,
        "pullback_dryup_score": pullback_dryup_score,
        "breakout_volume_score": breakout_volume_score,
        "up_down_volume_ratio": up_down_volume_ratio,
        "volume_trend_score": volume_trend_score,
        "accumulation_day_count": accumulation_day_count,
        "distribution_day_count": distribution_day_count,
        "recent_pullback_volume_ratio": recent_pullback_volume_ratio,
        "breakout_day_volume_ratio": breakout_day_volume_ratio,
        "catalyst_score": catalyst_score,
        "earnings_event_score": earnings_event_score,
        "post_earnings_reaction_score": post_earnings_reaction_score,
        "revision_momentum_score": revision_momentum_score,
        "event_risk_score": event_risk_score,
        "catalyst_text": catalyst_text,
        "post_earnings_text": post_earnings_text,
        "event_phase_label": event_phase_label,
        "earnings_reaction_5d": earnings_reaction_5d,
        "earnings_reaction_10d": earnings_reaction_10d,
        "cashflow_stability_score": cashflow_stability_score,
        "margin_stability_score": margin_stability_score,
        "institutional_quality_score": institutional_quality_score,
        "institutional_quality_text": institutional_quality_text,
        "sector_etf_symbol": sector_etf_symbol if sector_etf_symbol else "-",
        "next_trigger": next_trigger,
        "trigger_reason": trigger_reason,
        "tb_signal": tb_signal,
        "tb_empf": tb_empf,
        "tb_df": tb_df,
        "tb_context_df": tb_context_df,
        "tb_details": tb_details,
        "tb_context": tb_context,
        "stb_score": stb_score,
        "stb_signal": stb_signal,
        "stb_empf": stb_empf,
        "stb_text": stb_text,
        "kb": kb,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "decision_summary": decision_summary,
        "mode_label": "Position" if position_mode else "Watchlist",
        "stock_style": stock_style,
        "market_bucket": infer_market_bucket(ticker, info),
        "hmap": hmap,
        "atr_stop": atr_stop,
        "stop_used": stop_used,
        "stop_dist": stop_dist,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp1_source": tp1_source,
        "tp2_source": tp2_source,
        "tp3_source": tp3_source,
        "technical_target_1": technical_target_1,
        "technical_target_2": technical_target_2,
        "stop_source": stop_source,
        "suggested_entry_zone": suggested_entry_zone,
        "entry_source": entry_source,
        "entry_quality": entry_quality,
        "tradeability_score": tradeability_score,
        "tradeability_text": tradeability_text,
        "setup_type": setup_type,
        "preferred_entry": preferred_entry,
        "investment_case_score": investment_case_score,
        "investment_case_text": investment_case_text,
        "trading_case_score": trading_case_score,
        "trading_case_text": trading_case_text,
        "setup_confidence": setup_confidence,
        "setup_confidence_text": setup_confidence_text,
        "trade_crv_score": crv_score,
        "trade_stop_score": stop_score,
        "trade_entry_score": entry_score,
        "trade_timing_score": timing_trade_score,
        "trade_market_score": market_trade_score,
        "crv": crv,
        "pos_size": pos_size,
        "risk_eur": risk_eur,
        "risk_pct": risk_pct,
        "time_stop": time_stop,
        "valid_trade_setup": valid_trade_setup,
        "short_term_score": short_term_score,
        "s3": s3,
        "s3a": s3a,
        "s3t": s3t,
        "s4": s4,
        "s4a": s4a,
        "s4t": s4t,
        "s5": s5,
        "s5a": s5a,
        "s5t": s5t,
        "s6": s6,
        "s6a": s6a,
        "s6t": s6t,
        "w52": w52,
        "dist52": dist52,
        "rs_score": rs_score,
        "rs_composite": rs_composite,
        "ret21": ret21,
        "ret63": ret63,
        "ret126": ret126,
        "bench_ret21": bench_ret21,
        "bench_ret63": bench_ret63,
        "bench_ret126": bench_ret126,
        "rs_vs_benchmark_21": rs_vs_benchmark_21,
        "rs_vs_benchmark_63": rs_vs_benchmark_63,
        "rs_vs_benchmark_126": rs_vs_benchmark_126,
        "ma10": ma10,
        "ma10_dist_pct": ma10_dist_pct,
        "ma10_timing_label": ma10_timing_label,
        "ma10_timing_text": ma10_timing_text,
        "ma20": ma20,
        "ma50": ma50,
        "ma150": ma150,
        "ma200": ma200,
        "history_days": history_days,
        "history_mode": history_mode,
        "new_listing": bool(history_days < 120),
        "rsi": rsi,
        "macd_v": macd_v,
        "signal_v": signal_v,
        "macd_hist_current": macd_hist_current,
        "adx": adx,
        "atr": atr,
        "atr_pct": atr_pct,
        "stoch_k_v": stoch_k_v,
        "stoch_d_v": stoch_d_v,
        "willr_v": willr_v,
        "roc20": roc20,
        "roc60": roc60,
        "high52": high52,
        "low52": low52,
        "profit_margin": profit_margin,
        "oper_margin": oper_margin,
        "gross_margin": gross_margin,
        "roe": roe,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "debt_to_equity": debt_to_equity,
        "pe": pe,
        "peg": peg,
        "ps": ps,
        "pb": pb,
        "rec_label": rec_label,
        "analysts": analysts,
        "rec_mean": rec_mean,
        "beta": beta,
        "short_pct": short_pct,
        "market_cap": market_cap,
        "short_thesis": short_thesis,
        "chart_bias_info": chart_bias_info,
        "chart_structures_analysis": chart_structures_analysis,
        "emp": emp,
        "conv": conv,
    }



def legacy_analyze_stock(
    ticker: str,
    horizon: str,
    depot: float,
    risk_pct: float,
    override: Any,
    buy_in_override: Any,
    smart_money_default: Any,
    strict_mode: Any,
):
    """Run the extracted v28.3 analysis pipeline with validated dependencies."""
    missing = missing_context()
    if missing:
        preview = ", ".join(missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise RuntimeError(
            "Analyse-Kontext ist unvollstaendig. Fehlende Abhaengigkeiten: "
            f"{preview}{suffix}"
        )
    return _legacy_analyze_stock_impl(
        ticker=ticker,
        horizon=horizon,
        depot=depot,
        risk_pct=risk_pct,
        override=override,
        buy_in_override=buy_in_override,
        smart_money_default=smart_money_default,
        strict_mode=strict_mode,
    )
