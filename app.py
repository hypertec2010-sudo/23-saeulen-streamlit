# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, date, timedelta
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="23-Saeulen-Modell v5.6", page_icon="📊", layout="wide")

st.markdown("""
<style>
.metric-card{
    background:#1e2130;
    border-radius:10px;
    padding:16px 20px;
    margin:6px 0;
    border-left:4px solid #4CAF50;
}
.metric-card.red{border-left-color:#f44336;}
.metric-card.yellow{border-left-color:#FFC107;}
.small-note{color:#9aa4b2;font-size:0.88rem;}
pre{white-space:pre-wrap !important;}
</style>
""", unsafe_allow_html=True)


def ampel(v, g=65, y=45):
    return "🟢" if v >= g else ("🟡" if v >= y else "🔴")


def ampel_tb(score):
    if score >= 9:
        return "🟢"
    if score >= 5:
        return "🟡"
    if score >= 3:
        return "🟠"
    return "🔴"


def ampel_crv(c):
    return "🟢" if c >= 2.5 else ("🟡" if c >= 1.5 else "🔴")


def card_class(score):
    return "" if score >= 70 else ("yellow" if score >= 45 else "red")


def safe_last(s, default=np.nan):
    try:
        v = s.iloc[-1]
        return default if pd.isna(v) else float(v)
    except Exception:
        return default


def clamp(v, lo=0, hi=100):
    return max(lo, min(hi, float(v)))


def fmt_num(x, digits=2, suffix=""):
    return f"{x:.{digits}f}{suffix}" if pd.notna(x) else "n/a"


def known_ratio(values):
    vals = [v for v in values if pd.notna(v)]
    return len(vals) / len(values) if values else 0


def rsi14(close):
    d = close.diff()
    g = d.where(d > 0, 0.0).rolling(14).mean()
    l = (-d.where(d < 0, 0.0)).rolling(14).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx14(h, l, c):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up = h.diff()
    dn = -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0).rolling(14).mean()
    ndm = dn.where((dn > up) & (dn > 0), 0.0).rolling(14).mean()
    pdi = 100 * pdm / atr.replace(0, np.nan)
    ndi = 100 * ndm / atr.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.rolling(14).mean()


def true_range(high, low, close):
    return pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1
    ).max(axis=1)


def stoch14(high, low, close, k_period=14, d_period=3):
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def williams_r(high, low, close, period=14):
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def bollinger_bands(close, period=20, num_std=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan) * 100
    return mid, upper, lower, width


def tb_signal_label(score):
    if score >= 9:
        return "LONG", "AKTIV HALTEN"
    if score >= 5:
        return "HOLD", "HALTEN"
    if score >= 3:
        return "WAIT", "ABWARTEN"
    return "SHORT", "STOPP PRÜFEN"


@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="3y", auto_adjust=True)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    return hist, info


with st.sidebar:
    st.title("📊 23-Saeulen-Modell")
    st.caption("v5.6 | Core + TradingBoard + Kurzfrist Hilfsboard")
    st.divider()

    ticker = st.text_input(
        "Ticker",
        value="AAPL",
        placeholder="AAPL, AMAT, BNP.PA, AIXA.DE"
    ).upper().strip()

    horizon = st.selectbox(
        "Zeithorizont",
        [
            "Kurzfrist (1-7 Tage)",
            "Swing (1-4 Wochen)",
            "Mittelfrist (1-3 Monate)",
            "Langfrist (1-2 Jahre)",
            "Sehr langfristig (2+ Jahre)",
        ],
    )

    st.divider()
    depot = st.number_input("Depotwert EUR", min_value=1000, value=10000, step=1000)
    risk_pct = st.slider("Risiko pro Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    st.divider()
    override = st.number_input("Kurs-Override (0 = auto)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    buy_in_override = st.number_input("Buy-in fuer TradingBoard (0 = Watchlist)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    smart_money_default = st.checkbox("TradingBoard: Smart Money = True", value=True)
    strict_mode = st.checkbox("Strenges 23-Saeulen-Mapping", value=True)

    st.divider()
    go = st.button("Analyse starten", use_container_width=True, type="primary")

st.title("📊 23-Saeulen-Modell v5.6")
st.caption(
    "Core-Modell und TradingBoard werden getrennt gerechnet. "
    "Die Core-Saeulen bleiben unveraendert; zusaetzlich gibt es jetzt eine separate Kurzfrist-Hilfsboard-Ampel und das TradingBoard zeigt die additive Logik mit S0-S20-Zeilen nahezu 1:1 an."
)

if not go:
    st.info("Ticker eingeben und Analyse starten klicken.")
    st.stop()

with st.spinner(f"Lade {ticker}..."):
    try:
        df, info = load_data(ticker)
    except Exception as e:
        st.error(str(e))
        st.stop()

if df.empty or len(df) < 220:
    st.error("Nicht genug Kursdaten fuer belastbare Analyse.")
    st.stop()

close = df["Close"]
high = df["High"]
low = df["Low"]
vol = df["Volume"]

price = float(override) if override > 0 else float(close.iloc[-1])
name = info.get("longName", ticker)
ccy = info.get("currency", "USD")
exch = info.get("exchange", "-")
ts = df.index[-1].strftime("%d.%m.%Y")
sector = info.get("sector", "-")
industry = info.get("industry", "-")

ma20 = safe_last(close.rolling(20).mean())
ma50 = safe_last(close.rolling(50).mean())
ma150 = safe_last(close.rolling(150).mean())
ma200 = safe_last(close.rolling(200).mean())

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

roc20 = safe_last(close.pct_change(20) * 100)
roc60 = safe_last(close.pct_change(60) * 100)
ret5 = safe_last(close.pct_change(5) * 100, 0)
ret20 = safe_last(close.pct_change(20) * 100, 0)
ret63 = safe_last(close.pct_change(63) * 100, 0)
ret126 = safe_last(close.pct_change(126) * 100, 0)

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

earnings_ts = info.get("earningsTimestamp")
days_earn = (earnings_ts - datetime.now(timezone.utc).timestamp()) / 86400 if earnings_ts else 999
sg_earn = "🟢" if days_earn > 30 else ("🟡" if days_earn > 7 else "🔴")
sg_earn_txt = f"Earnings in ~{int(days_earn)}d" if days_earn < 999 else "kein Datum"
earnings_warning = days_earn <= 7

if price > ma50 > ma150 > ma200:
    regime, reg_amp = "UPTREND", "🟢"
elif price < ma50 < ma150 < ma200:
    regime, reg_amp = "DOWNTREND", "🔴"
else:
    regime, reg_amp = "SIDEWAYS", "🟡"

# Core unveraendert
s3 = 100 if price > ma20 > ma50 > ma150 else (15 if price < ma20 < ma50 < ma150 else 52)
s3a = ampel(s3)
s3t = "Trend-Stack sauber" if s3 >= 80 else ("Trend gemischt" if s3 >= 45 else "Trend schwach")

rsi_s = 100 if 55 <= rsi <= 72 else (70 if 48 <= rsi < 55 or 72 < rsi <= 78 else 25)
macd_s = 100 if (macd_v > 0 and macd_up) else (68 if macd_up else 22)
adx_s = 100 if adx > 25 else (65 if adx > 18 else 30)
roc_s = 100 if roc20 > 4 else (72 if roc20 > 0 else (45 if roc20 > -4 else 20))
s4 = round(rsi_s * 0.25 + macd_s * 0.30 + adx_s * 0.20 + roc_s * 0.25)
s4a = ampel(s4)
s4t = f"RSI {rsi:.1f} | MACD {'up' if macd_up else 'dn'} | ADX {adx:.1f} | ROC20 {roc20:.1f}%"

if ret5 > 0 and vol_ratio > 1.12 and obv_trend == "steigend":
    s5, s5a, s5t = 100, "🟢", f"Vol {vol_ratio:.2f}x | OBV steigend"
elif ret20 > 0 and obv_trend == "steigend":
    s5, s5a, s5t = 68, "🟡", f"Vol {vol_ratio:.2f}x | Nachfrage ok"
elif ret20 > 0:
    s5, s5a, s5t = 52, "🟡", f"Momentum ok | OBV {obv_trend}"
else:
    s5, s5a, s5t = 28, "🔴", f"Momentum/Volumen schwach | OBV {obv_trend}"

if atr_pct < 2.8:
    s6, s6a, s6t = 92, "🟢", f"ATR {atr_pct:.1f}% niedrig"
elif atr_pct < 5.5:
    s6, s6a, s6t = 66, "🟡", f"ATR {atr_pct:.1f}% normal"
elif atr_pct < 8.0:
    s6, s6a, s6t = 44, "🟡", f"ATR {atr_pct:.1f}% erhoeht"
else:
    s6, s6a, s6t = 20, "🔴", f"ATR {atr_pct:.1f}% hoch"

w52 = 100 if 80 <= dist52 <= 98 else (72 if 70 <= dist52 < 80 else (55 if 98 < dist52 <= 101 else (35 if dist52 >= 55 else 15)))
rs_score = 100 if ret63 > 12 else (78 if ret63 > 4 else (55 if ret63 > -5 else 22))
kb = sum([s3 >= 65, s4 >= 65, s5 >= 65, s6 >= 65])

setup_raw = s3 * 0.24 + s4 * 0.24 + s5 * 0.18 + s6 * 0.12 + rs_score * 0.14 + w52 * 0.08
if strict_mode:
    if kb < 2:
        setup_raw = min(setup_raw, 44)
    elif kb == 2:
        setup_raw = min(setup_raw, 58)
setup = round(clamp(setup_raw))

fundamental_fields = [
    profit_margin, oper_margin, gross_margin, roe, roa,
    revenue_growth, earnings_growth, current_ratio, quick_ratio,
    debt_to_equity, fcf, op_cf, pe, peg, ps, pb,
    beta, short_pct, info.get("recommendationMean", np.nan),
    info.get("numberOfAnalystOpinions", np.nan), target
]
fund_cov = known_ratio(fundamental_fields)

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

valuation_parts = []
valuation_parts.append(86 if pd.notna(pe) and 0 < pe < 20 else (72 if pd.notna(pe) and pe < 28 else (58 if pd.notna(pe) and pe < 38 else 42)))
valuation_parts.append(84 if pd.notna(peg) and 0 < peg < 1.5 else (70 if pd.notna(peg) and peg < 2.2 else (55 if pd.notna(peg) and peg < 3.0 else 42)))
valuation_parts.append(82 if pd.notna(ps) and ps < 4 else (68 if pd.notna(ps) and ps < 8 else (55 if pd.notna(ps) and ps < 12 else 42)))
valuation_parts.append(82 if pd.notna(upside) and upside > 20 else (70 if pd.notna(upside) and upside > 10 else (55 if pd.notna(upside) and upside > 0 else 40)))
valuation_score = round(np.mean(valuation_parts))

balance_parts = []
balance_parts.append(88 if pd.notna(current_ratio) and current_ratio >= 1.5 else (72 if pd.notna(current_ratio) and current_ratio >= 1.1 else 48))
balance_parts.append(88 if pd.notna(quick_ratio) and quick_ratio >= 1.0 else (70 if pd.notna(quick_ratio) and quick_ratio >= 0.8 else 48))
balance_parts.append(90 if pd.notna(debt_to_equity) and debt_to_equity < 60 else (72 if pd.notna(debt_to_equity) and debt_to_equity < 120 else 45))
balance_score = round(np.mean(balance_parts))

rec = info.get("recommendationKey", "hold")
rec_mean = info.get("recommendationMean", np.nan)
analysts = info.get("numberOfAnalystOpinions", np.nan)

sentiment_parts = []
sentiment_parts.append(88 if rec in ["strong_buy", "buy"] else (65 if rec in ["hold"] else 40))
sentiment_parts.append(84 if pd.notna(analysts) and analysts >= 20 else (72 if pd.notna(analysts) and analysts >= 10 else (58 if pd.notna(analysts) and analysts >= 5 else 48)))
sentiment_parts.append(84 if pd.notna(rec_mean) and rec_mean <= 2.0 else (68 if pd.notna(rec_mean) and rec_mean <= 2.5 else (55 if pd.notna(rec_mean) and rec_mean <= 3.0 else 42)))
sentiment_score = round(np.mean(sentiment_parts))

risk_parts = []
risk_parts.append(80 if pd.notna(beta) and beta < 1.2 else (62 if pd.notna(beta) and beta < 1.6 else 45))
risk_parts.append(78 if pd.notna(short_pct) and short_pct < 0.03 else (62 if pd.notna(short_pct) and short_pct < 0.07 else 45))
risk_parts.append(82 if atr_pct < 3.5 else (65 if atr_pct < 6 else 45))
risk_score = round(np.mean(risk_parts))

base_company = round(
    quality_score * 0.24
    + growth_score * 0.20
    + valuation_score * 0.18
    + balance_score * 0.14
    + sentiment_score * 0.14
    + risk_score * 0.10
)

coverage_penalty = 0
if fund_cov < 0.35:
    coverage_penalty = 12
elif fund_cov < 0.55:
    coverage_penalty = 6

base_company = max(35, round(base_company - coverage_penalty))
if hd < 30:
    company = round(base_company * 0.55 + 50 * 0.45)
else:
    company = base_company
company = int(clamp(company))
investment = round(clamp(setup * ws + company * wc))

atr_stop = round(price - 1.8 * atr, 2)
struct_stop = round(ma50 * 0.965, 2)
stop_used = min(atr_stop, struct_stop)
stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0
if stop_used <= 0 or stop_used >= price:
    stop_used = round(price - max(price * 0.08, atr * 1.8), 2)
    stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0
risk_per_share = price - stop_used
tp1 = round(price + 1 * risk_per_share, 2)
tp2 = round(price + 2 * risk_per_share, 2)
tp3 = round(price + 3 * risk_per_share, 2)
crv = (tp2 - price) / (price - stop_used) if (price - stop_used) > 0 else 0
risk_eur = depot * (risk_pct / 100)
pos_size = int(risk_eur / risk_per_share) if risk_per_share > 0 else 0
time_stop = (date.today() + timedelta(days=hd)).strftime("%d.%m.%Y")

short_term_score = round(clamp(s4 * 0.45 + s5 * 0.30 + s6 * 0.20 + rs_score * 0.05))
swing_score = round(clamp(s3 * 0.28 + s4 * 0.30 + s5 * 0.18 + s6 * 0.10 + rs_score * 0.10 + w52 * 0.04))
mid_term_score = round(clamp(setup * 0.55 + company * 0.45))
long_term_score = round(clamp(company * 0.55 + growth_score * 0.20 + quality_score * 0.15 + valuation_score * 0.10))
very_long_term_score = round(clamp(company * 0.45 + quality_score * 0.22 + growth_score * 0.18 + valuation_score * 0.15))

hmap = {
    "Kurzfrist": short_term_score,
    "Swing": swing_score,
    "Mittelfrist": mid_term_score,
    "Langfrist": long_term_score,
    "Sehr langfristig": very_long_term_score,
}

if days_earn < 7:
    emp, conv = "VETO - Earnings < 7 Tage", "-"
elif investment >= 78 and kb >= 3:
    emp, conv = "BUY / ACCUMULATE", "HIGH"
elif investment >= 68:
    emp, conv = "WATCH / kleine Position", "MEDIUM"
elif investment >= 52:
    emp, conv = "HOLD / beobachten", "LOW-MEDIUM"
else:
    emp, conv = "AVOID / WAIT", "LOW"

# TradingBoard 1:1 naeher am Original
tb_score = 0
tb_details = []

tb_buy = buy_in_override if buy_in_override > 0 else 0.0
tb_basispreis = tb_buy if tb_buy > 0 else price
tb_perf = ((price - tb_buy) / tb_buy) * 100 if tb_buy > 0 else 0.0
tb_stop = price - (2.5 * atr)
tb_tp1 = tb_basispreis + (2.5 * atr)
tb_tp2 = target if pd.notna(target) and target > tb_tp1 else tb_basispreis + (5.0 * atr)

tb_details.append(f"S0: {price:.2f} {ccy}")

if price > ma200:
    tb_score += 1
    tb_details.append("S1: Über MA200 ✓")
else:
    tb_details.append("S1: Unter MA200 ❌")

if price > ma50:
    tb_score += 2
    tb_details.append("S2: Über MA50 (+2) ✓")
else:
    tb_score -= 1
    tb_details.append("S2: Unter MA50 (-1) ❌")

if ma50 > ma200:
    tb_score += 1
    tb_details.append("S3: Golden Cross ✓")
else:
    tb_details.append("S3: Trendstruktur schwach ❌")

if 40 < rsi < 60:
    tb_score += 1
    tb_details.append("S4: RSI ok ✓")
elif rsi < 30:
    tb_score += 1
    tb_details.append("S4: RSI oversold ✓")
else:
    tb_details.append("S4: RSI hoch/niedrig ❌")

if tb_perf > 5:
    tb_score += 1
    tb_details.append(f"S5: +{tb_perf:.1f}% ✓")
else:
    tb_details.append(f"S5: {tb_perf:.1f}% ❌")

if 20 < rsi < 80:
    tb_score += 1
    tb_details.append("S6: Vola ok ✓")

if macd_hist_current > macd_hist_prev:
    tb_score += 1
    tb_details.append("S7: Momentum steigt ✓")
else:
    tb_details.append("S7: Momentum fällt ❌")

if macd_bull_cross:
    tb_score += 1
    tb_details.append("MACD Bull-Cross! 🚀")

if smart_money_default:
    tb_score += 1
    tb_details.append("S8: Smart Money sammelt ein ✓")
else:
    tb_details.append("S8: Smart Money verkauft ❌")

if adx > 25:
    tb_score += 1
    tb_details.append("S9: ADX>25 starker Trend ✓")
else:
    tb_details.append("S9: ADX<25 Seitwärts ❌")

if stoch_k_v < 20 and stoch_d_v < 20 and stoch_k_v > stoch_d_v:
    tb_score += 1
    tb_details.append("S10: Stoch Oversold Cross ✓")
elif stoch_k_v > 80:
    tb_details.append("S10: Stoch überkauft ❌")
else:
    tb_details.append("S10: Stoch neutral ❌")

if willr_v < -80:
    tb_score += 1
    tb_details.append("S11: Williams%R extrem Oversold ✓")
elif willr_v > -20:
    tb_details.append("S11: Williams%R überkauft ❌")
else:
    tb_details.append("S11: Williams%R neutral ❌")

if obv_trend == "steigend" and vol_ratio >= 1.0:
    tb_score += 1
    tb_details.append("S12: OBV/Volumen bestaetigt ✓")
else:
    tb_details.append("S12: OBV/Volumen schwach ❌")

if pd.notna(prev20_high) and price > prev20_high:
    tb_score += 1
    tb_details.append("S13: 20D Breakout ✓")
elif pd.notna(prev20_low) and price < prev20_low:
    tb_details.append("S13: 20D Breakdown ❌")
else:
    tb_details.append("S13: Range intakt ❌")

if pd.notna(bb_upper) and price > bb_upper:
    tb_score += 1
    tb_details.append("S14: BB Breakout UP ✓")
elif bb_squeeze:
    tb_score += 1
    tb_details.append("S14: BB Squeeze Achtung ✓")
elif pd.notna(bb_lower) and price < bb_lower:
    tb_details.append("S14: BB Breakout DOWN ❌")
else:
    tb_details.append("S14: BB neutral ❌")

if pd.notna(target) and target > 0 and price > 0:
    tb_potenzial = ((target - price) / price) * 100
    if tb_potenzial > 15:
        tb_score += 1
        tb_details.append(f"S15: Target +{tb_potenzial:.1f}% ✓")
    elif tb_potenzial < 0:
        tb_score -= 1
        tb_details.append(f"S15: Target -{abs(tb_potenzial):.1f}% ❌")

current_month = datetime.now().month
if current_month in [8, 9]:
    tb_score -= 1
    tb_details.append("S16: Seasonality schlecht (-1) ❌")
elif current_month in [11, 12, 1]:
    tb_score += 1
    tb_details.append("S16: Seasonality stark (+1) ✓")
else:
    tb_details.append("S16: Seasonality neutral ❌")

if crv >= 2.0:
    tb_score += 1
    tb_details.append("S17: CRV >= 2.0 ✓")
elif crv < 1.5:
    tb_details.append("S17: CRV schwach ❌")
else:
    tb_details.append("S17: CRV ok/neutral ❌")

if earnings_warning:
    tb_score -= 3
    tb_details.insert(0, "⚠️ EARNINGS IN <7 TAGEN (Vorsicht!)")

short_squeeze = pd.notna(short_pct) and short_pct > 0.12 and ret5 > 0 and vol_ratio > 1.2
if short_squeeze:
    tb_score += 1
    tb_details.append("S18: 🚀 SHORT SQUEEZE POTENZIAL ✓")
else:
    tb_details.append("S18: kein Short-Squeeze-Signal ❌")

insider_trend = "NEUTRAL"
if insider_trend == "BUY":
    tb_score += 1
    tb_details.append("S19: 👔 INSIDER KAUFEN ✓")
elif insider_trend == "SELL":
    tb_score -= 1
    tb_details.append("S19: 👔 INSIDER VERKAUFEN ❌")
else:
    tb_details.append("S19: 👔 Insider neutral / keine Daten ❌")

if pd.notna(pe) and 0 < pe < 15:
    tb_score += 1
    tb_details.append(f"S20: 🟢 VALUE KGV ({pe:.1f}) ✓")
elif pd.notna(pe) and pe > 50:
    tb_details.append(f"S20: 🔴 TEUER KGV>50 ({pe:.1f}) ❌")

tb_signal, tb_empf = tb_signal_label(tb_score)

# Kurzfrist-Hilfsboard nur aus kurzfristigen Trading-Signalen
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

tb_details_text = "\n".join(tb_details)

# strukturierte Ansicht
rows = []
for line in tb_details:
    if line.startswith("⚠️") or line.startswith("MACD "):
        rows.append({"Punkt": "Info", "Detail": line})
    else:
        k, v = line.split(": ", 1)
        rows.append({"Punkt": k, "Detail": v})
tb_df = pd.DataFrame(rows)

st.markdown(f"## {name} `{ticker}` — {exch} ({ccy})")
st.markdown(f"<div class='small-note'>Sektor: {sector} | Industrie: {industry}</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Kurs (Adj. Close)", f"{price:.2f} {ccy}", ts)
c2.metric("Trend-Regime", regime, reg_amp)
c3.metric("Earnings", sg_earn_txt, sg_earn)
c4.metric("Analysten-Target", fmt_num(target, 2, f" {ccy}"), fmt_num(upside, 1, "%"))

st.divider()

st.subheader("Scores")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Company Quality", f"{company}/100", ampel(company))
c2.metric("Setup Quality", f"{setup}/100", ampel(setup))
c3.metric("Kurzfrist Core", f"{short_term_score}/100", ampel(short_term_score))
c4.metric("Kurzfrist Hilfsboard", f"{stb_score} Pkt", stb_signal)
c5.metric("Investment Score", f"{investment}/100", ampel(investment))
c6.metric("TradingBoard Score", f"{tb_score} Punkte", ampel_tb(tb_score))
c7.metric("Konfluenz", f"{kb}/4", "Robust" if kb >= 3 else ("Fragil" if kb == 2 else "Schwach"))

st.caption(
    "Die App trennt jetzt drei Sichtweisen: Company/Core, Kurzfrist Core vs. Kurzfrist Hilfsboard und das volle additive TradingBoard. "
    "So sieht man sofort, ob ein Wert kurzfristig tradbar ist, obwohl das breite Board noch hinterherhaengt oder umgekehrt."
)

st.divider()

t1, t2, t3, t4, t5, t6 = st.tabs(["Technik", "Kurzfrist-Vergleich", "TradingBoard", "Fundamental", "Safeguards", "Trade-Setup"])

with t1:
    cols = st.columns(2)
    items = [
        ("S3 Trend", s3a, s3, s3t),
        ("S4 Momentum", s4a, s4, s4t),
        ("S5 Volumen", s5a, s5, s5t),
        ("S6 Volatilitaet", s6a, s6, s6t),
        ("52W-Lage", ampel(w52), w52, f"{dist52:.1f}% vom 52W-Hoch"),
        ("Rel. Staerke", ampel(rs_score), rs_score, f"3M: {ret63:.1f}%"),
    ]
    for i, (lab, ico, score, com) in enumerate(items):
        with cols[i % 2]:
            st.markdown(
                f'<div class="metric-card {card_class(score)}"><b>{ico} {lab}</b>'
                f'<span style="float:right;font-size:1.3rem;font-weight:700">{score}</span>'
                f'<br><small style="color:#aaa">{com}</small></div>',
                unsafe_allow_html=True,
            )

    st.dataframe(
        pd.DataFrame(
            {
                "Indikator": [
                    "Kurs", "MA20", "MA50", "MA150", "MA200",
                    "RSI(14)", "MACD", "Signal", "MACD-Hist", "ADX", "ATR", "ATR%",
                    "Stoch %K", "Stoch %D", "Williams %R", "ROC20", "ROC60",
                    "52W-Hoch", "52W-Tief", "Dist 52W"
                ],
                "Wert": [
                    f"{price:.2f}", f"{ma20:.2f}", f"{ma50:.2f}", f"{ma150:.2f}", f"{ma200:.2f}",
                    f"{rsi:.1f}", f"{macd_v:.3f}", f"{signal_v:.3f}", f"{macd_hist_current:.3f}", f"{adx:.1f}",
                    f"{atr:.3f}", f"{atr_pct:.1f}%", f"{stoch_k_v:.1f}", f"{stoch_d_v:.1f}", f"{willr_v:.1f}",
                    f"{roc20:.1f}%", f"{roc60:.1f}%", f"{high52:.2f}", f"{low52:.2f}", f"{dist52:.1f}%"
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with t2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kurzfrist Core", f"{short_term_score}/100", ampel(short_term_score))
    c2.metric("Kurzfrist Hilfsboard", f"{stb_score} Punkte", stb_signal)
    c3.metric("Core Fokus", "Momentum/Volumen")
    c4.metric("Board Fokus", "Diskrete Checks")

    st.dataframe(
        pd.DataFrame(
            {
                "Kennzahl": ["Kurzfrist Core", "Kurzfrist Hilfsboard", "Board-Signal", "Board-Treiber"],
                "Wert": [f"{short_term_score}/100", str(stb_score), stb_signal, stb_text],
                "Kommentar": [
                    "S4 Momentum 45%, S5 Volumen 30%, S6 Volatilitaet 20%, RS 5%",
                    "Additive Kurzfrist-Punkte aus MA/RSI/Momentum/ADX/Stoch/Williams",
                    stb_empf,
                    stb_text
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with t3:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TradingBoard Score", str(tb_score), tb_signal)
    c2.metric("Signal", tb_signal, tb_empf)
    c3.metric("TB Stop", f"{tb_stop:.2f} {ccy}")
    c4.metric("TB TP2", f"{tb_tp2:.2f} {ccy}")

    st.dataframe(tb_df, hide_index=True, use_container_width=True)

    st.markdown("**TradingBoard Details (1:1 Stil)**")
    st.text(tb_details_text)

with t4:
    st.markdown(
        f"<div class='small-note'>Datenabdeckung Fundamentaldaten: {fund_cov*100:.0f}%</div>",
        unsafe_allow_html=True
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Fundament-Block": [
                    "Qualitaet", "Wachstum", "Bewertung",
                    "Bilanz", "Sentiment", "Risiko"
                ],
                "Score": [
                    quality_score, growth_score, valuation_score,
                    balance_score, sentiment_score, risk_score
                ],
                "Kommentar": [
                    f"PM {fmt_num(profit_margin*100 if pd.notna(profit_margin) else np.nan,1,'%')} | OM {fmt_num(oper_margin*100 if pd.notna(oper_margin) else np.nan,1,'%')} | ROE {fmt_num(roe*100 if pd.notna(roe) else np.nan,1,'%')} | FCF {'ok' if pd.notna(fcf) and fcf > 0 else ('neg' if pd.notna(fcf) else 'n/a')}",
                    f"Rev {fmt_num(revenue_growth*100 if pd.notna(revenue_growth) else np.nan,1,'%')} | EPS {fmt_num(earnings_growth*100 if pd.notna(earnings_growth) else np.nan,1,'%')} | 6M {fmt_num(ret126,1,'%')}",
                    f"PE {fmt_num(pe,1)} | PEG {fmt_num(peg,2)} | P/S {fmt_num(ps,2)} | Upside {fmt_num(upside,1,'%')}",
                    f"CR {fmt_num(current_ratio,2)} | QR {fmt_num(quick_ratio,2)} | D/E {fmt_num(debt_to_equity,1)}",
                    f"{rec} | Analysten {fmt_num(analysts,0)} | RecMean {fmt_num(rec_mean,2)}",
                    f"Beta {fmt_num(beta,2)} | Short {fmt_num(short_pct*100 if pd.notna(short_pct) else np.nan,1,'%')} | ATR {fmt_num(atr_pct,1,'%')}",
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with t5:
    st.dataframe(
        pd.DataFrame(
            {
                "Safeguard": [
                    "S0 Currency/Exchange",
                    "S0 Preis-Typ-Lock",
                    "S1 Earnings",
                    "S2 Regime",
                    "S3 Konfluenz-Cap",
                    "S4 Datenabdeckung"
                ],
                "Status": [
                    "🟢",
                    "🟢",
                    sg_earn,
                    reg_amp,
                    "🟢" if kb >= 3 else ("🟡" if kb == 2 else "🔴"),
                    "🟢" if fund_cov >= 0.55 else ("🟡" if fund_cov >= 0.35 else "🔴"),
                ],
                "Kommentar": [
                    f"{ccy} | {exch}",
                    "auto_adjust=True Yahoo Finance",
                    sg_earn_txt,
                    regime,
                    f"{kb}/4 Kernbloecke",
                    f"Fundamental-Coverage {fund_cov*100:.0f}%"
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with t6:
    c1, c2, c3 = st.columns(3)
    c1.metric("Entry", f"{price:.2f} {ccy}")
    c2.metric("ATR-Stop", f"{atr_stop:.2f} {ccy}", f"-{(price-atr_stop)/price*100:.1f}%" if atr_stop < price else "-")
    c3.metric("Aktiver Stop", f"{stop_used:.2f} {ccy}", f"-{stop_dist:.1f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("TP1 (1R)", f"{tp1:.2f} {ccy}", f"+{(tp1/price-1)*100:.1f}%")
    c5.metric("TP2 (2R)", f"{tp2:.2f} {ccy}", f"+{(tp2/price-1)*100:.1f}%")
    c6.metric("TP3 (3R)", f"{tp3:.2f} {ccy}", f"+{(tp3/price-1)*100:.1f}%")

    c7, c8, c9 = st.columns(3)
    c7.metric(f"CRV {ampel_crv(crv)}", f"{crv:.1f}:1")
    c8.metric("Positionsgroesse", f"{pos_size} Stueck", f"Risiko {risk_eur:.0f} EUR ({risk_pct}%)")
    c9.metric("Zeit-Stop", time_stop, "wenn kein Follow-through")

st.divider()
st.subheader("5 Zeithorizont-Ampeln")
cols = st.columns(5)
for col, (lab, scv) in zip(cols, hmap.items()):
    col.markdown(
        f"<div style='text-align:center'><div style='font-size:2rem'>{ampel(scv)}</div>"
        f"<small>{lab}<br><b>{scv}/100</b></small></div>",
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("Handlungsempfehlung")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Core Empfehlung", emp)
c2.metric("Core Conviction", conv)
c3.metric("Kurzfrist Core", f"{short_term_score}/100")
c4.metric("Kurzfrist Hilfsboard", stb_signal, str(stb_score))
c5.metric("TradingBoard Urteil", tb_empf)

st.caption(
    "Die App zeigt bewusst drei getrennte Sichtweisen: das strenge 23-Saeulen-Core-Modell, "
    "die kurzfristige Hilfsboard-Ampel und das additive TradingBoard mit den originalnahen S-Zeilen."
)
