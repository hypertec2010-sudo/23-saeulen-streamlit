# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, date, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="23-Saeulen-Modell v5.2", page_icon="ðŸ“Š", layout="wide")

st.markdown("""
<style>
.metric-card{background:#1e2130;border-radius:10px;padding:16px 20px;margin:6px 0;border-left:4px solid #4CAF50;}
.metric-card.red{border-left-color:#f44336;}
.metric-card.yellow{border-left-color:#FFC107;}
.small-note{color:#9aa4b2;font-size:0.88rem;}
</style>
""", unsafe_allow_html=True)


def ampel(v, g=65, y=45):
    return "ðŸŸ¢" if v >= g else ("ðŸŸ¡" if v >= y else "ðŸ”´")


def ampel_crv(c):
    return "ðŸŸ¢" if c >= 2.5 else ("ðŸŸ¡" if c >= 1.5 else "ðŸ”´")


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
    return pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)


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


@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="2y", auto_adjust=True)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    return hist, info


with st.sidebar:
    st.title("ðŸ“Š 23-Saeulen-Modell")
    st.caption("v5.2 | Core-Saeulen + Technik-Overlay")
    st.divider()
    ticker = st.text_input("Ticker", value="AAPL", placeholder="AAPL, AMAT, AIXA.DE").upper().strip()
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
    strict_mode = st.checkbox("Strenges 23-Saeulen-Mapping", value=True)
    go = st.button("Analyse starten", use_container_width=True, type="primary")

st.title("ðŸ“Š 23-Saeulen-Modell v5.2")
st.caption("Technik + fundamentale Proxy-Saeulen + Safeguards. Bestehende Saeulen bleiben unveraendert; zusaetzlich gibt es ein Technik-Overlay fuer Trading-Board-naehere Signale.")

if not go:
    st.info("Ticker eingeben und Analyse starten klicken.")
    st.stop()

with st.spinner(f"Lade {ticker}..."):
    try:
        df, info = load_data(ticker)
    except Exception as e:
        st.error(str(e))
        st.stop()

if df.empty or len(df) < 120:
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
stoch_k_v = safe_last(stoch_k)
stoch_d_v = safe_last(stoch_d)
willr_v = safe_last(williams_r(high, low, close))
ma50_rising = ma50 > safe_last(close.rolling(50).mean().shift(10), ma50)
ma200_rising = ma200 > safe_last(close.rolling(200).mean().shift(20), ma200)
golden_cross = ma50 > ma200
smart_money_ok = obv_trend == "steigend" and vol_ratio >= 0.95
momentum_rising = macd_up and roc20 > 0

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
sg_earn = "ðŸŸ¢" if days_earn > 30 else ("ðŸŸ¡" if days_earn > 7 else "ðŸ”´")
sg_earn_txt = f"Earnings in ~{int(days_earn)}d" if days_earn < 999 else "kein Datum"

if price > ma50 > ma150 > ma200:
    regime, reg_amp = "UPTREND", "ðŸŸ¢"
elif price < ma50 < ma150 < ma200:
    regime, reg_amp = "DOWNTREND", "ðŸ”´"
else:
    regime, reg_amp = "SIDEWAYS", "ðŸŸ¡"

# --- Technische Proxy-Saeulen ---
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
    s5, s5a, s5t = 100, "ðŸŸ¢", f"Vol {vol_ratio:.2f}x | OBV steigend"
elif ret20 > 0 and obv_trend == "steigend":
    s5, s5a, s5t = 68, "ðŸŸ¡", f"Vol {vol_ratio:.2f}x | Nachfrage ok"
elif ret20 > 0:
    s5, s5a, s5t = 52, "ðŸŸ¡", f"Momentum ok | OBV {obv_trend}"
else:
    s5, s5a, s5t = 28, "ðŸ”´", f"Momentum/Volumen schwach | OBV {obv_trend}"

if atr_pct < 2.8:
    s6, s6a, s6t = 92, "ðŸŸ¢", f"ATR {atr_pct:.1f}% niedrig"
elif atr_pct < 5.5:
    s6, s6a, s6t = 66, "ðŸŸ¡", f"ATR {atr_pct:.1f}% normal"
elif atr_pct < 8.0:
    s6, s6a, s6t = 44, "ðŸŸ¡", f"ATR {atr_pct:.1f}% erhoeht"
else:
    s6, s6a, s6t = 20, "ðŸ”´", f"ATR {atr_pct:.1f}% hoch"

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

# --- Technisches Overlay (ohne Aenderung bestehender Saeulen) ---
tb_ma200 = 100 if price > ma200 and ma200_rising else (78 if price > ma200 else 25)
tb_ma50 = 100 if price > ma50 else 30
tb_cross = 100 if golden_cross else 25
tb_rsi_reversal = 100 if 28 <= rsi <= 40 else (72 if 40 < rsi <= 52 else (52 if 52 < rsi <= 68 else 25))
tb_momentum = 100 if momentum_rising else (68 if macd_up or roc20 > 0 else 25)
tb_smart_money = 100 if smart_money_ok else (65 if obv_trend == "steigend" else 25)
tb_adx = 100 if adx > 25 else (65 if adx > 18 else 25)
tb_stoch = 100 if pd.notna(stoch_k_v) and pd.notna(stoch_d_v) and stoch_k_v < 35 and stoch_k_v > stoch_d_v else (55 if pd.notna(stoch_k_v) and 20 <= stoch_k_v <= 80 else 25)
tb_williams = 100 if pd.notna(willr_v) and willr_v <= -80 else (55 if pd.notna(willr_v) and -80 < willr_v < -20 else 25)

overlay_items = [
    ("TB1 Ueber MA200", tb_ma200, f"Kurs {fmt_num(price,2)} vs MA200 {fmt_num(ma200,2)}"),
    ("TB2 Ueber MA50", tb_ma50, f"Kurs {fmt_num(price,2)} vs MA50 {fmt_num(ma50,2)}"),
    ("TB3 Golden Cross", tb_cross, f"MA50 {fmt_num(ma50,2)} vs MA200 {fmt_num(ma200,2)}"),
    ("TB4 RSI-Reversal", tb_rsi_reversal, f"RSI {fmt_num(rsi,1)}"),
    ("TB7 Momentum steigt", tb_momentum, f"MACD {'up' if macd_up else 'dn'} | ROC20 {fmt_num(roc20,1,'%')}"),
    ("TB8 Smart Money", tb_smart_money, f"OBV {obv_trend} | Vol {fmt_num(vol_ratio,2)}x"),
    ("TB9 ADX-Trend", tb_adx, f"ADX {fmt_num(adx,1)}"),
    ("TB10 Stochastik", tb_stoch, f"%K {fmt_num(stoch_k_v,1)} | %D {fmt_num(stoch_d_v,1)}"),
    ("TB11 Williams %R", tb_williams, f"%R {fmt_num(willr_v,1)}"),
]
tech_overlay_score = round(np.mean([x[1] for x in overlay_items]))
tech_overlay_delta = int(round((tech_overlay_score - 50) * 0.24))
setup_plus = int(clamp(setup + tech_overlay_delta))
if hd < 30:
    investment_plus = int(clamp(investment + round(tech_overlay_delta * 0.85)))
elif hd < 120:
    investment_plus = int(clamp(investment + round(tech_overlay_delta * 0.60)))
else:
    investment_plus = int(clamp(investment + round(tech_overlay_delta * 0.35)))

# --- Fundamentale Proxy-Saeulen ---
rec = info.get("recommendationKey", "hold")
rec_mean = info.get("recommendationMean", np.nan)
analysts = info.get("numberOfAnalystOpinions", np.nan)
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

fundamental_fields = [
    profit_margin, oper_margin, gross_margin, roe, roa,
    revenue_growth, earnings_growth, current_ratio, quick_ratio,
    debt_to_equity, fcf, op_cf, pe, peg, ps, pb,
    beta, short_pct, rec_mean, analysts, target
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

# --- Stark getrennte Zeithorizonte ---
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

if days_earn < 7:
    emp_plus, conv_plus = "VETO - Earnings < 7 Tage", "-"
elif investment_plus >= 78 and (kb >= 3 or tech_overlay_score >= 72):
    emp_plus, conv_plus = "BUY / ACCUMULATE", "HIGH"
elif investment_plus >= 68:
    emp_plus, conv_plus = "WATCH / kleine Position", "MEDIUM"
elif investment_plus >= 52:
    emp_plus, conv_plus = "HOLD / beobachten", "LOW-MEDIUM"
else:
    emp_plus, conv_plus = "AVOID / WAIT", "LOW"

st.markdown(f"## {name} `{ticker}` â€” {exch} ({ccy})")
st.markdown(f"<div class='small-note'>Sektor: {sector} | Industrie: {industry}</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kurs (Adj. Close)", f"{price:.2f} {ccy}", ts)
c2.metric("Trend-Regime", regime, reg_amp)
c3.metric("Earnings", sg_earn_txt, sg_earn)
c4.metric("Analysten-Target", fmt_num(target, 2, f" {ccy}"), fmt_num(upside, 1, "%"))
st.divider()

st.subheader("Scores")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Company Quality", f"{company}/100", ampel(company))
c2.metric("Setup Quality", f"{setup}/100", ampel(setup))
c3.metric("Tech Overlay", f"{tech_overlay_score}/100", f"Delta {tech_overlay_delta:+d}")
c4.metric("Setup Plus", f"{setup_plus}/100", ampel(setup_plus))
c5.metric("Investment Score", f"{investment}/100", ampel(investment))
c6.metric("Investment Plus", f"{investment_plus}/100", ampel(investment_plus))
st.caption(f"Konfluenz Kernmodell: {kb}/4 | Overlay nutzt zusaetzliche Trading-Board-nahe Techniksignale, ohne die bestehenden Saeulen S3-S6 zu aendern.")
st.divider()

t1, t2, t3, t4, t5 = st.tabs(["Technik", "Technik-Overlay", "Fundamental", "Safeguards", "Trade-Setup"])
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
                f'<div class="metric-card {card_class(score)}"><b>{ico} {lab}</b><span style="float:right;font-size:1.3rem;font-weight:700">{score}</span><br><small style="color:#aaa">{com}</small></div>',
                unsafe_allow_html=True,
            )
    st.dataframe(
        pd.DataFrame(
            {
                "Indikator": ["Kurs", "MA20", "MA50", "MA150", "MA200", "RSI(14)", "MACD", "Signal", "ADX", "ATR", "ATR%", "ROC20", "ROC60", "52W-Hoch", "52W-Tief", "Dist 52W"],
                "Wert": [f"{price:.2f}", f"{ma20:.2f}", f"{ma50:.2f}", f"{ma150:.2f}", f"{ma200:.2f}", f"{rsi:.1f}", f"{macd_v:.3f}", f"{signal_v:.3f}", f"{adx:.1f}", f"{atr:.3f}", f"{atr_pct:.1f}%", f"{roc20:.1f}%", f"{roc60:.1f}%", f"{high52:.2f}", f"{low52:.2f}", f"{dist52:.1f}%"],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
with t2:
    cols = st.columns(3)
    for i, (lab, score, com) in enumerate(overlay_items):
        with cols[i % 3]:
            st.markdown(
                f'<div class="metric-card {card_class(score)}"><b>{ampel(score)} {lab}</b><span style="float:right;font-size:1.3rem;font-weight:700">{score}</span><br><small style="color:#aaa">{com}</small></div>',
                unsafe_allow_html=True,
            )
    st.dataframe(
        pd.DataFrame(
            {
                "Overlay-Punkt": [x[0] for x in overlay_items],
                "Score": [x[1] for x in overlay_items],
                "Kommentar": [x[2] for x in overlay_items],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
with t3:
    st.markdown(f"<div class='small-note'>Datenabdeckung Fundamentaldaten: {fund_cov*100:.0f}%</div>", unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(
            {
                "Fundament-Block": ["Qualitaet", "Wachstum", "Bewertung", "Bilanz", "Sentiment", "Risiko"],
                "Score": [quality_score, growth_score, valuation_score, balance_score, sentiment_score, risk_score],
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
with t4:
    st.dataframe(
        pd.DataFrame(
            {
                "Safeguard": ["S0 Currency/Exchange", "S0 Preis-Typ-Lock", "S1 Earnings", "S2 Regime", "S3 Konfluenz-Cap", "S4 Datenabdeckung"],
                "Status": ["ðŸŸ¢", "ðŸŸ¢", sg_earn, reg_amp, "ðŸŸ¢" if kb >= 3 else ("ðŸŸ¡" if kb == 2 else "ðŸ”´"), "ðŸŸ¢" if fund_cov >= 0.55 else ("ðŸŸ¡" if fund_cov >= 0.35 else "ðŸ”´")],
                "Kommentar": [f"{ccy} | {exch}", "auto_adjust=True Yahoo Finance", sg_earn_txt, regime, f"{kb}/4 Kernbloecke", f"Fundamental-Coverage {fund_cov*100:.0f}%"],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
with t5:
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
        f"<div style='text-align:center'><div style='font-size:2rem'>{ampel(scv)}</div><small>{lab}<br><b>{scv}/100</b></small></div>",
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("Handlungsempfehlung")
c1, c2, c3 = st.columns(3)
c1.metric("Empfehlung Core", emp)
c2.metric("Empfehlung Plus", emp_plus, conv_plus)
c3.metric("Zeithorizont", horizon.split("(")[0].strip())

st.caption("Hinweis: Diese App ist eine datengetriebene Naeherung des 23-Saeulen-Modells. Einzelne KI-Urteile koennen abweichen, wenn qualitative Punkte wie Managementqualitaet, Guidance-Ton, Produktzyklus, regulatorische Risiken oder Makro-/Sektor-Story relevant sind.")
