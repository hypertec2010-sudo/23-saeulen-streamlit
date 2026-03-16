# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, date, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="23-Saeulen-Modell v5.0", page_icon="📊", layout="wide")

st.markdown("""
<style>
.metric-card{background:#1e2130;border-radius:10px;padding:16px 20px;margin:6px 0;border-left:4px solid #4CAF50;}
.metric-card.red{border-left-color:#f44336;}
.metric-card.yellow{border-left-color:#FFC107;}
.small-note{color:#9aa4b2;font-size:0.88rem;}
</style>
""", unsafe_allow_html=True)


def ampel(v, g=65, y=45):
    return "🟢" if v >= g else ("🟡" if v >= y else "🔴")


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


def score_from_bands(value, bands):
    for threshold, score in bands:
        if value >= threshold:
            return score
    return bands[-1][1]


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
    st.title("📊 23-Saeulen-Modell")
    st.caption("v5.0 | Naeher am klassischen KI-Urteil")
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

st.title("📊 23-Saeulen-Modell v5.0")
st.caption("Technik + fundamentale Proxy-Saeulen + Safeguards. Fokus: aehnlichere Urteile wie das klassische KI-Modell, aber mit Yahoo-Datenbeschraenkung.")

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

if "1-7" in horizon:
    hd, ws, wc = 7, 0.80, 0.20
elif "1-4" in horizon:
    hd, ws, wc = 21, 0.68, 0.32
elif "1-3" in horizon:
    hd, ws, wc = 60, 0.52, 0.48
elif "1-2" in horizon:
    hd, ws, wc = 365, 0.32, 0.68
else:
    hd, ws, wc = 730, 0.20, 0.80

earnings_ts = info.get("earningsTimestamp")
days_earn = (earnings_ts - datetime.now(timezone.utc).timestamp()) / 86400 if earnings_ts else 999
sg_earn = "🟢" if days_earn > 30 else ("🟡" if days_earn > 7 else "🔴")
sg_earn_txt = f"Earnings in ~{int(days_earn)}d" if days_earn < 999 else "kein Datum"

if price > ma50 > ma150 > ma200:
    regime, reg_amp = "UPTREND", "🟢"
elif price < ma50 < ma150 < ma200:
    regime, reg_amp = "DOWNTREND", "🔴"
else:
    regime, reg_amp = "SIDEWAYS", "🟡"

# --- Technische Saeulen ---
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

# --- Fundamentale Proxy-Saeulen ---
rec = info.get("recommendationKey", "hold")
rec_mean = info.get("recommendationMean")
analysts = info.get("numberOfAnalystOpinions", 0) or 0
target = info.get("targetMeanPrice", price) or price
upside = (target / price - 1) * 100 if price else 0
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

quality_parts = []
quality_parts.append(90 if pd.notna(profit_margin) and profit_margin > 0.20 else (75 if pd.notna(profit_margin) and profit_margin > 0.10 else (55 if pd.notna(profit_margin) and profit_margin > 0 else 40)))
quality_parts.append(90 if pd.notna(oper_margin) and oper_margin > 0.25 else (75 if pd.notna(oper_margin) and oper_margin > 0.15 else (55 if pd.notna(oper_margin) and oper_margin > 0.08 else 40)))
quality_parts.append(92 if pd.notna(roe) and roe > 0.25 else (78 if pd.notna(roe) and roe > 0.15 else (58 if pd.notna(roe) and roe > 0.08 else 42)))
quality_parts.append(85 if pd.notna(fcf) and fcf > 0 else 45)
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
valuation_parts.append(82 if upside > 20 else (70 if upside > 10 else (55 if upside > 0 else 40)))
valuation_score = round(np.mean(valuation_parts))

balance_parts = []
balance_parts.append(88 if pd.notna(current_ratio) and current_ratio >= 1.5 else (72 if pd.notna(current_ratio) and current_ratio >= 1.1 else 48))
balance_parts.append(88 if pd.notna(quick_ratio) and quick_ratio >= 1.0 else (70 if pd.notna(quick_ratio) and quick_ratio >= 0.8 else 48))
balance_parts.append(90 if pd.notna(debt_to_equity) and debt_to_equity < 60 else (72 if pd.notna(debt_to_equity) and debt_to_equity < 120 else 45))
balance_score = round(np.mean(balance_parts))

sentiment_parts = []
sentiment_parts.append(88 if rec in ["strong_buy", "buy"] else (65 if rec in ["hold"] else 40))
sentiment_parts.append(84 if analysts >= 20 else (72 if analysts >= 10 else (58 if analysts >= 5 else 48)))
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

short_term_score = round(clamp(s4 * 0.42 + s5 * 0.30 + s6 * 0.18 + rs_score * 0.10))
swing_score = round(clamp(s3 * 0.26 + s4 * 0.28 + s5 * 0.20 + s6 * 0.10 + rs_score * 0.10 + w52 * 0.06))
mid_term_score = investment
long_term_score = round(clamp(company * 0.65 + s3 * 0.15 + growth_score * 0.10 + valuation_score * 0.10))
very_long_term_score = round(clamp(company * 0.72 + growth_score * 0.12 + quality_score * 0.10 + valuation_score * 0.06))

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

st.markdown(f"## {name} `{ticker}` — {exch} ({ccy})")
st.markdown(f"<div class='small-note'>Sektor: {sector} | Industrie: {industry}</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kurs (Adj. Close)", f"{price:.2f} {ccy}", ts)
c2.metric("Trend-Regime", regime, reg_amp)
c3.metric("Earnings", sg_earn_txt, sg_earn)
c4.metric("Analysten-Target", f"{target:.2f} {ccy}", f"{upside:.1f}% Upside")
st.divider()

st.subheader("Scores")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Company Quality", f"{company}/100", ampel(company))
c2.metric("Setup Quality", f"{setup}/100", ampel(setup))
c3.metric("Investment Score", f"{investment}/100", ampel(investment))
c4.metric("Konfluenz", f"{kb}/4", "Robust" if kb >= 3 else ("Fragil" if kb == 2 else "Schwach"))
st.divider()

t1, t2, t3, t4 = st.tabs(["Technik", "Fundamental", "Safeguards", "Trade-Setup"])
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
    st.dataframe(
        pd.DataFrame(
            {
                "Fundament-Block": ["Qualitaet", "Wachstum", "Bewertung", "Bilanz", "Sentiment", "Risiko"],
                "Score": [quality_score, growth_score, valuation_score, balance_score, sentiment_score, risk_score],
                "Kommentar": [
                    f"Margen/ROE/FCF",
                    f"Rev {0 if pd.isna(revenue_growth) else revenue_growth*100:.1f}% | EPS {0 if pd.isna(earnings_growth) else earnings_growth*100:.1f}%",
                    f"PE {0 if pd.isna(pe) else pe:.1f} | PEG {0 if pd.isna(peg) else peg:.2f} | Upside {upside:.1f}%",
                    f"CR {0 if pd.isna(current_ratio) else current_ratio:.2f} | QR {0 if pd.isna(quick_ratio) else quick_ratio:.2f} | D/E {0 if pd.isna(debt_to_equity) else debt_to_equity:.1f}",
                    f"{rec} | Analysten {analysts}",
                    f"Beta {0 if pd.isna(beta) else beta:.2f} | Short {0 if pd.isna(short_pct) else short_pct*100:.1f}%",
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
with t3:
    st.dataframe(
        pd.DataFrame(
            {
                "Safeguard": ["S0 Currency/Exchange", "S0 Preis-Typ-Lock", "S1 Earnings", "S2 Regime", "S3 Konfluenz-Cap", "S4 Datenabdeckung"],
                "Status": ["🟢", "🟢", sg_earn, reg_amp, "🟢" if kb >= 3 else ("🟡" if kb == 2 else "🔴"), "🟢" if len(df) >= 252 else "🟡"],
                "Kommentar": [f"{ccy} | {exch}", "auto_adjust=True Yahoo Finance", sg_earn_txt, regime, f"{kb}/4 Kernbloecke", f"{len(df)} Handelstage Historie"],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
with t4:
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
c1.metric("Empfehlung", emp)
c2.metric("Conviction", conv)
c3.metric("Zeithorizont", horizon.split("(")[0].strip())

st.caption("Hinweis: Diese App ist eine datengetriebene Naeherung des 23-Saeulen-Modells. Einzelne KI-Urteile koennen abweichen, wenn qualitative Punkte wie Managementqualitaet, Produktzyklus, Guidance-Ton oder Makro-/Sektor-Story relevant sind.")
