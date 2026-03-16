{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # -*- coding: utf-8 -*-\
import streamlit as st\
import yfinance as yf\
import pandas as pd\
import numpy as np\
from datetime import datetime, timezone, date, timedelta\
import warnings\
warnings.filterwarnings("ignore")\
\
st.set_page_config(page_title="23-Saeulen-Modell v4.1", page_icon="\uc0\u55357 \u56522 ", layout="wide")\
\
st.markdown("""\
<style>\
.metric-card\{background:#1e2130;border-radius:10px;padding:16px 20px;margin:6px 0;border-left:4px solid #4CAF50;\}\
.metric-card.red\{border-left-color:#f44336;\}\
.metric-card.yellow\{border-left-color:#FFC107;\}\
</style>\
""", unsafe_allow_html=True)\
\
def ampel(v, g=70, y=50):\
    return "\uc0\u55357 \u57314 " if v >= g else ("\u55357 \u57313 " if v >= y else "\u55357 \u56628 ")\
\
def ampel_crv(c):\
    return "\uc0\u55357 \u57314 " if c >= 2.5 else ("\u55357 \u57313 " if c >= 1.5 else "\u55357 \u56628 ")\
\
def card_class(score):\
    return "" if score >= 70 else ("yellow" if score >= 45 else "red")\
\
def safe(s, d=np.nan):\
    try:\
        v = s.iloc[-1]\
        return d if pd.isna(v) else float(v)\
    except Exception:\
        return d\
\
def rsi14(close):\
    d = close.diff()\
    g = d.where(d > 0, 0.0).rolling(14).mean()\
    l = (-d.where(d < 0, 0.0)).rolling(14).mean()\
    rs = g / l.replace(0, np.nan)\
    return 100 - (100 / (1 + rs))\
\
def adx14(h, l, c):\
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)\
    atr = tr.rolling(14).mean()\
    up = h.diff()\
    dn = -l.diff()\
    pdm = up.where((up > dn) & (up > 0), 0.0).rolling(14).mean()\
    ndm = dn.where((dn > up) & (dn > 0), 0.0).rolling(14).mean()\
    pdi = 100 * pdm / atr.replace(0, np.nan)\
    ndi = 100 * ndm / atr.replace(0, np.nan)\
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)\
    return dx.rolling(14).mean()\
\
@st.cache_data(ttl=300, show_spinner=False)\
def load_data(ticker):\
    t = yf.Ticker(ticker)\
    df = t.history(period="1y", auto_adjust=True)\
    try:\
        info = t.info or \{\}\
    except Exception:\
        info = \{\}\
    return df, info\
\
with st.sidebar:\
    st.title("\uc0\u55357 \u56522  23-Saeulen-Modell")\
    st.caption("v4.1 | Maerz 2026")\
    st.divider()\
    ticker = st.text_input("Ticker", value="AAPL", placeholder="AAPL, LNTH, AIXA.DE").upper().strip()\
    horizon = st.selectbox("Zeithorizont", [\
        "Kurzfrist (1-7 Tage)",\
        "Swing (1-4 Wochen)",\
        "Mittelfrist (1-3 Monate)",\
        "Langfrist (1-2 Jahre)",\
        "Sehr langfristig (2+ Jahre)"\
    ])\
    st.divider()\
    depot = st.number_input("Depotwert EUR", min_value=1000, value=10000, step=1000)\
    risk_pct = st.slider("Risiko pro Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)\
    st.divider()\
    override = st.number_input("Kurs-Override (0 = auto)", min_value=0.0, value=0.0, step=0.01, format="%.2f")\
    go = st.button("Analyse starten", use_container_width=True, type="primary")\
\
st.title("\uc0\u55357 \u56522  23-Saeulen-Modell v4.1")\
\
if not go:\
    st.info("Ticker eingeben und Analyse starten klicken.")\
    st.stop()\
\
with st.spinner(f"Lade \{ticker\}..."):\
    try:\
        df, info = load_data(ticker)\
    except Exception as e:\
        st.error(str(e))\
        st.stop()\
\
if df.empty or len(df) < 50:\
    st.error("Nicht genug Daten fuer diesen Ticker.")\
    st.stop()\
\
close = df["Close"]\
high = df["High"]\
low = df["Low"]\
vol = df["Volume"]\
\
price = float(override) if override > 0 else float(close.iloc[-1])\
name = info.get("longName", ticker)\
ccy = info.get("currency", "USD")\
exch = info.get("exchange", "-")\
ts = df.index[-1].strftime("%d.%m.%Y")\
\
ma20 = safe(close.rolling(20).mean())\
ma50 = safe(close.rolling(50).mean())\
ma200 = safe(close.rolling(200).mean())\
rsi = safe(rsi14(close))\
ema12 = close.ewm(span=12, adjust=False).mean()\
ema26 = close.ewm(span=26, adjust=False).mean()\
macd = ema12 - ema26\
signal = macd.ewm(span=9, adjust=False).mean()\
macd_v = safe(macd)\
signal_v = safe(signal)\
macd_up = macd_v > signal_v\
atr = safe(pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1).rolling(14).mean())\
atr_pct = atr / price * 100 if price else 0\
adx = safe(adx14(high, low, close))\
roc = safe(close.pct_change(12) * 100)\
vol20 = safe(vol.rolling(20).mean(), 1)\
vol5 = safe(vol.rolling(5).mean(), 1)\
ret5 = float(close.pct_change(5).iloc[-1] * 100) if len(close) >= 6 else 0\
high52 = safe(close.rolling(252).max(), float(close.max()))\
low52 = safe(close.rolling(252).min(), float(close.min()))\
dist52 = price / high52 * 100 if high52 else 50\
ret3m = (price / float(close.iloc[-63]) - 1) * 100 if len(close) >= 63 else 0\
obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()\
obv_trend = "steigend" if float(obv.iloc[-1]) > float(obv.iloc[-20]) else "fallend"\
\
if "1-7" in horizon:\
    hd, ws, wc = 7, 0.80, 0.20\
elif "1-4" in horizon:\
    hd, ws, wc = 21, 0.70, 0.30\
elif "1-3" in horizon:\
    hd, ws, wc = 60, 0.50, 0.50\
elif "1-2" in horizon:\
    hd, ws, wc = 365, 0.30, 0.70\
else:\
    hd, ws, wc = 730, 0.20, 0.80\
\
earnings_ts = info.get("earningsTimestamp")\
days_earn = (earnings_ts - datetime.now(timezone.utc).timestamp()) / 86400 if earnings_ts else 999\
sg_earn = "\uc0\u55357 \u57314 " if days_earn > 30 else ("\u55357 \u57313 " if days_earn > 7 else "\u55357 \u56628 ")\
sg_earn_txt = f"Earnings in ~\{int(days_earn)\}d" if days_earn < 999 else "kein Datum"\
\
if price > ma50 > ma200:\
    regime, reg_amp = "UPTREND", "\uc0\u55357 \u57314 "\
elif price < ma50 < ma200:\
    regime, reg_amp = "DOWNTREND", "\uc0\u55357 \u56628 "\
else:\
    regime, reg_amp = "SIDEWAYS", "\uc0\u55357 \u57313 "\
\
if price > ma20 > ma50 > ma200:\
    s3, s3a, s3t = 100, "\uc0\u55357 \u57314 ", "Bull Stack"\
elif price < ma20 < ma50 < ma200:\
    s3, s3a, s3t = 10, "\uc0\u55357 \u56628 ", "Bear Stack"\
else:\
    s3, s3a, s3t = 50, "\uc0\u55357 \u57313 ", "Gemischt"\
\
rsi_s = 100 if rsi > 58 else (20 if rsi < 38 else 55)\
macd_s = 100 if (macd_v > 0 and macd_up) else (70 if macd_up else 20)\
adx_s = 100 if adx > 25 else (50 if adx > 20 else 20)\
roc_s = 100 if roc > 0 else 20\
s4 = round(rsi_s * 0.35 + macd_s * 0.30 + adx_s * 0.20 + roc_s * 0.15)\
s4a = ampel(s4)\
s4t = f"RSI \{rsi:.1f\} | MACD \{'up' if macd_up else 'dn'\} | ADX \{adx:.1f\} | ROC \{roc:.1f\}%"\
\
if ret5 > 0 and vol5 > vol20 * 1.15:\
    s5, s5a, s5t = 100, "\uc0\u55357 \u57314 ", f"Vol \{vol5/vol20:.1f\}x | OBV \{obv_trend\}"\
elif ret5 > 0:\
    s5, s5a, s5t = 65, "\uc0\u55357 \u57313 ", f"Vol normal | OBV \{obv_trend\}"\
elif vol5 > vol20 * 1.15:\
    s5, s5a, s5t = 20, "\uc0\u55357 \u56628 ", "Vol hoch bei Kursrueckgang"\
else:\
    s5, s5a, s5t = 40, "\uc0\u55357 \u57313 ", "Durchschnittliches Volumen"\
\
if atr_pct < 3:\
    s6, s6a, s6t = 100, "\uc0\u55357 \u57314 ", f"ATR \{atr_pct:.1f\}% niedrig"\
elif atr_pct < 7:\
    s6, s6a, s6t = 65, "\uc0\u55357 \u57313 ", f"ATR \{atr_pct:.1f\}% normal"\
else:\
    s6, s6a, s6t = 20, "\uc0\u55357 \u56628 ", f"ATR \{atr_pct:.1f\}% hoch"\
\
w52 = 100 if 80 <= dist52 <= 95 else (70 if 95 < dist52 <= 99 else (30 if dist52 > 99 else (40 if dist52 >= 60 else 10)))\
rs_score = 100 if ret3m > 10 else (70 if ret3m > 0 else (40 if ret3m > -10 else 10))\
kb = sum([s3 >= 60, s4 >= 60, s5 >= 60, s6 >= 60])\
setup_raw = s3*0.28 + s4*0.25 + s5*0.18 + s6*0.10 + rs_score*0.12 + w52*0.07\
if kb < 2:\
    setup_raw = min(setup_raw, 45)\
elif kb == 2:\
    setup_raw = min(setup_raw, 55)\
setup = round(setup_raw)\
\
rec = info.get("recommendationKey", "hold")\
pe = info.get("forwardPE", 99) or 99\
target = info.get("targetMeanPrice", price) or price\
upside = (target / price - 1) * 100 if price else 0\
if hd < 30:\
    company = 50\
else:\
    company = 45\
    if rec in ["buy", "strong_buy"]:\
        company += 18\
    if upside > 20:\
        company += 12\
    if pe < 25:\
        company += 10\
    elif pe < 35:\
        company += 5\
    company = min(95, company)\
\
investment = round(setup * ws + company * wc)\
\
atr_stop = round(price - 1.8 * atr, 2)\
struct_stop = round(ma50 * 0.965, 2)\
stop_used = min(atr_stop, struct_stop)\
stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0\
tp1 = round(price + 1 * (price - stop_used), 2)\
tp2 = round(price + 2 * (price - stop_used), 2)\
tp3 = round(price + 3 * (price - stop_used), 2)\
crv = (tp2 - price) / (price - stop_used) if (price - stop_used) > 0 else 0\
risk_eur = depot * (risk_pct / 100)\
risk_per_share = price - stop_used\
pos_size = int(risk_eur / risk_per_share) if risk_per_share > 0 else 0\
time_stop = (date.today() + timedelta(days=hd)).strftime("%d.%m.%Y")\
\
st.markdown(f"## \{name\} `\{ticker\}` \'97 \{exch\} (\{ccy\})")\
c1, c2, c3, c4 = st.columns(4)\
c1.metric("Kurs (Adj. Close)", f"\{price:.2f\} \{ccy\}", ts)\
c2.metric("MA-Quelle", "Yahoo Finance", "Adjusted Close OK")\
c3.metric("Trend-Regime", regime, reg_amp)\
c4.metric("Earnings", sg_earn_txt, sg_earn)\
st.divider()\
\
st.subheader("Scores")\
c1, c2, c3, c4 = st.columns(4)\
c1.metric("Company Quality", f"\{company\}/100", ampel(company))\
c2.metric("Setup Quality", f"\{setup\}/100", ampel(setup))\
c3.metric("Investment Score", f"\{investment\}/100", ampel(investment))\
c4.metric("Konfluenz", f"\{kb\}/4", "Robust" if kb >= 3 else ("Fragil" if kb == 2 else "Schwach"))\
st.divider()\
\
t1, t2, t3 = st.tabs(["Technik", "Safeguards", "Trade-Setup"])\
with t1:\
    cols = st.columns(2)\
    items = [\
        ("S3 Trend", s3a, s3, s3t),\
        ("S4 Momentum", s4a, s4, s4t),\
        ("S5 Volumen", s5a, s5, s5t),\
        ("S6 Volatilitaet", s6a, s6, s6t),\
        ("52W-Lage", ampel(w52), w52, f"\{dist52:.1f\}% vom 52W-Hoch"),\
        ("Rel. Staerke", ampel(rs_score), rs_score, f"3M: \{ret3m:.1f\}%")\
    ]\
    for i, (lab, ico, score, com) in enumerate(items):\
        with cols[i % 2]:\
            st.markdown(f'<div class="metric-card \{card_class(score)\}"><b>\{ico\} \{lab\}</b><span style="float:right;font-size:1.3rem;font-weight:700">\{score\}</span><br><small style="color:#aaa">\{com\}</small></div>', unsafe_allow_html=True)\
    st.dataframe(pd.DataFrame(\{\
        "Indikator": ["Kurs","MA20","MA50","MA200","RSI(14)","MACD","Signal","ADX","ATR","ATR%","ROC(12)","52W-Hoch","52W-Tief","Dist 52W"],\
        "Wert": [f"\{price:.2f\}",f"\{ma20:.2f\}",f"\{ma50:.2f\}",f"\{ma200:.2f\}",f"\{rsi:.1f\}",f"\{macd_v:.3f\}",f"\{signal_v:.3f\}",f"\{adx:.1f\}",f"\{atr:.3f\}",f"\{atr_pct:.1f\}%",f"\{roc:.1f\}%",f"\{high52:.2f\}",f"\{low52:.2f\}",f"\{dist52:.1f\}%"]\
    \}), hide_index=True, use_container_width=True)\
with t2:\
    st.dataframe(pd.DataFrame(\{\
        "Safeguard": ["S0 Currency Lock","S0 Preis-Typ-Lock","S1 Earnings","S2 Regime","S3 Fundamentals-Lockout","S7 Konfluenz-Cap"],\
        "Status": ["\uc0\u55357 \u57314 ","\u55357 \u57314 ",sg_earn,reg_amp,"\u55357 \u57314 " if hd >= 30 else "\u9899 ","\u55357 \u57314 " if kb >= 3 else ("\u55357 \u57313 " if kb == 2 else "\u55357 \u56628 ")],\
        "Kommentar": [f"\{ccy\} \{exch\}","auto_adjust=True Yahoo Finance",sg_earn_txt,regime,f"Company-Gewicht \{int(wc*100)\}%",f"\{kb\}/4 gruene Kernbloecke"]\
    \}), hide_index=True, use_container_width=True)\
with t3:\
    c1, c2, c3 = st.columns(3)\
    c1.metric("Entry", f"\{price:.2f\} \{ccy\}")\
    c2.metric("ATR-Stop", f"\{atr_stop:.2f\} \{ccy\}", f"-\{stop_dist:.1f\}%")\
    c3.metric("Struktur-Stop", f"\{struct_stop:.2f\} \{ccy\}", "MA50 x 0.965")\
    c4, c5, c6 = st.columns(3)\
    c4.metric("TP1 (1R)", f"\{tp1:.2f\} \{ccy\}", f"+\{(tp1/price-1)*100:.1f\}%")\
    c5.metric("TP2 (2R)", f"\{tp2:.2f\} \{ccy\}", f"+\{(tp2/price-1)*100:.1f\}%")\
    c6.metric("TP3 (3R)", f"\{tp3:.2f\} \{ccy\}", f"+\{(tp3/price-1)*100:.1f\}%")\
    c7, c8, c9 = st.columns(3)\
    c7.metric(f"CRV \{ampel_crv(crv)\}", f"\{crv:.1f\}:1")\
    c8.metric("Positionsgroesse", f"\{pos_size\} Stueck", f"Risiko \{risk_eur:.0f\} EUR (\{risk_pct\}%)")\
    c9.metric("Zeit-Stop", time_stop, "wenn kein Ausbruch")\
\
st.divider()\
st.subheader("5 Zeithorizont-Ampeln")\
hmap = \{\
    "Kurzfrist": setup,\
    "Swing": round(setup*0.7 + company*0.3),\
    "Mittelfrist": investment,\
    "Langfrist": round(setup*0.3 + company*0.7),\
    "Sehr langfristig": company\
\}\
cols = st.columns(5)\
for col, (lab, scv) in zip(cols, hmap.items()):\
    col.markdown(f"<div style='text-align:center'><div style='font-size:2rem'>\{ampel(scv)\}</div><small>\{lab\}<br><b>\{scv\}/100</b></small></div>", unsafe_allow_html=True)\
\
st.divider()\
st.subheader("Handlungsempfehlung")\
if days_earn < 7:\
    emp, conv = "VETO - Earnings < 7 Tage", "-"\
elif investment >= 75:\
    emp, conv = "BUY / ACCUMULATE", ("HIGH" if kb >= 3 else "MEDIUM")\
elif investment >= 60:\
    emp, conv = "WATCH / kleine Position", "MEDIUM"\
elif investment >= 45:\
    emp, conv = "HOLD / beobachten", "LOW"\
else:\
    emp, conv = "AVOID / WAIT", "LOW"\
\
c1, c2, c3 = st.columns(3)\
c1.metric("Empfehlung", emp)\
c2.metric("Conviction", conv)\
c3.metric("Zeithorizont", horizon.split("(")[0].strip())\
}