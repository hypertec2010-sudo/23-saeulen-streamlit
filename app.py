
# -*- coding: utf-8 -*-
import re
import warnings
from datetime import datetime, timezone, date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

APP_VERSION = "v9.0"

st.set_page_config(
    page_title=f"Capital-Hill-Score-Modell {APP_VERSION}",
    page_icon="📊",
    layout="wide"
)


def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Passwort eingeben:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Passwort eingeben:", type="password", on_change=password_entered, key="password")
        st.error("😕 Falsches Passwort")
        return False
    return True


if not check_password():
    st.stop()

# ---------- Session State ----------
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

if "search_input" not in st.session_state:
    st.session_state.search_input = "AAPL"

if "selected_search_label" not in st.session_state:
    st.session_state.selected_search_label = None

if "analysis_ticker" not in st.session_state:
    st.session_state.analysis_ticker = "AAPL"

if "analysis_requested" not in st.session_state:
    st.session_state.analysis_requested = False

if "batch_input" not in st.session_state:
    st.session_state.batch_input = ""

if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "Einzelanalyse"

if "analysis_mode_run" not in st.session_state:
    st.session_state.analysis_mode_run = "Einzelanalyse"

if "ranking_df" not in st.session_state:
    st.session_state.ranking_df = pd.DataFrame()

if "ranking_results" not in st.session_state:
    st.session_state.ranking_results = {}

if "selected_ranking_ticker" not in st.session_state:
    st.session_state.selected_ranking_ticker = "AAPL"

if "last_mode_label" not in st.session_state:
    st.session_state.last_mode_label = "Watchlist"

# ---------- CSS ----------
st.markdown("""
<style>
.metric-card{
    background:linear-gradient(180deg,#111827 0%, #0f172a 100%);
    border-radius:16px;
    padding:16px 20px;
    margin:8px 0;
    border:1px solid #243042;
    border-left:4px solid #4CAF50;
    box-shadow:0 10px 24px rgba(0,0,0,0.18);
}
.metric-card.red{border-left-color:#f44336;}
.metric-card.yellow{border-left-color:#FFC107;}
.small-note{color:#9aa4b2;font-size:0.88rem;}
pre{white-space:pre-wrap !important;}
.score-card{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:16px 16px 14px 16px;
    min-height:124px;
    box-shadow:0 10px 24px rgba(0,0,0,0.18);
}
.score-label{
    color:#94a3b8;
    font-size:0.80rem;
    text-transform:uppercase;
    letter-spacing:0.03em;
    line-height:1.2;
    margin-bottom:10px;
}
.score-value{
    color:#f8fafc;
    font-size:1.55rem;
    font-weight:800;
    line-height:1.15;
}
.score-delta{
    color:#cbd5e1;
    font-size:0.88rem;
    line-height:1.2;
    margin-top:10px;
}
.score-card.company{border-left:4px solid #3b82f6;}
.score-card.setup{border-left:4px solid #22c55e;}
.score-card.short{border-left:4px solid #f59e0b;}
.score-card.helper{border-left:4px solid #a78bfa;}
.score-card.investment{border-left:4px solid #14b8a6;}
.score-card.board{border-left:4px solid #ef4444; box-shadow:0 12px 28px rgba(239,68,68,0.22);}
.score-card.kb{border-left:4px solid #14b8a6; box-shadow:0 12px 28px rgba(20,184,166,0.22);}
.section-card{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:16px 18px;
    box-shadow:0 10px 24px rgba(0,0,0,0.18);
    margin:10px 0;
}
.premium-card{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:16px 18px;
    box-shadow:0 10px 24px rgba(0,0,0,0.18);
}
.premium-title{
    color:#94a3b8;
    font-size:0.80rem;
    text-transform:uppercase;
    letter-spacing:0.03em;
    margin-bottom:8px;
}
.premium-value{
    color:#f8fafc;
    font-size:1.15rem;
    font-weight:700;
    line-height:1.25;
}
.premium-sub{
    color:#cbd5e1;
    font-size:0.88rem;
    margin-top:8px;
    line-height:1.2;
}
.reco-card{
    background:linear-gradient(180deg,#0f172a 0%, #111827 100%);
    border:1px solid #243042;
    border-radius:18px;
    padding:14px 16px;
    min-height:132px;
    box-shadow:0 10px 24px rgba(0,0,0,0.20);
    display:flex;
    flex-direction:column;
    justify-content:space-between;
}
.reco-top{
    display:flex;
    align-items:flex-start;
    justify-content:space-between;
    gap:10px;
}
.reco-label{
    color:#94a3b8;
    font-size:0.79rem;
    line-height:1.2;
    margin-bottom:8px;
    letter-spacing:0.02em;
    text-transform:uppercase;
}
.reco-icon{
    font-size:1.1rem;
    line-height:1;
    opacity:0.95;
}
.reco-value{
    color:#f8fafc;
    font-size:1.05rem;
    font-weight:700;
    line-height:1.28;
    white-space:normal;
    word-break:break-word;
}
.reco-delta{
    color:#cbd5e1;
    font-size:0.85rem;
    margin-top:10px;
    line-height:1.2;
}
.reco-chip{
    display:inline-block;
    margin-top:10px;
    padding:4px 10px;
    border-radius:999px;
    background:#1e293b;
    color:#e2e8f0;
    font-size:0.78rem;
    font-weight:600;
    width:fit-content;
}
.reco-card.context{border-left:4px solid #60a5fa;}
.reco-card.main{border-left:4px solid #22c55e;}
.reco-card.conviction{border-left:4px solid #a78bfa;}
.reco-card.signal{border-left:4px solid #f59e0b;}
.reco-card.market{border-left:4px solid #14b8a6;}
.score-pill{
    border-radius:999px;
    padding:2px 10px;
    font-size:0.82rem;
    font-weight:600;
    display:inline-block;
}
</style>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
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
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def fmt_num(x, digits=2, suffix=""):
    return f"{x:.{digits}f}{suffix}" if pd.notna(x) else "n/a"


def known_ratio(values):
    vals = [v for v in values if pd.notna(v)]
    return len(vals) / len(values) if values else 0


def normalize_missing(v):
    if v is None:
        return np.nan
    if isinstance(v, str) and v.strip().lower() in {"", "none", "nan", "null"}:
        return np.nan
    try:
        if pd.isna(v):
            return np.nan
    except Exception:
        pass
    return v


def is_missing_scalar(v):
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in {"", "none", "nan", "null", "n/a"}:
        return True
    try:
        if np.isscalar(v) and pd.isna(v):
            return True
    except Exception:
        pass
    return False


def merge_info(base, extra):
    base = dict(base or {})
    for k, v in dict(extra or {}).items():
        cur = base.get(k)
        if k not in base or is_missing_scalar(cur):
            nv = normalize_missing(v)
            if not is_missing_scalar(nv):
                base[k] = nv
    return base


def split_batch_input(text):
    parts = re.split(r"[\n,;|]+", str(text or ""))
    cleaned = []
    seen = set()
    for p in parts:
        x = p.strip()
        if not x:
            continue
        key = x.lower()
        if key not in seen:
            cleaned.append(x)
            seen.add(key)
    return cleaned[:25]


def score_badge(score):
    if score >= 80:
        return "Exzellent", "#16a34a"
    if score >= 70:
        return "Stark", "#65a30d"
    if score >= 60:
        return "Solide", "#ca8a04"
    if score >= 50:
        return "Gemischt", "#ea580c"
    return "Schwach", "#dc2626"


def style_ranking_df(df):
    score_cols = [
        "Company Quality", "Setup Quality", "Investment Score",
        "Tradeability", "Kurzfrist-Timing", "TradingBoard Score", "Fundamental-Confidence"
    ]

    def cell_style(v):
        try:
            _, color = score_badge(float(v))
            return f"background-color: {color}; color: white;"
        except Exception:
            return ""

    styler = df.style
    for c in score_cols:
        if c in df.columns:
            styler = styler.map(cell_style, subset=[c])
    return styler


def market_regime_label(regime):
    mapping = {
        "POSITIV": "Positiv",
        "NEGATIV": "Negativ",
        "NEUTRAL": "Neutral",
        "UNBEKANNT": "Keine belastbaren Benchmark-Daten",
    }
    return mapping.get(str(regime or "").upper(), str(regime or "-"))


def shorten_text(value, max_len=42):
    if value is None:
        return "-"
    txt = str(value).strip()
    if not txt:
        return "-"
    if len(txt) <= max_len:
        return txt
    clipped = txt[: max_len - 3]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    clipped = clipped.rstrip(" ,;:-")
    return clipped + "..."


def display_mode_label(mode_label):
    mapping = {
        "Watchlist": "Beobachtung",
        "Position": "Bestehende Position",
    }
    return mapping.get(str(mode_label or ""), str(mode_label or "-"))


def display_emp_label(emp):
    mapping = {
        "BUY / ACCUMULATE": "Kauf / Aufbau",
        "WATCH / EINSTIEG PRÜFEN": "Beobachten / Einstieg prüfen",
        "BEOBACHTEN": "Weiter beobachten",
        "AVOID / WAIT": "Aktuell kein Einstieg",
        "HALTEN / AUSBAUEN": "Halten / ggf. ausbauen",
        "HALTEN / ENGE BEOBACHTUNG": "Halten / eng beobachten",
        "HALTEN / RISIKO PRÜFEN": "Halten / Risiko prüfen",
        "RISIKO REDUZIEREN / STOPP PRÜFEN": "Risiko senken / Stop prüfen",
        "NO TRADE": "Kein valides Setup",
        "NO TRADE / WAIT": "Noch kein valides Setup",
        "VETO - Earnings < 7 Tage": "Kein Trade vor Zahlen",
    }
    return mapping.get(str(emp or ""), str(emp or "-"))


def display_conv_label(conv):
    mapping = {
        "HIGH": "hoch",
        "MEDIUM": "mittel",
        "LOW-MEDIUM": "eher verhalten",
        "LOW": "niedrig",
        "NONE": "keine",
        "-": "-",
    }
    return mapping.get(str(conv or ""), str(conv or "-"))


def display_stb_label(signal):
    mapping = {
        "LONG": "Bullisch",
        "HOLD": "Neutral / Halten",
        "WAIT": "Abwarten",
        "SHORT": "Schwach / defensiv",
    }
    return mapping.get(str(signal or ""), str(signal or "-"))


def render_score_card(label, value, subtitle="", variant="company"):
    st.markdown(
        f"""
        <div class="score-card {variant}">
            <div class="score-label">{label}</div>
            <div class="score-value">{value}</div>
            <div class="score-delta">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def trading_timing_label(score):
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 9:
        return "stark"
    if s >= 5:
        return "konstruktiv"
    if s >= 3:
        return "abwartend"
    return "schwach"


def normalize_tb_score_100(score):
    try:
        s = float(score)
    except Exception:
        return np.nan
    return int(round(clamp((s + 3) / 12 * 100, 0, 100)))


def tradeability_label(score):
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 78:
        return "hoch handelbar"
    if s >= 62:
        return "brauchbar"
    if s >= 45:
        return "eingeschränkt"
    return "schwach"


def investment_case_label(score):
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 80:
        return "sehr attraktiv"
    if s >= 70:
        return "attraktiv"
    if s >= 55:
        return "okay"
    if s >= 40:
        return "schwach"
    return "unattraktiv"


def trading_case_label(score):
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 80:
        return "sehr attraktiv"
    if s >= 68:
        return "attraktiv"
    if s >= 52:
        return "brauchbar"
    if s >= 40:
        return "eher abwarten"
    return "nicht attraktiv"


def setup_confidence_label(score):
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 80:
        return "hoch"
    if s >= 60:
        return "mittel"
    if s >= 40:
        return "moderat"
    return "niedrig"


def linear_score(value, low, high, floor=10, ceiling=95):
    if pd.isna(value):
        return np.nan
    if high == low:
        return ceiling
    scaled = (float(value) - low) / (high - low)
    return clamp(floor + scaled * (ceiling - floor), floor, ceiling)


def ideal_range_score(value, ideal_low, ideal_high, hard_low, hard_high, floor=15, ceiling=92):
    if pd.isna(value):
        return np.nan
    v = float(value)
    if ideal_low <= v <= ideal_high:
        return ceiling
    if v < ideal_low:
        if v <= hard_low:
            return floor
        scaled = (v - hard_low) / (ideal_low - hard_low)
        return clamp(floor + scaled * (ceiling - floor), floor, ceiling)
    if v >= hard_high:
        return floor
    scaled = (hard_high - v) / (hard_high - ideal_high)
    return clamp(floor + scaled * (ceiling - floor), floor, ceiling)


def entry_quality_score(entry_quality, price, entry_low, entry_high):
    quality = str(entry_quality or "").lower()
    if quality == "gut":
        return 88
    if quality == "abwarten":
        if pd.notna(entry_high) and entry_high > 0 and pd.notna(price):
            gap_pct = (float(price) / float(entry_high) - 1) * 100
            return clamp(62 - gap_pct * 4.5, 20, 62)
        return 45
    if quality == "früh":
        if pd.notna(entry_low) and entry_low > 0 and pd.notna(price):
            gap_pct = (float(entry_low) / float(price) - 1) * 100
            return clamp(55 - max(gap_pct, 0) * 2.5, 25, 55)
        return 40
    return 45


def format_price_zone(low, high, ccy):
    if pd.isna(low) and pd.isna(high):
        return "-"
    if pd.notna(low) and pd.notna(high):
        if abs(float(low) - float(high)) < 0.01:
            return f"{float(low):.2f} {ccy}"
        return f"{float(low):.2f} - {float(high):.2f} {ccy}"
    if pd.notna(low):
        return f"ab {float(low):.2f} {ccy}"
    return f"bis {float(high):.2f} {ccy}"


# ---------- Indicators ----------
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


# ---------- Domain Helpers ----------
def infer_display_currency(ticker, info, fallback="USD"):
    suffix = ticker.split(".")[-1].upper() if "." in ticker else ""
    exchange = str(info.get("exchange", "") or "").upper()
    info_ccy = str(info.get("currency", fallback) or fallback).upper()

    eur_suffixes = {"DE", "PA", "AS", "BR", "MI", "MC", "HE", "VI", "LS", "F"}
    if suffix in eur_suffixes:
        return "EUR"
    if suffix == "L":
        return "GBP"
    if suffix == "SW":
        return "CHF"
    if suffix == "ST":
        return "SEK"
    if suffix == "OL":
        return "NOK"
    if suffix == "CO":
        return "DKK"
    if suffix == "WA":
        return "PLN"
    if suffix == "PR":
        return "CZK"

    if exchange in {"XETRA", "GER", "PAR", "AMS", "MIL", "MAD", "HEL", "VIE", "BRU", "EURONEXT"}:
        return "EUR"
    if exchange in {"LSE"}:
        return "GBP"
    if exchange in {"SIX"}:
        return "CHF"

    return info_ccy if info_ccy else fallback


def analyst_label(rec_key):
    mapping = {
        "strong_buy": "Starker Kauf",
        "buy": "Kauf",
        "hold": "Halten",
        "underperform": "Unterdurchschnittlich",
        "sell": "Verkaufen",
        "strong_sell": "Starker Verkauf",
    }
    return mapping.get(str(rec_key).lower(), str(rec_key))


def tb_signal_label(score):
    if score >= 9:
        return "LONG", "AKTIV HALTEN"
    if score >= 5:
        return "HOLD", "HALTEN"
    if score >= 3:
        return "WAIT", "ABWARTEN"
    return "SHORT", "STOPP PRÜFEN"


def first_existing_row(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for name in names:
        if name in df.index:
            row = df.loc[name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return pd.to_numeric(row, errors="coerce")
    return None


def latest_valid(series_like):
    if series_like is None:
        return np.nan
    s = pd.to_numeric(pd.Series(series_like), errors="coerce").dropna()
    return float(s.iloc[0]) if len(s) else np.nan


def previous_valid(series_like):
    if series_like is None:
        return np.nan
    s = pd.to_numeric(pd.Series(series_like), errors="coerce").dropna()
    return float(s.iloc[1]) if len(s) > 1 else np.nan


def select_benchmark(ticker, info=None):
    suffix = ticker.split(".")[-1].upper() if "." in ticker else ""
    exchange = str((info or {}).get("exchange", "") or "").upper()

    german_suffixes = {"DE", "F", "HM", "BE", "DU", "MU", "SG"}
    europe_suffixes = {"PA", "AS", "BR", "MI", "MC", "HE", "VI", "LS"}
    swiss_suffixes = {"SW"}
    uk_suffixes = {"L"}
    nordic_suffixes = {"ST", "OL", "CO"}

    if suffix in german_suffixes or exchange in {"XETRA", "GER"}:
        return "^GDAXI", "DAX"
    if suffix in europe_suffixes or exchange in {"PAR", "AMS", "MIL", "MAD", "HEL", "VIE", "BRU", "EURONEXT"}:
        return "^STOXX50E", "STOXX 50"
    if suffix in swiss_suffixes or exchange in {"SIX"}:
        return "^SSMI", "SMI"
    if suffix in uk_suffixes or exchange in {"LSE"}:
        return "^FTSE", "FTSE 100"
    if suffix in nordic_suffixes:
        return "^STOXX50E", "STOXX 50"

    return "SPY", "S&P 500"


def calc_return_metrics(close_series):
    return {
        "ret21": safe_last(close_series.pct_change(21) * 100, np.nan),
        "ret63": safe_last(close_series.pct_change(63) * 100, np.nan),
        "ret126": safe_last(close_series.pct_change(126) * 100, np.nan),
    }


def evaluate_market_filter(benchmark_df):
    if benchmark_df is None or benchmark_df.empty or "Close" not in benchmark_df.columns:
        return {
            "price": np.nan,
            "ma50": np.nan,
            "ma200": np.nan,
            "ret21": np.nan,
            "ret63": np.nan,
            "ret126": np.nan,
            "regime": "UNBEKANNT",
            "ampel": "⚪",
            "score": 50
        }

    close = benchmark_df["Close"]
    price = safe_last(close, np.nan)
    ma50 = safe_last(close.rolling(50).mean(), np.nan)
    ma200 = safe_last(close.rolling(200).mean(), np.nan)

    rets = calc_return_metrics(close)

    if pd.notna(price) and pd.notna(ma50) and pd.notna(ma200):
        if price > ma50 and price > ma200 and ma50 > ma200:
            regime, ampel_icon, score = "POSITIV", "🟢", 100
        elif price < ma50 and price < ma200 and ma50 < ma200:
            regime, ampel_icon, score = "NEGATIV", "🔴", 30
        else:
            regime, ampel_icon, score = "NEUTRAL", "🟡", 60
    else:
        regime, ampel_icon, score = "UNBEKANNT", "⚪", 50

    return {
        "price": price,
        "ma50": ma50,
        "ma200": ma200,
        "ret21": rets["ret21"],
        "ret63": rets["ret63"],
        "ret126": rets["ret126"],
        "regime": regime,
        "ampel": ampel_icon,
        "score": score
    }


def infer_market_bucket(ticker, info):
    suffix = ticker.split(".")[-1].upper() if "." in ticker else ""
    exchange = str((info or {}).get("exchange", "") or "").upper()
    ccy = str((info or {}).get("currency", "") or "").upper()

    us_ex = {"NASDAQ", "NASDAQGS", "NYQ", "NYSE", "AMEX", "PCX"}
    eu_suffixes = {"DE", "PA", "AS", "BR", "MI", "MC", "HE", "VI", "LS", "F", "SW", "L", "ST", "OL", "CO"}

    if exchange in us_ex or ccy == "USD" and suffix == "":
        return "USA"
    if suffix in eu_suffixes or ccy in {"EUR", "GBP", "CHF", "SEK", "NOK", "DKK"}:
        return "Europa"
    return "Andere"


def infer_stock_style_advanced(revenue_growth, earnings_growth, pe, pb, beta, debt_to_equity, roe, profit_margin, sector):
    sector = str(sector or "").lower()
    if pd.notna(revenue_growth) and revenue_growth > 0.15 and pd.notna(pe) and pe > 25:
        return "Growth"
    if pd.notna(roe) and roe > 0.18 and pd.notna(profit_margin) and profit_margin > 0.15:
        return "Quality"
    if pd.notna(pe) and pe < 15 and pd.notna(pb) and pb < 2:
        return "Value"
    if pd.notna(beta) and beta > 1.5:
        return "Cyclical"
    if pd.notna(earnings_growth) and earnings_growth < 0 and pd.notna(debt_to_equity) and debt_to_equity > 150:
        return "Turnaround / Risk"
    if "utility" in sector or "consumer defensive" in sector:
        return "Defensive"
    return "Neutral"


def get_style_sector_adjustment(style, sector):
    sector = str(sector or "").lower()
    weights = {
        "quality": 1.0,
        "growth": 1.0,
        "valuation": 1.0,
        "balance": 1.0,
        "momentum": 1.0,
        "trend": 1.0,
    }

    if style == "Growth":
        weights["growth"] = 1.18
        weights["momentum"] = 1.10
        weights["valuation"] = 0.82
    elif style == "Quality":
        weights["quality"] = 1.15
        weights["balance"] = 1.08
    elif style == "Value":
        weights["valuation"] = 1.20
        weights["balance"] = 1.08
        weights["growth"] = 0.88
    elif style == "Defensive":
        weights["balance"] = 1.16
        weights["momentum"] = 0.90
    elif style == "Cyclical":
        weights["trend"] = 1.12
        weights["momentum"] = 1.12

    if "technology" in sector or "software" in sector or "semiconductor" in sector:
        weights["growth"] *= 1.10
        weights["trend"] *= 1.07
        weights["valuation"] *= 0.86
    elif "financial" in sector or "bank" in sector or "insurance" in sector:
        weights["balance"] *= 1.15
        weights["valuation"] *= 1.12
        weights["growth"] *= 0.90
    elif "health" in sector or "pharma" in sector or "biotech" in sector:
        weights["quality"] *= 1.08
        weights["balance"] *= 1.05
    elif "energy" in sector or "materials" in sector:
        weights["trend"] *= 1.12
        weights["momentum"] *= 1.05
    elif "consumer defensive" in sector or "utilities" in sector:
        weights["balance"] *= 1.18
        weights["momentum"] *= 0.88

    return weights


def build_red_flags(
    revenue_growth,
    earnings_growth,
    profit_margin,
    fcf,
    op_cf,
    debt_to_equity,
    current_ratio,
    quick_ratio,
    has_upcoming_earnings,
    days_earn
):
    items = []

    def add_item(category, detail, penalty):
        items.append({
            "Kategorie": category,
            "Status": "🔴" if penalty >= 6 else "🟡",
            "Detail": detail,
            "Penalty": penalty
        })

    if pd.notna(earnings_growth) and earnings_growth < -0.15:
        add_item("Ertrags-Risiko", "Gewinnwachstum stark negativ", 8)
    if pd.notna(profit_margin) and profit_margin < 0:
        add_item("Ertrags-Risiko", "Gewinnmarge negativ", 8)
    if pd.notna(revenue_growth) and revenue_growth < -0.10:
        add_item("Umsatz-/Geschäfts-Risiko", "Umsatzwachstum negativ", 6)
    if pd.notna(fcf) and fcf < 0:
        add_item("Cashflow-Risiko", "Freier Cashflow negativ", 6)
    if pd.notna(op_cf) and op_cf < 0:
        add_item("Cashflow-Risiko", "Operativer Cashflow negativ", 5)
    if pd.notna(debt_to_equity) and debt_to_equity > 180:
        add_item("Bilanz-Risiko", "Verschuldung sehr hoch", 8)
    if pd.notna(current_ratio) and current_ratio < 1.0:
        add_item("Bilanz-Risiko", "Liquidität schwach (Current Ratio < 1.0)", 5)
    if pd.notna(quick_ratio) and quick_ratio < 0.8:
        add_item("Bilanz-Risiko", "Quick Ratio schwach (< 0.8)", 4)
    if has_upcoming_earnings and pd.notna(days_earn) and days_earn <= 7:
        add_item("Event-Risiko", f"Earnings in {int(days_earn)} Tagen", 6)

    total_penalty = sum(x["Penalty"] for x in items)
    return items, total_penalty


def build_decision_explanation(
    setup,
    company,
    investment,
    market_regime,
    rs_vs_benchmark_63,
    quality_score,
    growth_score,
    valuation_score,
    balance_score,
    red_flag_items,
    earnings_warning,
    kb,
    position_mode
):
    strengths = []
    weaknesses = []

    if setup >= 75:
        strengths.append("Technisches Setup ist stark.")
    elif setup < 50:
        weaknesses.append("Technisches Setup ist derzeit schwach.")

    if company >= 70:
        strengths.append("Unternehmensqualität ist solide bis stark.")
    elif company < 50:
        weaknesses.append("Unternehmensqualität ist eher schwach.")

    if pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 > 5:
        strengths.append("Die Aktie zeigt klare Outperformance gegenüber dem Benchmark.")
    elif pd.notna(rs_vs_benchmark_63) and rs_vs_benchmark_63 < -5:
        weaknesses.append("Die Aktie underperformt ihren Benchmark spürbar.")

    if quality_score >= 75:
        strengths.append("Profitabilität und Kapitalrendite sind überzeugend.")
    if growth_score >= 75:
        strengths.append("Wachstum ist aktuell stark.")
    if valuation_score >= 72:
        strengths.append("Bewertung wirkt noch akzeptabel bis attraktiv.")
    elif valuation_score < 50:
        weaknesses.append("Bewertung wirkt anspruchsvoll.")

    if balance_score >= 72:
        strengths.append("Bilanzqualität ist ordentlich.")
    elif balance_score < 50:
        weaknesses.append("Bilanzqualität ist belastet.")

    if market_regime == "POSITIV":
        strengths.append("Das Marktumfeld unterstützt Long-Setups.")
    elif market_regime == "NEGATIV":
        weaknesses.append("Das Marktumfeld ist aktuell klar negativ.")

    if kb >= 3:
        strengths.append("Hohe Konfluenz der Kernblöcke.")
    elif kb <= 1:
        weaknesses.append("Zu wenig Konfluenz der Kernblöcke.")

    if earnings_warning:
        weaknesses.append("Kurzfristiges Earnings-Risiko erhöht die Unsicherheit.")

    for item in red_flag_items[:3]:
        weaknesses.append(f"{item['Kategorie']}: {item['Detail']}")

    strengths = strengths[:5]
    weaknesses = weaknesses[:5]

    if position_mode:
        if investment >= 75 and market_regime != "NEGATIV":
            summary = "Bestehende Position wirkt insgesamt stark. Halten oder selektiv ausbauen ist plausibel."
        elif investment >= 60:
            summary = "Bestehende Position ist grundsätzlich okay, aber nicht frei von Risiken. Eng beobachten."
        else:
            summary = "Bestehende Position wirkt anfällig. Risiko-Management und Stopps prüfen."
    else:
        if investment >= 75 and market_regime == "POSITIV":
            summary = "Guter Watchlist-Kandidat mit unterstützendem Marktumfeld."
        elif investment >= 60:
            summary = "Interessanter Kandidat, aber Timing oder Teilbereiche sind noch nicht ideal."
        else:
            summary = "Aktuell eher Beobachtung statt Einstieg."

    return strengths, weaknesses, summary


def infer_data_source_flags(info):
    direct_fields = [
        "profitMargins", "operatingMargins", "grossMargins", "returnOnEquity", "returnOnAssets",
        "revenueGrowth", "earningsGrowth", "currentRatio", "quickRatio", "debtToEquity",
        "freeCashflow", "operatingCashflow", "forwardPE", "pegRatio", "priceToSalesTrailing12Months",
        "priceToBook", "beta", "shortPercentOfFloat", "recommendationMean",
        "numberOfAnalystOpinions", "targetMeanPrice"
    ]
    loaded = int(info.get("_fund_fields_loaded", 0) or 0)
    total = len(direct_fields)
    derived = 0
    coverage = loaded / total if total else 0
    if coverage >= 0.75:
        confidence = "Hoch"
        confidence_icon = "🟢"
    elif coverage >= 0.50:
        confidence = "Mittel"
        confidence_icon = "🟡"
    else:
        confidence = "Niedrig"
        confidence_icon = "🔴"
    return {
        "loaded": loaded,
        "total": total,
        "coverage": coverage,
        "derived_estimate": derived,
        "confidence": confidence,
        "confidence_icon": confidence_icon
    }


def build_candlestick_chart(chart_df, ticker, ccy):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name=ticker
        ),
        row=1,
        col=1
    )

    if "MA20" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["MA20"], mode="lines", name="MA20"),
            row=1,
            col=1
        )
    if "MA50" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["MA50"], mode="lines", name="MA50"),
            row=1,
            col=1
        )
    if "MA200" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["MA200"], mode="lines", name="MA200"),
            row=1,
            col=1
        )

    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"],
            name="Volumen"
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        title=f"{ticker} Candlestick-Chart ({ccy})",
        xaxis_rangeslider_visible=False,
        height=650,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_yaxes(title_text=f"Kurs ({ccy})", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)
    return fig


# ---------- Data Enrichment ----------
def extract_analyst_data(ticker_obj, info):
    info = dict(info or {})

    try:
        recs = getattr(ticker_obj, "recommendations", None)
    except Exception:
        recs = None

    if recs is not None and not getattr(recs, "empty", True):
        cols = {str(c).lower(): c for c in recs.columns}
        strong_buy_col = cols.get("strongbuy") or cols.get("strong_buy")
        buy_col = cols.get("buy")
        hold_col = cols.get("hold")
        sell_col = cols.get("sell")
        strong_sell_col = cols.get("strongsell") or cols.get("strong_sell")

        if all(c is not None for c in [strong_buy_col, buy_col, hold_col, sell_col, strong_sell_col]):
            row = recs.iloc[-1]
            sb = pd.to_numeric(row.get(strong_buy_col), errors="coerce")
            b = pd.to_numeric(row.get(buy_col), errors="coerce")
            h = pd.to_numeric(row.get(hold_col), errors="coerce")
            s = pd.to_numeric(row.get(sell_col), errors="coerce")
            ss = pd.to_numeric(row.get(strong_sell_col), errors="coerce")
            vals = [sb, b, h, s, ss]
            if sum(pd.notna(v) for v in vals) >= 3:
                sb = 0 if pd.isna(sb) else float(sb)
                b = 0 if pd.isna(b) else float(b)
                h = 0 if pd.isna(h) else float(h)
                s = 0 if pd.isna(s) else float(s)
                ss = 0 if pd.isna(ss) else float(ss)
                total = sb + b + h + s + ss
                if total > 0:
                    mean = (1 * sb + 2 * b + 3 * h + 4 * s + 5 * ss) / total
                    if pd.isna(info.get("recommendationMean")):
                        info["recommendationMean"] = mean
                    if pd.isna(info.get("numberOfAnalystOpinions")):
                        info["numberOfAnalystOpinions"] = int(total)
                    if pd.isna(info.get("recommendationKey")) or str(info.get("recommendationKey", "")).lower() in {"", "none", "nan"}:
                        if mean <= 1.5:
                            info["recommendationKey"] = "strong_buy"
                        elif mean <= 2.5:
                            info["recommendationKey"] = "buy"
                        elif mean <= 3.5:
                            info["recommendationKey"] = "hold"
                        elif mean <= 4.5:
                            info["recommendationKey"] = "sell"
                        else:
                            info["recommendationKey"] = "strong_sell"

    if pd.isna(info.get("targetMeanPrice")):
        try:
            tinfo = ticker_obj.get_info() or {}
            v = normalize_missing(tinfo.get("targetMeanPrice"))
            if not pd.isna(v):
                info["targetMeanPrice"] = v
        except Exception:
            pass

    return info


def extract_earnings_data(ticker_obj, info):
    info = dict(info or {})

    ts = normalize_missing(info.get("earningsTimestamp"))
    if not pd.isna(ts):
        return info

    try:
        cal = ticker_obj.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal:
            dates = cal["Earnings Date"]
            if isinstance(dates, list) and len(dates) > 0:
                dt = pd.to_datetime(dates[0], errors="coerce", utc=True)
                if pd.notna(dt):
                    info["earningsTimestamp"] = int(dt.timestamp())
                    return info
    except Exception:
        pass

    try:
        ed = ticker_obj.get_earnings_dates(limit=8)
        if ed is not None and not getattr(ed, "empty", True):
            idx = ed.index
            idx = pd.to_datetime(pd.Series(idx), errors="coerce", utc=True)
            now_utc = pd.Timestamp.now(tz="UTC")
            future_idx = [x for x in idx if pd.notna(x) and x >= now_utc]
            chosen = min(future_idx) if future_idx else None
            if chosen is None:
                past_idx = [x for x in idx if pd.notna(x)]
                chosen = max(past_idx) if past_idx else None
            if chosen is not None and pd.notna(chosen):
                info["earningsTimestamp"] = int(chosen.timestamp())
    except Exception:
        pass

    return info


def derive_fundamentals_from_statements(ticker_obj, info):
    info = dict(info or {})

    try:
        income = getattr(ticker_obj, "income_stmt", None)
    except Exception:
        income = None
    try:
        q_income = getattr(ticker_obj, "quarterly_income_stmt", None)
    except Exception:
        q_income = None
    try:
        balance = getattr(ticker_obj, "balance_sheet", None)
    except Exception:
        balance = None
    try:
        q_balance = getattr(ticker_obj, "quarterly_balance_sheet", None)
    except Exception:
        q_balance = None
    try:
        cashflow = getattr(ticker_obj, "cashflow", None)
    except Exception:
        cashflow = None
    try:
        q_cashflow = getattr(ticker_obj, "quarterly_cashflow", None)
    except Exception:
        q_cashflow = None

    revenue_row = first_existing_row(income, ["Total Revenue", "Operating Revenue", "Revenue"])
    op_income_row = first_existing_row(income, ["Operating Income", "EBIT", "OperatingIncome"])
    net_income_row = first_existing_row(income, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
    gross_profit_row = first_existing_row(income, ["Gross Profit", "GrossProfit"])
    diluted_eps_row = first_existing_row(income, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"])

    q_revenue_row = first_existing_row(q_income, ["Total Revenue", "Operating Revenue", "Revenue"])
    q_net_income_row = first_existing_row(q_income, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
    q_diluted_eps_row = first_existing_row(q_income, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"])

    total_assets_row = first_existing_row(balance, ["Total Assets", "TotalAssets"])
    total_equity_row = first_existing_row(balance, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    current_assets_row = first_existing_row(balance, ["Current Assets", "Total Current Assets", "CurrentAssets"])
    current_liab_row = first_existing_row(balance, ["Current Liabilities", "Total Current Liabilities", "CurrentLiabilities"])
    inventory_row = first_existing_row(balance, ["Inventory", "Inventories"])
    debt_row = first_existing_row(balance, ["Total Debt", "TotalDebt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"])

    q_current_assets_row = first_existing_row(q_balance, ["Current Assets", "Total Current Assets", "CurrentAssets"])
    q_current_liab_row = first_existing_row(q_balance, ["Current Liabilities", "Total Current Liabilities", "CurrentLiabilities"])
    q_inventory_row = first_existing_row(q_balance, ["Inventory", "Inventories"])
    q_total_equity_row = first_existing_row(q_balance, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    q_total_assets_row = first_existing_row(q_balance, ["Total Assets", "TotalAssets"])
    q_debt_row = first_existing_row(q_balance, ["Total Debt", "TotalDebt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"])

    op_cf_row = first_existing_row(cashflow, ["Operating Cash Flow", "OperatingCashFlow", "Cash Flow From Continuing Operating Activities"])
    fcf_row = first_existing_row(cashflow, ["Free Cash Flow", "FreeCashFlow"])

    revenue = latest_valid(revenue_row)
    op_income = latest_valid(op_income_row)
    net_income = latest_valid(net_income_row)
    gross_profit = latest_valid(gross_profit_row)
    diluted_eps = latest_valid(diluted_eps_row)

    prev_revenue = previous_valid(revenue_row)
    prev_net_income = previous_valid(net_income_row)
    prev_eps = previous_valid(diluted_eps_row)

    q_revenue = latest_valid(q_revenue_row)
    q_prev_revenue = previous_valid(q_revenue_row)
    q_net_income = latest_valid(q_net_income_row)
    q_prev_net_income = previous_valid(q_net_income_row)
    q_eps = latest_valid(q_diluted_eps_row)
    q_prev_eps = previous_valid(q_diluted_eps_row)

    total_assets = latest_valid(total_assets_row)
    total_equity = latest_valid(total_equity_row)
    current_assets = latest_valid(q_current_assets_row if q_current_assets_row is not None else current_assets_row)
    current_liab = latest_valid(q_current_liab_row if q_current_liab_row is not None else current_liab_row)
    inventory = latest_valid(q_inventory_row if q_inventory_row is not None else inventory_row)
    debt = latest_valid(q_debt_row if q_debt_row is not None else debt_row)
    q_total_assets = latest_valid(q_total_assets_row)
    q_total_equity = latest_valid(q_total_equity_row)

    operating_cf = latest_valid(op_cf_row)
    free_cf = latest_valid(fcf_row)

    if pd.isna(info.get("profitMargins")) and pd.notna(net_income) and pd.notna(revenue) and revenue != 0:
        info["profitMargins"] = net_income / revenue
    if pd.isna(info.get("operatingMargins")) and pd.notna(op_income) and pd.notna(revenue) and revenue != 0:
        info["operatingMargins"] = op_income / revenue
    if pd.isna(info.get("grossMargins")) and pd.notna(gross_profit) and pd.notna(revenue) and revenue != 0:
        info["grossMargins"] = gross_profit / revenue

    eq_for_roe = q_total_equity if pd.notna(q_total_equity) else total_equity
    assets_for_roa = q_total_assets if pd.notna(q_total_assets) else total_assets
    if pd.isna(info.get("returnOnEquity")) and pd.notna(net_income) and pd.notna(eq_for_roe) and eq_for_roe != 0:
        info["returnOnEquity"] = net_income / eq_for_roe
    if pd.isna(info.get("returnOnAssets")) and pd.notna(net_income) and pd.notna(assets_for_roa) and assets_for_roa != 0:
        info["returnOnAssets"] = net_income / assets_for_roa

    rev_growth = np.nan
    if pd.notna(q_revenue) and pd.notna(q_prev_revenue) and q_prev_revenue != 0:
        rev_growth = q_revenue / q_prev_revenue - 1
    elif pd.notna(revenue) and pd.notna(prev_revenue) and prev_revenue != 0:
        rev_growth = revenue / prev_revenue - 1
    if pd.isna(info.get("revenueGrowth")) and pd.notna(rev_growth):
        info["revenueGrowth"] = rev_growth

    earn_growth = np.nan
    if pd.notna(q_eps) and pd.notna(q_prev_eps) and q_prev_eps != 0:
        earn_growth = q_eps / q_prev_eps - 1
    elif pd.notna(q_net_income) and pd.notna(q_prev_net_income) and q_prev_net_income != 0:
        earn_growth = q_net_income / q_prev_net_income - 1
    elif pd.notna(diluted_eps) and pd.notna(prev_eps) and prev_eps != 0:
        earn_growth = diluted_eps / prev_eps - 1
    elif pd.notna(net_income) and pd.notna(prev_net_income) and prev_net_income != 0:
        earn_growth = net_income / prev_net_income - 1
    if pd.isna(info.get("earningsGrowth")) and pd.notna(earn_growth):
        info["earningsGrowth"] = earn_growth

    if pd.isna(info.get("currentRatio")) and pd.notna(current_assets) and pd.notna(current_liab) and current_liab != 0:
        info["currentRatio"] = current_assets / current_liab
    if pd.isna(info.get("quickRatio")) and pd.notna(current_assets) and pd.notna(current_liab) and current_liab != 0:
        inv = inventory if pd.notna(inventory) else 0
        info["quickRatio"] = (current_assets - inv) / current_liab
    if pd.isna(info.get("debtToEquity")) and pd.notna(debt) and pd.notna(eq_for_roe) and eq_for_roe != 0:
        info["debtToEquity"] = debt / eq_for_roe * 100

    if pd.isna(info.get("operatingCashflow")) and pd.notna(operating_cf):
        info["operatingCashflow"] = operating_cf
    if pd.isna(info.get("freeCashflow")) and pd.notna(free_cf):
        info["freeCashflow"] = free_cf

    return info


# ---------- IO ----------
@st.cache_data(ttl=120, show_spinner=False)
def load_data(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="3y", auto_adjust=True)

    info = {}
    try:
        info = merge_info(info, getattr(t, "fast_info", {}) or {})
    except Exception:
        pass
    try:
        info = merge_info(info, t.get_info() or {})
    except Exception:
        pass
    try:
        info = merge_info(info, t.info or {})
    except Exception:
        pass

    info = derive_fundamentals_from_statements(t, info)
    info = extract_analyst_data(t, info)
    info = extract_earnings_data(t, info)

    try:
        info["_fund_fields_loaded"] = int(sum(pd.notna(normalize_missing(info.get(k))) for k in [
            "profitMargins", "operatingMargins", "grossMargins", "returnOnEquity", "returnOnAssets",
            "revenueGrowth", "earningsGrowth", "currentRatio", "quickRatio", "debtToEquity",
            "freeCashflow", "operatingCashflow", "forwardPE", "pegRatio", "priceToSalesTrailing12Months",
            "priceToBook", "beta", "shortPercentOfFloat", "recommendationMean",
            "numberOfAnalystOpinions", "targetMeanPrice"
        ]))
    except Exception:
        info["_fund_fields_loaded"] = 0

    return hist, info


@st.cache_data(ttl=120, show_spinner=False)
def load_benchmark_data(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y", auto_adjust=True)
        return hist
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def search_tickers(query, max_results=8):
    query = str(query or "").strip()
    if not query:
        return []

    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": max_results, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    results = []
    for item in data.get("quotes", []):
        symbol = item.get("symbol")
        name = item.get("shortname") or item.get("longname") or "-"
        exchange = item.get("exchange") or "-"
        quote_type = item.get("quoteType") or "-"
        exch_disp = item.get("exchDisp") or exchange

        if not symbol:
            continue
        if quote_type not in {"EQUITY", "ETF"}:
            continue

        results.append({
            "symbol": symbol,
            "name": name,
            "shortname": item.get("shortname"),
            "longname": item.get("longname"),
            "exchange": exch_disp,
            "type": quote_type,
            "label": f"{name} ({symbol}) - {exch_disp}"
        })

    seen = set()
    clean = []
    for r in results:
        if r["symbol"] not in seen:
            clean.append(r)
            seen.add(r["symbol"])
    return clean


def looks_like_real_ticker(user_input):
    raw = str(user_input or "").strip()
    if not raw:
        return False

    upper = raw.upper()

    if re.fullmatch(r"[A-Z0-9]{1,5}([.\-][A-Z0-9]{1,5})?", upper):
        # Wörter wie Apple, Siemens, Nvidia nicht blind als Ticker behandeln,
        # außer der Nutzer hat sie bewusst in echter Ticker-Schreibweise eingegeben.
        if raw.isalpha() and raw != upper:
            return False
        return True

    return False


def score_search_result(query, item):
    query = str(query or "").strip().lower()

    symbol = str(item.get("symbol", "") or "").strip().lower()
    name = str(item.get("name", "") or item.get("shortname", "") or item.get("longname", "") or "").strip().lower()
    exchange = str(item.get("exchange", "") or "").strip().lower()
    quote_type = str(item.get("type", "") or item.get("quoteType", "") or "").strip().upper()

    score = 0

    if query == symbol:
        score += 120
    if query == name:
        score += 110
    if query in name and name:
        score += 55
    if query in symbol and symbol:
        score += 35

    q_tokens = {t for t in re.split(r"\W+", query) if t}
    n_tokens = {t for t in re.split(r"\W+", name) if t}
    overlap = len(q_tokens & n_tokens)
    score += overlap * 12

    if quote_type == "EQUITY":
        score += 12
    elif quote_type == "ETF":
        score += 6

    if exchange in {"nasdaq", "nasdaqgs", "nyq", "nyse", "xetra", "par", "ams", "mil", "six", "lse"}:
        score += 4

    if not name:
        score -= 10

    return score


def resolve_input_to_ticker(user_input, fallback=None):
    user_input = str(user_input or "").strip()
    if not user_input:
        return fallback

    raw = user_input.strip()
    upper = raw.upper()

    if looks_like_real_ticker(raw):
        return upper

    results = search_tickers(raw, max_results=8)
    if results:
        ranked = sorted(results, key=lambda x: score_search_result(raw, x), reverse=True)
        best = ranked[0]
        symbol = best.get("symbol")
        if symbol:
            return str(symbol).upper()

    return fallback if fallback else None

def build_short_thesis(investment, tb_score, market_regime, top_red_flag, position_mode):
    if position_mode:
        if investment >= 78 and market_regime != "NEGATIV":
            txt = "Starke Position mit solidem Halte-/Ausbauprofil."
        elif investment >= 60:
            txt = "Ordentliche Position, aber mit Beobachtungsbedarf."
        else:
            txt = "Position aktuell verletzlich; aktives Risikomanagement wichtig."
    else:
        if investment >= 78 and market_regime == "POSITIV":
            txt = "Attraktiver Watchlist-/Einstiegskandidat."
        elif investment >= 60:
            txt = "Interessanter Kandidat, aber Timing noch nicht perfekt."
        else:
            txt = "Derzeit eher Beobachtung statt Aktion."

    if tb_score >= 9:
        txt += " TradingBoard bestätigt Stärke."
    elif tb_score < 3:
        txt += " TradingBoard ist klar defensiv."

    if isinstance(top_red_flag, str) and top_red_flag and top_red_flag != "-":
        txt += f" Wichtigste Bremse: {top_red_flag}."
    return txt


def build_ranking_table(results):
    rows = []
    for r in results:
        market_info = r.get("market_info", {}) or {}
        confidence_info = r.get("confidence_info", {}) or {}
        full_red_flag = r.get("top_red_flag", "-")
        full_thesis = r.get("short_thesis", r.get("decision_summary", "-"))

        rows.append({
            "Ticker": r.get("ticker", "-"),
            "Name": shorten_text(r.get("name", r.get("ticker", "-")), 28),
            "Benchmark": r.get("benchmark_label", "-"),
            "Marktregime": market_regime_label(market_info.get("regime", "UNBEKANNT")),
            "Company Quality": r.get("company", np.nan),
            "Setup Quality": r.get("setup_adj", np.nan),
            "Investment Score": r.get("investment", np.nan),
            "Investment-Attraktivität": r.get("investment_case_score", np.nan),
            "Trading-Attraktivität": r.get("trading_case_score", np.nan),
            "Tradeability": r.get("tradeability_score", np.nan),
            "Kurzfrist-Timing": r.get("tb_score_100", normalize_tb_score_100(r.get("tb_score", np.nan))),
            "Fundamental-Confidence": round(confidence_info.get("coverage", 0) * 100),
            "Top Red Flag": shorten_text(full_red_flag, 34),
            "Kurzfazit": shorten_text(full_thesis, 52),
            "_Top Red Flag Full": full_red_flag if full_red_flag else "-",
            "_Kurzfazit Full": full_thesis if full_thesis else "-",
        })

    df = pd.DataFrame(rows)

    if "_Top Red Flag Full" not in df.columns:
        df["_Top Red Flag Full"] = df.get("Top Red Flag", "-")
    if "_Kurzfazit Full" not in df.columns:
        df["_Kurzfazit Full"] = df.get("Kurzfazit", "-")

    if not df.empty:
        df = df.sort_values(
            by=["Investment-Attraktivität", "Trading-Attraktivität", "Tradeability"],
            ascending=False
        ).reset_index(drop=True)
        df.index = df.index + 1
    return df

def compute_chart_df(df, chart_range):
    if chart_range == "3 Monate":
        chart_df = df.tail(63).copy()
    elif chart_range == "6 Monate":
        chart_df = df.tail(126).copy()
    elif chart_range == "1 Jahr":
        chart_df = df.tail(252).copy()
    else:
        chart_df = df.copy()
    chart_df["MA20"] = chart_df["Close"].rolling(20).mean()
    chart_df["MA50"] = chart_df["Close"].rolling(50).mean()
    chart_df["MA200"] = chart_df["Close"].rolling(200).mean()
    return chart_df


def analyze_stock(
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

    if df.empty or len(df) < 220:
        raise ValueError("Nicht genug Kursdaten fuer belastbare Analyse. Prüfe den ausgewählten Ticker.")

    benchmark_symbol, benchmark_label = select_benchmark(ticker, info)
    benchmark_df = load_benchmark_data(benchmark_symbol)
    market_info = evaluate_market_filter(benchmark_df)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    price = float(override) if override > 0 else float(close.iloc[-1])
    name = info.get("longName", ticker)
    raw_ccy = info.get("currency", "USD")
    ccy = infer_display_currency(ticker, info, raw_ccy)
    exch = info.get("exchange", "-")
    ts = df.index[-1].strftime("%d.%m.%Y")
    sector = info.get("sector", "-")
    industry = info.get("industry", "-")

    company_summary = info.get("longBusinessSummary", "")
    if not company_summary:
        company_summary = "Keine Unternehmensbeschreibung verfügbar."

    confidence_info = infer_data_source_flags(info)

    # ---------- Technicals ----------
    ma20_series = close.rolling(20).mean()
    ma50_series = close.rolling(50).mean()
    ma150_series = close.rolling(150).mean()
    ma200_series = close.rolling(200).mean()

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
    red_flag_notes = [f"{x['Kategorie']}: {x['Detail']}" for x in red_flag_items]

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

    investment = round(clamp(setup_adj * ws + company * wc))

    if pd.notna(prev20_high) and price > prev20_high:
        setup_type = "Breakout"
        preferred_entry = "Breakout über 20T-Hoch"
    elif price > ma50 and pd.notna(ma20) and abs(price - ma20) / price <= 0.03:
        setup_type = "Pullback im Aufwärtstrend"
        preferred_entry = "Pullback nahe MA20 / Trendfortsetzung"
    elif price > ma200 and rsi < 40:
        setup_type = "Rebound im Aufwärtstrend"
        preferred_entry = "Rebound nach Stabilisierung"
    elif price > ma50 and price > ma200:
        setup_type = "Trendfolge"
        preferred_entry = "Trendfolge bei Rücksetzer"
    else:
        setup_type = "Kein sauberes Setup"
        preferred_entry = "Aktuell kein sauberer Einstieg"

    valid_trade_setup = (
        investment >= 60
        and setup_adj >= 55
        and kb >= 2
        and market_info["regime"] != "NEGATIV"
        and not (has_upcoming_earnings and pd.notna(days_earn) and days_earn < 7)
    )

    # ---------- Trade Setup ----------
    if valid_trade_setup:
        atr_stop = round(price - 1.8 * atr, 2)
        struct_stop = round(ma50 * 0.965, 2)
        stop_used = min(atr_stop, struct_stop)
        stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0
        if stop_used <= 0 or stop_used >= price:
            stop_used = round(price - max(price * 0.08, atr * 1.8), 2)
            stop_dist = (price - stop_used) / price * 100 if price > stop_used else 0

        risk_per_share = price - stop_used

        tp1 = round(price + 1 * risk_per_share, 2)
        tp1_source = "1R vom Stop"

        if pd.notna(target) and target > price:
            tp2 = round(target, 2)
            tp2_source = "Analysten-Target"
        elif pd.notna(high52) and high52 > price:
            tp2 = round(high52, 2)
            tp2_source = "52W-Hoch"
        else:
            tp2 = round(price + 2 * risk_per_share, 2)
            tp2_source = "2R-Fallback"

        if pd.notna(high52) and high52 > tp2:
            tp3 = round(high52, 2)
            tp3_source = "52W-Hoch"
        else:
            tp3 = round(max(price + 3 * risk_per_share, tp2 + risk_per_share), 2)
            tp3_source = "Erweitertes R-Ziel"

        if setup_type == "Breakout":
            anchor_price = prev20_high if pd.notna(prev20_high) else price
            entry_low = max(anchor_price, price * 0.995)
            entry_high = max(anchor_price * 1.015, price * 1.005)
            entry_source = "Breakout-Zone über dem Ausbruchsniveau"
        elif setup_type == "Pullback im Aufwärtstrend":
            anchor_price = ma20 if pd.notna(ma20) else ma50
            entry_low = anchor_price * 0.99 if pd.notna(anchor_price) else price * 0.98
            entry_high = anchor_price * 1.01 if pd.notna(anchor_price) else price
            entry_source = "Pullback-Zone nahe MA20 / Trendlinie"
        elif setup_type == "Rebound im Aufwärtstrend":
            anchor_price = prev20_low if pd.notna(prev20_low) else ma20
            entry_low = anchor_price * 1.00 if pd.notna(anchor_price) else price * 0.98
            entry_high = anchor_price * 1.03 if pd.notna(anchor_price) else price
            entry_source = "Rebound-Zone nach Stabilisierung"
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

        crv = (tp2 - price) / (price - stop_used) if (price - stop_used) > 0 else 0
        timing_trade_score = round(clamp(s4 * 0.45 + s5 * 0.25 + rs_score * 0.20 + s6 * 0.10))
        stop_score = ideal_range_score(stop_dist, ideal_low=3.0, ideal_high=7.5, hard_low=1.0, hard_high=13.0)
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
        suggested_entry_zone = "-"
        entry_source = "-"
        entry_quality = "-"
        crv = np.nan
        tradeability_score = np.nan
        tradeability_text = "-"
        crv_score = np.nan
        stop_score = np.nan
        entry_score = np.nan
        timing_trade_score = np.nan
        market_trade_score = np.nan
        setup_confidence = np.nan
        setup_confidence_text = "-"
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
        trading_case_text = trading_case_label(trading_case_score)
        risk_eur = depot * (risk_pct / 100)
        pos_size = 0
        time_stop = "-"

    short_term_score = round(clamp(s4 * 0.45 + s5 * 0.28 + s6 * 0.17 + rs_score * 0.10))
    swing_score = round(clamp(s3 * 0.26 + s4 * 0.28 + s5 * 0.16 + s6 * 0.10 + rs_score * 0.12 + w52 * 0.08))
    mid_term_score = round(clamp(setup_adj * 0.55 + company * 0.45))
    long_term_score = round(clamp(company * 0.50 + growth_score * 0.15 + growth_quality * 0.10 + quality_score * 0.15 + valuation_score * 0.10))
    very_long_term_score = round(clamp(company * 0.40 + quality_score * 0.20 + growth_score * 0.15 + growth_quality * 0.10 + valuation_score * 0.15))

    hmap = {
        "Kurzfrist": short_term_score,
        "Swing": swing_score,
        "Mittelfrist": mid_term_score,
        "Langfrist": long_term_score,
        "Sehr langfristig": very_long_term_score,
    }

    position_mode = buy_in_override > 0

    # ---------- Recommendations ----------
    if has_upcoming_earnings and days_earn < 7:
        emp, conv = ("VETO - Earnings < 7 Tage", "-")
    elif position_mode:
        if investment >= 78 and kb >= 3 and market_info["regime"] != "NEGATIV":
            emp, conv = ("HALTEN / AUSBAUEN", "HIGH")
        elif investment >= 65:
            emp, conv = ("HALTEN / ENGE BEOBACHTUNG", "MEDIUM")
        elif investment >= 52:
            emp, conv = ("HALTEN / RISIKO PRÜFEN", "LOW-MEDIUM")
        else:
            emp, conv = ("RISIKO REDUZIEREN / STOPP PRÜFEN", "LOW")
    else:
        if investment >= 78 and kb >= 3 and market_info["regime"] == "POSITIV":
            emp, conv = ("BUY / ACCUMULATE", "HIGH")
        elif investment >= 68:
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
        tb_details.append("S2: Über MA200 ✓")
    else:
        tb_details.append("S2: Unter MA200 ❌")

    if price > ma50:
        tb_score += 1
        tb_details.append("S3: Über MA50 (+1) ✓")
    else:
        tb_score -= 1
        tb_details.append("S3: Unter MA50 (-1) ❌")

    if ma50 > ma200:
        tb_score += 1
        tb_details.append("S4: Golden Cross ✓")
    else:
        tb_details.append("S4: Trendstruktur schwach ❌")

    if 40 < rsi < 60 or rsi < 30:
        tb_score += 1
        tb_details.append("S5: RSI konstruktiv ✓")
    else:
        tb_details.append("S5: RSI hoch/niedrig ❌")

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
        tb_context.append("S8: Vola ok ✓")

    if macd_bull_cross:
        tb_context.append("S9: MACD Bull-Cross! 🚀")

    if smart_money_default:
        tb_context.append("S10: Smart Money sammelt ein ✓")
    else:
        tb_context.append("S10: Smart Money verkauft ❌")

    if adx > 25:
        tb_context.append("S11: ADX>25 starker Trend ✓")
    else:
        tb_context.append("S11: ADX<25 Seitwärts ❌")

    if stoch_k_v < 20 and stoch_d_v < 20 and stoch_k_v > stoch_d_v:
        tb_context.append("S12: Stoch Oversold Cross ✓")
    elif stoch_k_v > 80:
        tb_context.append("S12: Stoch überkauft ❌")
    else:
        tb_context.append("S12: Stoch neutral ❌")

    if willr_v < -80:
        tb_context.append("S13: Williams%R extrem Oversold ✓")
    elif willr_v > -20:
        tb_context.append("S13: Williams%R überkauft ❌")
    else:
        tb_context.append("S13: Williams%R neutral ❌")

    if obv_trend == "steigend" and vol_ratio >= 1.0:
        tb_context.append("S14: OBV/Volumen bestätigt ✓")
    else:
        tb_context.append("S14: OBV/Volumen schwach ❌")

    if pd.notna(prev20_high) and price > prev20_high:
        tb_context.append("S15: 20D Breakout ✓")
    elif pd.notna(prev20_low) and price < prev20_low:
        tb_context.append("S15: 20D Breakdown ❌")
    else:
        tb_context.append("S15: Range intakt ❌")

    if pd.notna(bb_upper) and price > bb_upper:
        tb_context.append("S16: BB Breakout UP ✓")
    elif bb_squeeze:
        tb_context.append("S16: BB Squeeze Achtung ✓")
    elif pd.notna(bb_lower) and price < bb_lower:
        tb_context.append("S16: BB Breakout DOWN ❌")
    else:
        tb_context.append("S16: BB neutral ❌")

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
        tb_context.append("S20: kein Short-Squeeze-Signal ❌")

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

    rows = []
    for line in tb_details:
        if ": " in line:
            k, v = line.split(": ", 1)
            rows.append({"Punkt": k, "Detail": v})
        else:
            rows.append({"Punkt": "Info", "Detail": line})
    tb_df = pd.DataFrame(rows)

    red_flags_df = pd.DataFrame(red_flag_items) if red_flag_items else pd.DataFrame(
        [{"Kategorie": "-", "Status": "🟢", "Detail": "Keine relevanten Red Flags erkannt", "Penalty": 0}]
    )

    top_red_flag = red_flag_items[0]["Detail"] if red_flag_items else "-"
    short_thesis = build_short_thesis(investment, tb_score, market_info["regime"], top_red_flag, position_mode)

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
        "benchmark_symbol": benchmark_symbol,
        "benchmark_label": benchmark_label,
        "price": price,
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
        "tb_signal": tb_signal,
        "tb_empf": tb_empf,
        "tb_df": tb_df,
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
        "ma20": ma20,
        "ma50": ma50,
        "ma150": ma150,
        "ma200": ma200,
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
        "emp": emp,
        "conv": conv,
    }


# ---------- Sidebar ----------
with st.sidebar:
    st.title(f"📊 Capital-Hill-Score-Modell {APP_VERSION}")
    st.caption(f"{APP_VERSION} | v9.0 startet mit Investment-Case, Trading-Case, Setup-Confidence und Executive Summary")
    st.divider()

    st.markdown("### 1) Was möchtest du analysieren?")
    analysis_mode = st.radio(
        "Analyseart",
        ["Einzelanalyse", "Mehrfach-Ranking"],
        horizontal=True,
        index=0 if st.session_state.analysis_mode == "Einzelanalyse" else 1,
        key="analysis_mode_widget"
    )
    st.session_state.analysis_mode = analysis_mode

    search_results = []
    ticker = st.session_state.selected_ticker

    if analysis_mode == "Einzelanalyse":
        search_input = st.text_input(
            "Unternehmen oder Ticker",
            value=st.session_state.search_input,
            placeholder="z. B. Apple, Siemens, AAPL, SAP.DE",
            key="search_input_widget"
        ).strip()
        st.session_state.search_input = search_input
        st.caption("Beispiele: Apple, Siemens, AAPL, SAP.DE")

        if search_input:
            looks_like_ticker = looks_like_real_ticker(search_input)

            if looks_like_ticker:
                ticker = search_input.upper()
                st.session_state.selected_ticker = ticker
                st.session_state.selected_search_label = None
            else:
                search_results = search_tickers(search_input)

                if len(search_results) == 1:
                    ticker = search_results[0]["symbol"]
                    st.session_state.selected_ticker = ticker
                    st.session_state.selected_search_label = search_results[0]["label"]
                    st.caption(f"Automatisch gefunden: {search_results[0]['label']}")
                elif len(search_results) > 1:
                    labels = [r["label"] for r in search_results]

                    if st.session_state.selected_search_label not in labels:
                        st.session_state.selected_search_label = labels[0]

                    selected_label = st.selectbox(
                        "Passenden Treffer auswählen",
                        options=labels,
                        index=labels.index(st.session_state.selected_search_label),
                        key="search_result_select"
                    )

                    st.session_state.selected_search_label = selected_label
                    ticker = next(r["symbol"] for r in search_results if r["label"] == selected_label)
                    st.session_state.selected_ticker = ticker
                else:
                    st.warning("Kein passender Ticker gefunden. Bitte Namen präzisieren oder den Ticker direkt eingeben.")
                    ticker = st.session_state.selected_ticker
        else:
            ticker = st.session_state.selected_ticker

        st.caption(f"Aktuelle Auswahl: {st.session_state.selected_ticker}")

    else:
        batch_input = st.text_area(
            "Mehrere Unternehmen oder Ticker",
            value=st.session_state.batch_input,
            placeholder="z. B.\nApple\nMicrosoft\nSAP.DE\nASML",
            height=140,
            key="batch_input_widget"
        )
        st.session_state.batch_input = batch_input
        st.caption("Ein Wert pro Zeile oder getrennt durch Komma bzw. Semikolon.")

    st.divider()
    st.markdown("### 2) In welchem Kontext bewerten?")
    horizon = st.selectbox(
        "Geplanter Zeithorizont",
        [
            "Kurzfrist (1-7 Tage)",
            "Swing (1-4 Wochen)",
            "Mittelfrist (1-3 Monate)",
            "Langfrist (1-2 Jahre)",
            "Sehr langfristig (2+ Jahre)",
        ],
    )

    context_choice = st.radio(
        "Analysekontext",
        ["Beobachtung", "Bestehende Position"],
        horizontal=True,
        index=1 if st.session_state.last_mode_label == "Position" else 0,
        key="context_choice_widget"
    )

    if context_choice == "Bestehende Position":
        buy_in_override = st.number_input(
            "Dein Einstiegskurs",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            help="Nur ausfüllen, wenn du die Aktie bereits hältst."
        )
        mode_label = "Position"
    else:
        buy_in_override = 0.0
        mode_label = "Watchlist"
        st.caption("Kontext: Beobachtung / noch nicht investiert")

    position_mode = buy_in_override > 0
    st.session_state.last_mode_label = mode_label

    st.divider()
    st.markdown("### 3) Risikoeinstellungen")
    depot = st.number_input("Depotwert in EUR", min_value=1000, value=10000, step=1000)
    risk_pct = st.slider("Maximales Risiko pro Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    with st.expander("Erweiterte Einstellungen", expanded=False):
        override = st.number_input(
            "Kurs-Override (optional)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            help="Nur verwenden, wenn du einen anderen Einstiegskurs simulieren möchtest."
        )
        smart_money_default = st.checkbox("Smart-Money-Signal standardmäßig positiv", value=True)
        strict_mode = st.checkbox("Strenges 23-Säulen-Mapping", value=True)

        if st.button("Cache leeren", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache geleert. Bitte Analyse neu starten.")

    st.divider()
    run_analysis = st.button("Analyse starten", use_container_width=True, type="primary")
    if run_analysis:
        st.session_state.analysis_mode_run = analysis_mode
        st.session_state.analysis_requested = True
        if analysis_mode == "Einzelanalyse":
            st.session_state.analysis_ticker = st.session_state.selected_ticker
        else:
            st.session_state.analysis_ticker = st.session_state.selected_ticker


# ---------- Main ----------
st.title(f"📊 Capital-Hill-Score-Modell {APP_VERSION}")
st.caption(
    "Core-Modell und TradingBoard werden getrennt gerechnet. "
    "Zusätzlich sind Candlestick-Chart mit Volumen, Fundamental-Confidence, Multi-Ranking "
    "und stil-/sektorabhängige Feinkalibrierung integriert."
)

if not st.session_state.analysis_requested:
    st.info("Wähle links eine Analyseart, gib Unternehmen oder Ticker ein und starte dann die Analyse.")
    st.stop()

# ---------- Batch / Ranking Run ----------
analysis_mode_run = st.session_state.get("analysis_mode_run", "Einzelanalyse")

if analysis_mode_run == "Einzelanalyse":
    entries = [st.session_state.analysis_ticker]
else:
    entries = split_batch_input(st.session_state.batch_input)

if not entries:
    entries = [st.session_state.analysis_ticker]

resolved_entries = []
resolved_input_rows = []
for e in entries:
    fallback = st.session_state.analysis_ticker if analysis_mode_run == "Einzelanalyse" else None
    resolved = resolve_input_to_ticker(e, fallback=fallback)
    resolved_input_rows.append({
        "Eingabe": e,
        "Aufgelöst zu": resolved if resolved else "Nicht gefunden"
    })
    if resolved and resolved not in resolved_entries:
        resolved_entries.append(resolved)

results = []
errors = []

progress = st.progress(0)
status = st.empty()

for i, tkr in enumerate(resolved_entries, start=1):
    status.info(f"Analysiere {tkr} ({i}/{len(resolved_entries)}) …")
    try:
        result = analyze_stock(
            ticker=tkr,
            horizon=horizon,
            depot=depot,
            risk_pct=risk_pct,
            override=override if analysis_mode_run == "Einzelanalyse" and len(resolved_entries) == 1 else 0.0,
            buy_in_override=buy_in_override if analysis_mode_run == "Einzelanalyse" and len(resolved_entries) == 1 else 0.0,
            smart_money_default=smart_money_default,
            strict_mode=strict_mode
        )
        results.append(result)
    except Exception as e:
        errors.append((tkr, str(e)))
    progress.progress(i / len(resolved_entries))

progress.empty()
status.empty()

if not results:
    st.error("Keine belastbaren Ergebnisse. Prüfe die Eingaben.")
    if errors:
        st.write(pd.DataFrame(errors, columns=["Ticker", "Fehler"]))
    st.stop()

if analysis_mode_run == "Einzelanalyse":
    st.caption("Aktiver Modus: Einzelanalyse")
else:
    st.caption("Aktiver Modus: Mehrfach-Ranking")

ranking_df = build_ranking_table(results)
results_map = {r["ticker"]: r for r in results}
st.session_state.ranking_df = ranking_df
st.session_state.ranking_results = results_map

if st.session_state.selected_ranking_ticker not in results_map and not ranking_df.empty:
    st.session_state.selected_ranking_ticker = ranking_df.iloc[0]["Ticker"]

# Prefer explicit single-analysis ticker if present in results
if st.session_state.analysis_ticker in results_map:
    selected_display_ticker = st.session_state.analysis_ticker
else:
    selected_display_ticker = st.session_state.selected_ranking_ticker

if selected_display_ticker not in results_map and not ranking_df.empty:
    selected_display_ticker = ranking_df.iloc[0]["Ticker"]

# ---------- Ranking Section ----------
st.subheader("Ranking mehrerer Aktien")
st.caption(
    "Ranking-Tabelle für Multi-Screening. In der Einzelanalyse wird automatisch nur ein Wert verarbeitet. "
    "Red Flag und Kurzfazit stehen darunter in einer Kartenansicht vollständig und lesbar."
)

ranking_display_cols = [
    "Ticker",
    "Name",
    "Benchmark",
    "Marktregime",
    "Company Quality",
    "Setup Quality",
    "Investment Score",
    "Investment-Attraktivität",
    "Trading-Attraktivität",
    "Tradeability",
    "Kurzfrist-Timing",
    "Fundamental-Confidence",
]

ranking_display_df = ranking_df.copy()

if "Kurzfrist-Timing" not in ranking_display_df.columns and "TradingBoard Score" in ranking_display_df.columns:
    ranking_display_df = ranking_display_df.rename(columns={"TradingBoard Score": "Kurzfrist-Timing"})

available_ranking_cols = [c for c in ranking_display_cols if c in ranking_display_df.columns]
ranking_display_df = ranking_display_df[available_ranking_cols].copy()

ranking_column_config = {
    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
    "Name": st.column_config.TextColumn("Name", width="medium"),
    "Benchmark": st.column_config.TextColumn("Benchmark", width="small"),
    "Marktregime": st.column_config.TextColumn("Marktregime", width="medium"),
    "Company Quality": st.column_config.NumberColumn("Company Quality", width="small", format="%.0f"),
    "Setup Quality": st.column_config.NumberColumn("Setup Quality", width="small", format="%.0f"),
    "Investment Score": st.column_config.NumberColumn("Investment Score", width="small", format="%.0f"),
    "Investment-Attraktivität": st.column_config.NumberColumn("Investment-Attraktivität", width="small", format="%.0f"),
    "Trading-Attraktivität": st.column_config.NumberColumn("Trading-Attraktivität", width="small", format="%.0f"),
    "Tradeability": st.column_config.NumberColumn("Tradeability", width="small", format="%.0f"),
    "Kurzfrist-Timing": st.column_config.NumberColumn("Kurzfrist-Timing", width="small", format="%.0f"),
    "Fundamental-Confidence": st.column_config.NumberColumn("Fundamental-Confidence", width="small", format="%.0f"),
}
ranking_column_config = {k: v for k, v in ranking_column_config.items() if k in ranking_display_df.columns}

st.dataframe(
    style_ranking_df(ranking_display_df),
    hide_index=False,
    use_container_width=True,
    height=420,
    column_config=ranking_column_config,
)

st.download_button(
    "Ranking als CSV exportieren",
    data=ranking_df.drop(columns=["_Top Red Flag Full", "_Kurzfazit Full"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
    file_name=f"capital_hill_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

if not ranking_df.empty:
    st.markdown("**Ranking-Details**")

    card_df = ranking_df.copy()
    if "_Top Red Flag Full" not in card_df.columns:
        card_df["_Top Red Flag Full"] = card_df.get("Top Red Flag", "-")
    if "_Kurzfazit Full" not in card_df.columns:
        card_df["_Kurzfazit Full"] = card_df.get("Kurzfazit", "-")

    for _, row in card_df.iterrows():
        score_val = row.get("Investment Score", np.nan)
        score_text, score_color = score_badge(score_val if pd.notna(score_val) else 0)
        st.markdown(
            f"""
<div class="section-card" style="margin:12px 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
    <div>
      <div style="font-size:1.08rem;font-weight:800;color:#f8fafc;">{row.get("Ticker", "-")} — {row.get("Name", "-")}</div>
      <div style="color:#94a3b8;font-size:0.90rem;margin-top:3px;">
        Benchmark: {row.get("Benchmark", "-")} | Marktumfeld: {row.get("Marktregime", "-")}
      </div>
    </div>
    <div style="background:{score_color};color:white;border-radius:999px;padding:7px 13px;font-weight:800;box-shadow:0 8px 18px rgba(0,0,0,0.18);">
      {score_text}: {row.get("Investment Score", "n/a")}
    </div>
  </div>

  <div style="margin-top:12px;color:#d1d5db;font-size:0.92rem;line-height:1.5;">
    <b>Scores:</b>
    Company {row.get("Company Quality", "n/a")} |
    Setup {row.get("Setup Quality", "n/a")} |
    Tradeability {row.get("Tradeability", "n/a")} |
    Kurzfrist-Timing {row.get("Kurzfrist-Timing", row.get("TradingBoard Score", "n/a"))}
  </div>

  <div style="margin-top:14px;padding:12px 14px;border-radius:14px;background:#0b1220;border:1px solid #1f2937;">
    <div style="font-weight:800;margin-bottom:5px;color:#f8fafc;">Red Flag</div>
    <div style="color:#e5e7eb;line-height:1.5;">{row.get("_Top Red Flag Full", "-")}</div>
  </div>

  <div style="margin-top:12px;padding:12px 14px;border-radius:14px;background:#0b1220;border:1px solid #1f2937;">
    <div style="font-weight:800;margin-bottom:5px;color:#f8fafc;">Kurzfazit</div>
    <div style="color:#e5e7eb;line-height:1.5;">{row.get("_Kurzfazit Full", "-")}</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

try:
    resolved_df = pd.DataFrame(resolved_input_rows)
    if not resolved_df.empty:
        with st.expander("Aufgelöste Eingaben", expanded=False):
            st.dataframe(resolved_df, hide_index=True, use_container_width=True)
except Exception:
    pass

sel_col1, sel_col2 = st.columns([2, 1])
with sel_col1:
    selected_display_ticker = st.selectbox(
        "Einzelanalyse aus Ranking auswählen",
        options=ranking_df["Ticker"].tolist(),
        index=ranking_df["Ticker"].tolist().index(selected_display_ticker),
        key="selected_ranking_ticker_widget"
    )
    st.session_state.selected_ranking_ticker = selected_display_ticker

with sel_col2:
    st.metric("Analysierte Werte", len(results))

if errors:
    with st.expander("Nicht analysierbare Eingaben", expanded=False):
        st.dataframe(pd.DataFrame(errors, columns=["Ticker", "Fehler"]), hide_index=True, use_container_width=True)

result = results_map[selected_display_ticker]
ticker = result["ticker"]
df = result["df"]
info = result["info"]
name = result["name"]
ccy = result["ccy"]
exch = result["exch"]
ts = result["ts"]
sector = result["sector"]
industry = result["industry"]
company_summary = result["company_summary"]
confidence_info = result["confidence_info"]
benchmark_label = result["benchmark_label"]
benchmark_symbol = result["benchmark_symbol"]
market_info = result["market_info"]
price = result["price"]
target = result["target"]
upside = result["upside"]
regime = result["regime"]
reg_amp = result["reg_amp"]
sg_earn = result["sg_earn"]
sg_earn_txt = result["sg_earn_txt"]
days_earn = result["days_earn"]
has_upcoming_earnings = result["has_upcoming_earnings"]
has_past_earnings = result["has_past_earnings"]
fund_cov = result["fund_cov"]
fund_fields_loaded = result["fund_fields_loaded"]
fund_data_warning = result["fund_data_warning"]
red_flag_items = result["red_flag_items"]
red_flags_df = result["red_flags_df"]
red_flag_notes = result["red_flag_notes"]
red_flag_penalty_total = result["red_flag_penalty_total"]
quality_score = result["quality_score"]
growth_score = result["growth_score"]
growth_quality = result["growth_quality"]
valuation_score = result["valuation_score"]
balance_score = result["balance_score"]
sentiment_score = result["sentiment_score"]
risk_score = result["risk_score"]
company = result["company"]
setup_adj = result["setup_adj"]
investment = result["investment"]
tb_score = result["tb_score"]
tb_score_100 = result["tb_score_100"]
tb_timing_text = result["tb_timing_text"]
tb_signal = result["tb_signal"]
tb_empf = result["tb_empf"]
tb_df = result["tb_df"]
tb_details = result["tb_details"]
tb_context = result["tb_context"]
stb_score = result["stb_score"]
stb_signal = result["stb_signal"]
stb_empf = result["stb_empf"]
stb_text = result["stb_text"]
kb = result["kb"]
strengths = result["strengths"]
weaknesses = result["weaknesses"]
decision_summary = result["decision_summary"]
mode_label = result["mode_label"]
stock_style = result["stock_style"]
hmap = result["hmap"]
atr_stop = result["atr_stop"]
stop_used = result["stop_used"]
stop_dist = result["stop_dist"]
tp1 = result["tp1"]
tp2 = result["tp2"]
tp3 = result["tp3"]
tp1_source = result["tp1_source"]
tp2_source = result["tp2_source"]
tp3_source = result["tp3_source"]
suggested_entry_zone = result["suggested_entry_zone"]
entry_source = result["entry_source"]
entry_quality = result["entry_quality"]
tradeability_score = result["tradeability_score"]
tradeability_text = result["tradeability_text"]
investment_case_score = result["investment_case_score"]
investment_case_text = result["investment_case_text"]
trading_case_score = result["trading_case_score"]
trading_case_text = result["trading_case_text"]
setup_type = result["setup_type"]
preferred_entry = result["preferred_entry"]
setup_confidence = result["setup_confidence"]
setup_confidence_text = result["setup_confidence_text"]
trade_crv_score = result["trade_crv_score"]
trade_stop_score = result["trade_stop_score"]
trade_entry_score = result["trade_entry_score"]
trade_timing_score = result["trade_timing_score"]
trade_market_score = result["trade_market_score"]
crv = result["crv"]
pos_size = result["pos_size"]
risk_eur = result["risk_eur"]
time_stop = result["time_stop"]
valid_trade_setup = result["valid_trade_setup"]
short_term_score = result["short_term_score"]
s3 = result["s3"]
s3a = result["s3a"]
s3t = result["s3t"]
s4 = result["s4"]
s4a = result["s4a"]
s4t = result["s4t"]
s5 = result["s5"]
s5a = result["s5a"]
s5t = result["s5t"]
s6 = result["s6"]
s6a = result["s6a"]
s6t = result["s6t"]
w52 = result["w52"]
dist52 = result["dist52"]
rs_score = result["rs_score"]
rs_composite = result["rs_composite"]
ret21 = result["ret21"]
ret63 = result["ret63"]
ret126 = result["ret126"]
bench_ret21 = result["bench_ret21"]
bench_ret63 = result["bench_ret63"]
bench_ret126 = result["bench_ret126"]
rs_vs_benchmark_21 = result["rs_vs_benchmark_21"]
rs_vs_benchmark_63 = result["rs_vs_benchmark_63"]
rs_vs_benchmark_126 = result["rs_vs_benchmark_126"]
ma20 = result["ma20"]
ma50 = result["ma50"]
ma150 = result["ma150"]
ma200 = result["ma200"]
rsi = result["rsi"]
macd_v = result["macd_v"]
signal_v = result["signal_v"]
macd_hist_current = result["macd_hist_current"]
adx = result["adx"]
atr = result["atr"]
atr_pct = result["atr_pct"]
stoch_k_v = result["stoch_k_v"]
stoch_d_v = result["stoch_d_v"]
willr_v = result["willr_v"]
roc20 = result["roc20"]
roc60 = result["roc60"]
high52 = result["high52"]
low52 = result["low52"]
profit_margin = result["profit_margin"]
oper_margin = result["oper_margin"]
gross_margin = result["gross_margin"]
roe = result["roe"]
revenue_growth = result["revenue_growth"]
earnings_growth = result["earnings_growth"]
current_ratio = result["current_ratio"]
quick_ratio = result["quick_ratio"]
debt_to_equity = result["debt_to_equity"]
pe = result["pe"]
peg = result["peg"]
ps = result["ps"]
pb = result["pb"]
rec_label = result["rec_label"]
analysts = result["analysts"]
rec_mean = result["rec_mean"]
beta = result["beta"]
short_pct = result["short_pct"]
market_cap = result["market_cap"]
short_thesis = result["short_thesis"]
top_red_flag = result["top_red_flag"]

# ---------- Header ----------
st.markdown(f"## {name} `{ticker}` — {exch} ({ccy})")
st.markdown(
    f"<div class='small-note'>Sektor: {sector} | Industrie: {industry} | Stil: {stock_style} | "
    f"Kontext: {display_mode_label(mode_label)} | Benchmark: {benchmark_label} | Marktumfeld: {market_regime_label(market_info['regime'])} | "
    f"Top Red Flag: {top_red_flag}</div>",
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(
        f"""
        <div class="premium-card">
            <div class="premium-title">Kurs (Adj. Close)</div>
            <div class="premium-value">{price:.2f} {ccy}</div>
            <div class="premium-sub">{ts}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="premium-card">
            <div class="premium-title">Trend-Regime</div>
            <div class="premium-value">{regime}</div>
            <div class="premium-sub">{reg_amp}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="premium-card">
            <div class="premium-title">Earnings-Datum</div>
            <div class="premium-value">{sg_earn_txt}</div>
            <div class="premium-sub">{sg_earn}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    if has_upcoming_earnings:
        countdown_value = f"{int(days_earn)}d"
    elif has_past_earnings:
        countdown_value = "vorbei"
    else:
        countdown_value = "kein Datum"
    st.markdown(
        f"""
        <div class="premium-card">
            <div class="premium-title">Earnings-Countdown</div>
            <div class="premium-value">{countdown_value}</div>
            <div class="premium-sub">{sg_earn}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        f"""
        <div class="premium-card">
            <div class="premium-title">Analysten-Target</div>
            <div class="premium-value">{fmt_num(target, 2, f" {ccy}")}</div>
            <div class="premium-sub">{fmt_num(upside, 1, "%")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

if fund_data_warning:
    st.warning(
        f"Fundamentaldaten nur teilweise geladen ({fund_cov*100:.0f}% Abdeckung, {fund_fields_loaded}/21 Felder). "
        "Der Company Quality Score kann dadurch zu niedrig ausfallen."
    )
elif fund_cov < 0.55:
    st.info(
        f"Fundamentaldaten teilweise vorhanden ({fund_cov*100:.0f}% Abdeckung, {fund_fields_loaded}/21 Felder)."
    )

if red_flag_items:
    st.warning("Red Flags erkannt: " + " | ".join(red_flag_notes[:4]))

# ---------- Scores ----------
st.subheader("Executive Summary")
sx1, sx2, sx3, sx4, sx5 = st.columns(5)
with sx1:
    render_score_card("Investment-Attraktivität", f"{investment_case_score}/100", investment_case_text, "investment")
with sx2:
    render_score_card("Trading-Attraktivität", f"{trading_case_score}/100", trading_case_text, "board")
with sx3:
    render_score_card("Setup-Typ", setup_type, preferred_entry, "setup")
with sx4:
    render_score_card("Setup-Confidence", f"{fmt_num(setup_confidence,0)}/100", setup_confidence_text, "helper")
with sx5:
    action_label = display_emp_label(result.get("emp", "-"))
    render_score_card("Handlung", action_label, market_regime_label(market_info["regime"]), "kb")

st.subheader("Scores")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
with c1:
    render_score_card("Company Quality", f"{company}/100", ampel(company), "company")
with c2:
    render_score_card("Setup Quality", f"{setup_adj}/100", ampel(setup_adj), "setup")
with c3:
    render_score_card("Investment Score", f"{investment}/100", ampel(investment), "investment")
with c4:
    render_score_card("Tradeability", f"{fmt_num(tradeability_score,0)}" + "/100", tradeability_text, "kb")
with c5:
    render_score_card("Kurzfrist-Timing", f"{tb_score_100}/100", f"{tb_timing_text} | Board: {tb_score} Punkte", "board")
with c6:
    render_score_card("Kurzfrist Core", f"{short_term_score}/100", ampel(short_term_score), "short")
with c7:
    render_score_card("Setup-Confidence", f"{fmt_num(setup_confidence,0)}/100", setup_confidence_text, "helper")

st.caption(
    "Diese Version ergänzt Candlestick-Chart mit Volumen, Ranking mehrerer Aktien, "
    "stil-/sektorabhängige Feinkalibrierung und eine transparentere Einschätzung der Qualität der Fundamentaldaten."
)

st.markdown("### Scores verständlich erklärt")
st.markdown(
    "- **Company Quality** bewertet Profitabilität, Wachstum, Bilanz, Bewertung, Sentiment und Risiko.\n"
    "- **Setup Quality** bewertet das technische Gesamtbild. Zusätzlich fließt ein kleiner Marktfilter ein.\n"
    "- **Kurzfrist Core** bewertet die kurzfristige technische Lage.\n"
    "- **Kurzfristiges Signalbild** ist eine ergänzende Kurzfrist-Ampel.\n"
    "- **Investment Score** ist die Gesamtbewertung aus technischer und fundamentaler Qualität.\n"
    "- **Investment-Attraktivität** bewertet, wie attraktiv die Aktie grundsätzlich als Investment-Case ist.\n"
    "- **Trading-Attraktivität** bewertet, wie attraktiv der Trade aktuell konkret ist.\n"
    "- **Tradeability** bewertet, wie gut der Case praktisch handelbar ist. In v9.0 fließt dieser Wert in die Trading-Attraktivität ein.\n"
    "- **Kurzfrist-Timing** zeigt, wie gut das aktuelle Timing für einen taktischen Einstieg wirkt.\n"
    "- **Setup-Confidence** zeigt, wie sauber und belastbar das erkannte Setup aktuell wirkt."
)

# ---------- Tabs ----------
st.divider()
t0, t1, t2, t3, t4, t5, t6 = st.tabs([
    "Profil & Chart",
    "Technik",
    "Kurzfrist-Vergleich",
    "TradingBoard",
    "Fundamental",
    "Safeguards",
    "Trade-Setup"
])

with t0:
    st.subheader("Unternehmensprofil")

    p1, p2, p3 = st.columns(3)
    p1.metric("Unternehmen", name)
    p2.metric("Sektor", sector if sector else "-")
    p3.metric("Industrie", industry if industry else "-")

    st.markdown("**Kurzbeschreibung**")
    summary_short = company_summary[:900] + "..." if len(company_summary) > 900 else company_summary
    st.write(summary_short)

    st.markdown("**Chartverlauf**")
    chart_range = st.selectbox(
        "Zeitraum",
        ["3 Monate", "6 Monate", "1 Jahr", "3 Jahre"],
        index=2,
        key="chart_range"
    )

    chart_df = compute_chart_df(df, chart_range)
    fig = build_candlestick_chart(chart_df, ticker, ccy)
    st.plotly_chart(fig, use_container_width=True)

    perf_start = float(chart_df["Close"].iloc[0]) if not chart_df.empty else np.nan
    perf_end = float(chart_df["Close"].iloc[-1]) if not chart_df.empty else np.nan
    perf_pct = ((perf_end / perf_start) - 1) * 100 if pd.notna(perf_start) and perf_start != 0 else np.nan

    p4, p5, p6 = st.columns(3)
    p4.metric("Start", fmt_num(perf_start, 2, f" {ccy}"))
    p5.metric("Aktuell", fmt_num(perf_end, 2, f" {ccy}"))
    p6.metric("Performance", fmt_num(perf_pct, 1, "%"))

    st.markdown("**Kurzfazit**")
    st.write(short_thesis)

with t1:
    st.markdown(f"**Benchmark:** {benchmark_label} (`{benchmark_symbol}`) | **Marktregime:** {market_info['ampel']} {market_regime_label(market_info['regime'])}")

    cols = st.columns(2)
    items = [
        ("S3 Trend", s3a, s3, s3t),
        ("S4 Momentum", s4a, s4, s4t),
        ("S5 Volumen", s5a, s5, s5t),
        ("S6 Volatilitaet", s6a, s6, s6t),
        ("52W-Lage", ampel(w52), w52, f"{dist52:.1f}% vom 52W-Hoch"),
        ("Relative Stärke", ampel(rs_score), rs_score, f"RS composite: {fmt_num(rs_composite,1,'%')}"),
    ]
    for i, (lab, ico, score, com) in enumerate(items):
        with cols[i % 2]:
            st.markdown(
                f'<div class="metric-card {card_class(score)}"><b>{ico} {lab}</b>'
                f'<span style="float:right;font-size:1.3rem;font-weight:700">{score}</span>'
                f'<br><small style="color:#aaa">{com}</small></div>',
                unsafe_allow_html=True,
            )

    tech_df = pd.DataFrame({
        "Indikator": [
            "Kurs", "MA20", "MA50", "MA150", "MA200",
            "RSI(14)", "MACD", "Signal", "MACD-Hist", "ADX", "ATR", "ATR in %",
            "Stoch %K", "Stoch %D", "Williams %R", "ROC20", "ROC60",
            "52W-Hoch", "52W-Tief", "Abstand zum 52W-Hoch",
            "1M Aktie", "1M Benchmark", "1M Outperformance",
            "3M Aktie", "3M Benchmark", "3M Outperformance",
            "6M Aktie", "6M Benchmark", "6M Outperformance",
            "RS Composite", "Marktregime"
        ],
        "Wert": [
            f"{price:.2f}", f"{ma20:.2f}", f"{ma50:.2f}", f"{ma150:.2f}", f"{ma200:.2f}",
            f"{rsi:.1f}", f"{macd_v:.3f}", f"{signal_v:.3f}", f"{macd_hist_current:.3f}", f"{adx:.1f}",
            f"{atr:.3f}", f"{atr_pct:.1f}%", f"{stoch_k_v:.1f}", f"{stoch_d_v:.1f}", f"{willr_v:.1f}",
            f"{roc20:.1f}%", f"{roc60:.1f}%", f"{high52:.2f}", f"{low52:.2f}", f"{dist52:.1f}%",
            fmt_num(ret21, 1, "%"), fmt_num(bench_ret21, 1, "%"), fmt_num(rs_vs_benchmark_21, 1, "%"),
            fmt_num(ret63, 1, "%"), fmt_num(bench_ret63, 1, "%"), fmt_num(rs_vs_benchmark_63, 1, "%"),
            fmt_num(ret126, 1, "%"), fmt_num(bench_ret126, 1, "%"), fmt_num(rs_vs_benchmark_126, 1, "%"),
            fmt_num(rs_composite, 1, "%"), market_regime_label(market_info["regime"])
        ],
    })
    st.dataframe(tech_df, hide_index=True, use_container_width=True)

with t2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kurzfrist Core", f"{short_term_score}/100", ampel(short_term_score))
    c2.metric("Kurzfrist Hilfsboard", f"{stb_score} Punkte", stb_signal)
    c3.metric("Core Fokus", "Momentum und Volumen")
    c4.metric("Board Fokus", "Einzelne Handelssignale")

    st.dataframe(
        pd.DataFrame({
            "Kennzahl": ["Kurzfrist Core", "Kurzfrist Hilfsboard", "Board-Signal", "Board-Treiber"],
            "Wert": [f"{short_term_score}/100", str(stb_score), stb_signal, stb_text],
            "Kommentar": [
                "S4 Momentum 45%, S5 Volumen 28%, S6 Volatilitaet 17%, RS 10%",
                "Additive Kurzfrist-Punkte aus MA/RSI/Momentum/ADX/Stoch/Williams",
                stb_empf,
                stb_text
            ],
        }),
        hide_index=True,
        use_container_width=True,
    )

with t3:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kurzfrist-Timing", f"{tb_score_100}/100", tb_timing_text)
    c2.metric("Board-Signal", tb_signal, tb_empf)
    c3.metric("TradingBoard Stop-Loss", f"{price - (2.5 * atr):.2f} {ccy}")
    c4.metric("TradingBoard Kursziel 2", f"{tp2:.2f} {ccy}")

    st.dataframe(tb_df, hide_index=True, use_container_width=True)

    st.markdown("**Board-Details (Kurzfrist-Timing)**")
    st.text("\n".join(tb_details))

    st.markdown("**Board-Kontext (nicht im Score)**")
    st.text("\n".join(tb_context))

with t4:
    st.markdown(
        f"<div class='small-note'>Datenabdeckung Fundamentaldaten: {fund_cov*100:.0f}% | Geladene Felder: {fund_fields_loaded}/21</div>",
        unsafe_allow_html=True
    )

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Fundamental-Confidence", confidence_info["confidence"], confidence_info["confidence_icon"])
    f2.metric("Coverage", f"{confidence_info['coverage']*100:.0f}%")
    f3.metric("Geladene Felder", f"{confidence_info['loaded']}/{confidence_info['total']}")
    f4.metric("Abgeleitete Felder", str(confidence_info["derived_estimate"]))

    fund_df = pd.DataFrame({
        "Fundament-Block": [
            "Qualitaet", "Wachstum", "Growth Quality", "Bewertung",
            "Bilanz", "Sentiment", "Risiko"
        ],
        "Score": [
            quality_score, growth_score, growth_quality, valuation_score,
            balance_score, sentiment_score, risk_score
        ],
        "Kommentar": [
            f"Gewinnmarge {fmt_num(profit_margin*100 if pd.notna(profit_margin) else np.nan,1,'%')} | Operative Marge {fmt_num(oper_margin*100 if pd.notna(oper_margin) else np.nan,1,'%')} | ROE {fmt_num(roe*100 if pd.notna(roe) else np.nan,1,'%')}",
            f"Umsatzwachstum {fmt_num(revenue_growth*100 if pd.notna(revenue_growth) else np.nan,1,'%')} | EPS-Wachstum {fmt_num(earnings_growth*100 if pd.notna(earnings_growth) else np.nan,1,'%')}",
            f"Wachstum + Cashflow + Margen | Stil: {stock_style}",
            f"KGV {fmt_num(pe,1)} | PEG {fmt_num(peg,2)} | KUV {fmt_num(ps,2)} | KBV {fmt_num(pb,2)}",
            f"Current Ratio {fmt_num(current_ratio,2)} | Quick Ratio {fmt_num(quick_ratio,2)} | D/E {fmt_num(debt_to_equity,1)}",
            f"Analystenmeinung {rec_label} | Anzahl {fmt_num(analysts,0)} | Mean {fmt_num(rec_mean,2)}",
            f"Beta {fmt_num(beta,2)} | Short-Quote {fmt_num(short_pct*100 if pd.notna(short_pct) else np.nan,1,'%')} | ATR% {fmt_num(atr_pct,1,'%')}",
        ],
    })
    st.dataframe(fund_df, hide_index=True, use_container_width=True)

    st.markdown("**Strukturierte Red Flags**")
    st.dataframe(red_flags_df, hide_index=True, use_container_width=True)

with t5:
    safeguard_df = pd.DataFrame({
        "Safeguard": [
            "S0 Currency/Exchange",
            "S0 Preis-Typ-Lock",
            "S1 Earnings",
            "S2 Regime",
            "S3 Konfluenz-Cap",
            "S4 Datenabdeckung",
            "S5 Marktfilter",
            "S6 Modus",
            "S7 Red Flags",
            "S8 Fundamental-Confidence"
        ],
        "Status": [
            "🟢",
            "🟢",
            sg_earn,
            reg_amp,
            "🟢" if kb >= 3 else ("🟡" if kb == 2 else "🔴"),
            "🟢" if fund_cov >= 0.55 else ("🟡" if fund_cov >= 0.35 else "🔴"),
            market_info["ampel"],
            "🟢",
            "🟢" if red_flag_penalty_total == 0 else ("🟡" if red_flag_penalty_total <= 10 else "🔴"),
            confidence_info["confidence_icon"],
        ],
        "Kommentar": [
            f"{ccy} | {exch}",
            "auto_adjust=True Yahoo Finance",
            sg_earn_txt,
            regime,
            f"{kb}/4 Kernbloecke",
            f"Fundamental-Coverage {fund_cov*100:.0f}%",
            f"{benchmark_label} | {market_regime_label(market_info['regime'])}",
            mode_label,
            f"Penalty {red_flag_penalty_total}",
            confidence_info["confidence"]
        ],
    })
    st.dataframe(safeguard_df, hide_index=True, use_container_width=True)

with t6:
    if not valid_trade_setup:
        st.error("Kein valides Trade-Setup: Score, Marktumfeld oder Konfluenz reichen aktuell nicht aus.")
        st.write(
            f"Aktuell: Investment Score {investment}/100 | "
            f"Setup Quality {setup_adj}/100 | "
            f"Konfluenz {kb}/4 | "
            f"Marktregime {market_regime_label(market_info['regime'])}"
        )
        if has_upcoming_earnings and pd.notna(days_earn) and days_earn < 7:
            st.write("Zusatzhinweis: Earnings-Veto aktiv.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Einstiegskurs", f"{price:.2f} {ccy}")
        c2.metric("ATR-basierter Stop-Loss", f"{atr_stop:.2f} {ccy}", f"-{(price-atr_stop)/price*100:.1f}%" if atr_stop < price else "-")
        c3.metric("Aktueller Stop-Loss", f"{stop_used:.2f} {ccy}", f"-{stop_dist:.1f}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("Kursziel 1 (1R)", f"{tp1:.2f} {ccy}", f"+{(tp1/price-1)*100:.1f}%")
        c5.metric("Kursziel 2 (Hauptziel)", f"{tp2:.2f} {ccy}", f"+{(tp2/price-1)*100:.1f}%")
        c6.metric("Kursziel 3", f"{tp3:.2f} {ccy}", f"+{(tp3/price-1)*100:.1f}%")

        c7, c8, c9 = st.columns(3)
        c7.metric(f"Chance-Risiko-Verhältnis {ampel_crv(crv)}", f"{crv:.1f}:1")
        c8.metric("Positionsgroesse", f"{pos_size} Stueck", f"Risiko {risk_eur:.0f} EUR ({risk_pct}%)")
        c9.metric("Zeitlicher Stop", time_stop, "wenn der Kurs nicht anschiebt")

        st.markdown("**Konkreter Einstiegsvorschlag**")
        e1, e2, e3 = st.columns(3)
        e1.metric("Entry-Zone", suggested_entry_zone)
        e2.metric("Entry-Herleitung", entry_source)
        e3.metric("Aktuelle Lage", entry_quality)

        st.markdown("**Case-Komponenten**")
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        tc1.metric("Investment-Case", f"{investment_case_score}/100", investment_case_text)
        tc2.metric("Trading-Case", f"{trading_case_score}/100", trading_case_text)
        tc3.metric("CRV-Score", fmt_num(trade_crv_score, 0))
        tc4.metric("Entry-Score", fmt_num(trade_entry_score, 0))
        tc5.metric("Setup-Confidence", fmt_num(setup_confidence, 0))

        st.markdown("**Zielherleitung**")
        st.write(f"• TP1: {tp1_source}")
        st.write(f"• TP2: {tp2_source}")
        st.write(f"• TP3: {tp3_source}")

# ---------- Horizon lamps ----------
st.divider()
st.subheader("5 Zeithorizont-Ampeln")
cols = st.columns(5)
for col, (lab, scv) in zip(cols, hmap.items()):
    col.markdown(
        f"<div style='text-align:center'><div style='font-size:2rem'>{ampel(scv)}</div>"
        f"<small>{lab}<br><b>{scv}/100</b></small></div>",
        unsafe_allow_html=True,
    )

# ---------- Recommendation + Why block ----------
st.divider()
st.subheader("Handlungsempfehlung")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class="reco-card context">
            <div>
                <div class="reco-top">
                    <div class="reco-label">Analysekontext</div>
                    <div class="reco-icon">🧭</div>
                </div>
                <div class="reco-value">{display_mode_label(mode_label)}</div>
            </div>
            <div class="reco-chip">Aktueller Rahmen</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="reco-card main">
            <div>
                <div class="reco-top">
                    <div class="reco-label">Haupteinschätzung</div>
                    <div class="reco-icon">🎯</div>
                </div>
                <div class="reco-value">{display_emp_label(result.get("emp", "-"))}</div>
            </div>
            <div class="reco-chip">Zentrale Aussage</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="reco-card conviction">
            <div>
                <div class="reco-top">
                    <div class="reco-label">Überzeugungsgrad</div>
                    <div class="reco-icon">📌</div>
                </div>
                <div class="reco-value">{display_conv_label(result.get("conv", "-"))}</div>
            </div>
            <div class="reco-chip">Vertrauen ins Setup</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="reco-card signal">
            <div>
                <div class="reco-top">
                    <div class="reco-label">Kurzfristsignal</div>
                    <div class="reco-icon">⚡</div>
                </div>
                <div class="reco-value">{display_stb_label(stb_signal)}</div>
            </div>
            <div class="reco-delta">Timing: {tb_timing_text} | Score: {stb_score}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"""
        <div class="reco-card market">
            <div>
                <div class="reco-top">
                    <div class="reco-label">Marktumfeld</div>
                    <div class="reco-icon">🌍</div>
                </div>
                <div class="reco-value">{market_regime_label(market_info["regime"])}</div>
            </div>
            <div class="reco-delta">{market_info["ampel"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Warum diese Einschätzung?")
w1, w2 = st.columns(2)

with w1:
    st.markdown("**Top-Stärken**")
    if strengths:
        for s in strengths:
            st.write(f"- {s}")
    else:
        st.write("- Keine klaren Stärken identifiziert.")

with w2:
    st.markdown("**Top-Schwächen**")
    if weaknesses:
        for w in weaknesses:
            st.write(f"- {w}")
    else:
        st.write("- Keine wesentlichen Schwächen identifiziert.")

st.markdown("**Kurzfazit**")
st.write(decision_summary)

st.caption(
    "Die App zeigt bewusst mehrere getrennte Sichtweisen: Core-Modell, kurzfristige Hilfsboard-Ampel, "
    "dashboardnahen TradingBoard-Referenzscore, Marktfilter, Fundamental-Confidence, Ranking-Tabelle "
    "und eine qualitative Begründung."
)
