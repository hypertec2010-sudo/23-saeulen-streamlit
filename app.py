# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, date, timedelta
import warnings

import streamlit as st

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

# Ab hier kommt dein normaler Code (st.set_page_config etc.)


warnings.filterwarnings("ignore")

APP_VERSION = "v5.8c"

st.set_page_config(
    page_title=f"Capital-Hill-Score-Modell {APP_VERSION}",
    page_icon="📊",
    layout="wide"
)

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
                    mean = (1*sb + 2*b + 3*h + 4*s + 5*ss) / total
                    if pd.isna(info.get("recommendationMean")):
                        info["recommendationMean"] = mean
                    if pd.isna(info.get("numberOfAnalystOpinions")):
                        info["numberOfAnalystOpinions"] = int(total)
                    if pd.isna(info.get("recommendationKey")) or str(info.get("recommendationKey","")).lower() in {"", "none", "nan"}:
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

    # 1. Versuch: yfinance info dictionary
    ts = normalize_missing(info.get("earningsTimestamp"))
    if not pd.isna(ts):
        return info

    # 2. Versuch: Ticker Calendar (oft zuverlässiger bei der neuen Yahoo-Struktur)
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

    # 3. Versuch: get_earnings_dates Fallback
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



def tb_signal_label(score):
    if score >= 9:
        return "LONG", "AKTIV HALTEN"
    if score >= 5:
        return "HOLD", "HALTEN"
    if score >= 3:
        return "WAIT", "ABWARTEN"
    return "SHORT", "STOPP PRÜFEN"


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


with st.sidebar:
    st.title(f"📊 Capital-Hill-Score-Modell {APP_VERSION}")
    st.caption(f"{APP_VERSION} | Core + TradingBoard Referenzscore + Analysten- und Earnings-Fallback")
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
    if st.button("Cache leeren", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache geleert. Bitte Analyse neu starten.")

    go = st.button("Analyse starten", use_container_width=True, type="primary")

st.title(f"📊 Capital-Hill-Score-Modell {APP_VERSION}")
st.caption(
    "Core-Modell und TradingBoard werden getrennt gerechnet. "
    "Die Core-Saeulen bleiben unveraendert; das TradingBoard ist jetzt als dashboardnaher Referenzscore modelliert, waehrend Zusatzsignale getrennt als Kontext angezeigt werden."
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
raw_ccy = info.get("currency", "USD")
ccy = infer_display_currency(ticker, info, raw_ccy)
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

if has_upcoming_earnings and days_earn < 7:
    emp, conv = "VETO - Earnings < 7 Tage", "-"
elif investment >= 78 and kb >= 3:
    emp, conv = "BUY / ACCUMULATE", "HIGH"
elif investment >= 68:
    emp, conv = "WATCH / kleine Position", "MEDIUM"
elif investment >= 52:
    emp, conv = "HOLD / beobachten", "LOW-MEDIUM"
else:
    emp, conv = "AVOID / WAIT", "LOW"

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
    if days_earn < 0:
        tb_details.append(f"S1 Earnings: vor {int(abs(days_earn))}d (Letzte Earnings am {sg_earn_txt})")
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

if tb_perf > 5:
    tb_score += 1
    tb_details.append(f"S6: +{tb_perf:.1f}% ✓")
else:
    tb_details.append(f"S6: {tb_perf:.1f}% ❌")

if macd_hist_current > macd_hist_prev:
    tb_score += 1
    tb_details.append("S7: Momentum steigt ✓")
else:
    tb_details.append("S7: Momentum fällt ❌")

if earnings_warning:
    tb_score -= 3
    tb_details.insert(0, "⚠️ EARNINGS IN <7 TAGEN (Vorsicht!)")

if 20 < rsi < 80:
    tb_context.append("S6: Vola ok ✓")

if macd_bull_cross:
    tb_context.append("MACD Bull-Cross! 🚀")

if smart_money_default:
    tb_context.append("S8: Smart Money sammelt ein ✓")
else:
    tb_context.append("S8: Smart Money verkauft ❌")

if adx > 25:
    tb_context.append("S9: ADX>25 starker Trend ✓")
else:
    tb_context.append("S9: ADX<25 Seitwärts ❌")

if stoch_k_v < 20 and stoch_d_v < 20 and stoch_k_v > stoch_d_v:
    tb_context.append("S10: Stoch Oversold Cross ✓")
elif stoch_k_v > 80:
    tb_context.append("S10: Stoch überkauft ❌")
else:
    tb_context.append("S10: Stoch neutral ❌")

if willr_v < -80:
    tb_context.append("S11: Williams%R extrem Oversold ✓")
elif willr_v > -20:
    tb_context.append("S11: Williams%R überkauft ❌")
else:
    tb_context.append("S11: Williams%R neutral ❌")

if obv_trend == "steigend" and vol_ratio >= 1.0:
    tb_context.append("S12: OBV/Volumen bestaetigt ✓")
else:
    tb_context.append("S12: OBV/Volumen schwach ❌")

if pd.notna(prev20_high) and price > prev20_high:
    tb_context.append("S13: 20D Breakout ✓")
elif pd.notna(prev20_low) and price < prev20_low:
    tb_context.append("S13: 20D Breakdown ❌")
else:
    tb_context.append("S13: Range intakt ❌")

if pd.notna(bb_upper) and price > bb_upper:
    tb_context.append("S14: BB Breakout UP ✓")
elif bb_squeeze:
    tb_context.append("S14: BB Squeeze Achtung ✓")
elif pd.notna(bb_lower) and price < bb_lower:
    tb_context.append("S14: BB Breakout DOWN ❌")
else:
    tb_context.append("S14: BB neutral ❌")

if pd.notna(target) and target > 0 and price > 0:
    tb_potenzial = ((target - price) / price) * 100
    if tb_potenzial > 15:
        tb_context.append(f"S15: Target +{tb_potenzial:.1f}% ✓")
    elif tb_potenzial < 0:
        tb_context.append(f"S15: Target -{abs(tb_potenzial):.1f}% ❌")
    else:
        tb_context.append(f"S15: Target +{tb_potenzial:.1f}% neutral ❌")
else:
    tb_context.append("S15: Kein valides Target ❌")

current_month = datetime.now().month
if current_month in [8, 9]:
    tb_context.append("S16: Seasonality schlecht (-1) ❌")
elif current_month in [11, 12, 1]:
    tb_context.append("S16: Seasonality stark (+1) ✓")
else:
    tb_context.append("S16: Seasonality neutral ❌")

if crv >= 2.0:
    tb_context.append("S17: CRV >= 2.0 ✓")
elif crv < 1.5:
    tb_context.append("S17: CRV schwach ❌")
else:
    tb_context.append("S17: CRV ok/neutral ❌")

short_squeeze = pd.notna(short_pct) and short_pct > 0.12 and ret5 > 0 and vol_ratio > 1.2
if short_squeeze:
    tb_context.append("S18: 🚀 SHORT SQUEEZE POTENZIAL ✓")
else:
    tb_context.append("S18: kein Short-Squeeze-Signal ❌")

insider_trend = "NEUTRAL"
if insider_trend == "BUY":
    tb_context.append("S19: 👔 INSIDER KAUFEN ✓")
elif insider_trend == "SELL":
    tb_context.append("S19: 👔 INSIDER VERKAUFEN ❌")
else:
    tb_context.append("S19: 👔 Insider neutral / keine Daten ❌")

if pd.notna(pe) and 0 < pe < 15:
    tb_context.append(f"S20: 🟢 VALUE KGV ({pe:.1f}) ✓")
elif pd.notna(pe) and pe > 50:
    tb_context.append(f"S20: 🔴 TEUER KGV>50 ({pe:.1f}) ❌")
else:
    tb_context.append(f"S20: Value neutral ({fmt_num(pe,1)}) ❌")

tb_signal, tb_empf = tb_signal_label(tb_score)

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
tb_context_text = "\n".join(tb_context)

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

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Kurs (Adj. Close)", f"{price:.2f} {ccy}", ts)
c2.metric("Trend-Regime", regime, reg_amp)
c3.metric("Earnings-Datum", sg_earn_txt, sg_earn)
if has_upcoming_earnings:
    c4.metric("Earnings-Countdown", f"{int(days_earn)}d", sg_earn)
elif has_past_earnings:
    c4.metric("Earnings-Countdown", "vorbei", sg_earn)
else:
    c4.metric("Earnings-Countdown", "kein Datum", sg_earn)
else:
    c4.metric("Earnings-Countdown", "kein Datum", sg_earn)
c5.metric("Analysten-Target", fmt_num(target, 2, f" {ccy}"), fmt_num(upside, 1, "%"))

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

st.subheader("Scores")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Company Quality", f"{company}/100", ampel(company))
c2.metric("Setup Quality", f"{setup}/100", ampel(setup))
c3.metric("Kurzfrist Core", f"{short_term_score}/100", ampel(short_term_score))
c4.metric("Kurzfrist Hilfsboard", f"{stb_score} Punkte", stb_signal)
c5.metric("Investment Score", f"{investment}/100", ampel(investment))
c6.metric("TradingBoard Score", f"{tb_score} Punkte", ampel_tb(tb_score))
c7.metric("Konfluenz", f"{kb}/4", "Robust" if kb >= 3 else ("Fragil" if kb == 2 else "Schwach"))

st.caption(
    "Die App trennt jetzt drei Sichtweisen: Company/Core, Kurzfrist Core vs. Kurzfrist Hilfsboard und das volle additive TradingBoard. "
    "So sieht man sofort, ob ein Wert kurzfristig tradbar ist, obwohl das breite Board noch hinterherhaengt oder umgekehrt."
)

st.markdown("### Scores verständlich erklärt")
st.markdown(
    "- **Company Quality** bewertet die Qualität des Unternehmens: Profitabilität, Wachstum, Bilanz, Bewertung, Analysteneinschätzung und Risiko.\n"
    "- **Setup Quality** bewertet das technische Gesamtbild im strengen 23-Säulen-Kernmodell. Wenn Trend, Momentum, Volumen und Volatilität nicht gemeinsam tragen, wird dieser Wert bewusst gedeckelt.\n"
    "- **Kurzfrist Core** bewertet die kurzfristige technische Lage mit Fokus auf Momentum, Volumen, Volatilität und relative Stärke.\n"
    "- **Kurzfrist Hilfsboard** ist eine zusätzliche Kurzfrist-Ampel aus einzelnen Handelssignalen. Dieser Wert ist hilfreich, ersetzt aber nicht den eigentlichen TradingBoard-Referenzscore.\n"
    "- **Investment Score** ist die Gesamtbewertung aus technischer Qualität und Unternehmensqualität. Die Gewichtung ändert sich mit dem gewählten Zeithorizont.\n"
    "- **TradingBoard Score** ist der dashboardnahe Referenzscore für die Trading-Entscheidung. Genau dieser Wert soll den Blick ins separate Trading-Dashboard möglichst ersetzen.\n"
    "- **Konfluenz** zeigt, wie viele der vier Kernbereiche Trend, Momentum, Volumen und Volatilität gleichzeitig tragfähig sind."
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
        ("Relative Stärke", ampel(rs_score), rs_score, f"3 Monate: {ret63:.1f}%"),
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
                    "RSI(14)", "MACD", "Signal", "MACD-Hist", "ADX", "ATR", "ATR in %",
                    "Stoch %K", "Stoch %D", "Williams %R", "ROC20", "ROC60",
                    "52W-Hoch", "52W-Tief", "Abstand zum 52W-Hoch"
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
    c3.metric("Core Fokus", "Momentum und Volumen")
    c4.metric("Board Fokus", "Einzelne Handelssignale")

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
    c3.metric("TradingBoard Stop-Loss", f"{tb_stop:.2f} {ccy}")
    c4.metric("TradingBoard Kursziel 2", f"{tb_tp2:.2f} {ccy}")

    st.dataframe(tb_df, hide_index=True, use_container_width=True)

    st.markdown("**TradingBoard Referenzscore**")
    st.text(tb_details_text)

    st.markdown("**Board-Kontext (nicht im Score)**")
    st.text(tb_context_text)

with t4:
    st.markdown(
        f"<div class='small-note'>Datenabdeckung Fundamentaldaten: {fund_cov*100:.0f}% | Geladene Felder: {fund_fields_loaded}/21</div>",
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
                    f"Gewinnmarge {fmt_num(profit_margin*100 if pd.notna(profit_margin) else np.nan,1,'%')} | Operative Marge {fmt_num(oper_margin*100 if pd.notna(oper_margin) else np.nan,1,'%')} | Eigenkapitalrendite {fmt_num(roe*100 if pd.notna(roe) else np.nan,1,'%')} | Freier Cashflow {'positiv' if pd.notna(fcf) and fcf > 0 else ('negativ' if pd.notna(fcf) else 'n/a')}",
                    f"Umsatzwachstum {fmt_num(revenue_growth*100 if pd.notna(revenue_growth) else np.nan,1,'%')} | Gewinnwachstum je Aktie {fmt_num(earnings_growth*100 if pd.notna(earnings_growth) else np.nan,1,'%')} | 6-Monats-Performance {fmt_num(ret126,1,'%')}",
                    f"KGV {fmt_num(pe,1)} | PEG-Verhältnis {fmt_num(peg,2)} | Kurs-Umsatz-Verhältnis {fmt_num(ps,2)} | Analysten-Potenzial {fmt_num(upside,1,'%')}",
                    f"Current Ratio {fmt_num(current_ratio,2)} | Quick Ratio {fmt_num(quick_ratio,2)} | Verschuldung zu Eigenkapital {fmt_num(debt_to_equity,1)}",
                    f"Analystenmeinung {rec_label} | Anzahl Analysten {fmt_num(analysts,0)} | Durchschnittliche Empfehlung {fmt_num(rec_mean,2)} | Kursziel {fmt_num(target,2)}",
                    f"Beta {fmt_num(beta,2)} | Short-Quote {fmt_num(short_pct*100 if pd.notna(short_pct) else np.nan,1,'%')} | ATR in Prozent {fmt_num(atr_pct,1,'%')}",
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
    c1.metric("Einstiegskurs", f"{price:.2f} {ccy}")
    c2.metric("ATR-basierter Stop-Loss", f"{atr_stop:.2f} {ccy}", f"-{(price-atr_stop)/price*100:.1f}%" if atr_stop < price else "-")
    c3.metric("Aktueller Stop-Loss", f"{stop_used:.2f} {ccy}", f"-{stop_dist:.1f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("Kursziel 1 (1R)", f"{tp1:.2f} {ccy}", f"+{(tp1/price-1)*100:.1f}%")
    c5.metric("Kursziel 2 (2R)", f"{tp2:.2f} {ccy}", f"+{(tp2/price-1)*100:.1f}%")
    c6.metric("Kursziel 3 (3R)", f"{tp3:.2f} {ccy}", f"+{(tp3/price-1)*100:.1f}%")

    c7, c8, c9 = st.columns(3)
    c7.metric(f"Chance-Risiko-Verhältnis {ampel_crv(crv)}", f"{crv:.1f}:1")
    c8.metric("Positionsgroesse", f"{pos_size} Stueck", f"Risiko {risk_eur:.0f} EUR ({risk_pct}%)")
    c9.metric("Zeitlicher Stop", time_stop, "wenn der Kurs nicht anschiebt")

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
    "die kurzfristige Hilfsboard-Ampel und den dashboardnahen TradingBoard-Referenzscore mit getrenntem Kontextblock."
)

