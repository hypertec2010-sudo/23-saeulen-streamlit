from __future__ import annotations

import pandas as pd

from modules import depot_transaction_import as dti
from modules import trade_journal as tj


class MemStore:
    def __init__(self):
        self.data = {}

    def load_namespace(self, name, default=None):
        return self.data.get(name, default)

    def save_namespace(self, name, value):
        self.data[name] = value
        return True


def t212_df(action="Market buy", ticker="NVDA", shares=26.0, price=230.72):
    return pd.DataFrame([{
        "Action": action,
        "Time (UTC)": "2026-09-29 15:17:36+00:00",
        "ISIN": "US67066G1040",
        "Ticker": ticker,
        "Name": "NVIDIA",
        "Notes": "",
        "ID": "EOF58107672845",
        "No. of shares": shares,
        "Price / share": price,
        "Currency (Price / share)": "USD",
        "Exchange rate": 1.17,
        "Result": None,
        "Currency (Result)": "EUR",
        "Gross Total": 5000.0,
        "Currency (Gross Total)": "EUR",
        "Withholding tax": None,
        "Currency (Withholding tax)": "",
        "Currency conversion fee": 0.0,
        "Currency (Currency conversion fee)": "EUR",
        "Taxes": None,
        "Currency (Taxes)": "",
        "Net Total": None,
        "Currency (Net Total)": "EUR",
    }])


def test_trading212_schema_and_stop_sell_are_recognised():
    raw = t212_df()
    assert dti.is_trading212_export(raw) is True
    raw.loc[0, "Action"] = "Stop sell"
    pkg = dti.normalize_transactions(raw)
    assert pkg["ok"] is True
    assert pkg["data"].iloc[0]["Action-Typ"] == "SELL"


def test_wrong_manual_sell_can_be_rebuilt_from_broker_buy(tmp_path):
    store = MemStore()
    tj.configure_context(storage=store, base_dir=tmp_path)
    store.save_namespace("trade_journal", {"entries": [{
        "ID": "manual-wrong-nvda",
        "Zeit": "29.09.2026 17:30:00",
        "Datum": "2026-09-29",
        "Watchlist": "WL",
        "Ticker": "NVDA",
        "Name": "NVIDIA",
        "Typ": "Position geschlossen",
        "Kurs": 230.72,
        "Stück": 26,
        "Verbleibend": 0,
        "Entry": 225.0,
        "Position vorher": {
            "ticker": "NVDA",
            "name": "NVIDIA",
            "entry": 225.0,
            "shares": 26.0,
            "stop": 220.0,
            "target": 250.0,
            "portfolio_group": "Semis",
            "entry_context": {"grade": "A"},
            "strategy_origin": "screener",
            "opened_at_iso": "2026-09-29T15:00:00+02:00",
        },
    }]})

    norm = dti.normalize_transactions(t212_df())["data"]
    preview = tj._v3021ab_trading212_correction_preview("WL", norm, {})
    row = preview["table"].iloc[0]
    assert row["Ticker"] == "NVDA"
    assert row["Status"] == "KORREKTUR MÖGLICH"
    assert row["Broker Endbestand"] == 26.0
    assert row["Manuelle Exit-Buchungen"] == 1

    prep = tj._v3021ab_prepare_correction_positions("WL", {}, ["NVDA"])
    plan = dti.apply_transactions(
        "WL", norm, prep["positions"], mode="rebuild", already_processed=set(),
        screener_only=False, broker_source="Trading 212 CSV",
    )
    assert plan["ok"] is True
    assert plan["anomalies"].empty
    assert plan["positions"]["NVDA"]["shares"] == 26.0
    assert plan["positions"]["NVDA"]["entry"] == 230.72
    assert plan["positions"]["NVDA"]["broker_source"] == "Trading 212 CSV"
    assert plan["positions"]["NVDA"]["entry_context"]["grade"] == "A"

    superseded = tj._v3021ab_supersede_manual_executions("WL", norm, ["NVDA"])
    assert superseded["ok"] is True
    assert superseded["updated"] == 1
    journal = store.load_namespace("trade_journal")["entries"]
    assert journal[0]["Typ"] == "Storniert · Broker-Korrektur"
    assert journal[0]["Ursprünglicher Typ"] == "Position geschlossen"


def test_unrelated_pie_ticker_is_not_offered_for_correction(tmp_path):
    store = MemStore()
    tj.configure_context(storage=store, base_dir=tmp_path)
    store.save_namespace("trade_journal", {"entries": []})
    norm = dti.normalize_transactions(t212_df(ticker="SAP", shares=0.05, price=185.0))["data"]
    preview = tj._v3021ab_trading212_correction_preview("WL", norm, {})
    assert preview["table"].empty
    assert preview["eligible_tickers"] == []


def test_incomplete_cycle_with_sell_first_is_blocked(tmp_path):
    store = MemStore()
    tj.configure_context(storage=store, base_dir=tmp_path)
    store.save_namespace("trade_journal", {"entries": []})
    norm = dti.normalize_transactions(t212_df(action="Market sell", shares=5.0, price=230.0))["data"]
    positions = {"NVDA": {"ticker": "NVDA", "entry": 220.0, "shares": 5.0, "strategy_origin": "screener"}}
    preview = tj._v3021ab_trading212_correction_preview("WL", norm, positions)
    row = preview["table"].iloc[0]
    assert row["Status"] == "BLOCKIERT"
    assert "NVDA" in preview["blocked_tickers"]
