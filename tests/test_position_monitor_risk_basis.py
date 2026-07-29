from __future__ import annotations

import pytest

from modules import position_monitor
from modules import trade_journal


def _safe_float(value, default=None):
    if value in (None, "", "n/a", "-"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def setup_module():
    position_monitor.configure_context(
        safe_float=_safe_float,
        price_text=lambda value, digits=2: "n/a" if value is None else f"{float(value):.{digits}f}",
    )
    trade_journal.configure_context(safe_float=_safe_float)


def test_r_multiple_uses_initial_stop_after_stop_moves_above_entry():
    result = position_monitor._v244_calc_trade_state(
        {
            "entry": 147.25,
            "stop": 150.00,
            "initial_stop": 140.00,
            "shares": 36,
            "last_price": 161.34,
        },
        {"Kurs": 161.34},
    )
    expected = (161.34 - 147.25) / (147.25 - 140.00)
    assert result["R-Multiple"] == pytest.approx(expected)
    assert result["R-Basis-Stop"] == pytest.approx(140.00)
    assert "Nicht berechenbar" not in result["Status"]
    assert "Gewinnschutz aktiv" in result["Stop-Hinweis"]


def test_risk_basis_can_be_recovered_from_stop_history():
    result = position_monitor._v244_calc_trade_state(
        {
            "entry": 100.0,
            "stop": 103.0,
            "shares": 10,
            "stop_history": [
                {"old_stop": 95.0, "new_stop": 100.0},
                {"old_stop": 100.0, "new_stop": 103.0},
            ],
        },
        {"Kurs": 110.0},
    )
    assert result["R-Basis-Stop"] == pytest.approx(95.0)
    assert result["R-Multiple"] == pytest.approx(2.0)


def test_missing_original_risk_has_precise_message():
    result = position_monitor._v244_calc_trade_state(
        {"entry": 100.0, "stop": 102.0, "shares": 10},
        {"Kurs": 105.0},
    )
    assert result["R-Multiple"] is None
    assert result["Status"] == "Initialrisiko fehlt"
    assert "ursprüngliche Initial-Stop" in result["Aktion"]


def test_trade_journal_uses_initial_stop_for_realized_r():
    entry, initial_stop, unit_risk = trade_journal._position_risk(
        {"entry": 100.0, "stop": 103.0, "initial_stop": 95.0}
    )
    assert entry == pytest.approx(100.0)
    assert initial_stop == pytest.approx(95.0)
    assert unit_risk == pytest.approx(5.0)
