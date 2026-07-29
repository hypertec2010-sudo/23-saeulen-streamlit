from __future__ import annotations

import modules.trade_journal as journal


def _safe_float(value, default=None):
    if value in (None, "", "n/a", "-"):
        return default
    return float(value)


def test_partial_exit_then_close_creates_persistent_entries(tmp_path) -> None:
    journal.configure_context(
        base_dir=tmp_path,
        safe_float=_safe_float,
        event_logger=lambda **kwargs: True,
        repository=None,
    )
    journal._v270_reset_trade_journal()
    positions = {
        "AAPL": {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "entry": 100,
            "stop": 95,
            "initial_stop": 95,
            "target": 115,
            "shares": 10,
            "initial_shares": 10,
        }
    }

    partial = journal._v270_partial_exit(
        positions,
        watchlist_name="CI",
        ticker="AAPL",
        exit_price=105,
        exit_shares=4,
        exit_date="2026-07-27",
        note="Teilgewinn",
    )
    assert partial["ok"] is True
    assert partial["positions"]["AAPL"]["shares"] == 6

    closed = journal._v270_close_position(
        partial["positions"],
        watchlist_name="CI",
        ticker="AAPL",
        exit_price=110,
        exit_date="2026-07-28",
        reason="Ziel erreicht",
    )
    assert closed["ok"] is True
    assert "AAPL" not in closed["positions"]

    frame = journal._v270_journal_entries_dataframe("CI")
    assert len(frame) == 2
    assert set(frame["Typ"]) == {"Teilverkauf", "Position geschlossen"}
    assert (tmp_path / ".trade_journal_v270.json").exists()
