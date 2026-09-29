import pandas as pd

from modules.classic_pivots import (
    build_classic_daily_pivot_package,
    calculate_classic_pivot_levels,
)


def _df():
    return pd.DataFrame(
        {
            "High": [100.0, 110.0, 120.0],
            "Low": [90.0, 100.0, 105.0],
            "Close": [95.0, 108.0, 115.0],
        },
        index=pd.to_datetime(["2026-09-27", "2026-09-28", "2026-09-29"]),
    )


def test_classic_formula():
    levels = calculate_classic_pivot_levels(110.0, 100.0, 108.0)
    pp = (110.0 + 100.0 + 108.0) / 3.0
    assert levels["PP"] == pp
    assert levels["R1"] == 2 * pp - 100.0
    assert levels["S1"] == 2 * pp - 110.0
    assert levels["R2"] == pp + 10.0
    assert levels["S2"] == pp - 10.0


def test_current_daily_bar_is_not_used_as_pivot_basis():
    pkg = build_classic_daily_pivot_package(
        _df(),
        current_price=109.0,
        now="2026-09-29T15:00:00Z",
    )
    assert pkg["available"] is True
    assert pkg["source_date"] == "2026-09-28"
    assert pkg["source_ohlc"] == {"high": 110.0, "low": 100.0, "close": 108.0}
    assert pkg["score_neutral"] is True


def test_before_new_session_latest_completed_bar_is_used():
    df = _df().iloc[:2]
    pkg = build_classic_daily_pivot_package(
        df,
        current_price=109.0,
        now="2026-09-29T08:00:00Z",
    )
    assert pkg["available"] is True
    assert pkg["source_date"] == "2026-09-28"


def test_pivot_package_detects_chsm_zone_confluence_without_score():
    structures = {
        "resistances": [{"low": 111.0, "high": 113.0, "mid": 112.0, "touches": 3}],
        "supports": [],
        "active_zones": [],
    }
    pkg = build_classic_daily_pivot_package(
        _df(),
        current_price=109.0,
        structures=structures,
        now="2026-09-29T15:00:00Z",
    )
    assert pkg["score_neutral"] is True
    assert "score" not in pkg
    assert any(item["zone_label"] == "CHSM-Widerstand" for item in pkg["confluences"])
    assert "Empfehlung" not in pkg["location"]
