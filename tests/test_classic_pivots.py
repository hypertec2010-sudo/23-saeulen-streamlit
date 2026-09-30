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


def test_action_guidance_near_pp_is_clear_and_hides_distant_r2_s2():
    from modules.classic_pivots import build_pivot_reading

    levels = {"PP": 542.52, "R1": 548.46, "R2": 557.33, "S1": 533.65, "S2": 527.71}
    reading = build_pivot_reading(levels, 544.17)
    guide = reading["guidance"]

    assert guide["near_decision_level"] is True
    assert guide["decision_level"] == "PP"
    assert "PP 542.52" in guide["current"]
    assert "R1 548.46" in guide["positive"]
    assert "S1 533.65" in guide["negative"]
    assert guide["relevant_level_names"] == ["PP", "R1", "S1"]
    assert "R2" not in guide["relevant_level_names"]
    assert "S2" not in guide["relevant_level_names"]
    assert "CHSM-Ampel" in guide["action"]


def test_between_levels_only_immediate_boundaries_are_displayed():
    from modules.classic_pivots import build_pivot_reading

    levels = {"PP": 100.0, "R1": 105.0, "R2": 111.0, "S1": 94.0, "S2": 89.0}
    reading = build_pivot_reading(levels, 102.5)
    guide = reading["guidance"]

    assert guide["near_decision_level"] is False
    assert guide["relevant_level_names"] == ["PP", "R1"]
    assert "R2 111.00" in guide["positive"]
    assert "S1 94.00" in guide["negative"]


def test_near_r1_makes_r2_relevant_but_not_s2():
    from modules.classic_pivots import build_pivot_reading

    levels = {"PP": 100.0, "R1": 105.0, "R2": 111.0, "S1": 94.0, "S2": 89.0}
    reading = build_pivot_reading(levels, 104.8)
    guide = reading["guidance"]

    assert guide["decision_level"] == "R1"
    assert guide["relevant_level_names"] == ["R1", "R2", "PP"]
    assert "S2" not in guide["relevant_level_names"]
    assert "Nicht direkt in R1 hinein nachlaufen" in guide["action"]


def test_package_exposes_guidance_without_score_effect():
    pkg = build_classic_daily_pivot_package(
        _df(),
        current_price=109.0,
        now="2026-09-29T15:00:00Z",
    )
    assert pkg["score_neutral"] is True
    assert "score" not in pkg
    assert isinstance(pkg["guidance"], dict)
    assert pkg["guidance"]["current"]
    assert pkg["guidance"]["action"]
