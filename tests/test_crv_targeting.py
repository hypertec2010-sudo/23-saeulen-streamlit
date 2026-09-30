from __future__ import annotations

import math

from modules.crv_targeting import projected_breakout_target, select_operational_tp2


def test_breakout_target_validates_projection_not_old_level() -> None:
    # Real breakout: price is already above the old 20D high. The projected
    # +3% target is nevertheless still above price and therefore valid.
    target = projected_breakout_target(100.0, 101.0, 1.03)
    assert target == 103.0


def test_breakout_target_disappears_when_already_exceeded() -> None:
    assert math.isnan(projected_breakout_target(100.0, 104.0, 1.03))


def test_tp2_ignores_analyst_target_by_design() -> None:
    # No analyst parameter exists in the operational selector. A distant
    # consensus target can therefore no longer inflate productive CRV.
    selected = select_operational_tp2(
        price=100.0,
        risk_per_share=5.0,
        setup_type="Trendfolge",
        technical_target=None,
        high52=None,
    )
    assert selected["value"] == 110.0
    assert selected["kind"] == "synthetic_2r_fallback"


def test_1_8r_floor_is_explicitly_marked_synthetic() -> None:
    selected = select_operational_tp2(
        price=100.0,
        risk_per_share=5.0,
        setup_type="Breakout",
        technical_target=106.0,
        high52=130.0,
    )
    assert selected["value"] == 109.0
    assert selected["synthetic"] is True
    assert "1,8R" in selected["source"]
    assert selected["base_target"] == 106.0


def test_real_technical_target_keeps_its_source() -> None:
    selected = select_operational_tp2(
        price=100.0,
        risk_per_share=5.0,
        setup_type="Breakout",
        technical_target=112.0,
        high52=130.0,
    )
    assert selected["value"] == 112.0
    assert selected["kind"] == "technical"
    assert selected["synthetic"] is False
