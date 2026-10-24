from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from plotly.subplots import make_subplots

from modules import chart_overlays
from modules.live_monitor import build_decision_state_v3021az
from modules.technical_crv import (
    build_technical_crv_package, build_crv_basis_comparison,
    crv_soft_gate_state, parse_price_zone,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def meta():
    return json.loads((ROOT / "tests/fixtures/meta_crv_basis.json").read_text(encoding="utf-8"))


@pytest.fixture
def radar_rr():
    # Run the actual UI-free Radar functions without executing the whole app.
    names = {
        "_radar_v18_clip", "_radar_v182_first_non_empty", "_radar_v182_num",
        "_radar_v182_parse_zone", "_radar_v182_get_price",
        "_radar_v182_entry_zone_text", "_radar_v182_stop_value",
        "build_radar_entry_rr_package_v182",
    }
    tree = ast.parse((ROOT / "legacy_app.py").read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    namespace = {
        "np": np, "pd": pd, "re": re,
        "_build_technical_crv_package_v3021aj": build_technical_crv_package,
        "_crv_soft_gate_state_v3021aj": crv_soft_gate_state,
        "_parse_crv_price_zone_v3021be": parse_price_zone,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "radar_functions", "exec"), namespace)
    return namespace["build_radar_entry_rr_package_v182"]


def decision(now, entry, **overrides):
    values = dict(
        valid_trade_setup=True, signal_packages_complete=True,
        timing_score=85.9, conf_score=70.9, crv_now=now, crv_entry=entry,
        trigger_status="Beobachten", entry_distance_pct=0,
        crv_now_eligible=True, crv_entry_eligible=True,
    )
    values.update(overrides)
    return build_decision_state_v3021az(**values)


def test_meta_chart_and_radar_use_identical_operational_inputs(meta, radar_rr):
    overlay = chart_overlays.build_trade_setup_overlay_v193(meta)
    radar = radar_rr(meta)
    assert overlay["crv"] == round(radar["crv"], 2) == 0.93
    assert round(overlay["crv_entry"], 2) == round(radar["crv_entry"], 2) == 0.48
    assert overlay["crv_target"] == radar["tp1"] == 742.06
    assert (overlay["entry_low"], overlay["entry_high"]) == (711.86, 726.24)
    now, entry, setup = overlay["crv_basis"]
    assert (now["reference"], now["stop"], now["target"]) == (718.67, 693.52, 742.06)
    assert (entry["reference"], entry["stop"]) == (726.24, 693.52)
    assert round(setup["crv"], 2) == 2.34
    assert setup["target"] == 777.52
    assert setup["operative"] is False
    assert decision(radar["crv"], radar["crv_entry"])["state"] == "plan"


def test_live_marker_does_not_silently_change_operational_crv(meta, radar_rr):
    meta["live_price"] = 735.0
    overlay = chart_overlays.build_trade_setup_overlay_v193(meta)
    radar = radar_rr(meta)
    assert overlay["price"] == 735.0
    assert overlay["crv"] == round(radar["crv"], 2) == 0.93
    assert radar["price"] == overlay["crv_basis"][0]["reference"] == 718.67
    assert radar["crv_price_reference"] == 718.67 and radar["crv_price_source"] == "price"
    assert overlay["crv_basis"][2]["reference"] == 735.0
    assert overlay["crv_basis"][2]["price_source"] == "live_price"
    assert radar["entry_distance_pct"] == 0


@pytest.mark.parametrize("text", [
    "711.86-726.24", "711.86 - 726.24", "711,86–726,24 USD",
    "711.86 — 726.24", (711.86, 726.24),
])
def test_price_zone_separators_are_not_negative_prices(text):
    assert parse_price_zone(text) == (711.86, 726.24)


@pytest.mark.parametrize("reward,expected", [(7.48, "Eng"), (7.5, "Ab 1,50"), (7.51, "Ab 1,50")])
def test_informational_crv_comparison_uses_full_precision(radar_rr, reward, expected):
    result = {
        "price": 100, "stop_used": 95, "suggested_entry_zone": (100, 100),
        "technical_target_1": 100 + reward, "setup_type": "Trendfolge",
    }
    radar = radar_rr(result)
    assert radar["crv"] == pytest.approx(reward / 5)
    ds=decision(radar["crv"], radar["crv_entry"], trigger_status="Aktiv")
    assert ds["state"] == "buy"
    assert ds["crv_assessment"].startswith(expected)


def test_synthetic_and_analyst_targets_cannot_create_operational_release():
    result = {
        "price": 100, "stop_used": 95, "tp2": 110,
        "tp2_source": "2R-Fallback", "tp2_is_synthetic": True,
        "target": 150, "suggested_entry_zone": "99–101",
    }
    pkg = build_technical_crv_package(result)
    assert math.isnan(pkg["technical_crv_now"])
    setup = build_crv_basis_comparison(result, pkg)[2]
    assert setup["crv"] == 2
    assert setup["synthetic"] is True and setup["operative"] is False
    assert decision(pkg["technical_crv_now"], pkg["technical_crv_entry"])["state"] == "plan"


def test_distant_setup_target_remains_separate_from_near_resistance(meta):
    meta["technical_target_1"] = 779.82
    pkg = build_technical_crv_package(meta)
    assert pkg["technical_crv_target"] == 742.06
    assert round(pkg["technical_crv_now"], 2) == 0.93
    assert pkg["technical_crv_alternative_target"] > pkg["technical_crv_target"]


def test_breakout_trigger_is_obstacle_and_next_resistance_is_target():
    result = {
        "price": 100, "stop_used": 95, "setup_type": "Breakout",
        "suggested_entry_zone": (100, 101),
        "chart_structures_analysis": {"zones": [
            {"low": 101, "high": 102, "touches": 2},
            {"low": 110, "high": 111, "touches": 3},
        ]},
    }
    pkg = build_technical_crv_package(result)
    assert pkg["technical_clearance_target"] == 101
    assert pkg["technical_crv_target"] == 110
    assert pkg["technical_crv_now"] == 2


@pytest.mark.parametrize("overrides,expected", [
    ({}, "plan"),
    ({"trigger_status": "Aktiv"}, "buy"),
    ({"trigger_status": "inaktiv"}, "plan"),
    ({"trigger_status": "Aktiv", "entry_hard_gate": True}, "blocked"),
    ({"trigger_status": "Aktiv", "invalidated": True}, "blocked"),
    ({"signal_packages_complete": False}, "observe"),
    ({"timing_score": 59.9}, "observe"),
    ({"conf_score": 59.9}, "observe"),
    ({"trigger_status": "Aktiv", "timing_score": 69.9}, "plan"),
    ({"trigger_status": "Aktiv", "conf_score": 64.9}, "plan"),
    ({"trigger_status": "Aktiv", "entry_distance_pct": 5.1}, "wait_entry"),
])
def test_planning_and_buy_remain_distinct(overrides, expected):
    ds = decision(2, 2, **overrides)
    assert ds["state"] == expected
    assert ds["buy_now"] == (expected == "buy")


def test_entry_crv_difference_is_context_not_a_location_gate():
    assert decision(0.9, 1.6, trigger_status="Aktiv")["state"] == "buy"


def test_missing_or_broken_stop_cannot_show_good_chart_crv(meta):
    meta["stop_used"] = 725
    assert chart_overlays.build_trade_setup_overlay_v193(meta)["crv"] is None


def test_chart_draws_the_actual_operational_target(meta):
    overlay = chart_overlays.build_trade_setup_overlay_v193(meta)
    df = pd.DataFrame(index=pd.date_range("2026-10-08", periods=2))
    fig = make_subplots(rows=1, cols=1)
    chart_overlays.add_trade_setup_overlay_to_plotly_v193(fig, df, overlay)
    target = next(trace for trace in fig.data if trace.name == "Operatives CRV-Ziel")
    assert tuple(target.y) == (742.06, 742.06)
