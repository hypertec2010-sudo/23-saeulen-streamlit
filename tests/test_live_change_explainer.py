from __future__ import annotations

from modules.live_change_explainer import build_change_explanation


def test_yellow_to_red_explains_non_price_trigger() -> None:
    previous = {
        "ampel": "🟡",
        "status": "Nahe am Trigger",
        "price": 100.00,
        "live_score": "62/100",
        "radar_bucket": "Nahe am Trigger",
        "entry_hard_gate": False,
        "invalidated": False,
        "final_release_ok": True,
        "timing_component": 68,
        "conf_component": 64,
        "chart_component": 61,
        "trigger_component": 64,
    }
    current = {
        "ampel": "🔴",
        "status": "Setup blockiert",
        "price": 100.03,
        "live_score": "38/100",
        "radar_bucket": "Warnsignale / meiden",
        "entry_hard_gate": True,
        "invalidated": False,
        "final_release_ok": False,
        "timing_component": 41,
        "conf_component": 46,
        "chart_component": 52,
        "trigger_component": 42,
        "reason": "Hartes Einstiegsgate aktiv.",
    }

    text = build_change_explanation(previous, current, "Verschlechtert")
    assert "Kurs nahezu unverändert" in text
    assert "hartes Einstiegsgate" in text
    assert "Radar-Bucket" in text
    assert "Live-Score 62→38" in text


def test_unchanged_status_has_no_noise() -> None:
    previous = {"ampel": "🟡", "status": "Nahe am Trigger", "price": 100}
    current = {"ampel": "🟡", "status": "Nahe am Trigger", "price": 100.2}
    assert build_change_explanation(previous, current, "Unverändert") == "-"


def test_first_state_is_identified() -> None:
    text = build_change_explanation({}, {"ampel": "🟡", "status": "Neu"}, "Neu")
    assert "Erster gespeicherter Vergleichsstand" in text
