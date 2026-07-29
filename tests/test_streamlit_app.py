from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st

if getattr(st, "__capital_hill_test_stub__", False):
    pytest.skip("Streamlit ist in dieser Umgebung nicht installiert", allow_module_level=True)

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.streamlit
def test_real_entrypoint_renders_login_gate_without_exception() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
    app.run()
    assert not app.exception
    assert app.title
    assert "23 Säulen Analyse" in app.title[0].value


@pytest.mark.streamlit
def test_cockpit_choice_survives_normal_reruns() -> None:
    app = AppTest.from_file(str(ROOT / "tests/fixtures/cockpit_state_app.py"), default_timeout=15)
    app.run()
    assert not app.exception
    assert app.radio[0].value == "📡 Live-Screener"

    app.radio[0].set_value("📓 Trade-Journal").run()
    assert not app.exception
    assert app.radio[0].value == "📓 Trade-Journal"

    app.button[0].click().run()
    assert not app.exception
    assert app.radio[0].value == "📓 Trade-Journal"

    app.radio[0].set_value("🧾 Historie & Details").run()
    assert app.radio[0].value == "🧾 Historie & Details"


@pytest.mark.streamlit
def test_refresh_policy_is_visible_and_becomes_due() -> None:
    app = AppTest.from_file(str(ROOT / "tests/fixtures/refresh_policy_app.py"), default_timeout=15)
    app.run()
    assert not app.exception
    assert app.metric[0].value == "Nein"

    app.number_input[1].set_value(900).run()
    assert not app.exception
    assert app.metric[0].value == "Ja"
