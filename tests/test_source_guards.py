from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_contains_ci_and_refresh_guards() -> None:
    legacy = (ROOT / "legacy_app.py").read_text(encoding="utf-8")
    runtime = (ROOT / "modules/page_runtime.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")

    assert 'APP_VERSION = "v28.4.3"' in legacy
    assert "_live_refresh_policy.evaluate_refresh" in legacy
    assert "run_every=_native_refresh_poll_seconds_v2832" in legacy
    assert "_live_screener_snapshot.load_snapshot" in legacy
    assert "Mobile-Modus" in legacy
    assert "Warum geändert?" in legacy
    assert (ROOT / "modules/live_change_explainer.py").exists()
    assert "if page_changed:" in runtime
    assert "python verify_deployment.py" in workflow
    assert "pytest -q" in workflow


def test_repository_contains_no_real_secrets() -> None:
    forbidden = ("service_role_key = \"ey", "sb_secret_", "client_secret = \"GOCSPX-")
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc", ".zip"}:
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in forbidden:
            assert marker not in text, f"Möglicher echter Secret-Wert in {path.relative_to(ROOT)}"
