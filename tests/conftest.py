from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("streamlit")
    stub.__capital_hill_test_stub__ = True
    stub.session_state = {}
    stub.query_params = {}
    sys.modules["streamlit"] = stub
