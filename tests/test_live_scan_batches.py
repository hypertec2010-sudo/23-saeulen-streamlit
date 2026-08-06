from __future__ import annotations

import pandas as pd

from modules import live_scan_batches as batches


def _tickers(count: int) -> list[str]:
    return [f"T{index:03d}" for index in range(count)]


def test_all_scope_analyzes_every_unique_value_and_reports_duplicates() -> None:
    source = _tickers(55) + ["T010", " t011 ", ""]
    plan = batches.build_scan_plan(source, "Alle Werte")

    assert plan.source_count == 57
    assert len(plan.unique_tickers) == 55
    assert len(plan.selected_tickers) == 55
    assert plan.deferred_tickers == ()
    assert plan.duplicate_tickers == ("T010", "T011")


def test_explicit_limit_is_visible_as_deferred_not_silent() -> None:
    plan = batches.build_scan_plan(_tickers(55), "40 Werte")

    assert len(plan.selected_tickers) == 40
    assert len(plan.deferred_tickers) == 15
    assert plan.deferred_tickers[0] == "T040"


def test_batches_are_stable_and_checkpoint_can_resume() -> None:
    plan = batches.build_scan_plan(_tickers(45), "Alle Werte")
    chunks = batches.split_batches(plan.selected_tickers, 20)
    assert [len(chunk) for chunk in chunks] == [20, 20, 5]

    live_df = pd.DataFrame({"Ticker": list(chunks[0])})
    errors = pd.DataFrame({"Ticker": [chunks[1][0]], "Fehler": ["Keine Daten"]})
    completed = batches.completed_tickers(live_df, errors)
    meta = batches.build_scan_meta(plan, completed=completed, complete=False, batch_size=20)

    assert meta["completed_count"] == 21
    assert meta["complete"] is False
    assert len(meta["pending_tickers"]) == 24
    assert chunks[1][0] not in meta["pending_tickers"]


def test_global_merge_keeps_one_result_per_ticker_and_sorts_ampels() -> None:
    first = pd.DataFrame([
        {"Ticker": "B", "Ampel": "🔴", "Live-Score": "30/100"},
        {"Ticker": "A", "Ampel": "🟡", "Live-Score": "60/100"},
    ])
    second = pd.DataFrame([
        {"Ticker": "A", "Ampel": "🟢", "Live-Score": "80/100"},
        {"Ticker": "C", "Ampel": "⚪", "Live-Score": "45/100"},
    ])

    merged = batches.sort_live_frame(batches.merge_frames(first, second))
    assert merged["Ticker"].tolist() == ["A", "C", "B"]
    assert merged.iloc[0]["Ampel"] == "🟢"
