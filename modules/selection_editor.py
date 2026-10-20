"""Stable checkbox editor inputs across Streamlit reruns (UI state only)."""
from uuid import uuid4


def prepare_selection_editor(state, frame, *, selection_column, state_key,
                             widget_key, revision=0, reset=False):
    """Freeze widget input until the view or explicit selection revision changes.

    data_editor keeps edits separately from its input. Feeding its last returned
    checkboxes back as input changes widget identity and loses every other click.
    A changed view starts a fresh widget seeded by the caller's ticker selection;
    row-index edits from the previous view must never reach reordered rows.
    """
    baseline_key = state_key + "::editor_baseline_v3021bs"
    view = frame.drop(columns=[selection_column]).reset_index(drop=True)
    record = None if reset else state.get(baseline_key)
    if (record is None or record["widget_key"] not in state
            or record["revision"] != revision
            or not record["view"].equals(view)):
        record = {"view": view.copy(deep=True), "revision": revision,
                  "frame": frame.reset_index(drop=True).copy(deep=True),
                  "widget_key": widget_key + "::" + uuid4().hex}
        state[baseline_key] = record
    return record["frame"].copy(deep=True), record["widget_key"]
