# -*- coding: utf-8 -*-
"""Deterministic batch planning helpers for the Live-Screener.

The module keeps ticker normalization, scan-limit selection, batch splitting,
checkpoint metadata and global result sorting independent from Streamlit. This
makes the formerly silent 40-item truncation visible and testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

SCAN_SCOPE_OPTIONS = ("40 Werte", "80 Werte", "120 Werte", "Alle Werte")
DEFAULT_SCAN_SCOPE = "Alle Werte"
DEFAULT_BATCH_SIZE = 20


@dataclass(frozen=True)
class ScanPlan:
    source_count: int
    unique_tickers: tuple[str, ...]
    selected_tickers: tuple[str, ...]
    deferred_tickers: tuple[str, ...]
    duplicate_tickers: tuple[str, ...]
    scope_label: str

    @property
    def total(self) -> int:
        return len(self.selected_tickers)


def normalize_tickers(tickers: Iterable[Any] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return stable unique symbols and duplicate occurrences.

    Empty values are ignored. Duplicate symbols are reported in occurrence
    order so the UI can explain why the source row count differs from the
    number of unique symbols.
    """
    unique: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for raw in tickers or []:
        ticker = str(raw or "").strip().upper()
        if not ticker:
            continue
        if ticker in seen:
            duplicates.append(ticker)
            continue
        seen.add(ticker)
        unique.append(ticker)
    return tuple(unique), tuple(duplicates)


def resolve_limit(scope_label: Any, unique_count: int) -> int:
    """Translate a UI scope to a safe item count."""
    total = max(0, int(unique_count or 0))
    label = str(scope_label or DEFAULT_SCAN_SCOPE).strip()
    if label == "Alle Werte":
        return total
    try:
        requested = int(label.split()[0])
    except (TypeError, ValueError, IndexError):
        requested = total
    return min(total, max(0, requested))


def build_scan_plan(tickers: Iterable[Any] | None, scope_label: Any) -> ScanPlan:
    raw_items = list(tickers or [])
    unique, duplicates = normalize_tickers(raw_items)
    label = str(scope_label or DEFAULT_SCAN_SCOPE).strip()
    if label not in SCAN_SCOPE_OPTIONS:
        label = DEFAULT_SCAN_SCOPE
    limit = resolve_limit(label, len(unique))
    selected = unique[:limit]
    deferred = unique[limit:]
    return ScanPlan(
        source_count=len([item for item in raw_items if str(item or "").strip()]),
        unique_tickers=unique,
        selected_tickers=selected,
        deferred_tickers=deferred,
        duplicate_tickers=duplicates,
        scope_label=label,
    )


def split_batches(tickers: Sequence[str] | Iterable[str], batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[tuple[str, ...], ...]:
    items = tuple(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
    return tuple(tuple(items[index:index + size]) for index in range(0, len(items), size))


def completed_tickers(live_df: Any, live_errors: Any) -> tuple[str, ...]:
    """Read already processed symbols from a checkpoint cache."""
    ordered: list[str] = []
    seen: set[str] = set()
    for frame_index, frame in enumerate((live_df, live_errors)):
        df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame or [])
        if "Ticker" not in df.columns:
            continue
        for row_index, raw in enumerate(df["Ticker"].tolist()):
            # v28.4.5c: provider rate limits are temporary. Do not mark those
            # tickers completed in the checkpoint; the next scan resumes them.
            if frame_index == 1 and "Temporär" in df.columns:
                try:
                    if bool(df.iloc[row_index].get("Temporär", False)):
                        continue
                except Exception:
                    pass
            ticker = str(raw or "").strip().upper()
            if ticker and ticker not in seen:
                seen.add(ticker)
                ordered.append(ticker)
    return tuple(ordered)


def merge_frames(*frames: Any) -> pd.DataFrame:
    valid = [frame.copy() for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not valid:
        return pd.DataFrame()
    result = pd.concat(valid, ignore_index=True, sort=False)
    if "Ticker" in result.columns:
        normalized = result["Ticker"].astype(str).str.strip().str.upper()
        result = result.assign(__ticker_normalized=normalized)
        result = result.drop_duplicates(subset=["__ticker_normalized"], keep="last")
        result = result.drop(columns=["__ticker_normalized"], errors="ignore")
    return result.reset_index(drop=True)


def sort_live_frame(frame: Any) -> pd.DataFrame:
    df = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame or [])
    if df.empty:
        return df

    def ampel_rank(value: Any) -> int:
        icon = str(value or "").strip()[:1]
        return {"🟢": 0, "🟡": 1, "🔵": 2, "⚪": 3, "🔴": 4}.get(icon, 5)

    def score_value(value: Any) -> float:
        text = str(value or "").strip().replace(",", ".")
        if "/" in text:
            text = text.split("/", 1)[0]
        try:
            return float(text)
        except (TypeError, ValueError):
            return -1.0

    df["__v2844_ampel_rank"] = df.get("Ampel", pd.Series(index=df.index, dtype=object)).map(ampel_rank)
    df["__v2844_score"] = df.get("Live-Score", pd.Series(index=df.index, dtype=object)).map(score_value)
    if "Ticker" not in df.columns:
        df["Ticker"] = ""
    return (
        df.sort_values(["__v2844_ampel_rank", "__v2844_score", "Ticker"], ascending=[True, False, True])
        .drop(columns=["__v2844_ampel_rank", "__v2844_score"], errors="ignore")
        .reset_index(drop=True)
    )


def build_scan_meta(
    plan: ScanPlan,
    *,
    completed: Iterable[Any] = (),
    complete: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    completed_set = {
        str(ticker or "").strip().upper()
        for ticker in completed or []
        if str(ticker or "").strip()
    }
    completed_ordered = [ticker for ticker in plan.selected_tickers if ticker in completed_set]
    pending = [ticker for ticker in plan.selected_tickers if ticker not in completed_set]
    return {
        "version": 1,
        "complete": bool(complete and not pending),
        "scope_label": plan.scope_label,
        "source_count": plan.source_count,
        "unique_count": len(plan.unique_tickers),
        "selected_count": len(plan.selected_tickers),
        "completed_count": len(completed_ordered),
        "completed_tickers": completed_ordered,
        "pending_tickers": pending,
        "deferred_tickers": list(plan.deferred_tickers),
        "duplicate_tickers": list(plan.duplicate_tickers),
        "batch_size": max(1, int(batch_size or DEFAULT_BATCH_SIZE)),
    }


def checkpoint_matches(meta: Any, plan: ScanPlan) -> bool:
    if not isinstance(meta, Mapping):
        return False
    return (
        str(meta.get("scope_label") or "") == plan.scope_label
        and int(meta.get("selected_count") or -1) == len(plan.selected_tickers)
        and tuple(str(item).strip().upper() for item in (meta.get("pending_tickers") or []) + (meta.get("completed_tickers") or []))
        != ()
    )

# v30.21d: Merge a provider-light selective re-scan into the last complete
# watchlist snapshot without pretending that failed refreshes are fresh.
def merge_selective_refresh(
    base_live_df: Any,
    base_errors_df: Any,
    refreshed_live_df: Any,
    refreshed_errors_df: Any,
    selected_tickers: Iterable[Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Merge fresh selected rows into a complete prior snapshot.

    Successful selected tickers replace their previous live rows. Selected
    tickers that fail the refresh keep their previous live row (and therefore
    its old Scan-Zeit) while the new error replaces an older error entry. This
    keeps the visible baseline usable without mislabelling stale data as fresh.
    """
    selected, _ = normalize_tickers(selected_tickers)
    selected_set = set(selected)

    base_live = base_live_df.copy() if isinstance(base_live_df, pd.DataFrame) else pd.DataFrame(base_live_df or [])
    base_errors = base_errors_df.copy() if isinstance(base_errors_df, pd.DataFrame) else pd.DataFrame(base_errors_df or [])
    fresh_live = refreshed_live_df.copy() if isinstance(refreshed_live_df, pd.DataFrame) else pd.DataFrame(refreshed_live_df or [])
    fresh_errors = refreshed_errors_df.copy() if isinstance(refreshed_errors_df, pd.DataFrame) else pd.DataFrame(refreshed_errors_df or [])

    # Defensive filter: even if a caller accidentally supplies extra provider rows,
    # a selective refresh may only mutate explicitly requested tickers.
    if not fresh_live.empty and "Ticker" in fresh_live.columns:
        fresh_live = fresh_live.loc[
            fresh_live["Ticker"].astype(str).str.strip().str.upper().isin(selected_set)
        ].reset_index(drop=True)
    if not fresh_errors.empty and "Ticker" in fresh_errors.columns:
        fresh_errors = fresh_errors.loc[
            fresh_errors["Ticker"].astype(str).str.strip().str.upper().isin(selected_set)
        ].reset_index(drop=True)

    def _ticker_set(frame: pd.DataFrame) -> set[str]:
        if frame.empty or "Ticker" not in frame.columns:
            return set()
        return {
            str(value or "").strip().upper()
            for value in frame["Ticker"].tolist()
            if str(value or "").strip()
        }

    success_set = _ticker_set(fresh_live) & selected_set
    error_set = (_ticker_set(fresh_errors) & selected_set) - success_set

    if not base_live.empty and "Ticker" in base_live.columns and success_set:
        keep_mask = ~base_live["Ticker"].astype(str).str.strip().str.upper().isin(success_set)
        base_live = base_live.loc[keep_mask].reset_index(drop=True)
    merged_live = merge_frames(base_live, fresh_live)

    # A fresh attempt supersedes an earlier error for every requested ticker.
    if not base_errors.empty and "Ticker" in base_errors.columns and selected_set:
        keep_error_mask = ~base_errors["Ticker"].astype(str).str.strip().str.upper().isin(selected_set)
        base_errors = base_errors.loc[keep_error_mask].reset_index(drop=True)
    if not fresh_errors.empty and "Ticker" in fresh_errors.columns and success_set:
        fresh_errors = fresh_errors.loc[
            ~fresh_errors["Ticker"].astype(str).str.strip().str.upper().isin(success_set)
        ].reset_index(drop=True)
    merged_errors = merge_frames(base_errors, fresh_errors)

    success_ordered = tuple(ticker for ticker in selected if ticker in success_set)
    error_ordered = tuple(ticker for ticker in selected if ticker in error_set)
    return merged_live, merged_errors, success_ordered, error_ordered



# v30.21e: Selection helpers for checkbox-driven selective re-scans.
def green_tickers(frame: Any) -> tuple[str, ...]:
    """Return visible green tickers in stable row order.

    The helper deliberately uses the already rendered/live Ampel state instead
    of recalculating any trading rule. It is a convenience preselection only.
    """
    df = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame or [])
    if df.empty or "Ticker" not in df.columns:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker") or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        ampel = str(row.get("Ampel") or "").strip().lower()
        if "🟢" in ampel or "grün" in ampel or "gruen" in ampel:
            seen.add(ticker)
            ordered.append(ticker)
    return tuple(ordered)


def selected_tickers_from_editor(frame: Any, selection_column: str = "🔄") -> tuple[str, ...]:
    """Read checked ticker rows from a Streamlit data-editor result."""
    df = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame or [])
    if df.empty or "Ticker" not in df.columns or selection_column not in df.columns:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        try:
            checked = bool(row.get(selection_column, False))
        except Exception:
            checked = False
        if not checked:
            continue
        ticker = str(row.get("Ticker") or "").strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    return tuple(ordered)
