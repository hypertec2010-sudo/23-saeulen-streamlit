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
    for frame in (live_df, live_errors):
        df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame or [])
        if "Ticker" not in df.columns:
            continue
        for raw in df["Ticker"].tolist():
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
