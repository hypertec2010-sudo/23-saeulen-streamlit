"""Ticker normalization helpers; network search stays injectable."""
from __future__ import annotations


def normalize_input(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_ticker(value: str) -> str:
    return normalize_input(value).upper()


def candidate_variants(value: str):
    raw = normalize_input(value)
    upper = raw.upper()
    variants = [upper]
    if upper and "." not in upper and "=" not in upper and not upper.startswith("^"):
        variants.extend([f"{upper}.DE", f"{upper}.MI", f"{upper}.PA"])
    return list(dict.fromkeys(v for v in variants if v))
