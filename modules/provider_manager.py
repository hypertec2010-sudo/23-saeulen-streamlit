"""Central market-data provider facade introduced in v28.4.5a.

Phase A intentionally keeps Yahoo/yfinance as the only active provider.  The
rest of the application calls this facade at the main history/info entry
points so retries, fallback providers and diagnostics can be added centrally
in later v28.4.5 releases without changing the trading logic again.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, Optional

import pandas as pd


class MarketDataError(RuntimeError):
    """Base exception for provider failures."""


class MarketDataRateLimitError(MarketDataError):
    """Raised when the upstream provider reports a rate limit."""


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    symbol: str
    operation: str
    ok: bool
    message: str = ""
    timestamp: float = 0.0


def _looks_rate_limited(exc: BaseException) -> bool:
    text = str(exc or "").lower()
    needles = (
        "too many requests",
        "rate limit",
        "rate-limited",
        "ratelimit",
        "http 429",
        "status code 429",
        "response code = 429",
    )
    return any(token in text for token in needles)


class YahooProvider:
    """Thin, defensive wrapper around yfinance.

    No long sleeps or automatic retries are performed in v28.4.5a.  This is
    deliberate: the live scanner already has checkpointing and should not be
    blocked by hidden waits.  v28.4.5c will add the explicit retry queue.
    """

    name = "yahoo"

    def __init__(self):
        import yfinance as yf

        self._yf = yf

    def ticker(self, symbol: str):
        return self._yf.Ticker(str(symbol or "").strip())

    def history(self, symbol: str, **kwargs) -> pd.DataFrame:
        try:
            frame = self.ticker(symbol).history(**kwargs)
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:
            if _looks_rate_limited(exc):
                raise MarketDataRateLimitError(str(exc)) from exc
            raise MarketDataError(str(exc)) from exc

    def download(self, symbol: str, **kwargs) -> pd.DataFrame:
        try:
            frame = self._yf.download(str(symbol or "").strip(), **kwargs)
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:
            if _looks_rate_limited(exc):
                raise MarketDataRateLimitError(str(exc)) from exc
            raise MarketDataError(str(exc)) from exc


class MarketDataProvider:
    """Single application facade for external market-data access."""

    def __init__(self, primary: Optional[YahooProvider] = None):
        self.primary = primary or YahooProvider()
        self._lock = threading.Lock()
        self._status: Dict[str, ProviderStatus] = {}

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return str(symbol or "").strip().upper()

    def _record(self, symbol: str, operation: str, ok: bool, message: str = "") -> None:
        key = f"{self.normalize_symbol(symbol)}::{operation}"
        status = ProviderStatus(
            provider=self.primary.name,
            symbol=self.normalize_symbol(symbol),
            operation=operation,
            ok=bool(ok),
            message=str(message or "")[:400],
            timestamp=time.time(),
        )
        with self._lock:
            self._status[key] = status

    def last_status(self, symbol: str, operation: str = "history") -> Optional[ProviderStatus]:
        key = f"{self.normalize_symbol(symbol)}::{operation}"
        with self._lock:
            return self._status.get(key)

    def get_ticker(self, symbol: str):
        clean = self.normalize_symbol(symbol)
        if not clean:
            raise MarketDataError("Ticker fehlt.")
        try:
            obj = self.primary.ticker(clean)
            self._record(clean, "ticker", True)
            return obj
        except Exception as exc:
            self._record(clean, "ticker", False, str(exc))
            if _looks_rate_limited(exc):
                raise MarketDataRateLimitError(str(exc)) from exc
            raise MarketDataError(str(exc)) from exc

    def get_history(self, symbol: str, **kwargs) -> pd.DataFrame:
        clean = self.normalize_symbol(symbol)
        if not clean:
            return pd.DataFrame()
        try:
            frame = self.primary.history(clean, **kwargs)
            self._record(clean, "history", True, f"rows={len(frame)}")
            return frame
        except MarketDataRateLimitError as exc:
            self._record(clean, "history", False, f"rate_limit: {exc}")
            raise
        except Exception as exc:
            self._record(clean, "history", False, str(exc))
            raise MarketDataError(str(exc)) from exc

    def get_info_bundle(self, symbol: str) -> tuple[Any, Dict[str, Any]]:
        """Return the native ticker object and a merged lightweight info dict.

        Keeping the ticker object is important because the existing analysis
        engine still derives statements/earnings from yfinance in this phase.
        """
        ticker = self.get_ticker(symbol)
        info: Dict[str, Any] = {}
        for getter in (
            lambda: getattr(ticker, "fast_info", {}) or {},
            lambda: ticker.get_info() or {},
            lambda: ticker.info or {},
        ):
            try:
                part = getter()
                if isinstance(part, dict):
                    for key, value in part.items():
                        if key not in info or info.get(key) in (None, ""):
                            info[key] = value
            except Exception as exc:
                if _looks_rate_limited(exc):
                    self._record(symbol, "info", False, f"rate_limit: {exc}")
                    # Preserve partial information. The full retry policy comes
                    # in v28.4.5c rather than aborting the analysis here.
                    continue
        self._record(symbol, "info", True, f"fields={len(info)}")
        return ticker, info


_default_provider: Optional[MarketDataProvider] = None
_default_lock = threading.Lock()


def get_market_data_provider() -> MarketDataProvider:
    global _default_provider
    if _default_provider is None:
        with _default_lock:
            if _default_provider is None:
                _default_provider = MarketDataProvider()
    return _default_provider
