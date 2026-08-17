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

from modules.ticker_resolver import resolve_ticker


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

    @staticmethod
    def resolve_symbol(symbol: str) -> str:
        return resolve_ticker(symbol).provider_symbol

    @staticmethod
    def resolve(symbol: str):
        return resolve_ticker(symbol)

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
        resolved = self.resolve_symbol(clean)
        if not resolved:
            raise MarketDataError("Ticker fehlt.")
        try:
            obj = self.primary.ticker(resolved)
            self._record(clean, "ticker", True)
            return obj
        except Exception as exc:
            self._record(clean, "ticker", False, str(exc))
            if _looks_rate_limited(exc):
                raise MarketDataRateLimitError(str(exc)) from exc
            raise MarketDataError(str(exc)) from exc

    def get_history(self, symbol: str, **kwargs) -> pd.DataFrame:
        clean = self.normalize_symbol(symbol)
        resolution = self.resolve(clean)
        resolved = resolution.provider_symbol
        if not resolved:
            return pd.DataFrame()

        request_kwargs = dict(kwargs)
        # New/recycled listings must start at the current security's actual
        # listing date. This is crucial for SPCX, whose ticker was previously
        # used by a different ETF, and also avoids an empty long-period query
        # on very young listings such as SKHY.
        if resolution.history_start and not request_kwargs.get("interval"):
            request_kwargs.pop("period", None)
            request_kwargs["start"] = resolution.history_start

        def _clean_frame(frame):
            if not isinstance(frame, pd.DataFrame):
                return pd.DataFrame()
            out = frame.copy()
            try:
                if hasattr(out.columns, "nlevels") and out.columns.nlevels > 1:
                    out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
            except Exception:
                pass
            try:
                out = out[~out.index.duplicated(keep="last")].sort_index()
            except Exception:
                pass
            return out

        try:
            frame = _clean_frame(self.primary.history(resolved, **request_kwargs))

            # Defensive second path for young listings if Ticker.history()
            # returns an unexpectedly short/empty frame. yf.download uses a
            # separate yfinance path and often succeeds when Ticker.history
            # temporarily does not. Do not use period=max for recycled tickers.
            is_daily = str(request_kwargs.get("interval") or "1d").lower() in {"1d", "1day", "day"}
            if is_daily and len(frame) < 20:
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.setdefault("interval", "1d")
                retry_kwargs.setdefault("progress", False)
                retry_kwargs.setdefault("threads", False)
                try:
                    alt = _clean_frame(self.primary.download(resolved, **retry_kwargs))
                    if len(alt) > len(frame):
                        frame = alt
                except MarketDataRateLimitError:
                    raise
                except Exception:
                    pass

            # For ordinary (non-recycled) symbols only, a final max-history
            # retry can recover provider quirks without risking stale security
            # history.
            if is_daily and len(frame) < 20 and not resolution.history_start:
                try:
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs.pop("start", None)
                    fallback_kwargs.pop("end", None)
                    fallback_kwargs["period"] = "max"
                    alt = _clean_frame(self.primary.history(resolved, **fallback_kwargs))
                    if len(alt) > len(frame):
                        frame = alt
                except MarketDataRateLimitError:
                    raise
                except Exception:
                    pass

            self._record(
                clean,
                "history",
                True,
                f"provider_symbol={resolved}; rows={len(frame)}; start={resolution.history_start or '-'}",
            )
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
