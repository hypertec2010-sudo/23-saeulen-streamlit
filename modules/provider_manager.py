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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd
import requests

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



def _period_start_epoch(period: str) -> int:
    now = datetime.now(timezone.utc)
    text = str(period or "3y").strip().lower()
    try:
        if text.endswith("d"):
            delta = timedelta(days=max(1, int(text[:-1])))
        elif text.endswith("mo"):
            delta = timedelta(days=max(1, int(text[:-2])) * 31)
        elif text.endswith("y"):
            delta = timedelta(days=max(1, int(text[:-1])) * 366)
        elif text == "max":
            return 0
        else:
            delta = timedelta(days=3 * 366)
    except Exception:
        delta = timedelta(days=3 * 366)
    return int((now - delta).timestamp())


def _raw_yahoo_chart_history(symbol: str, **kwargs) -> pd.DataFrame:
    """Direct Yahoo chart endpoint fallback.

    This is intentionally independent of yfinance's parsing/cache path. It is
    especially useful for very new/recycled symbols where ``Ticker.history``
    can temporarily return an empty frame even though Yahoo already exposes
    chart candles.
    """
    interval = str(kwargs.get("interval") or "1d")
    start = kwargs.get("start")
    end = kwargs.get("end")
    period = kwargs.get("period") or "3y"
    if start:
        try:
            period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
        except Exception:
            period1 = _period_start_epoch(period)
    else:
        period1 = _period_start_epoch(period)
    if end:
        try:
            period2 = int(pd.Timestamp(end, tz="UTC").timestamp())
        except Exception:
            period2 = int(datetime.now(timezone.utc).timestamp()) + 86400
    else:
        period2 = int(datetime.now(timezone.utc).timestamp()) + 86400

    params = {
        "period1": period1,
        "period2": period2,
        "interval": interval,
        "events": "div,splits,capitalGains",
        "includeAdjustedClose": "true",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    last_exc = None
    for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{symbol}"
        try:
            response = requests.get(url, params=params, headers=headers, timeout=12)
            if response.status_code == 429:
                raise MarketDataRateLimitError("Yahoo chart endpoint HTTP 429")
            response.raise_for_status()
            payload = response.json() or {}
            chart = payload.get("chart") or {}
            err = chart.get("error")
            if err:
                raise MarketDataError(str(err))
            results = chart.get("result") or []
            if not results:
                continue
            result = results[0] or {}
            ts = result.get("timestamp") or []
            indicators = result.get("indicators") or {}
            quotes = indicators.get("quote") or []
            if not ts or not quotes:
                continue
            q = quotes[0] or {}
            adj_sets = indicators.get("adjclose") or []
            adj = (adj_sets[0] or {}).get("adjclose") if adj_sets else None
            idx = pd.to_datetime(ts, unit="s", utc=True)
            data = {
                "Open": q.get("open", []),
                "High": q.get("high", []),
                "Low": q.get("low", []),
                "Close": q.get("close", []),
                "Volume": q.get("volume", []),
            }
            frame = pd.DataFrame(data, index=idx)
            if adj and len(adj) == len(frame):
                frame["Adj Close"] = adj
                if bool(kwargs.get("auto_adjust", False)):
                    raw_close = pd.to_numeric(frame["Close"], errors="coerce")
                    adj_close = pd.to_numeric(frame["Adj Close"], errors="coerce")
                    ratio = adj_close / raw_close.replace(0, pd.NA)
                    for col in ("Open", "High", "Low"):
                        frame[col] = pd.to_numeric(frame[col], errors="coerce") * ratio
                    frame["Close"] = adj_close
            frame = frame.dropna(subset=["Close"]).sort_index()
            return frame
        except MarketDataRateLimitError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            last_exc = exc
            continue
    if isinstance(last_exc, MarketDataRateLimitError):
        raise last_exc
    return pd.DataFrame()

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
            frame = pd.DataFrame()
            last_rate_exc = None
            # Controlled retry: enough to survive a short Yahoo throttle without
            # blocking a whole watchlist scan for minutes.
            for attempt, delay in enumerate((0.0, 2.0, 6.0)):
                if delay:
                    time.sleep(delay)
                try:
                    frame = _clean_frame(self.primary.history(resolved, **request_kwargs))
                    if not frame.empty:
                        break
                except MarketDataRateLimitError as exc:
                    last_rate_exc = exc
                    continue

            is_daily = str(request_kwargs.get("interval") or "1d").lower() in {"1d", "1day", "day"}

            # Direct Yahoo chart fallback. This bypasses yfinance's history
            # parser/cache and is the preferred recovery path for SKHY/SPCX.
            if is_daily and len(frame) < 10:
                try:
                    raw = _clean_frame(_raw_yahoo_chart_history(resolved, **request_kwargs))
                    if len(raw) > len(frame):
                        frame = raw
                except MarketDataRateLimitError as exc:
                    last_rate_exc = exc
                except Exception:
                    pass

            # Defensive second path for young listings if Ticker.history()
            # returns an unexpectedly short/empty frame. yf.download uses a
            # separate yfinance path and often succeeds when Ticker.history
            # temporarily does not. Do not use period=max for recycled tickers.
            if is_daily and len(frame) < 10:
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
            if is_daily and len(frame) < 10 and not resolution.history_start:
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

            if len(frame) == 0 and last_rate_exc is not None:
                raise last_rate_exc

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
