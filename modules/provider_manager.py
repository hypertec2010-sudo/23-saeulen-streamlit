"""Central market-data provider facade introduced in v28.4.5a.

Phase A intentionally keeps Yahoo/yfinance as the only active provider.  The
rest of the application calls this facade at the main history/info entry
points so retries, fallback providers and diagnostics can be added centrally
in later v28.4.5 releases without changing the trading logic again.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
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


@dataclass(frozen=True)
class ProviderHealth:
    provider: str
    symbol: str
    checked_at: float
    overall: str
    history_ok: Optional[bool]
    info_ok: Optional[bool]
    rate_limited: bool
    history_message: str = ""
    info_message: str = ""
    yfinance_version: str = ""
    curl_cffi_installed: bool = False
    curl_cffi_active: Optional[bool] = None
    http_backend: str = ""


def _safe_health_message(exc: BaseException) -> str:
    """Return a compact, non-sensitive provider-health message."""
    text = str(exc or "").replace("\n", " ").replace("\r", " ").strip()
    lowered = text.lower()
    if _looks_rate_limited(exc):
        return "Yahoo begrenzt die Abfragen voruebergehend (Rate-Limit)."
    if "timeout" in lowered or "timed out" in lowered:
        return "Provider-Abfrage hat das Zeitlimit ueberschritten."
    if "connection" in lowered or "network" in lowered:
        return "Provider-Verbindung konnte nicht hergestellt werden."
    if not text:
        return type(exc).__name__
    return f"{type(exc).__name__}: {text[:160]}"


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
        # v28.4.5c: one shared throttle/cooldown for all Yahoo requests.
        # This avoids a large watchlist hammering Yahoo immediately after a 429.
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0
        self._cooldown_until = 0.0
        self._min_request_gap = 0.45
        # v30.21r: a tiny cached health probe. It deliberately runs far less
        # often than normal app reruns so the diagnostic itself cannot create
        # provider pressure.
        self._health_lock = threading.Lock()
        self._health_cache: Optional[ProviderHealth] = None
        self._health_cache_ttl = 15 * 60.0

    def _pace_request(self) -> None:
        with self._request_lock:
            now = time.monotonic()
            wait_for = max(0.0, self._cooldown_until - now, self._min_request_gap - (now - self._last_request_at))
            if wait_for > 0:
                time.sleep(min(wait_for, 12.0))
            self._last_request_at = time.monotonic()

    def _note_rate_limit(self, attempt: int = 0) -> None:
        # Short exponential cooldown. The scanner retries only the affected ticker;
        # no full-watchlist restart is necessary.
        cooldown = (2.0, 5.0, 10.0)[min(max(int(attempt), 0), 2)]
        with self._request_lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + cooldown)

    def is_rate_limit_error(self, exc: BaseException) -> bool:
        return isinstance(exc, MarketDataRateLimitError) or _looks_rate_limited(exc)

    def mark_rate_limited(self, symbol: str = "AAPL", *, source: str = "Yahoo") -> ProviderHealth:
        """Publish a rate-limit observation from any app path into health state."""
        clean = self.normalize_symbol(symbol) or "AAPL"
        yf_version, curl_installed, curl_active, backend = self._http_backend_health()
        result = ProviderHealth(
            provider=self.primary.name,
            symbol=clean,
            checked_at=time.time(),
            overall="rate_limited",
            history_ok=False,
            info_ok=None,
            rate_limited=True,
            history_message=f"{source}: Yahoo begrenzt die Abfragen voruebergehend (Rate-Limit).",
            info_message="Wegen aktivem Rate-Limit nicht zusaetzlich getestet.",
            yfinance_version=yf_version,
            curl_cffi_installed=bool(curl_installed),
            curl_cffi_active=curl_active,
            http_backend=backend,
        )
        with self._health_lock:
            self._health_cache = result
        self._note_rate_limit(2)
        return result

    def _http_backend_health(self) -> tuple[str, bool, Optional[bool], str]:
        """Return yfinance/curl_cffi backend metadata without network access."""
        yf_obj = getattr(self.primary, "_yf", None)
        yf_version = str(getattr(yf_obj, "__version__", "") or "")
        try:
            curl_installed = importlib.util.find_spec("curl_cffi") is not None
        except Exception:
            curl_installed = False

        backend = ""
        curl_active: Optional[bool] = None
        try:
            yf_data = importlib.import_module("yfinance.data")
            requests_obj = getattr(yf_data, "requests", None)
            module_name = str(getattr(requests_obj, "__name__", "") or "")
            if module_name:
                backend = module_name
                curl_active = "curl_cffi" in module_name.lower()
        except Exception:
            pass

        if not backend:
            try:
                yf_data = importlib.import_module("yfinance.data")
                data_cls = getattr(yf_data, "YfData", None)
                data_obj = data_cls() if data_cls is not None else None
                session = getattr(data_obj, "_session", None)
                if session is not None:
                    backend = f"{type(session).__module__}.{type(session).__name__}"
                    curl_active = "curl_cffi" in backend.lower()
            except Exception:
                pass

        if curl_active is None and not curl_installed:
            curl_active = False
        return yf_version, curl_installed, curl_active, backend

    def health_check(self, *, force: bool = False, symbol: str = "AAPL") -> ProviderHealth:
        """Run a low-cost Yahoo/yfinance health probe with a 15-minute cache.

        The probe intentionally performs at most one lightweight history call
        and, only when history succeeds, one fast-info lookup. It never enters
        the normal multi-retry/fallback history path.
        """
        now = time.time()
        with self._health_lock:
            cached = self._health_cache
            if (
                not force
                and cached is not None
                and (now - float(cached.checked_at or 0.0)) < self._health_cache_ttl
            ):
                return cached

            clean = self.normalize_symbol(symbol) or "AAPL"
            resolved = self.resolve_symbol(clean) or clean
            yf_version, curl_installed, curl_active, backend = self._http_backend_health()

            history_ok: Optional[bool] = None
            info_ok: Optional[bool] = None
            history_message = ""
            info_message = ""
            rate_limited = False

            try:
                self._pace_request()
                frame = self.primary.history(
                    resolved,
                    period="5d",
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                )
                history_ok = isinstance(frame, pd.DataFrame) and not frame.empty
                history_message = (
                    f"{len(frame)} Tageszeilen empfangen."
                    if history_ok
                    else "Yahoo lieferte keine Kurszeilen."
                )
            except Exception as exc:
                history_ok = False
                rate_limited = self.is_rate_limit_error(exc)
                history_message = _safe_health_message(exc)
                if rate_limited:
                    self._note_rate_limit(2)

            # Do not create an additional Yahoo request while an active rate
            # limit is already established by the history probe.
            if rate_limited:
                info_ok = None
                info_message = "Wegen aktivem Rate-Limit nicht zusaetzlich getestet."
            elif history_ok:
                try:
                    ticker = self.primary.ticker(resolved)
                    self._pace_request()
                    fast = getattr(ticker, "fast_info", None)
                    last_price = None
                    if fast is not None:
                        try:
                            last_price = fast.get("last_price")
                        except Exception:
                            try:
                                last_price = fast["last_price"]
                            except Exception:
                                last_price = None
                    info_ok = last_price is not None
                    info_message = (
                        "Fast-Info erreichbar."
                        if info_ok
                        else "Fast-Info lieferte keinen letzten Kurs."
                    )
                except Exception as exc:
                    info_ok = False
                    rate_limited = rate_limited or self.is_rate_limit_error(exc)
                    info_message = _safe_health_message(exc)
                    if self.is_rate_limit_error(exc):
                        self._note_rate_limit(2)
            else:
                info_ok = None
                info_message = "Nicht getestet, weil der Kursdaten-Test bereits fehlgeschlagen ist."

            if rate_limited:
                overall = "rate_limited"
            elif history_ok and info_ok:
                overall = "ok"
            elif history_ok:
                overall = "partial"
            else:
                overall = "down"

            result = ProviderHealth(
                provider=self.primary.name,
                symbol=clean,
                checked_at=now,
                overall=overall,
                history_ok=history_ok,
                info_ok=info_ok,
                rate_limited=bool(rate_limited),
                history_message=history_message,
                info_message=info_message,
                yfinance_version=yf_version,
                curl_cffi_installed=bool(curl_installed),
                curl_cffi_active=curl_active,
                http_backend=backend,
            )
            self._health_cache = result
            return result

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
            for attempt in range(3):
                try:
                    self._pace_request()
                    frame = _clean_frame(self.primary.history(resolved, **request_kwargs))
                    if not frame.empty:
                        break
                except MarketDataRateLimitError as exc:
                    last_rate_exc = exc
                    self._note_rate_limit(attempt)
                    continue

            is_daily = str(request_kwargs.get("interval") or "1d").lower() in {"1d", "1day", "day"}

            # Direct Yahoo chart fallback. This bypasses yfinance's history
            # parser/cache and is the preferred recovery path for SKHY/SPCX.
            if is_daily and len(frame) < 10:
                try:
                    self._pace_request()
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
                    self._pace_request()
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
                    self._pace_request()
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
            self.mark_rate_limited(clean, source="Kursdaten")
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
            for attempt in range(2):
                try:
                    self._pace_request()
                    part = getter()
                    if isinstance(part, dict):
                        for key, value in part.items():
                            if key not in info or info.get(key) in (None, ""):
                                info[key] = value
                    break
                except Exception as exc:
                    if _looks_rate_limited(exc):
                        self._note_rate_limit(attempt)
                        self._record(symbol, "info", False, f"rate_limit retry {attempt + 1}: {exc}")
                        continue
                    break
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
