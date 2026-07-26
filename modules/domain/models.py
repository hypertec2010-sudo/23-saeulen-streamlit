"""Typed domain models for the trading application.

The UI and older modules still exchange legacy dictionaries.  These models form
an explicit boundary: repositories validate and normalise persisted payloads,
while ``to_legacy_dict`` keeps all currently used field names compatible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _number(value: Any, default=None):
    if value in (None, "", "n/a", "-"):
        return default
    try:
        number = float(value)
        if number != number:  # NaN
            return default
        return number
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    number = _number(value, None)
    return int(number) if number is not None else int(default)


def _iso_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return _text(value)


@dataclass(slots=True)
class Position:
    ticker: str
    name: str = ""
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    shares: int = 0
    initial_stop: float | None = None
    initial_shares: int = 0
    last_price: float | None = None
    realized_pnl: float = 0.0
    realized_shares: int = 0
    realized_r_weighted: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    stop_history: list[dict[str, Any]] = field(default_factory=list)
    journal_notes: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict, repr=False)
    _present: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    @classmethod
    def from_legacy_dict(cls, payload: Mapping[str, Any] | None, *, ticker: str = "") -> "Position":
        raw = dict(payload or {})
        resolved_ticker = _text(raw.get("ticker") or raw.get("Ticker") or ticker).upper()
        known = {
            "ticker", "Ticker", "name", "Name", "entry", "stop", "target", "shares",
            "initial_stop", "initial_shares", "last_price", "realized_pnl",
            "realized_shares", "realized_r_weighted", "created_at", "updated_at",
            "stop_history", "journal_notes",
        }
        return cls(
            ticker=resolved_ticker,
            name=_text(raw.get("name") or raw.get("Name") or resolved_ticker),
            entry=_number(raw.get("entry")),
            stop=_number(raw.get("stop")),
            target=_number(raw.get("target")),
            shares=_integer(raw.get("shares"), 0),
            initial_stop=_number(raw.get("initial_stop")),
            initial_shares=_integer(raw.get("initial_shares"), 0),
            last_price=_number(raw.get("last_price")),
            realized_pnl=float(_number(raw.get("realized_pnl"), 0.0) or 0.0),
            realized_shares=_integer(raw.get("realized_shares"), 0),
            realized_r_weighted=float(_number(raw.get("realized_r_weighted"), 0.0) or 0.0),
            created_at=_text(raw.get("created_at")),
            updated_at=_text(raw.get("updated_at")),
            stop_history=list(raw.get("stop_history") or []),
            journal_notes=list(raw.get("journal_notes") or []),
            extra={k: v for k, v in raw.items() if k not in known},
            _present=frozenset(raw.keys()),
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        out = dict(self.extra)
        values = {
            "ticker": self.ticker,
            "name": self.name,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "shares": self.shares,
            "initial_stop": self.initial_stop,
            "initial_shares": self.initial_shares,
            "last_price": self.last_price,
            "realized_pnl": self.realized_pnl,
            "realized_shares": self.realized_shares,
            "realized_r_weighted": self.realized_r_weighted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stop_history": list(self.stop_history),
            "journal_notes": list(self.journal_notes),
        }
        defaults = {
            "name": "", "entry": None, "stop": None, "target": None, "shares": 0,
            "initial_stop": None, "initial_shares": 0, "last_price": None,
            "realized_pnl": 0.0, "realized_shares": 0, "realized_r_weighted": 0.0,
            "created_at": "", "updated_at": "", "stop_history": [], "journal_notes": [],
        }
        for key, value in values.items():
            if key == "ticker" or key in self._present or value != defaults.get(key):
                out[key] = value
        return out

    @property
    def unit_risk(self) -> float | None:
        reference_stop = self.initial_stop if self.initial_stop is not None else self.stop
        if self.entry is None or reference_stop is None or self.entry <= reference_stop:
            return None
        return self.entry - reference_stop

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.ticker:
            errors.append("Ticker fehlt")
        if self.shares < 0:
            errors.append("Stueckzahl darf nicht negativ sein")
        if self.entry is not None and self.entry <= 0:
            errors.append("Entry muss groesser als 0 sein")
        if self.stop is not None and self.stop <= 0:
            errors.append("Stop muss groesser als 0 sein")
        return errors


@dataclass(slots=True)
class JournalEntry:
    entry_id: str
    time_text: str = ""
    action_date: str = ""
    watchlist: str = "Standard"
    ticker: str = ""
    name: str = ""
    action_type: str = "Journal-Eintrag"
    price: float | None = None
    shares: int = 0
    remaining_shares: int = 0
    entry_price: float | None = None
    initial_stop: float | None = None
    current_stop: float | None = None
    old_stop: float | None = None
    new_stop: float | None = None
    realized_pnl: float | None = None
    realized_pct: float | None = None
    realized_r: float | None = None
    total_pnl: float | None = None
    total_r: float | None = None
    note: str = ""
    learning: str = ""
    details: str = ""
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_legacy_dict(cls, payload: Mapping[str, Any] | None) -> "JournalEntry":
        raw = dict(payload or {})
        known = {
            "ID", "Zeit", "Datum", "Watchlist", "Ticker", "Name", "Typ", "Kurs",
            "Stueck", "Stück", "Verbleibend", "Entry", "Initial-Stop", "Aktueller Stop",
            "Alter Stop", "Neuer Stop", "Realisiert P/L", "Realisiert %", "Realisiert R",
            "Gesamt P/L", "Gesamt R", "Notiz", "Erkenntnis", "Details",
        }
        return cls(
            entry_id=_text(raw.get("ID")),
            time_text=_text(raw.get("Zeit")),
            action_date=_iso_date(raw.get("Datum")),
            watchlist=_text(raw.get("Watchlist"), "Standard") or "Standard",
            ticker=_text(raw.get("Ticker")).upper(),
            name=_text(raw.get("Name")),
            action_type=_text(raw.get("Typ"), "Journal-Eintrag") or "Journal-Eintrag",
            price=_number(raw.get("Kurs")),
            shares=_integer(raw.get("Stück", raw.get("Stueck")), 0),
            remaining_shares=_integer(raw.get("Verbleibend"), 0),
            entry_price=_number(raw.get("Entry")),
            initial_stop=_number(raw.get("Initial-Stop")),
            current_stop=_number(raw.get("Aktueller Stop")),
            old_stop=_number(raw.get("Alter Stop")),
            new_stop=_number(raw.get("Neuer Stop")),
            realized_pnl=_number(raw.get("Realisiert P/L")),
            realized_pct=_number(raw.get("Realisiert %")),
            realized_r=_number(raw.get("Realisiert R")),
            total_pnl=_number(raw.get("Gesamt P/L")),
            total_r=_number(raw.get("Gesamt R")),
            note=_text(raw.get("Notiz")),
            learning=_text(raw.get("Erkenntnis")),
            details=_text(raw.get("Details")),
            extra={k: v for k, v in raw.items() if k not in known},
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "ID": self.entry_id,
            "Zeit": self.time_text,
            "Datum": self.action_date,
            "Watchlist": self.watchlist,
            "Ticker": self.ticker,
            "Name": self.name,
            "Typ": self.action_type,
            "Kurs": self.price,
            "Stück": self.shares,
            "Verbleibend": self.remaining_shares,
            "Entry": self.entry_price,
            "Initial-Stop": self.initial_stop,
            "Aktueller Stop": self.current_stop,
            "Alter Stop": self.old_stop,
            "Neuer Stop": self.new_stop,
            "Realisiert P/L": self.realized_pnl,
            "Realisiert %": self.realized_pct,
            "Realisiert R": self.realized_r,
            "Gesamt P/L": self.total_pnl,
            "Gesamt R": self.total_r,
            "Notiz": self.note,
            "Erkenntnis": self.learning,
            "Details": self.details,
        }


@dataclass(slots=True)
class SignalEvent:
    time_text: str
    watchlist: str
    ticker: str
    event_type: str
    source: str = "-"
    status: str = "-"
    trade_state: str = "-"
    price: float | None = None
    score: Any = None
    details: str = "-"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_legacy_dict(cls, payload: Mapping[str, Any] | None) -> "SignalEvent":
        raw = dict(payload or {})
        known = {"Zeit", "Watchlist", "Ticker", "Ereignis", "Quelle", "Status", "Trade-State", "Kurs", "Live-Score", "Details"}
        return cls(
            time_text=_text(raw.get("Zeit")),
            watchlist=_text(raw.get("Watchlist"), "default") or "default",
            ticker=_text(raw.get("Ticker")).upper(),
            event_type=_text(raw.get("Ereignis")),
            source=_text(raw.get("Quelle"), "-") or "-",
            status=_text(raw.get("Status"), "-") or "-",
            trade_state=_text(raw.get("Trade-State"), "-") or "-",
            price=_number(raw.get("Kurs")),
            score=raw.get("Live-Score"),
            details=_text(raw.get("Details"), "-") or "-",
            extra={k: v for k, v in raw.items() if k not in known},
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "Zeit": self.time_text,
            "Watchlist": self.watchlist,
            "Ticker": self.ticker,
            "Ereignis": self.event_type,
            "Quelle": self.source,
            "Status": self.status,
            "Trade-State": self.trade_state,
            "Kurs": self.price,
            "Live-Score": self.score,
            "Details": self.details,
        }


@dataclass(slots=True)
class WatchlistItem:
    watchlist_name: str
    ticker: str
    watchlist_type: str = "Watchlist"
    alert_mode: str = "Standard"
    check_frequency: str = ""
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_legacy_dict(cls, payload: Mapping[str, Any] | None) -> "WatchlistItem":
        raw = dict(payload or {})
        known = {"Watchlist_Name", "Ticker", "Watchlist_Type", "Alert_Mode", "Check_Frequency"}
        return cls(
            watchlist_name=_text(raw.get("Watchlist_Name")),
            ticker=_text(raw.get("Ticker")).upper(),
            watchlist_type=_text(raw.get("Watchlist_Type"), "Watchlist") or "Watchlist",
            alert_mode=_text(raw.get("Alert_Mode"), "Standard") or "Standard",
            check_frequency=_text(raw.get("Check_Frequency")),
            extra={k: v for k, v in raw.items() if k not in known},
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "Watchlist_Name": self.watchlist_name,
            "Ticker": self.ticker,
            "Watchlist_Type": self.watchlist_type,
            "Alert_Mode": self.alert_mode,
            "Check_Frequency": self.check_frequency,
        }
