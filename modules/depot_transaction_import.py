"""v30.16a provider-free broker/depot transaction import.

Adds a reconciliation guard for overlapping manually maintained positions while
preserving the v30.13 storage namespace so existing Broker-ID/hash history remains
valid across the upgrade.

Reads transaction exports (xlsx/csv), normalises the supplied broker columns,
classifies buy/sell rows, protects against duplicate booking, and reconstructs
open positions with weighted-average price/share. The module never fetches
market data and never places orders.

The caller remains responsible for persisting returned positions/journal rows.
A small import ledger is stored here so repeated uploads of the same broker file
are idempotent.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
import hashlib
import json
import math
import re
from typing import Any

import pandas as pd

_NAMESPACE = "depot_transaction_import_v3013"
_SCHEMA = "depot-import-v30.13"
_MAX_IDS_PER_WATCHLIST = 60000
_MAX_ARCHIVE_ROWS_PER_WATCHLIST = 30000

_storage = None
_time_provider = lambda: datetime.now(timezone.utc)

_EXPECTED_COLUMNS = [
    "Action", "Time (UTC)", "ISIN", "Ticker", "Name", "Notes", "ID",
    "No. of shares", "Price / share", "Currency (Price / share)",
    "Exchange rate", "Result", "Currency (Result)", "Gross Total",
    "Currency (Gross Total)", "Withholding tax", "Currency (Withholding tax)",
    "Currency conversion fee", "Currency (Currency conversion fee)", "Taxes",
    "Currency (Taxes)", "Net Total", "Currency (Net Total)",
]

_ALIASES = {
    "action": "Action",
    "time (utc)": "Time (UTC)",
    "time": "Time (UTC)",
    "date": "Time (UTC)",
    "isin": "ISIN",
    "ticker": "Ticker",
    "symbol": "Ticker",
    "name": "Name",
    "notes": "Notes",
    "note": "Notes",
    "id": "ID",
    "transaction id": "ID",
    "no. of shares": "No. of shares",
    "number of shares": "No. of shares",
    "shares": "No. of shares",
    "quantity": "No. of shares",
    "price / share": "Price / share",
    "price/share": "Price / share",
    "price per share": "Price / share",
    "currency (price / share)": "Currency (Price / share)",
    "exchange rate": "Exchange rate",
    "result": "Result",
    "currency (result)": "Currency (Result)",
    "gross total": "Gross Total",
    "currency (gross total)": "Currency (Gross Total)",
    "withholding tax": "Withholding tax",
    "currency (withholding tax)": "Currency (Withholding tax)",
    "currency conversion fee": "Currency conversion fee",
    "currency (currency conversion fee)": "Currency (Currency conversion fee)",
    "taxes": "Taxes",
    "currency (taxes)": "Currency (Taxes)",
    "net total": "Net Total",
    "currency (net total)": "Currency (Net Total)",
}


def configure_context(*, storage=None, time_provider=None):
    global _storage, _time_provider
    if storage is not None:
        _storage = storage
    if callable(time_provider):
        _time_provider = time_provider


def _now():
    try:
        return _time_provider()
    except Exception:
        return datetime.now(timezone.utc)


def _now_iso():
    try:
        return _now().isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _num(value: Any, default=None):
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        try:
            out = float(value)
            return out if math.isfinite(out) else default
        except Exception:
            return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a", "na", "-", "null"}:
        return default
    text = text.replace("\u2212", "-").replace("\u00a0", " ").strip()
    # Handle common decimal/thousands formats conservatively.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    text = re.sub(r"[^0-9eE+\-.]", "", text)
    try:
        out = float(text)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _txt(value: Any, default="") -> str:
    try:
        text = str(value if value is not None else "").strip()
    except Exception:
        text = ""
    if text.lower() in {"nan", "none", "null"}:
        text = ""
    return text or default


def _colkey(value: Any) -> str:
    text = _txt(value).lower().replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expected_columns() -> list[str]:
    return list(_EXPECTED_COLUMNS)


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    errors = []
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            text = raw.decode(enc)
        except Exception as exc:
            errors.append(str(exc))
            continue
        for sep in (None, ";", ",", "\t"):
            try:
                if sep is None:
                    df = pd.read_csv(StringIO(text), sep=None, engine="python")
                else:
                    df = pd.read_csv(StringIO(text), sep=sep)
                if isinstance(df, pd.DataFrame) and len(df.columns) >= 2:
                    return df
            except Exception as exc:
                errors.append(str(exc))
    raise ValueError("CSV konnte nicht gelesen werden.")


def read_transaction_file(raw: bytes, filename: str = "") -> pd.DataFrame:
    """Read xlsx/xlsm/csv bytes; no network/provider access."""
    if not raw:
        raise ValueError("Die Importdatei ist leer.")
    name = _txt(filename).lower()
    if name.endswith((".csv", ".txt")):
        df = _read_csv_bytes(raw)
    else:
        try:
            df = pd.read_excel(BytesIO(raw), sheet_name=0, engine="openpyxl")
        except Exception as exc:
            if name.endswith(".xls"):
                raise ValueError("Altes .xls-Format wird nicht unterstützt. Bitte im Depot als .xlsx oder .csv exportieren.") from exc
            # Some broker downloads have no reliable extension; try CSV once.
            try:
                df = _read_csv_bytes(raw)
            except Exception:
                raise ValueError(f"Excel-Datei konnte nicht gelesen werden: {exc}") from exc
    if df is None or df.empty:
        raise ValueError("Die Datei enthält keine Transaktionszeilen.")
    return df


def _rename_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    work = df.copy()
    rename = {}
    mapping = {}
    for col in work.columns:
        key = _colkey(col)
        canonical = _ALIASES.get(key)
        if canonical:
            rename[col] = canonical
            mapping[str(col)] = canonical
    if rename:
        work = work.rename(columns=rename)
    # If duplicates arose from aliases, coalesce left-to-right.
    for canonical in set(rename.values()):
        idxs = [i for i, c in enumerate(work.columns) if c == canonical]
        if len(idxs) <= 1:
            continue
        vals = work.iloc[:, idxs].bfill(axis=1).iloc[:, 0]
        keep = [i for i, c in enumerate(work.columns) if c != canonical]
        reduced = work.iloc[:, keep].copy()
        reduced[canonical] = vals
        work = reduced
    return work, mapping


def classify_action(value: Any) -> str:
    raw = _txt(value).lower()
    norm = re.sub(r"[^a-z0-9äöüß]+", " ", raw)
    words = set(norm.split())
    if "buy" in words or "kauf" in words or "gekauft" in words or "purchase" in words:
        return "BUY"
    if "sell" in words or "verkauf" in words or "verkauft" in words or "sale" in words:
        return "SELL"
    if "dividend" in words or "dividende" in words:
        return "DIVIDEND"
    if "deposit" in words or "einzahlung" in words:
        return "DEPOSIT"
    if "withdrawal" in words or "auszahlung" in words:
        return "WITHDRAWAL"
    if "interest" in words or "zinsen" in words or "interest" in raw:
        return "INTEREST"
    return "OTHER"


def _parse_utc(value: Any):
    if value is None or _txt(value) == "":
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts)
    except Exception:
        return None


def _tx_fingerprint(row: dict[str, Any]) -> str:
    explicit = _txt(row.get("ID"))
    if explicit:
        return "id:" + explicit
    fields = [
        _txt(row.get("Action")), _txt(row.get("Time (UTC)")), _txt(row.get("ISIN")),
        _txt(row.get("Ticker")), _txt(row.get("No. of shares")), _txt(row.get("Price / share")),
        _txt(row.get("Gross Total")), _txt(row.get("Net Total")), _txt(row.get("Result")),
        _txt(row.get("Currency (Price / share)")), _txt(row.get("Currency (Net Total)")),
    ]
    raw = "|".join(fields)
    return "hash:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def normalize_transactions(df: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"ok": False, "error": "Keine Datenzeilen vorhanden.", "data": pd.DataFrame(), "warnings": []}
    work, mapping = _rename_columns(df)
    required = ["Action", "Time (UTC)", "Ticker", "No. of shares", "Price / share"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        return {
            "ok": False,
            "error": "Pflichtspalten fehlen: " + ", ".join(missing),
            "data": pd.DataFrame(),
            "warnings": [],
            "column_mapping": mapping,
            "available_columns": [str(c) for c in work.columns],
        }
    for c in _EXPECTED_COLUMNS:
        if c not in work.columns:
            work[c] = None

    rows = []
    warnings = []
    seen_file = set()
    for idx, series in work.iterrows():
        row = series.to_dict()
        action_raw = _txt(row.get("Action"))
        action = classify_action(action_raw)
        ticker = _txt(row.get("Ticker")).upper()
        name = _txt(row.get("Name"), ticker)
        shares = _num(row.get("No. of shares"), None)
        price = _num(row.get("Price / share"), None)
        signed_shares_normalized = False
        if action in {"BUY", "SELL"} and shares is not None and shares < 0:
            # Some broker exports encode sells with a negative quantity even though
            # Action already carries the direction. Position arithmetic uses magnitude.
            shares = abs(shares)
            signed_shares_normalized = True
        ts_utc = _parse_utc(row.get("Time (UTC)"))
        txid = _tx_fingerprint(row)
        duplicate_in_file = txid in seen_file
        seen_file.add(txid)
        if ts_utc is not None:
            try:
                ts_berlin = ts_utc.tz_convert("Europe/Berlin")
                time_utc = ts_utc.isoformat()
                time_berlin = ts_berlin.isoformat()
                date_berlin = ts_berlin.date().isoformat()
            except Exception:
                time_utc = ts_utc.isoformat()
                time_berlin = time_utc
                date_berlin = ts_utc.date().isoformat()
        else:
            time_utc = ""
            time_berlin = ""
            date_berlin = ""

        status = "OK"
        problem = ""
        if action in {"BUY", "SELL"}:
            if not ticker:
                status, problem = "FEHLER", "Ticker fehlt"
            elif shares is None or shares <= 0:
                status, problem = "FEHLER", "Stückzahl fehlt/ungültig"
            elif price is None or price <= 0:
                status, problem = "FEHLER", "Preis/Aktie fehlt/ungültig"
            elif ts_utc is None:
                status, problem = "FEHLER", "UTC-Zeit fehlt/ungültig"
        elif action == "OTHER":
            status, problem = "ARCHIV", "Action wird archiviert, verändert aber keine Position"
        else:
            status, problem = "ARCHIV", f"{action.title()} wird nur archiviert"
        if signed_shares_normalized and status == "OK":
            problem = "Negative Broker-Stückzahl als Betrag normalisiert"
        if duplicate_in_file:
            status, problem = "DUPLIKAT", "Doppelte Transaktions-ID innerhalb der Datei"

        rows.append({
            "Import-ID": txid,
            "Zeile": int(idx) + 2,
            "Action": action_raw,
            "Action-Typ": action,
            "Time UTC": time_utc,
            "Zeit Berlin": time_berlin,
            "Datum Berlin": date_berlin,
            "ISIN": _txt(row.get("ISIN")),
            "Ticker": ticker,
            "Name": name,
            "Notes": _txt(row.get("Notes")),
            "Broker-ID": _txt(row.get("ID")),
            "Stück": shares,
            "Preis/Aktie": price,
            "Preis-Währung": _txt(row.get("Currency (Price / share)")),
            "Wechselkurs": _num(row.get("Exchange rate"), None),
            "Result": _num(row.get("Result"), None),
            "Result-Währung": _txt(row.get("Currency (Result)")),
            "Brutto": _num(row.get("Gross Total"), None),
            "Brutto-Währung": _txt(row.get("Currency (Gross Total)")),
            "Quellensteuer": _num(row.get("Withholding tax"), None),
            "Quellensteuer-Währung": _txt(row.get("Currency (Withholding tax)")),
            "FX-Gebühr": _num(row.get("Currency conversion fee"), None),
            "FX-Gebühr-Währung": _txt(row.get("Currency (Currency conversion fee)")),
            "Steuern": _num(row.get("Taxes"), None),
            "Steuern-Währung": _txt(row.get("Currency (Taxes)")),
            "Netto": _num(row.get("Net Total"), None),
            "Netto-Währung": _txt(row.get("Currency (Net Total)")),
            "Import-Status": status,
            "Import-Hinweis": problem,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return {"ok": False, "error": "Keine lesbaren Transaktionen.", "data": out, "warnings": warnings}
    return {
        "ok": True,
        "error": "",
        "data": out,
        "warnings": warnings,
        "column_mapping": mapping,
        "available_columns": [str(c) for c in work.columns],
    }


def _eur_transaction_volume(row: dict[str, Any]) -> tuple[float | None, str]:
    """Return absolute EUR transaction volume when the export supports it.

    Priority is a broker total already denominated in EUR. Falling back to
    shares * price is only safe when the price currency itself is EUR. We do
    not guess FX directions or make provider requests.
    """
    net = _num(row.get("Netto"), None)
    net_ccy = _txt(row.get("Netto-Währung")).upper()
    if net is not None and net_ccy == "EUR":
        return abs(float(net)), "Net Total (EUR)"

    gross = _num(row.get("Brutto"), None)
    gross_ccy = _txt(row.get("Brutto-Währung")).upper()
    if gross is not None and gross_ccy == "EUR":
        return abs(float(gross)), "Gross Total (EUR)"

    qty = _num(row.get("Stück"), None)
    price = _num(row.get("Preis/Aktie"), None)
    price_ccy = _txt(row.get("Preis-Währung")).upper()
    if qty is not None and price is not None and price_ccy == "EUR":
        return abs(float(qty) * float(price)), "Stück × Preis/Aktie (EUR)"

    return None, "EUR-Volumen nicht eindeutig bestimmbar"


def filter_min_eur_transaction_volume(
    normalized: pd.DataFrame,
    *,
    enabled: bool = False,
    minimum_eur: float = 500.0,
) -> dict[str, Any]:
    """Optionally exclude small BUY/SELL rows before preview and booking.

    Only rows whose EUR transaction volume is reliably available are filtered.
    Archive-only actions (dividends, interest, etc.) remain unchanged. Unknown
    currency rows are retained rather than silently discarded.
    """
    df = normalized.copy() if isinstance(normalized, pd.DataFrame) else pd.DataFrame()
    try:
        minimum = max(0.0, float(minimum_eur or 0.0))
    except Exception:
        minimum = 500.0

    if df.empty:
        return {
            "data": df,
            "excluded": pd.DataFrame(),
            "enabled": bool(enabled),
            "minimum_eur": minimum,
            "excluded_rows": 0,
            "excluded_tickers": [],
            "unknown_volume_rows": 0,
        }

    volumes = []
    sources = []
    for _, series in df.iterrows():
        value, source = _eur_transaction_volume(series.to_dict())
        volumes.append(value)
        sources.append(source)
    df["Transaktionsvolumen EUR"] = volumes
    df["Volumen-Quelle"] = sources

    trade_mask = df.get("Action-Typ", pd.Series("", index=df.index)).astype(str).isin(["BUY", "SELL"])
    unknown_mask = trade_mask & df["Transaktionsvolumen EUR"].isna()
    if not enabled:
        return {
            "data": df,
            "excluded": df.iloc[0:0].copy(),
            "enabled": False,
            "minimum_eur": minimum,
            "excluded_rows": 0,
            "excluded_tickers": [],
            "unknown_volume_rows": int(unknown_mask.sum()),
        }

    small_mask = trade_mask & df["Transaktionsvolumen EUR"].notna() & (df["Transaktionsvolumen EUR"] < minimum - 1e-9)
    excluded = df.loc[small_mask].copy()
    kept = df.loc[~small_mask].copy().reset_index(drop=True)
    tickers = sorted({str(x).strip().upper() for x in excluded.get("Ticker", pd.Series(dtype=str)).tolist() if str(x).strip()})
    return {
        "data": kept,
        "excluded": excluded.reset_index(drop=True),
        "enabled": True,
        "minimum_eur": minimum,
        "excluded_rows": int(len(excluded)),
        "excluded_tickers": tickers,
        "unknown_volume_rows": int(unknown_mask.sum()),
    }


def _empty_store():
    return {"schema": _SCHEMA, "watchlists": {}, "updated_at": _now_iso()}


def _load_store():
    payload = None
    if _storage is not None:
        try:
            payload = _storage.load_namespace(_NAMESPACE, default=None)
        except Exception:
            payload = None
    if not isinstance(payload, dict):
        payload = _empty_store()
    payload.setdefault("schema", _SCHEMA)
    payload.setdefault("watchlists", {})
    return payload


def _save_store(payload):
    if _storage is None:
        return False
    try:
        payload = dict(payload or {})
        payload["schema"] = _SCHEMA
        payload["updated_at"] = _now_iso()
        return bool(_storage.save_namespace(_NAMESPACE, payload))
    except Exception:
        return False


def processed_ids(watchlist_name: str) -> set[str]:
    store = _load_store()
    bucket = dict((store.get("watchlists") or {}).get(str(watchlist_name or "Standard")) or {})
    return {str(x) for x in list(bucket.get("processed_ids") or []) if str(x)}


def import_history(watchlist_name: str) -> pd.DataFrame:
    store = _load_store()
    bucket = dict((store.get("watchlists") or {}).get(str(watchlist_name or "Standard")) or {})
    rows = list(bucket.get("archive") or [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "Zeit Berlin" in df.columns:
        try:
            df = df.sort_values("Zeit Berlin", ascending=False, na_position="last")
        except Exception:
            pass
    return df.reset_index(drop=True)


def mark_imported(watchlist_name: str, normalized_rows: pd.DataFrame, *, filename: str = "", result_summary: dict | None = None) -> bool:
    # An empty frame is a valid no-op: it means every row in the uploaded file
    # was already present in the import ledger. Treat this as success so a
    # repeated import is not reported as a storage warning.
    if not isinstance(normalized_rows, pd.DataFrame):
        return False
    if normalized_rows.empty:
        return True
    wl = str(watchlist_name or "Standard")
    store = _load_store()
    watchlists = dict(store.get("watchlists") or {})
    bucket = dict(watchlists.get(wl) or {})
    ids = list(bucket.get("processed_ids") or [])
    known = set(str(x) for x in ids)
    archive = list(bucket.get("archive") or [])
    for _, series in normalized_rows.iterrows():
        rec = series.to_dict()
        txid = str(rec.get("Import-ID") or "")
        if txid and txid not in known:
            ids.append(txid)
            known.add(txid)
            safe = {}
            for k, v in rec.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    safe[str(k)] = v.isoformat()
                elif isinstance(v, float) and not math.isfinite(v):
                    safe[str(k)] = None
                elif pd.isna(v) if not isinstance(v, (list, dict, tuple)) else False:
                    safe[str(k)] = None
                else:
                    safe[str(k)] = v.item() if hasattr(v, "item") else v
            safe["Import-Datei"] = _txt(filename)
            safe["Importiert am"] = _now_iso()
            archive.append(safe)
    bucket["processed_ids"] = ids[-_MAX_IDS_PER_WATCHLIST:]
    bucket["archive"] = archive[-_MAX_ARCHIVE_ROWS_PER_WATCHLIST:]
    bucket["last_import_at"] = _now_iso()
    bucket["last_filename"] = _txt(filename)
    bucket["last_summary"] = dict(result_summary or {})
    watchlists[wl] = bucket
    store["watchlists"] = watchlists
    return _save_store(store)


def _position_metadata(old: dict[str, Any] | None) -> dict[str, Any]:
    old = dict(old or {})
    keep = [
        "stop", "initial_stop", "target", "stop_history", "journal_notes",
        "portfolio_group", "entry_context", "last_context", "last_price",
        "strategy_origin", "execution_status", "planned_entry", "planned_shares",
    ]
    return {k: old.get(k) for k in keep if k in old}


def _shares_display(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _journal_record(*, watchlist_name, ticker, name, typ, date_text, time_text, price, shares, remaining,
                    entry, realized_pnl, realized_pct, total_realized_pnl, currency, import_id,
                    notes="", broker_result=None, broker_result_currency="", details=""):
    return {
        "ID": "broker-" + hashlib.sha1(str(import_id).encode("utf-8")).hexdigest()[:24],
        "Zeit": time_text,
        "Datum": date_text,
        "Watchlist": str(watchlist_name or "Standard"),
        "Ticker": ticker,
        "Name": name or ticker,
        "Typ": typ,
        "Kurs": price,
        "Stück": shares,
        "Verbleibend": remaining,
        "Entry": entry,
        "Initial-Stop": None,
        "Aktueller Stop": None,
        "Alter Stop": None,
        "Neuer Stop": None,
        "Realisiert P/L": realized_pnl,
        "Realisiert %": realized_pct,
        "Realisiert R": None,
        "Gesamt P/L": total_realized_pnl,
        "Gesamt R": None,
        "Notiz": notes,
        "Erkenntnis": "",
        "Details": details,
        "Broker Import ID": import_id,
        "Broker Quelle": "Depot-Excel",
        "Broker Preis-Währung": currency,
        "Realisiert P/L-Währung": currency,
        "Broker Result": broker_result,
        "Broker Result-Währung": broker_result_currency,
    }


def _parse_any_time(value: Any):
    text = _txt(value)
    if not text:
        return None
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    try:
        if ts.tzinfo is None:
            ts = ts.tz_localize("Europe/Berlin")
        else:
            ts = ts.tz_convert("Europe/Berlin")
    except Exception:
        pass
    return ts


def _is_broker_position(position: dict[str, Any] | None) -> bool:
    src = _txt((position or {}).get("broker_source")).lower().replace("_", " ").replace("-", " ")
    return "depot" in src and "excel" in src


def reconciliation_guard(
    normalized: pd.DataFrame,
    positions: dict[str, Any] | None,
    *,
    mode: str = "incremental",
    already_processed: set[str] | None = None,
) -> dict[str, Any]:
    """Assess overlap between incoming broker rows and existing open positions.

    The guard is intentionally conservative. A manually maintained open position
    has no Broker-ID lineage, so incoming broker rows cannot be proven to be new
    relative to that manual state. Such overlaps require explicit user review.

    For rebuild mode, the file is additionally replayed from zero per ticker. A
    sell that occurs before enough file-contained buys is a hard blocker because
    the file cannot reconstruct that ticker from zero.
    """
    df = normalized.copy() if isinstance(normalized, pd.DataFrame) else pd.DataFrame()
    current = {str(k).upper(): dict(v or {}) for k, v in dict(positions or {}).items()}
    processed = {str(x) for x in set(already_processed or set()) if str(x)}
    mode_key = str(mode or "incremental").lower()
    if df.empty:
        return {
            "table": pd.DataFrame(), "flagged_tickers": [], "blockers": [],
            "requires_confirmation": False, "hard_block": False,
            "summary": {"tickers": 0, "manual_overlaps": 0, "blockers": 0},
        }

    work = df[df.get("Import-Status", pd.Series("", index=df.index)).astype(str).eq("OK")].copy()
    if "Action-Typ" not in work.columns:
        return {
            "table": pd.DataFrame(), "flagged_tickers": [], "blockers": [],
            "requires_confirmation": False, "hard_block": False,
            "summary": {"tickers": 0, "manual_overlaps": 0, "blockers": 0},
        }
    work = work[work["Action-Typ"].astype(str).isin(["BUY", "SELL"])].copy()
    if mode_key != "rebuild" and "Import-ID" in work.columns:
        work = work[~work["Import-ID"].astype(str).isin(processed)].copy()
    if work.empty:
        return {
            "table": pd.DataFrame(), "flagged_tickers": [], "blockers": [],
            "requires_confirmation": False, "hard_block": False,
            "summary": {"tickers": 0, "manual_overlaps": 0, "blockers": 0},
        }

    if "Zeit Berlin" in work.columns:
        work["__guard_time"] = pd.to_datetime(work["Zeit Berlin"], errors="coerce", utc=True)
    else:
        work["__guard_time"] = pd.NaT
    work = work.sort_values(["Ticker", "__guard_time", "Zeile"], ascending=[True, True, True], na_position="last")

    rows = []
    flagged = []
    blockers = []
    for ticker, part in work.groupby(work["Ticker"].astype(str).str.upper(), sort=True):
        ticker = str(ticker or "").strip().upper()
        if not ticker:
            continue
        part = part.copy().sort_values(["__guard_time", "Zeile"], ascending=[True, True], na_position="last")
        old = dict(current.get(ticker) or {})
        existing_qty = _num(old.get("shares"), 0.0) or 0.0
        existing_entry = _num(old.get("entry"), None)
        existing_source = _txt(old.get("broker_source"), "Manuell / nicht als Broker-Import markiert") if old else "Keine offene Position"
        manual_existing = bool(old and existing_qty > 1e-12 and not _is_broker_position(old))
        broker_existing = bool(old and existing_qty > 1e-12 and _is_broker_position(old))

        buy_qty = float(pd.to_numeric(part.loc[part["Action-Typ"] == "BUY", "Stück"], errors="coerce").fillna(0).sum())
        sell_qty = float(pd.to_numeric(part.loc[part["Action-Typ"] == "SELL", "Stück"], errors="coerce").fillna(0).sum())
        net_qty = buy_qty - sell_qty
        first = part.iloc[0].to_dict()
        first_action = _txt(first.get("Action-Typ"), "-")
        first_time = _parse_any_time(first.get("Zeit Berlin"))
        last_time = _parse_any_time(part.iloc[-1].to_dict().get("Zeit Berlin"))
        opened_time = _parse_any_time(old.get("opened_at_iso")) if old else None

        # Can the file reconstruct this ticker from zero?
        replay_balance = 0.0
        replay_problem = ""
        for _, ser in part.iterrows():
            act = _txt(ser.get("Action-Typ"))
            qty = _num(ser.get("Stück"), 0.0) or 0.0
            if act == "BUY":
                replay_balance += qty
            elif act == "SELL":
                if qty > replay_balance + 1e-8:
                    replay_problem = (
                        f"Datei ist ab Null nicht vollständig: Verkauf {_shares_display(qty)} Stück, "
                        f"zuvor in Datei nur {_shares_display(replay_balance)} Stück aufgebaut."
                    )
                    break
                replay_balance = max(0.0, replay_balance - qty)

        starts_after_manual_open = False
        if opened_time is not None and first_time is not None:
            try:
                starts_after_manual_open = bool(first_time > opened_time + pd.Timedelta(minutes=1))
            except Exception:
                starts_after_manual_open = False

        level = "OK"
        guidance = "Keine Überschneidung mit einer manuell gepflegten offenen Position erkannt."
        if mode_key == "rebuild" and replay_problem:
            level = "BLOCKIERT"
            guidance = replay_problem + " Für Rebuild vollständige Historie ab Positionsbeginn exportieren."
            blockers.append(ticker)
        elif manual_existing:
            level = "ABGLEICH"
            flagged.append(ticker)
            if mode_key == "incremental":
                if first_action == "BUY":
                    guidance = (
                        "Manuelle offene Position + neuer Kauf in Datei: Doppelzählung ist möglich, falls der aktuelle "
                        "Tool-Bestand diesen Kauf bereits enthält. Nur bestätigen, wenn der Tool-Bestand dem Stand direkt "
                        "vor der ersten noch nicht importierten Datei-Transaktion entspricht."
                    )
                else:
                    guidance = (
                        "Manuelle offene Position + Datei beginnt mit Verkauf: Der manuelle Bestand kann der nötige Anfangsbestand sein. "
                        "Nur bestätigen, wenn dieser Verkauf im aktuell gespeicherten Tool-Bestand noch nicht berücksichtigt ist."
                    )
                if starts_after_manual_open:
                    guidance += " Die Datei beginnt nach dem gespeicherten manuellen Positionsbeginn."
            else:
                if starts_after_manual_open:
                    guidance = (
                        "Rebuild-Datei beginnt nach dem gespeicherten manuellen Positionsbeginn. Dadurch kann ein älterer Anfangsbestand "
                        "fehlen. Nur freigeben, wenn die Datei trotzdem die vollständige Kauf-/Verkaufshistorie dieses Positionszyklus enthält."
                    )
                elif opened_time is None:
                    guidance = (
                        "Manuelle Position ohne belastbaren Eröffnungszeitpunkt. Für Rebuild kann nicht automatisch bewiesen werden, "
                        "dass die Datei den gesamten Positionszyklus enthält. Vollständigkeit explizit bestätigen."
                    )
                else:
                    guidance = (
                        "Manuelle Position wird im Rebuild ersetzt. Die Datei beginnt spätestens zum gespeicherten Positionsbeginn und "
                        "ist aus Käufen/Verkäufen ab Null replay-fähig; Vollständigkeit trotzdem explizit bestätigen."
                    )
        elif broker_existing:
            guidance = (
                "Bestehende Position stammt bereits aus Depot-Excel; Broker-ID/Hash-Dublettenschutz wird berücksichtigt."
                if mode_key != "rebuild" else
                "Bestehende Broker-Position wird aus den Datei-Transaktionen neu aufgebaut."
            )
        elif mode_key == "rebuild" and not replay_problem:
            guidance = "Datei ist für diesen Ticker aus Käufen/Verkäufen ab Null replay-fähig."

        def fmt_ts(ts):
            if ts is None:
                return "-"
            try:
                return ts.strftime("%d.%m.%Y %H:%M")
            except Exception:
                return _txt(ts, "-")

        rows.append({
            "Ticker": ticker,
            "Abgleich": level,
            "Tool-Stück": round(float(existing_qty), 8) if old else 0.0,
            "Tool-Entry": None if existing_entry is None else round(float(existing_entry), 6),
            "Tool-Quelle": existing_source,
            "Tool-Eröffnung": fmt_ts(opened_time),
            "Datei von": fmt_ts(first_time),
            "Datei bis": fmt_ts(last_time),
            "Erste Datei-Aktion": first_action,
            "Datei Käufe": round(buy_qty, 8),
            "Datei Verkäufe": round(sell_qty, 8),
            "Datei Netto-Stück": round(net_qty, 8),
            "Rebuild ab Null": "Nein" if replay_problem else "Ja",
            "Hinweis": guidance,
        })

    table = pd.DataFrame(rows)
    return {
        "table": table,
        "flagged_tickers": sorted(set(flagged)),
        "blockers": sorted(set(blockers)),
        "requires_confirmation": bool(flagged),
        "hard_block": bool(blockers),
        "summary": {
            "tickers": int(len(table)),
            "manual_overlaps": int(len(set(flagged))),
            "blockers": int(len(set(blockers))),
        },
    }


def apply_transactions(
    watchlist_name: str,
    normalized: pd.DataFrame,
    positions: dict[str, Any] | None,
    *,
    mode: str = "incremental",
    already_processed: set[str] | None = None,
    screener_only: bool = False,
) -> dict[str, Any]:
    """Apply BUY/SELL rows to a copy of the position store.

    mode='incremental': only transaction IDs not present in the ledger are applied
    to the current position state.

    mode='rebuild': all valid BUY/SELL rows in this file are replayed from zero
    for the tickers contained in the file. Manual stop/target/group/context fields
    from currently open matching positions are retained if the ticker remains open.
    The caller can still de-duplicate returned journal rows by Broker Import ID.
    """
    work = normalized.copy() if isinstance(normalized, pd.DataFrame) else pd.DataFrame()
    current = {str(k).upper(): dict(v or {}) for k, v in dict(positions or {}).items()}
    processed = set(already_processed or set())
    if work.empty:
        return {"ok": False, "error": "Keine normalisierten Transaktionen.", "positions": current, "journal_entries": []}

    valid = work[work["Import-Status"].isin(["OK", "ARCHIV"])].copy()
    if "Zeit Berlin" in valid.columns:
        valid["__sort_time"] = pd.to_datetime(valid["Zeit Berlin"], errors="coerce", utc=True)
    else:
        valid["__sort_time"] = pd.NaT
    valid = valid.sort_values(["__sort_time", "Zeile"], ascending=[True, True], na_position="last")

    trade_tickers = sorted(set(valid.loc[valid["Action-Typ"].isin(["BUY", "SELL"]), "Ticker"].astype(str).str.upper()))
    preserved = {tk: _position_metadata(current.get(tk)) for tk in trade_tickers}
    if str(mode).lower() == "rebuild":
        for tk in trade_tickers:
            current.pop(tk, None)

    journal_entries = []
    applied_rows = []
    skipped_rows = []
    anomalies = []
    stats = {
        "buy_rows": 0, "sell_rows": 0, "other_rows": 0,
        "new_positions": 0, "closed_positions": 0, "partial_sales": 0,
        "external_rows": 0, "baseline_rows": 0, "mixed_sales": 0, "external_shares_ignored": 0.0,
    }

    for _, s in valid.iterrows():
        row = s.to_dict()
        txid = str(row.get("Import-ID") or "")
        action = str(row.get("Action-Typ") or "OTHER")
        if str(mode).lower() != "rebuild" and txid in processed:
            row["Import-Hinweis"] = "Bereits früher importiert"
            skipped_rows.append(row)
            continue
        if action not in {"BUY", "SELL"}:
            stats["other_rows"] += 1
            applied_rows.append(row)  # archived, but position-neutral
            continue

        ticker = str(row.get("Ticker") or "").strip().upper()
        qty = _num(row.get("Stück"), None)
        price = _num(row.get("Preis/Aktie"), None)
        if not ticker or qty is None or qty <= 0 or price is None or price <= 0:
            row["Import-Hinweis"] = "Ungültige Trade-Zeile"
            skipped_rows.append(row)
            continue
        name = _txt(row.get("Name"), ticker)
        currency = _txt(row.get("Preis-Währung"))
        action_date = _txt(row.get("Datum Berlin"))
        berlin_iso = _txt(row.get("Zeit Berlin"))
        try:
            berlin_pretty = pd.Timestamp(berlin_iso).strftime("%d.%m.%Y %H:%M:%S") if berlin_iso else _now().strftime("%d.%m.%Y %H:%M:%S")
        except Exception:
            berlin_pretty = _now().strftime("%d.%m.%Y %H:%M:%S")

        pos = dict(current.get(ticker) or {})
        open_qty = _num(pos.get("shares"), 0.0) or 0.0
        avg_entry = _num(pos.get("entry"), None)
        realized_before = _num(pos.get("realized_pnl"), 0.0) or 0.0
        is_screener_position = bool(
            pos and (
                _txt(pos.get("strategy_origin")).lower() == "screener"
                or isinstance(pos.get("entry_context"), dict) and bool(pos.get("entry_context"))
                or _txt(pos.get("broker_source")).lower() != "depot-excel"
            )
        )
        row_time = _parse_any_time(row.get("Zeit Berlin"))
        opened_time = _parse_any_time(pos.get("opened_at_iso")) if pos else None
        before_screener_open = False
        if row_time is not None and opened_time is not None:
            try:
                before_screener_open = bool(row_time < opened_time - pd.Timedelta(minutes=5))
            except Exception:
                before_screener_open = False

        if action == "BUY":
            stats["buy_rows"] += 1
            if screener_only and (not is_screener_position or before_screener_open):
                row["Import-Hinweis"] = (
                    "Extern/Pie: Transaktion liegt vor dem Screener-Trade" if before_screener_open else
                    "Extern/Pie: kein vorgemerkter Screener-Trade; nicht in Screener-Position gebucht"
                )
                row["Screener-Klassifizierung"] = "EXTERN/PIE"
                stats["external_rows"] += 1
                applied_rows.append(row)
                continue
            if screener_only and open_qty > 1e-12 and _txt(pos.get("broker_source")).lower() != "depot-excel":
                row["Import-Hinweis"] = "Legacy-Bestand: Kauf ist bereits in der manuell geführten Screener-Stückzahl enthalten; nicht doppelt gebucht"
                row["Screener-Klassifizierung"] = "BASELINE"
                stats["baseline_rows"] += 1
                applied_rows.append(row)
                continue
            new_qty = open_qty + qty
            if new_qty <= 0:
                anomalies.append({**row, "Problem": "Ungültige resultierende Stückzahl"})
                continue
            new_entry = price if open_qty <= 1e-12 or avg_entry is None else ((avg_entry * open_qty) + (price * qty)) / new_qty
            is_new_cycle = open_qty <= 1e-12
            base = {
                "ticker": ticker,
                "name": name,
                "entry": float(new_entry),
                "shares": float(new_qty),
                "initial_shares": float(new_qty) if is_new_cycle else float(_num(pos.get("initial_shares"), open_qty) or open_qty) + float(qty),
                "realized_pnl": 0.0 if is_new_cycle else realized_before,
                "realized_shares": 0.0 if is_new_cycle else float(_num(pos.get("realized_shares"), 0.0) or 0.0),
                "realized_r_weighted": 0.0 if is_new_cycle else float(_num(pos.get("realized_r_weighted"), 0.0) or 0.0),
                "stop_history": list(pos.get("stop_history") or []),
                "journal_notes": list(pos.get("journal_notes") or []),
                "created_at": pos.get("created_at") or berlin_pretty[:16],
                "updated_at": berlin_pretty[:16],
                "opened_at_iso": pos.get("opened_at_iso") or berlin_iso,
                "entry_context": dict(pos.get("entry_context") or {}),
                "last_context": dict(pos.get("last_context") or {}),
                "last_price": _num(pos.get("last_price"), price) or price,
                "portfolio_group": _txt(pos.get("portfolio_group")),
                "price_currency": currency or _txt(pos.get("price_currency")),
                "broker_isin": _txt(row.get("ISIN")) or _txt(pos.get("broker_isin")),
                "broker_source": "Depot-Excel",
                "broker_last_import_id": txid,
                "strategy_origin": _txt(pos.get("strategy_origin"), "screener" if screener_only else ""),
                "execution_status": "open",
                "planned_entry": _num(pos.get("planned_entry"), None),
                "planned_shares": _num(pos.get("planned_shares"), None),
            }
            # Preserve manual management fields. New broker-only positions have no
            # invented stop/target; zero keeps legacy UI compatible.
            meta = preserved.get(ticker, {}) if str(mode).lower() == "rebuild" and is_new_cycle else _position_metadata(pos)
            base["stop"] = _num(meta.get("stop"), 0.0) or 0.0
            base["initial_stop"] = _num(meta.get("initial_stop"), None)
            base["target"] = _num(meta.get("target"), 0.0) or 0.0
            if meta.get("portfolio_group"):
                base["portfolio_group"] = _txt(meta.get("portfolio_group"))
            if isinstance(meta.get("entry_context"), dict) and meta.get("entry_context"):
                base["entry_context"] = dict(meta.get("entry_context"))
            if isinstance(meta.get("last_context"), dict) and meta.get("last_context"):
                base["last_context"] = dict(meta.get("last_context"))
            if isinstance(meta.get("stop_history"), list) and meta.get("stop_history"):
                base["stop_history"] = list(meta.get("stop_history"))
            if isinstance(meta.get("journal_notes"), list) and meta.get("journal_notes"):
                base["journal_notes"] = list(meta.get("journal_notes"))
            if is_new_cycle:
                stats["new_positions"] += 1
            current[ticker] = base
            applied_rows.append(row)
            continue

        # SELL
        stats["sell_rows"] += 1
        if screener_only and before_screener_open:
            row["Import-Hinweis"] = "Extern/Pie: Verkauf liegt vor dem gespeicherten Screener-Positionsbeginn"
            row["Screener-Klassifizierung"] = "EXTERN/PIE"
            stats["external_rows"] += 1
            applied_rows.append(row)
            continue
        if open_qty <= 1e-12 or avg_entry is None:
            if screener_only:
                row["Import-Hinweis"] = "Extern/Pie: keine offene Screener-Position; Verkauf archiviert, aber nicht gebucht"
                row["Screener-Klassifizierung"] = "EXTERN/PIE"
                stats["external_rows"] += 1
                applied_rows.append(row)
                continue
            anomalies.append({**row, "Problem": "Verkauf ohne bekannte offene Stückzahl; Historie vermutlich unvollständig"})
            continue
        external_excess = max(0.0, float(qty) - float(open_qty))
        if external_excess > 1e-8 and not screener_only:
            anomalies.append({**row, "Problem": f"Verkauf {_shares_display(qty)} > offen {_shares_display(open_qty)}; nicht gebucht"})
            continue
        sold = min(qty, open_qty)
        if screener_only and external_excess > 1e-8:
            row["Import-Hinweis"] = (
                f"GEMISCHT: {_shares_display(sold)} Screener-Stück gebucht; "
                f"{_shares_display(external_excess)} externe/Pie-Stück ignoriert"
            )
            row["Screener-Klassifizierung"] = "GEMISCHT"
            stats["mixed_sales"] += 1
            stats["external_shares_ignored"] += float(external_excess)
        remaining = max(0.0, open_qty - sold)
        calc_pnl = (price - avg_entry) * sold
        calc_pct = (price / avg_entry - 1.0) * 100.0 if avg_entry else None
        total_realized = realized_before + calc_pnl
        pos["realized_pnl"] = total_realized
        pos["realized_shares"] = float(_num(pos.get("realized_shares"), 0.0) or 0.0) + sold
        pos["last_exit_price"] = price
        pos["updated_at"] = berlin_pretty[:16]
        pos["broker_last_import_id"] = txid

        is_close = remaining <= 1e-8
        typ = "Position geschlossen" if is_close else "Teilverkauf"
        details = (
            f"Broker-Import: {_shares_display(sold)} Screener-Stück verkauft; "
            + ("Position geschlossen." if is_close else f"{_shares_display(remaining)} Screener-Stück verbleiben.")
        )
        if screener_only and external_excess > 1e-8:
            details += f" {_shares_display(external_excess)} externe/Pie-Stück aus derselben Broker-Ausführung wurden ignoriert."
        broker_result = _num(row.get("Result"), None)
        broker_result_currency = _txt(row.get("Result-Währung"))
        if broker_result is not None:
            details += f" Broker-Result {broker_result:.2f} {broker_result_currency or ''}".rstrip()
        journal_entries.append(_journal_record(
            watchlist_name=watchlist_name,
            ticker=ticker,
            name=name,
            typ=typ,
            date_text=action_date,
            time_text=berlin_pretty,
            price=price,
            shares=float(sold),
            remaining=float(remaining),
            entry=float(avg_entry),
            realized_pnl=float(calc_pnl),
            realized_pct=calc_pct,
            total_realized_pnl=float(total_realized),
            currency=currency,
            import_id=txid,
            notes=_txt(row.get("Notes")),
            broker_result=broker_result,
            broker_result_currency=broker_result_currency,
            details=details,
        ))
        if is_close:
            current.pop(ticker, None)
            stats["closed_positions"] += 1
        else:
            pos["shares"] = float(remaining)
            pos["execution_status"] = "open"
            current[ticker] = pos
            stats["partial_sales"] += 1
        applied_rows.append(row)

    applied_df = pd.DataFrame(applied_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    anomaly_df = pd.DataFrame(anomalies)
    open_tickers = sorted(current.keys())
    fractional = sorted([
        tk for tk, pos in current.items()
        if abs((_num(pos.get("shares"), 0.0) or 0.0) - round(_num(pos.get("shares"), 0.0) or 0.0)) > 1e-8
    ])
    return {
        "ok": True,
        "error": "",
        "positions": current,
        "journal_entries": journal_entries,
        "applied_rows": applied_df,
        "skipped_rows": skipped_df,
        "anomalies": anomaly_df,
        "open_tickers": open_tickers,
        "fractional_tickers": fractional,
        "stats": stats,
    }


def preview_summary(normalized: pd.DataFrame, watchlist_name: str = "") -> dict[str, Any]:
    df = normalized if isinstance(normalized, pd.DataFrame) else pd.DataFrame()
    known = processed_ids(watchlist_name) if watchlist_name else set()
    if df.empty:
        return {"rows": 0, "buys": 0, "sells": 0, "other": 0, "errors": 0, "already_imported": 0, "tickers": 0}
    action = df.get("Action-Typ", pd.Series("", index=df.index)).astype(str)
    status = df.get("Import-Status", pd.Series("", index=df.index)).astype(str)
    ids = df.get("Import-ID", pd.Series("", index=df.index)).astype(str)
    tickers = df.loc[action.isin(["BUY", "SELL"]), "Ticker"].astype(str) if "Ticker" in df.columns else pd.Series(dtype=str)
    return {
        "rows": int(len(df)),
        "buys": int((action == "BUY").sum()),
        "sells": int((action == "SELL").sum()),
        "other": int((~action.isin(["BUY", "SELL"])).sum()),
        "errors": int((status == "FEHLER").sum()),
        "duplicates_in_file": int((status == "DUPLIKAT").sum()),
        "already_imported": int(ids.isin(known).sum()),
        "tickers": int(tickers[tickers.str.strip() != ""].nunique()),
    }
