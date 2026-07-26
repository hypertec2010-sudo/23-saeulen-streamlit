"""Watchlist repository compatible with the former Google-Sheets function API."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd


WATCHLIST_COLUMNS = [
    "Watchlist_Name",
    "Watchlist_Type",
    "Ticker",
    "Added_At",
    "Alert_Mode",
    "Check_Frequency",
]


class WatchlistRepository:
    namespace = "watchlists"

    def __init__(self, storage, *, time_provider=None):
        self.storage = storage
        self.time_provider = time_provider or datetime.now

    def _now(self) -> str:
        try:
            return self.time_provider().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _normalise_ticker(value) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _normalise_name(value) -> str:
        return str(value or "").strip()

    def _load_rows(self) -> list[dict]:
        payload = self.storage.load_namespace(self.namespace, default={})
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        return [dict(row) for row in rows if isinstance(row, dict)]

    def _save_rows(self, rows: list[dict]) -> bool:
        clean = []
        for row in rows:
            item = {col: row.get(col, "") for col in WATCHLIST_COLUMNS}
            for key, value in row.items():
                if key not in item:
                    item[key] = value
            clean.append(item)
        return self.storage.save_namespace(self.namespace, {"rows": clean, "version": 1})

    def load_watchlists_df(self):
        try:
            rows = self._load_rows()
            df = pd.DataFrame(rows)
            for col in WATCHLIST_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            if df.empty:
                df = pd.DataFrame(columns=WATCHLIST_COLUMNS)
            return df, None
        except Exception as exc:
            return pd.DataFrame(columns=WATCHLIST_COLUMNS), str(exc)

    def get_watchlist_catalog_df(self):
        df, err = self.load_watchlists_df()
        if err:
            return pd.DataFrame(columns=["Watchlist_Name", "Watchlist_Type"]), err
        if df.empty:
            return pd.DataFrame(columns=["Watchlist_Name", "Watchlist_Type"]), None
        catalog = (
            df[["Watchlist_Name", "Watchlist_Type"]]
            .fillna("")
            .astype(str)
            .query("Watchlist_Name != ''")
            .drop_duplicates()
            .sort_values(["Watchlist_Name", "Watchlist_Type"])
            .reset_index(drop=True)
        )
        return catalog, None

    def get_watchlist_tickers(self, watchlist_name):
        name = self._normalise_name(watchlist_name)
        if not name:
            return [], "Watchlist-Name fehlt."
        df, err = self.load_watchlists_df()
        if err:
            return [], err
        if df.empty:
            return [], None
        mask = df["Watchlist_Name"].astype(str).str.strip().str.lower() == name.lower()
        tickers = []
        seen = set()
        for value in df.loc[mask, "Ticker"].tolist():
            ticker = self._normalise_ticker(value)
            if ticker and ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
        return tickers, None

    def create_watchlist(self, watchlist_name, watchlist_type="Watchlist", check_frequency="4x täglich"):
        name = self._normalise_name(watchlist_name)
        wl_type = self._normalise_name(watchlist_type) or "Watchlist"
        frequency = self._normalise_name(check_frequency) or "4x täglich"
        if not name:
            return False, "Bitte einen Watchlist-Namen eingeben."
        rows = self._load_rows()
        if any(self._normalise_name(row.get("Watchlist_Name")).lower() == name.lower() for row in rows):
            return False, "Diese Watchlist existiert bereits."
        rows.append({
            "Watchlist_Name": name,
            "Watchlist_Type": wl_type,
            "Ticker": "",
            "Added_At": self._now(),
            "Alert_Mode": "Standard",
            "Check_Frequency": frequency,
        })
        if not self._save_rows(rows):
            return False, "Watchlist konnte nicht gespeichert werden."
        return True, f"Watchlist '{name}' wurde angelegt."

    def delete_watchlist(self, watchlist_name):
        name = self._normalise_name(watchlist_name)
        rows = self._load_rows()
        kept = [row for row in rows if self._normalise_name(row.get("Watchlist_Name")).lower() != name.lower()]
        if len(kept) == len(rows):
            return False, "Watchlist nicht gefunden."
        if not self._save_rows(kept):
            return False, "Watchlist konnte nicht gelöscht werden."
        return True, f"Watchlist '{name}' wurde gelöscht."

    def _settings_for(self, rows: list[dict], name: str, wl_type: str, frequency: str):
        matching = [row for row in rows if self._normalise_name(row.get("Watchlist_Name")).lower() == name.lower()]
        if matching:
            first = matching[0]
            return (
                self._normalise_name(first.get("Watchlist_Type")) or wl_type,
                self._normalise_name(first.get("Alert_Mode")) or "Standard",
                self._normalise_name(first.get("Check_Frequency")) or frequency,
            )
        return wl_type, "Standard", frequency

    def add_entries_to_watchlist(self, watchlist_name, watchlist_type, tickers: Iterable[str], check_frequency="4x täglich"):
        name = self._normalise_name(watchlist_name)
        wl_type = self._normalise_name(watchlist_type) or "Watchlist"
        frequency = self._normalise_name(check_frequency) or "4x täglich"
        if not name:
            return False, "Watchlist-Name fehlt."
        rows = self._load_rows()
        actual_type, alert_mode, actual_frequency = self._settings_for(rows, name, wl_type, frequency)
        existing = {
            self._normalise_ticker(row.get("Ticker"))
            for row in rows
            if self._normalise_name(row.get("Watchlist_Name")).lower() == name.lower()
        }
        added = 0
        for raw in tickers or []:
            ticker = self._normalise_ticker(raw)
            if not ticker or ticker in existing:
                continue
            rows.append({
                "Watchlist_Name": name,
                "Watchlist_Type": actual_type,
                "Ticker": ticker,
                "Added_At": self._now(),
                "Alert_Mode": alert_mode,
                "Check_Frequency": actual_frequency,
            })
            existing.add(ticker)
            added += 1
        if not any(self._normalise_name(row.get("Watchlist_Name")).lower() == name.lower() for row in rows):
            rows.append({
                "Watchlist_Name": name,
                "Watchlist_Type": actual_type,
                "Ticker": "",
                "Added_At": self._now(),
                "Alert_Mode": alert_mode,
                "Check_Frequency": actual_frequency,
            })
        if not self._save_rows(rows):
            return False, "Watchlist-Einträge konnten nicht gespeichert werden."
        if added == 0:
            return True, "Keine neuen Ticker; vorhandene Einträge blieben unverändert."
        return True, f"{added} Ticker zur Watchlist '{name}' hinzugefügt."

    def remove_ticker_from_watchlist(self, watchlist_name, ticker):
        name = self._normalise_name(watchlist_name)
        symbol = self._normalise_ticker(ticker)
        rows = self._load_rows()
        kept = [
            row for row in rows
            if not (
                self._normalise_name(row.get("Watchlist_Name")).lower() == name.lower()
                and self._normalise_ticker(row.get("Ticker")) == symbol
            )
        ]
        if len(kept) == len(rows):
            return False, "Ticker nicht gefunden."
        if not self._save_rows(kept):
            return False, "Ticker konnte nicht entfernt werden."
        return True, f"{symbol} wurde aus '{name}' entfernt."

    def _first_setting(self, name: str, field: str, default: str) -> str:
        for row in self._load_rows():
            if self._normalise_name(row.get("Watchlist_Name")).lower() == name.lower():
                return self._normalise_name(row.get(field)) or default
        return default

    def get_watchlist_alert_mode(self, watchlist_name):
        return self._first_setting(self._normalise_name(watchlist_name), "Alert_Mode", "Standard")

    def get_watchlist_check_frequency(self, watchlist_name):
        return self._first_setting(self._normalise_name(watchlist_name), "Check_Frequency", "4x täglich")

    def _update_setting(self, watchlist_name, field: str, value: str):
        name = self._normalise_name(watchlist_name)
        rows = self._load_rows()
        changed = False
        for row in rows:
            if self._normalise_name(row.get("Watchlist_Name")).lower() == name.lower():
                row[field] = self._normalise_name(value)
                changed = True
        if not changed:
            return False, "Watchlist nicht gefunden."
        if not self._save_rows(rows):
            return False, "Einstellung konnte nicht gespeichert werden."
        return True, "Einstellung gespeichert."

    def update_watchlist_alert_mode(self, watchlist_name, alert_mode):
        return self._update_setting(watchlist_name, "Alert_Mode", alert_mode)

    def update_watchlist_check_frequency(self, watchlist_name, check_frequency):
        return self._update_setting(watchlist_name, "Check_Frequency", check_frequency)

    def get_due_watchlists_for_slot(self, slot_label):
        slot = self._normalise_name(slot_label)
        due_by_frequency = {
            "2x täglich": {"10:30", "18:30"},
            "3x täglich": {"10:30", "18:30", "22:10"},
            "4x täglich": {"10:30", "15:40", "18:30", "22:10"},
        }
        df, err = self.load_watchlists_df()
        if err:
            return pd.DataFrame(), err
        if df.empty or not slot:
            return pd.DataFrame(columns=["Watchlist_Name", "Watchlist_Type", "Alert_Mode", "Check_Frequency"]), None
        catalog = (
            df[["Watchlist_Name", "Watchlist_Type", "Alert_Mode", "Check_Frequency"]]
            .fillna("")
            .drop_duplicates(subset=["Watchlist_Name"], keep="first")
        )
        mask = catalog["Check_Frequency"].astype(str).map(lambda freq: slot in due_by_frequency.get(freq.strip(), set()))
        return catalog.loc[mask].reset_index(drop=True), None

    def import_dataframe(self, dataframe: pd.DataFrame, *, merge=True):
        if dataframe is None or dataframe.empty:
            return True, "Keine Watchlist-Daten zum Importieren."
        incoming = dataframe.fillna("").to_dict(orient="records")
        rows = self._load_rows() if merge else []
        by_key = {}
        for row in rows:
            key = (
                self._normalise_name(row.get("Watchlist_Name")).lower(),
                self._normalise_ticker(row.get("Ticker")),
            )
            by_key[key] = dict(row)
        for row in incoming:
            name = self._normalise_name(row.get("Watchlist_Name"))
            if not name:
                continue
            ticker = self._normalise_ticker(row.get("Ticker"))
            key = (name.lower(), ticker)
            existing = by_key.get(key, {})
            existing.update({k: v for k, v in row.items() if v not in (None, "")})
            existing.setdefault("Watchlist_Name", name)
            existing.setdefault("Watchlist_Type", "Watchlist")
            existing.setdefault("Ticker", ticker)
            existing.setdefault("Added_At", self._now())
            existing.setdefault("Alert_Mode", "Standard")
            existing.setdefault("Check_Frequency", "4x täglich")
            by_key[key] = existing
        merged_rows = list(by_key.values())
        if not self._save_rows(merged_rows):
            return False, "Watchlist-Import konnte nicht gespeichert werden."
        return True, f"{len(merged_rows)} Watchlist-Zeilen im neuen Speicher verfügbar."
