"""Compact presentation adapter for the unchanged bz watcher.

Only the watcher page receives this view object. It delegates all Streamlit
operations and widget results, replacing presentation of the wide overview.
No global monkeypatching, provider calls, storage writes or signal calculations.
"""
from __future__ import annotations

import importlib
import math

import pandas as pd

REQUIRED = {"Ticker", "Status", "Kurs", "Entry", "Invalidierung", "Ziel"}
MAIN_COLUMNS = ["Ticker", "Status", "Kurs", "Entry-Bereich", "Invalidierung", "Ziel", "Einheit"]


def _level(value):
    try:
        value = float(value)
        if not math.isfinite(value):
            return "–"
        digits = 4 if abs(value) >= 1 else 6
        return f"{value:.{digits}f}".rstrip("0").rstrip(".").replace(".", ",")
    except (ValueError, TypeError):
        return "–"


def compact_frame(frame):
    """Keep the full entry band in view without changing any underlying level."""
    if not isinstance(frame, pd.DataFrame) or not REQUIRED.issubset(frame.columns):
        return None
    result = frame[["Ticker", "Status", "Kurs", "Entry", "Invalidierung", "Ziel"]].copy(deep=True)
    ceilings = frame["Entry maximal"] if "Entry maximal" in frame else frame["Entry"]
    result["Entry-Bereich"] = [f"{_level(low)} – {_level(high)}" for low, high in zip(frame["Entry"], ceilings)]
    result["Einheit"] = frame["Einheit"] if "Einheit" in frame else "–"
    return result[MAIN_COLUMNS]


def data_notes(frame):
    """Hidden timestamps must not hide a stale/error/gap warning."""
    alerts, pending = [], []
    for row in frame.to_dict("records"):
        ticker = str(row.get("Ticker") or "–")
        health = str(row.get("Daten") or "Datenzustand unbekannt")
        if health.startswith("Aktuelle 1m-Daten"):
            continue
        text = ticker + ": " + health
        if health.startswith(("Noch nicht geprüft", "Wartet auf erste")):
            pending.append(text)
        else:
            alerts.append(text)
    return alerts, pending


class CompactWatcherView:
    """Page-scoped view; every non-overview call reaches the original UI intact."""
    def __init__(self, streamlit):
        self._st = streamlit

    def __getattr__(self, name):
        return getattr(self._st, name)

    def dataframe(self, data=None, *args, **kwargs):
        compact = compact_frame(data)
        if compact is None:
            return self._st.dataframe(data, *args, **kwargs)
        display = dict(kwargs)
        display.pop("column_order", None)
        if isinstance(display.get("column_config"), dict):
            display["column_config"] = {key: val for key, val in display["column_config"].items() if key in MAIN_COLUMNS}
        shown = self._st.dataframe(compact, *args, **display)
        self._st.caption("Entry-Bereich = Auslöser bis oberste Einstiegsgrenze. Das Ziel ist das gespeicherte Kursniveau; ein leeres Ziel bedeutet: kein Ziel hinterlegt.")
        alerts, pending = data_notes(data)
        if alerts:
            text = " · ".join(alerts[:3])
            rest = len(alerts) - 3
            if rest > 0:
                text += f" · {rest} weitere Datenhinweise in den Details."
            self._st.warning("Daten prüfen: " + text)
        if pending:
            self._st.info("Noch ohne verwertbare Prüfung: " + " · ".join(pending[:3]))
        with self._st.expander("Datenzeiten & technische Details", expanded=False):
            columns = [key for key in data.columns if key not in {"Status", "Kurs", "Entry", "Invalidierung", "Ziel", "Einheit"}]
            self._st.dataframe(data[columns].copy(deep=True), hide_index=True, use_container_width=True)
            self._st.caption("Hier stehen insbesondere Datenzustand, Kurszeit, Abrufzeit, Signalzeit, Regel, Ablauf und Setup-Herkunft. Ein gespeicherter Trigger bleibt ein historisches Ereignis.")
        return shown


def render_compact_monitor(st, *, renderer=None):
    """bz is a required base; do not accidentally restore bx free-form entry."""
    if renderer is None:
        try:
            setup = importlib.import_module("modules.watch_setups")
        except ImportError:
            st.error("Diese Layout-Ergänzung benötigt die installierte v30.21bz. Bitte zuerst das bz-Paket vollständig installieren.")
            return
        if getattr(setup, "POLICY", "") != "chart_watch_v30.21bz":
            st.error("Die vorhandene Chart-Erkennung passt nicht zu dieser Layout-Ergänzung. Basis v30.21bz erforderlich.")
            return
        from modules.trade_watch_ui import render_monitor
        renderer = render_monitor
    renderer(CompactWatcherView(st))
