"""Watchlist-Startkurse, Batch-Queue und gebuendelte Speicherung."""
from __future__ import annotations
import re
from datetime import datetime
import pandas as pd
import streamlit as st

_CONTEXT = {}

def configure_context(**kwargs):
    global _CONTEXT
    _CONTEXT.update(kwargs)
    globals().update(kwargs)

def _v228_norm_watchlist_ticker(value):
    try:
        txt = str(value or "").strip().upper()
    except Exception:
        return ""
    return txt.replace(" ", "")


def _v2214_watchlist_key(watchlist_name, ticker):
    wl = str(watchlist_name or "default").strip() or "default"
    tk = _v228_norm_watchlist_ticker(ticker)
    return f"{wl}::{tk}"


def _v2214_set_start_price(watchlist_name, ticker, price=None, *, source="Backfill", added_at=None, force=False):
    wl = str(watchlist_name or "").strip()
    tk = _v228_norm_watchlist_ticker(ticker)
    if not wl or not tk:
        return False
    store = _v2214_load_start_price_store()
    key = _v2214_watchlist_key(wl, tk)
    existing = store.get(key) if isinstance(store.get(key), dict) else {}
    if existing and _v2214_valid_price(existing.get("Startkurs")) is not None and not force:
        return True
    price = _v2214_valid_price(price)
    if price is None:
        price = _v2214_get_current_price_for_ticker(tk)
    if price is None:
        return False
    try:
        now_txt = get_current_berlin_time().strftime("%d.%m.%Y %H:%M:%S")
    except Exception:
        now_txt = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    store[key] = {
        "Watchlist_Name": wl,
        "Ticker": tk,
        "Startkurs": round(float(price), 4),
        "Start_Price": round(float(price), 4),
        "Added_At": added_at or existing.get("Added_At") or now_txt,
        "Startkurs_Quelle": str(source or "Backfill"),
        "Startkurs_Gesetzt_Am": now_txt,
    }
    return _v2214_save_start_price_store(store)


def _v2214_get_start_price_meta_map(watchlist_name):
    wl = str(watchlist_name or "").strip()
    out = {}
    if not wl:
        return out
    store = _v2214_load_start_price_store()
    prefix = f"{wl}::".lower()
    for key, item in (store or {}).items():
        if not str(key).lower().startswith(prefix) or not isinstance(item, dict):
            continue
        tk = _v228_norm_watchlist_ticker(item.get("Ticker") or str(key).split("::")[-1])
        if tk:
            out[tk] = dict(item)
    return out


def _v232_delete_current_baselines_for_watchlist(watchlist_name):
    """Entfernt automatisch erzeugte Aktuell-Baselines, die wie echte Einstandskurse wirken koennen."""
    wl = str(watchlist_name or "").strip()
    if not wl:
        return 0
    store = _v2214_load_start_price_store()
    if not isinstance(store, dict):
        return 0
    prefix = f"{wl}::".lower()
    removed = 0
    for key in list(store.keys()):
        item = store.get(key)
        if not str(key).lower().startswith(prefix) or not isinstance(item, dict):
            continue
        source = item.get("Startkurs_Quelle") or item.get("Start_Source") or item.get("Quelle")
        if _v232_is_current_baseline_source(source):
            store.pop(key, None)
            removed += 1
    if removed:
        _v2214_save_start_price_store(store)
    return removed


def _v234_set_current_baselines_for_missing(watchlist_name, tickers):
    """Setzt bewusst eine Baseline ab jetzt fuer fehlende Startkurse.

    Wichtig: Das passiert nur auf Button-Klick, nicht automatisch im Live-Monitor.
    Damit wird klar: alte Performance ist unbekannt, ab jetzt wird sauber verfolgt.
    """
    wl = str(watchlist_name or "").strip()
    if not wl:
        return False, "Keine Watchlist ausgewaehlt."
    meta_map = _v2214_get_start_price_meta_map(wl)
    added, skipped, failed = 0, 0, []
    try:
        now_txt = get_current_berlin_time().strftime("%d.%m.%Y %H:%M:%S")
    except Exception:
        now_txt = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    for raw in tickers or []:
        tk = _v228_norm_watchlist_ticker(raw)
        if not tk:
            continue
        existing = meta_map.get(tk, {}) if isinstance(meta_map, dict) else {}
        if _v2214_valid_price(existing.get("Startkurs")) is not None:
            skipped += 1
            continue
        px = _v2214_get_current_price_for_ticker(tk)
        if px is None:
            failed.append(tk)
            continue
        if _v2214_set_start_price(wl, tk, px, source="Baseline ab jetzt (manuell)", added_at=now_txt, force=True):
            added += 1
        else:
            failed.append(tk)
    msg = f"{added} Baseline(s) ab jetzt gesetzt"
    if skipped:
        msg += f" · {skipped} bereits mit Startkurs"
    if failed:
        msg += f" · nicht abrufbar: {', '.join(failed[:8])}{' ...' if len(failed) > 8 else ''}"
    return added > 0, msg


def backfill_watchlist_start_prices_v2214(watchlist_name, tickers, *, force=False, meta_by_ticker=None, prefer_historical=True, allow_current_fallback=False):
    wl = str(watchlist_name or "").strip()
    added = 0
    skipped = 0
    historical = 0
    current_fallback = 0
    failed = []
    store = _v2214_load_start_price_store()
    meta_by_ticker = meta_by_ticker if isinstance(meta_by_ticker, dict) else {}

    for raw in tickers or []:
        tk = _v228_norm_watchlist_ticker(raw)
        if not tk:
            continue
        key = _v2214_watchlist_key(wl, tk)
        existing = store.get(key) if isinstance(store.get(key), dict) else {}
        if existing and _v2214_valid_price(existing.get("Startkurs")) is not None and not force:
            skipped += 1
            continue

        meta = meta_by_ticker.get(tk, {}) if isinstance(meta_by_ticker.get(tk, {}), dict) else {}
        added_at = _v2216_get_added_at_from_meta(meta) or existing.get("Added_At")
        price = None
        source = "Backfill aktuell"

        if prefer_historical and added_at:
            price = _v2216_get_historical_price_for_ticker(tk, added_at)
            if price is not None:
                source = "Historischer Backfill"
                historical += 1

        # v23.2: Bei bestehenden Watchlist-Werten keinen aktuellen Kurs mehr automatisch
        # als scheinbaren Einstand verwenden. Das erzeugte irrefuehrende Anzeigen wie
        # Startkurs = aktueller Kurs und +0.0%. Aktuelle Baseline nur, wenn explizit erlaubt.
        if price is None and allow_current_fallback:
            price = _v2214_get_current_price_for_ticker(tk)
            source = "Baseline ab jetzt"
            if price is not None:
                current_fallback += 1

        if price is None:
            failed.append(tk)
            continue
        if _v2214_set_start_price(wl, tk, price, source=source, added_at=added_at, force=force):
            added += 1
        else:
            failed.append(tk)

    msg = f"{added} Startkurs(e) nachgetragen"
    if historical:
        msg += f" · {historical} historisch"
    if current_fallback:
        msg += f" · {current_fallback} mit aktuellem Kurs"
    if skipped:
        msg += f" · {skipped} bereits vorhanden"
    if failed:
        msg += f" · nicht abrufbar: {', '.join(failed[:8])}{' ...' if len(failed) > 8 else ''}"
    return added > 0 or skipped > 0, msg


def _v228_get_pending_watchlist_adds():
    pending = st.session_state.get("pending_watchlist_adds_v228", [])
    if not isinstance(pending, list):
        pending = []
        st.session_state.pending_watchlist_adds_v228 = pending
    return pending


def _v228_pending_for_watchlist(watchlist_name):
    name_l = str(watchlist_name or "").strip().lower()
    out = []
    for item in _v228_get_pending_watchlist_adds():
        if str(item.get("Watchlist_Name", "")).strip().lower() == name_l:
            out.append(item)
    return out


def _v228_pending_tickers_for_watchlist(watchlist_name):
    return [_v228_norm_watchlist_ticker(x.get("Ticker")) for x in _v228_pending_for_watchlist(watchlist_name) if _v228_norm_watchlist_ticker(x.get("Ticker"))]


def queue_entries_to_watchlist_v228(watchlist_name, watchlist_type, entries, *, source="Manuell", check_frequency="4x täglich", existing_tickers=None):
    """Sammelt Watchlist-Aenderungen lokal, ohne Google Sheets sofort zu beschreiben."""
    watchlist_name = str(watchlist_name or "").strip()
    watchlist_type = str(watchlist_type or "Watchlist").strip() or "Watchlist"
    check_frequency = str(check_frequency or "4x täglich").strip() or "4x täglich"
    if not watchlist_name:
        return False, "Bitte zuerst eine Ziel-Watchlist auswaehlen."
    if entries is None:
        entries = []
    if isinstance(entries, str):
        entries = [entries]

    existing = {_v228_norm_watchlist_ticker(x) for x in (existing_tickers or []) if _v228_norm_watchlist_ticker(x)}
    pending = _v228_get_pending_watchlist_adds()
    pending_keys = {
        (str(x.get("Watchlist_Name", "")).strip().lower(), _v228_norm_watchlist_ticker(x.get("Ticker")))
        for x in pending
    }

    added = []
    skipped_existing = []
    skipped_duplicate = []
    try:
        now_txt = get_current_berlin_time().strftime("%d.%m.%Y %H:%M:%S")
    except Exception:
        now_txt = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    for raw in entries:
        ticker = _v228_norm_watchlist_ticker(raw)
        if not ticker:
            continue
        key = (watchlist_name.lower(), ticker)
        if ticker in existing:
            skipped_existing.append(ticker)
            continue
        if key in pending_keys:
            skipped_duplicate.append(ticker)
            continue
        # v22.17: Startkurs sofort lokal erfassen, aber ohne zusaetzlichen Google-Sheets-Write.
        start_price = _v2214_get_current_price_for_ticker(ticker)
        if start_price is not None:
            _v2214_set_start_price(watchlist_name, ticker, start_price, source=str(source or "Manuell"), added_at=now_txt)
        pending.append({
            "Watchlist_Name": watchlist_name,
            "Watchlist_Type": watchlist_type,
            "Ticker": ticker,
            "Quelle": str(source or "Manuell"),
            "Check_Frequency": check_frequency,
            "Vorgemerkt_Am": now_txt,
            "Added_At": now_txt,
            "Startkurs": round(float(start_price), 4) if start_price is not None else "n/a",
            "Start_Price": round(float(start_price), 4) if start_price is not None else "n/a",
        })
        pending_keys.add(key)
        added.append(ticker)

    st.session_state.pending_watchlist_adds_v228 = pending
    parts = []
    if added:
        parts.append(f"{len(added)} Wert(e) vorgemerkt: {', '.join(added[:8])}{' ...' if len(added) > 8 else ''}")
    if skipped_existing:
        parts.append(f"bereits vorhanden: {len(skipped_existing)}")
    if skipped_duplicate:
        parts.append(f"bereits vorgemerkt: {len(skipped_duplicate)}")
    if not parts:
        return False, "Keine neuen Werte zum Vormerken erkannt."
    return True, " · ".join(parts)


def save_pending_watchlist_adds_v228(*, watchlist_name=None):
    """Speichert vorgemerkte Eintraege gebuendelt. Pro Watchlist entsteht nur ein Sheets-Write."""
    pending = _v228_get_pending_watchlist_adds()
    if not pending:
        return False, "Keine ausstehenden Watchlist-Aenderungen vorhanden."
    target_l = str(watchlist_name or "").strip().lower()
    to_save = []
    keep = []
    for item in pending:
        if target_l and str(item.get("Watchlist_Name", "")).strip().lower() != target_l:
            keep.append(item)
        else:
            to_save.append(item)
    if not to_save:
        return False, "Fuer diese Watchlist gibt es keine ausstehenden Aenderungen."

    grouped = {}
    for item in to_save:
        wl = str(item.get("Watchlist_Name", "")).strip()
        wt = str(item.get("Watchlist_Type", "Watchlist")).strip() or "Watchlist"
        freq = str(item.get("Check_Frequency", "4x täglich")).strip() or "4x täglich"
        ticker = _v228_norm_watchlist_ticker(item.get("Ticker"))
        if not wl or not ticker:
            continue
        grouped.setdefault((wl, wt, freq), set()).add(ticker)

    ok_all = True
    messages = []
    saved_keys = set()
    for (wl, wt, freq), tickers in grouped.items():
        tickers_list = sorted(tickers)
        ok, msg = add_entries_to_watchlist(wl, wt, tickers_list, check_frequency=freq)
        if ok:
            saved_keys.update((wl.lower(), t) for t in tickers_list)
            messages.append(f"{wl}: {len(tickers_list)} Wert(e) gespeichert")
        else:
            ok_all = False
            messages.append(f"{wl}: Fehler - {msg}")

    new_pending = []
    for item in keep + to_save:
        wl = str(item.get("Watchlist_Name", "")).strip().lower()
        tk = _v228_norm_watchlist_ticker(item.get("Ticker"))
        if (wl, tk) not in saved_keys:
            new_pending.append(item)
    st.session_state.pending_watchlist_adds_v228 = new_pending
    return ok_all, " · ".join(messages) if messages else "Keine gueltigen Aenderungen gespeichert."


def clear_pending_watchlist_adds_v228(*, watchlist_name=None):
    pending = _v228_get_pending_watchlist_adds()
    target_l = str(watchlist_name or "").strip().lower()
    if not target_l:
        n = len(pending)
        st.session_state.pending_watchlist_adds_v228 = []
        return n
    keep = [x for x in pending if str(x.get("Watchlist_Name", "")).strip().lower() != target_l]
    removed = len(pending) - len(keep)
    st.session_state.pending_watchlist_adds_v228 = keep
    return removed


def render_pending_watchlist_adds_v228(*, selected_watchlist_name=None):
    pending = _v228_get_pending_watchlist_adds()
    if not pending:
        return
    df = pd.DataFrame(pending)
    if selected_watchlist_name:
        view_df = df[df["Watchlist_Name"].astype(str).str.strip().str.lower() == str(selected_watchlist_name).strip().lower()].copy()
    else:
        view_df = df.copy()
    if view_df.empty:
        return
    st.markdown("**Ausstehende Watchlist-Aenderungen**")
    st.caption("Diese Werte sind sofort lokal nutzbar, werden aber erst mit 'Aenderungen speichern' gebuendelt in Google Sheets geschrieben.")
    cols = [c for c in ["Watchlist_Name", "Ticker", "Startkurs", "Quelle", "Vorgemerkt_Am"] if c in view_df.columns]
    st.dataframe(view_df[cols], hide_index=True, use_container_width=True, height=min(260, 42 * len(view_df) + 50))
    b1, b2, b3 = st.columns([1.1, 1.0, 1.2])
    safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", str(selected_watchlist_name or "all"))
    with b1:
        if st.button("Aenderungen speichern", use_container_width=True, key=f"save_pending_watchlist_adds_v228_{safe_key}"):
            ok, msg = save_pending_watchlist_adds_v228(watchlist_name=selected_watchlist_name)
            if ok:
                st.success(msg)
                trigger_ui_refresh()
            else:
                st.error(msg)
    with b2:
        if st.button("Queue leeren", use_container_width=True, key=f"clear_pending_watchlist_adds_v228_{safe_key}"):
            n = clear_pending_watchlist_adds_v228(watchlist_name=selected_watchlist_name)
            st.info(f"{n} vorgemerkte Aenderung(en) geloescht.")
            trigger_ui_refresh()
    with b3:
        st.caption(f"Ausstehend: {len(view_df)}")
