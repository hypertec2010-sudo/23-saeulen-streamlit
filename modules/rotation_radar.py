"""v30.1 Investment Rotation Radar.

Pure calculation/persistence helpers for cross-asset, sector and industry rotation.
The module deliberately does not fetch market data itself. The Streamlit app owns
provider access and passes one atomic daily-close frame into this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import math
import numpy as np
import pandas as pd

_STORAGE = None
_NAMESPACE = "rotation_radar_snapshot_v301"


def configure_storage(storage_manager=None) -> None:
    global _STORAGE
    _STORAGE = storage_manager


@dataclass(frozen=True)
class RotationSpec:
    symbol: str
    name: str
    level: str
    peer_group: str
    region: str
    benchmark: str
    breadth: tuple[str, ...] = ()


UNIVERSE: tuple[RotationSpec, ...] = (
    # Cross-asset / regions
    RotationSpec("SPY", "USA Aktien", "Investmentklasse / Region", "Aktienregionen", "USA", "ACWI"),
    RotationSpec("QQQ", "US Growth / Nasdaq 100", "Investmentklasse / Region", "Aktienregionen", "USA", "ACWI"),
    RotationSpec("VGK", "Europa Aktien", "Investmentklasse / Region", "Aktienregionen", "Europa", "ACWI"),
    RotationSpec("EWG", "Deutschland Aktien", "Investmentklasse / Region", "Aktienregionen", "Deutschland", "ACWI"),
    RotationSpec("EEM", "Emerging Markets", "Investmentklasse / Region", "Aktienregionen", "Global", "ACWI"),
    RotationSpec("TLT", "US Staatsanleihen lang", "Investmentklasse / Region", "Assetklassen", "USA", "ACWI"),
    RotationSpec("HYG", "High Yield Anleihen", "Investmentklasse / Region", "Assetklassen", "USA", "ACWI"),
    RotationSpec("GLD", "Gold", "Investmentklasse / Region", "Assetklassen", "Global", "ACWI"),
    RotationSpec("DBC", "Breite Rohstoffe", "Investmentklasse / Region", "Assetklassen", "Global", "ACWI"),
    RotationSpec("USO", "Öl", "Investmentklasse / Region", "Assetklassen", "Global", "ACWI"),
    RotationSpec("CPER", "Kupfer", "Investmentklasse / Region", "Assetklassen", "Global", "ACWI"),

    # US sectors
    RotationSpec("XLK", "Technologie", "Sektor", "US Sektoren", "USA", "SPY", ("MSFT", "AAPL", "NVDA", "AVGO", "CRM")),
    RotationSpec("XLF", "Finanzen", "Sektor", "US Sektoren", "USA", "SPY", ("JPM", "BAC", "WFC", "GS", "MS")),
    RotationSpec("XLI", "Industrie", "Sektor", "US Sektoren", "USA", "SPY", ("GE", "HON", "CAT", "UNP", "RTX")),
    RotationSpec("XLV", "Healthcare", "Sektor", "US Sektoren", "USA", "SPY", ("LLY", "JNJ", "ABBV", "UNH", "MRK")),
    RotationSpec("XLE", "Energie", "Sektor", "US Sektoren", "USA", "SPY", ("XOM", "CVX", "COP", "SLB", "EOG")),
    RotationSpec("XLB", "Materialien", "Sektor", "US Sektoren", "USA", "SPY", ("LIN", "SHW", "FCX", "NEM", "ECL")),
    RotationSpec("XLY", "Zyklischer Konsum", "Sektor", "US Sektoren", "USA", "SPY", ("AMZN", "TSLA", "HD", "MCD", "LOW")),
    RotationSpec("XLP", "Basiskonsum", "Sektor", "US Sektoren", "USA", "SPY", ("WMT", "COST", "PG", "KO", "PM")),
    RotationSpec("XLU", "Versorger", "Sektor", "US Sektoren", "USA", "SPY", ("NEE", "SO", "DUK", "CEG", "AEP")),
    RotationSpec("XLRE", "Immobilien", "Sektor", "US Sektoren", "USA", "SPY", ("PLD", "AMT", "EQIX", "WELL", "SPG")),
    RotationSpec("XLC", "Kommunikation", "Sektor", "US Sektoren", "USA", "SPY", ("META", "GOOGL", "NFLX", "TMUS", "DIS")),

    # Industries / themes
    RotationSpec("SMH", "Halbleiter", "Branche / Thema", "US Branchen & Themen", "Global/USA", "SPY", ("NVDA", "AVGO", "AMD", "TSM", "ASML")),
    RotationSpec("IGV", "Software", "Branche / Thema", "US Branchen & Themen", "USA", "SPY", ("MSFT", "ORCL", "CRM", "ADBE", "NOW")),
    RotationSpec("CIBR", "Cybersecurity", "Branche / Thema", "US Branchen & Themen", "Global/USA", "SPY", ("CRWD", "PANW", "FTNT", "ZS", "OKTA")),
    RotationSpec("ITA", "Defense / Aerospace", "Branche / Thema", "US Branchen & Themen", "USA", "SPY", ("RTX", "GE", "LMT", "NOC", "GD")),
    RotationSpec("XBI", "Biotech", "Branche / Thema", "US Branchen & Themen", "USA", "SPY", ("MRNA", "VRTX", "REGN", "BIIB", "ALNY")),
    RotationSpec("KRE", "Regionalbanken", "Branche / Thema", "US Branchen & Themen", "USA", "SPY", ("CFG", "KEY", "MTB", "RF", "HBAN")),
    RotationSpec("XHB", "Hausbau", "Branche / Thema", "US Branchen & Themen", "USA", "SPY", ("DHI", "LEN", "PHM", "TOL", "NVR")),
    RotationSpec("IYT", "Transport", "Branche / Thema", "US Branchen & Themen", "USA", "SPY", ("UNP", "UPS", "FDX", "CSX", "NSC")),
    RotationSpec("ICLN", "Clean Energy", "Branche / Thema", "US Branchen & Themen", "Global", "SPY", ("FSLR", "ENPH", "SEDG", "PLUG", "NOVA")),
    RotationSpec("TAN", "Solar", "Branche / Thema", "US Branchen & Themen", "Global", "SPY", ("FSLR", "ENPH", "SEDG", "NOVA", "ARRY")),
    RotationSpec("GDX", "Goldminen", "Branche / Thema", "US Branchen & Themen", "Global", "SPY", ("NEM", "GOLD", "AEM", "KGC", "WPM")),
    RotationSpec("COPX", "Kupferminen", "Branche / Thema", "US Branchen & Themen", "Global", "SPY", ("FCX", "SCCO", "TECK", "HBM", "ERO")),
)

_SPEC_BY_SYMBOL = {x.symbol: x for x in UNIVERSE}


def universe_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Ticker": s.symbol,
            "Name": s.name,
            "Ebene": s.level,
            "Peer-Gruppe": s.peer_group,
            "Region": s.region,
            "Benchmark": s.benchmark,
            "Breadth verfügbar": bool(s.breadth),
        }
        for s in UNIVERSE
    ])


def core_tickers() -> list[str]:
    out = {s.symbol for s in UNIVERSE}
    out.update(s.benchmark for s in UNIVERSE)
    return sorted(out)


def breadth_candidates() -> list[str]:
    return [s.symbol for s in UNIVERSE if s.breadth]


def breadth_tickers(symbols: Iterable[str]) -> list[str]:
    out = {"SPY"}
    for symbol in symbols or []:
        spec = _SPEC_BY_SYMBOL.get(str(symbol).upper())
        if spec:
            out.update(spec.breadth)
    return sorted(out)


def _safe_float(v, default=np.nan) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _clamp(v, lo=0.0, hi=100.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def _ret_pct(s: pd.Series, days: int, offset: int = 0) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    end = len(s) - 1 - int(offset)
    start = end - int(days)
    if start < 0 or end < 0:
        return np.nan
    a = _safe_float(s.iloc[start])
    b = _safe_float(s.iloc[end])
    if not math.isfinite(a) or not math.isfinite(b) or a <= 0:
        return np.nan
    return (b / a - 1.0) * 100.0


def _ma(s: pd.Series, window: int, offset: int = 0) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    end = len(s) - int(offset)
    start = end - int(window)
    if start < 0 or end <= 0:
        return np.nan
    return _safe_float(s.iloc[start:end].mean())


def _last(s: pd.Series, offset: int = 0) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    idx = len(s) - 1 - int(offset)
    if idx < 0:
        return np.nan
    return _safe_float(s.iloc[idx])


def _score_from_pct(value: float, scale: float) -> float:
    if not math.isfinite(_safe_float(value)):
        return 50.0
    return _clamp(50.0 + float(value) * float(scale))


def _metrics_for(spec: RotationSpec, prices: pd.DataFrame, offset: int = 0) -> dict[str, Any] | None:
    if spec.symbol not in prices.columns or spec.benchmark not in prices.columns:
        return None
    s = prices[spec.symbol]
    b = prices[spec.benchmark]
    px = _last(s, offset)
    if not math.isfinite(px) or px <= 0:
        return None

    r5 = _ret_pct(s, 5, offset)
    r21 = _ret_pct(s, 21, offset)
    r63 = _ret_pct(s, 63, offset)
    r126 = _ret_pct(s, 126, offset)
    b5 = _ret_pct(b, 5, offset)
    b21 = _ret_pct(b, 21, offset)
    b63 = _ret_pct(b, 63, offset)
    rs5 = r5 - b5 if math.isfinite(r5) and math.isfinite(b5) else np.nan
    rs21 = r21 - b21 if math.isfinite(r21) and math.isfinite(b21) else np.nan
    rs63 = r63 - b63 if math.isfinite(r63) and math.isfinite(b63) else np.nan

    ma20 = _ma(s, 20, offset)
    ma50 = _ma(s, 50, offset)
    ma200 = _ma(s, 200, offset)
    trend = 0.0
    if math.isfinite(ma20):
        trend += 25.0 if px >= ma20 else 0.0
    if math.isfinite(ma50):
        trend += 35.0 if px >= ma50 else 0.0
    if math.isfinite(ma200):
        trend += 40.0 if px >= ma200 else 0.0

    leadership = (
        0.40 * _score_from_pct(rs63, 2.4)
        + 0.25 * _score_from_pct(rs21, 3.5)
        + 0.15 * _score_from_pct(r21, 2.2)
        + 0.20 * trend
    )

    # Acceleration compares recent per-day relative/absolute progress with the
    # slower windows. This intentionally rewards an inflection before the
    # 63-day leadership score has fully caught up.
    def per_day(x, n):
        return (x / n) if math.isfinite(_safe_float(x)) else 0.0

    accel = (
        0.50 * (per_day(rs5, 5) - per_day(rs21, 21))
        + 0.30 * (per_day(rs21, 21) - per_day(rs63, 63))
        + 0.20 * (per_day(r5, 5) - per_day(r21, 21))
    )
    rotation = 50.0 + 45.0 * math.tanh(accel / 0.35)

    return {
        "Ticker": spec.symbol,
        "Name": spec.name,
        "Ebene": spec.level,
        "Peer-Gruppe": spec.peer_group,
        "Region": spec.region,
        "Benchmark": spec.benchmark,
        "Kurs": round(px, 4),
        "Perf 5T %": r5,
        "Perf 21T %": r21,
        "Perf 63T %": r63,
        "Perf 126T %": r126,
        "RS 5T %": rs5,
        "RS 21T %": rs21,
        "RS 63T %": rs63,
        "Trend-Score": round(_clamp(trend), 1),
        "Leadership": round(_clamp(leadership), 1),
        "Rotation": round(_clamp(rotation), 1),
    }


def _phase(row: pd.Series) -> str:
    lead = _safe_float(row.get("Leadership"), 50.0)
    rot = _safe_float(row.get("Rotation"), 50.0)
    trend = _safe_float(row.get("Trend-Score"), 50.0)
    rs21 = _safe_float(row.get("RS 21T %"), 0.0)
    if lead >= 70 and rot >= 52 and trend >= 60:
        return "🟢 Leading"
    if rot >= 65 and lead < 72 and trend >= 35:
        return "🟣 Emerging"
    if lead >= 66 and rot < 52:
        return "🟡 Mature"
    if (lead <= 42 and rot <= 38) or (rs21 <= -5 and trend <= 35 and rot < 45):
        return "🔴 Rotating Out"
    return "🟠 Cooling"


def _signal_text(row: pd.Series) -> str:
    phase = str(row.get("Phase", ""))
    dlead = _safe_float(row.get("Leadership Δ5T"), 0.0)
    drot = _safe_float(row.get("Rotation Δ5T"), 0.0)
    drank = _safe_float(row.get("Rang Δ5T"), 0.0)
    if "Emerging" in phase:
        if drank >= 3 or dlead >= 8:
            return "Früher positiver Trendshift · Rang/Leadership beschleunigt"
        return "Frühe Rotation hinein · Bestätigung beobachten"
    if "Leading" in phase:
        if drot < -10:
            return "Leader, aber Rotation verliert Tempo"
        return "Bestätigte Leadership / Kapital-Tailwind"
    if "Mature" in phase:
        return "Noch stark, aber Momentum flacht ab"
    if "Rotating Out" in phase:
        return "Leadership-Verlust / Kapital rotiert heraus"
    if drot <= -10 or dlead <= -8:
        return "Abkühlung beschleunigt"
    return "Übergang / abkühlender Kapitalfluss"


def build_radar(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build current radar plus provider errors from one atomic close-price frame."""
    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        return pd.DataFrame(), pd.DataFrame([{"Ticker": "-", "Fehler": "Keine Kursdaten"}]), {}
    frame = prices.copy()
    frame.columns = [str(c).upper() for c in frame.columns]
    offsets = (0, 1, 5, 20)
    by_offset: dict[int, pd.DataFrame] = {}
    errors: list[dict[str, str]] = []

    for off in offsets:
        rows = []
        for spec in UNIVERSE:
            metric = _metrics_for(spec, frame, off)
            if metric is not None:
                rows.append(metric)
            elif off == 0:
                errors.append({"Ticker": spec.symbol, "Name": spec.name, "Fehler": "Kurs-/Benchmark-Historie unvollständig"})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["Rang"] = df.groupby("Peer-Gruppe")["Leadership"].rank(method="min", ascending=False).astype(int)
        by_offset[off] = df

    cur = by_offset[0].copy()
    if cur.empty:
        return cur, pd.DataFrame(errors), {}

    for off, suffix in ((1, "1T"), (5, "5T"), (20, "20T")):
        hist = by_offset[off]
        if hist.empty:
            continue
        indexed = hist.set_index("Ticker")
        cur[f"Leadership {suffix}"] = cur["Ticker"].map(indexed["Leadership"])
        cur[f"Rotation {suffix}"] = cur["Ticker"].map(indexed["Rotation"])
        cur[f"Rang {suffix}"] = cur["Ticker"].map(indexed["Rang"])
        cur[f"Leadership Δ{suffix}"] = (cur["Leadership"] - cur[f"Leadership {suffix}"]).round(1)
        cur[f"Rotation Δ{suffix}"] = (cur["Rotation"] - cur[f"Rotation {suffix}"]).round(1)
        # Positive means rank improved (e.g. old 8 -> current 3 = +5)
        cur[f"Rang Δ{suffix}"] = (cur[f"Rang {suffix}"] - cur["Rang"]).round(0)

    for col in ("Leadership Δ1T", "Leadership Δ5T", "Leadership Δ20T", "Rotation Δ1T", "Rotation Δ5T", "Rotation Δ20T", "Rang Δ1T", "Rang Δ5T", "Rang Δ20T"):
        if col not in cur.columns:
            cur[col] = np.nan

    cur["Phase"] = cur.apply(_phase, axis=1)
    cur["Trendshift"] = cur.apply(_signal_text, axis=1)
    cur["Breadth"] = np.nan
    cur["Breadth-Status"] = "Nicht geprüft"
    cur["Breadth n"] = 0

    phase_order = {
        "🟣 Emerging": 0,
        "🟢 Leading": 1,
        "🟡 Mature": 2,
        "🟠 Cooling": 3,
        "🔴 Rotating Out": 4,
    }
    cur["_phase_order"] = cur["Phase"].map(phase_order).fillna(9)
    cur = cur.sort_values(["_phase_order", "Rotation", "Leadership"], ascending=[True, False, False]).drop(columns=["_phase_order"]).reset_index(drop=True)

    meta = {
        "processed": len(UNIVERSE),
        "success": int(len(cur)),
        "errors": int(len(errors)),
        "coverage_pct": round(100.0 * len(cur) / max(1, len(UNIVERSE)), 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "atomic": True,
    }
    return cur, pd.DataFrame(errors), meta


def build_breadth(prices: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    """Representative-member breadth for selected sectors/themes only."""
    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        return pd.DataFrame()
    p = prices.copy()
    p.columns = [str(c).upper() for c in p.columns]
    spy = p["SPY"] if "SPY" in p.columns else None
    rows = []
    for raw_symbol in symbols or []:
        symbol = str(raw_symbol).upper()
        spec = _SPEC_BY_SYMBOL.get(symbol)
        if not spec or not spec.breadth:
            continue
        member_rows = []
        for member in spec.breadth:
            if member not in p.columns:
                continue
            s = p[member]
            px = _last(s)
            ma20 = _ma(s, 20)
            ma50 = _ma(s, 50)
            ret21 = _ret_pct(s, 21)
            rs21 = np.nan
            if spy is not None:
                b21 = _ret_pct(spy, 21)
                if math.isfinite(ret21) and math.isfinite(b21):
                    rs21 = ret21 - b21
            if not math.isfinite(px):
                continue
            member_rows.append({
                "member": member,
                "above20": bool(math.isfinite(ma20) and px >= ma20),
                "above50": bool(math.isfinite(ma50) and px >= ma50),
                "pos21": bool(math.isfinite(ret21) and ret21 > 0),
                "rs21pos": bool(math.isfinite(rs21) and rs21 > 0),
            })
        n = len(member_rows)
        if not n:
            rows.append({"Ticker": symbol, "Breadth": np.nan, "Breadth-Status": "Keine Daten", "Breadth n": 0})
            continue
        md = pd.DataFrame(member_rows)
        pct20 = float(md["above20"].mean() * 100.0)
        pct50 = float(md["above50"].mean() * 100.0)
        pctpos = float(md["pos21"].mean() * 100.0)
        pctrsp = float(md["rs21pos"].mean() * 100.0)
        score = 0.25 * pct20 + 0.25 * pct50 + 0.25 * pctpos + 0.25 * pctrsp
        if score >= 70:
            status = "🟢 Bestätigt"
        elif score >= 50:
            status = "🟡 Gemischt"
        else:
            status = "🔴 Schwach"
        rows.append({
            "Ticker": symbol,
            "Breadth": round(score, 1),
            "Breadth-Status": status,
            "Breadth n": n,
            "% > MA20": round(pct20, 1),
            "% > MA50": round(pct50, 1),
            "% +21T": round(pctpos, 1),
            "% RS21 > 0": round(pctrsp, 1),
        })
    return pd.DataFrame(rows)


def merge_breadth(radar: pd.DataFrame, breadth: pd.DataFrame) -> pd.DataFrame:
    if radar is None or radar.empty or breadth is None or breadth.empty:
        return radar.copy() if isinstance(radar, pd.DataFrame) else pd.DataFrame()
    out = radar.copy()
    b = breadth.set_index("Ticker")
    for col in ["Breadth", "Breadth-Status", "Breadth n", "% > MA20", "% > MA50", "% +21T", "% RS21 > 0"]:
        if col in b.columns:
            mapped = out["Ticker"].map(b[col])
            if col not in out.columns:
                out[col] = mapped
            else:
                out[col] = mapped.where(mapped.notna(), out[col])
    return out


def top_rotation_candidates(radar: pd.DataFrame, limit: int = 5) -> list[str]:
    if radar is None or radar.empty:
        return []
    df = radar[radar["Ticker"].isin(breadth_candidates())].copy()
    if df.empty:
        return []
    phase_priority = df["Phase"].astype(str).map(lambda x: 0 if "Emerging" in x else (1 if "Leading" in x else 2))
    df["_p"] = phase_priority
    df = df.sort_values(["_p", "Rotation", "Leadership"], ascending=[True, False, False])
    return df["Ticker"].head(max(1, int(limit))).tolist()


def summarize(radar: pd.DataFrame) -> dict[str, Any]:
    if radar is None or radar.empty:
        return {}
    emerging = radar[radar["Phase"].astype(str).str.contains("Emerging", na=False)]
    leading = radar[radar["Phase"].astype(str).str.contains("Leading", na=False)]
    out = radar[radar["Phase"].astype(str).str.contains("Rotating Out", na=False)]
    top_rot = radar.sort_values("Rotation", ascending=False).iloc[0]
    rank_df = radar.dropna(subset=["Rang Δ5T"]).sort_values("Rang Δ5T", ascending=False)
    climber = rank_df.iloc[0] if not rank_df.empty else top_rot
    cooling = radar.sort_values("Rotation Δ5T", ascending=True).iloc[0]
    return {
        "emerging": int(len(emerging)),
        "leading": int(len(leading)),
        "rotating_out": int(len(out)),
        "top_rotation_name": str(top_rot.get("Name", "-")),
        "top_rotation_score": _safe_float(top_rot.get("Rotation"), np.nan),
        "climber_name": str(climber.get("Name", "-")),
        "climber_rank_delta": _safe_float(climber.get("Rang Δ5T"), np.nan),
        "cooling_name": str(cooling.get("Name", "-")),
        "cooling_delta": _safe_float(cooling.get("Rotation Δ5T"), np.nan),
    }


def _plain_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    clean = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict(orient="records")


def save_snapshot(radar: pd.DataFrame, errors: pd.DataFrame | None = None, meta: dict[str, Any] | None = None) -> bool:
    if radar is None or not isinstance(radar, pd.DataFrame) or radar.empty:
        return False
    payload = {
        "schema": "rotation-v30.1-atomic",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "radar": _plain_records(radar),
        "errors": _plain_records(errors if isinstance(errors, pd.DataFrame) else pd.DataFrame()),
        "meta": dict(meta or {}),
    }
    if _STORAGE is None:
        return False
    try:
        return bool(_STORAGE.save_namespace(_NAMESPACE, payload))
    except Exception:
        return False


def load_snapshot() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    if _STORAGE is None:
        return pd.DataFrame(), pd.DataFrame(), {}, ""
    try:
        payload = _STORAGE.load_namespace(_NAMESPACE, default={}) or {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or payload.get("schema") != "rotation-v30.1-atomic":
        return pd.DataFrame(), pd.DataFrame(), {}, ""
    try:
        radar = pd.DataFrame(payload.get("radar") or [])
        errors = pd.DataFrame(payload.get("errors") or [])
        meta = dict(payload.get("meta") or {})
        saved_at = str(payload.get("saved_at") or "")
        return radar, errors, meta, saved_at
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), {}, ""
