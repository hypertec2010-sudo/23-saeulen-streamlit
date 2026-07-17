"""Chart-, Overlay- und Chartstruktur-Helfer, ausgelagert aus app.py (v25.4)."""
from __future__ import annotations

import math
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_CTX = {}
_CHART_LARGE_LABELS_V207 = False

def configure_context(**kwargs):
    _CTX.update(kwargs)
    globals().update(kwargs)

def _chart_annotation_style_v206(kind="neutral"):
    """Einheitliche, kontrastreiche Chart-Badges fuer v21.1.

    v21.1: Labels koennen optional groesser dargestellt werden. Zusaetzlich
    bekommen alle Badge-Labels unsichtbare Hover-Punkte mit gut lesbaren
    Detail-Tooltips.
    """
    kind = str(kind or "neutral").lower()
    large = bool(globals().get("_CHART_LARGE_LABELS_V207", False))
    small_size = 10 if large else 8
    trend_size = 11 if large else 9
    price_size = 13 if large else 11
    pad_small = 4 if large else 2
    styles = {
        "fib": dict(font=dict(size=small_size, color="#111827"), bgcolor="rgba(255,255,255,0.90)", bordercolor="rgba(31,41,55,0.38)", borderwidth=1, borderpad=pad_small),
        "support": dict(font=dict(size=small_size, color="#052e16"), bgcolor="rgba(220,252,231,0.92)", bordercolor="rgba(22,163,74,0.42)", borderwidth=1, borderpad=pad_small),
        "resistance": dict(font=dict(size=small_size, color="#450a0a"), bgcolor="rgba(254,226,226,0.92)", bordercolor="rgba(220,38,38,0.42)", borderwidth=1, borderpad=pad_small),
        "active": dict(font=dict(size=small_size, color="#082f49"), bgcolor="rgba(224,242,254,0.92)", bordercolor="rgba(14,165,233,0.42)", borderwidth=1, borderpad=pad_small),
        "trend": dict(font=dict(size=trend_size, color="#111827"), bgcolor="rgba(255,255,255,0.88)", bordercolor="rgba(59,130,246,0.35)", borderwidth=1, borderpad=4 if large else 3),
        "price": dict(font=dict(size=price_size, color="white"), bgcolor="rgba(2,132,199,0.94)", bordercolor="rgba(226,232,240,0.90)", borderwidth=1, borderpad=5 if large else 4),
    }
    return styles.get(kind, dict(font=dict(size=small_size, color="#111827"), bgcolor="rgba(255,255,255,0.88)", bordercolor="rgba(31,41,55,0.30)", borderwidth=1, borderpad=pad_small))

def _chart_hover_text_v207(text, kind="neutral"):
    try:
        clean = str(text or "").replace("<br>", " ").strip()
        title = str(kind or "Info").replace("_", " ").title()
        return f"<b>{clean}</b><br>{title}<extra></extra>"
    except Exception:
        return str(text or "") + "<extra></extra>"

def _chart_add_hover_point_v207(fig, *, x, y, text, kind="neutral", row=1, col=1):
    """Unsichtbarer Hover-Trefferpunkt fuer Annotationen.

    Plotly-Annotationen selbst besitzen keinen verlaesslichen Hover-/Mouseover-
    Stil. Dieser sehr transparente Marker erzeugt einen gut lesbaren Tooltip,
    ohne den Chart sichtbar zu ueberladen.
    """
    try:
        if x is None or y is None:
            return
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[float(y)],
                mode="markers",
                name=str(text or "Info"),
                marker=dict(size=22 if bool(globals().get("_CHART_LARGE_LABELS_V207", False)) else 18, color="rgba(15,23,42,0.01)", line=dict(width=0)),
                hovertemplate=_chart_hover_text_v207(text, kind),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    except Exception:
        pass

def _chart_label_yshift_v206(index, total=1, amplitude=13):
    """Kleine Pixel-Staffelung gegen Label-Ueberlagerungen.

    Plotly-Annotationen bleiben am korrekten Preis verankert, werden aber
    optisch leicht versetzt. Dadurch kollidieren nahe Fib-/S/R-Labels nicht
    mehr direkt miteinander.
    """
    try:
        i = int(index)
        total = max(1, int(total or 1))
        if total <= 1:
            return 0
        pattern = [0, 12, -12, 24, -24, 36, -36]
        return pattern[i % len(pattern)]
    except Exception:
        return 0

def _chart_add_annotation_v206(fig, *, x, y, text, kind="neutral", xanchor="left", yanchor="middle", yshift=0, row=1, col=1):
    """Robuste Chart-Beschriftung mit einheitlichem Kontrast."""
    try:
        style = _chart_annotation_style_v206(kind)
        fig.add_annotation(
            x=x,
            y=y,
            text=str(text or ""),
            showarrow=False,
            xanchor=xanchor,
            yanchor=yanchor,
            align="left",
            yshift=yshift,
            row=row,
            col=col,
            **style,
        )
        # v21.1: Mouseover ueber dem Label zeigt groesseren Detail-Tooltip.
        _chart_add_hover_point_v207(fig, x=x, y=y, text=text, kind=kind, row=row, col=col)
    except Exception:
        pass

def _chart_zone_compact_label_v205(prefix, idx, z, ccy=""):
    """Kompaktes Zonenlabel fuer den Chart; Details bleiben im Hover/unterhalb des Charts."""
    try:
        lo = float(z.get("low"))
        hi = float(z.get("high"))
        touches = int(z.get("touches", 0) or 0)
        cur = f" {ccy}" if str(ccy or "").strip() else ""
        touch_txt = f" · {touches}x" if touches else ""
        if abs(hi - lo) <= max(abs(hi), 1.0) * 0.002:
            return f"{prefix}{idx} {hi:.2f}{cur}{touch_txt}"
        return f"{prefix}{idx} {lo:.2f}-{hi:.2f}{cur}{touch_txt}"
    except Exception:
        return format_chart_zone_label(prefix, idx, z, ccy=ccy)

def add_fibonacci_levels_to_plotly_v1533(fig, chart_df, fib_pkg, show_labels=True):
    """Optionale Fibonacci-Linien im Chart. In kompakten Ansichten ohne rechte Dauerlabels."""
    try:
        if not isinstance(fib_pkg, dict):
            return fig
        levels = fib_pkg.get("levels") or []
        if not levels:
            return fig

        def _fmt_fib_price_v1535_1(value):
            try:
                v = float(value)
                if not pd.notna(v):
                    return "n/a"
                if abs(v) >= 100:
                    return f"{v:,.2f}"
                if abs(v) >= 10:
                    return f"{v:,.2f}"
                if abs(v) >= 1:
                    return f"{v:,.2f}"
                return f"{v:,.4f}"
            except Exception:
                return str(value or "n/a")

        # Plotlys add_hline annotation can truncate labels in some Streamlit/browser
        # combinations. Therefore we draw the line and the label separately.
        try:
            x0 = chart_df.index[0]
            x1 = chart_df.index[-1]
        except Exception:
            x0 = None
            x1 = None

        _fib_items = sorted(levels, key=lambda it: float(it.get("Kurszone") or 0), reverse=True)
        for _fib_idx, item in enumerate(_fib_items):
            y = item.get("Kurszone")
            name = str(item.get("Level") or "").strip()
            if y is None or str(y).strip().lower() in {"", "n/a", "nan", "none"}:
                continue
            y_float = float(y)
            price_txt = _fmt_fib_price_v1535_1(y_float)
            label = f"Fib {name} · {price_txt}" if name else f"Fib · {price_txt}"

            if x0 is not None and x1 is not None:
                fig.add_shape(
                    type="line",
                    x0=x0,
                    x1=x1,
                    y0=y_float,
                    y1=y_float,
                    xref="x",
                    yref="y",
                    line=dict(width=1, dash="dot", color="rgba(220,220,220,0.55)"),
                    row=1,
                    col=1,
                )
                if show_labels:
                    _chart_add_annotation_v206(
                        fig,
                        x=_chart_label_xpos_v205(chart_df, "left") or x0,
                        y=y_float,
                        text=label,
                        kind="fib",
                        xanchor="left",
                        yanchor="middle",
                        yshift=_chart_label_yshift_v206(_fib_idx, len(_fib_items)),
                        row=1,
                        col=1,
                    )
            else:
                fig.add_hline(
                    y=y_float,
                    line_width=1,
                    line_dash="dot",
                    opacity=0.45,
                    row=1,
                    col=1,
                )
        return fig
    except Exception:
        return fig

def format_chart_zone_label(prefix, idx, zone, ccy=""):
    try:
        low = float(zone.get("low", np.nan))
        high = float(zone.get("high", np.nan))
        touches = int(zone.get("touches", 0))
        ccy_suffix = f" {ccy}".strip()
        return f"{prefix}{idx} ({touches}x) - {low:.2f} bis {high:.2f}{(' ' + ccy) if ccy else ''}"
    except Exception:
        touches = zone.get("touches", "?")
        return f"{prefix}{idx} ({touches}x)"

def add_sr_zones_to_plotly(fig, df, supports, resistances, active_zones=None, ccy="", show_labels=True):
    if df is None or df.empty:
        return
    x0 = df.index.min()
    x1 = df.index.max()

    for idx, z in enumerate(supports, start=1):
        label = _chart_zone_compact_label_v205("S", idx, z, ccy=ccy)
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=z["low"],
            y1=z["high"],
            line=dict(width=0),
            fillcolor="rgba(34,197,94,0.24)",
            layer="below",
            row=1,
            col=1
        )
        if show_labels:
            _chart_add_annotation_v206(
                fig,
                x=_chart_label_xpos_v205(df, "left") or x0,
                y=z["mid"],
                text=label,
                kind="support",
                xanchor="left",
                yanchor="middle",
                yshift=_chart_label_yshift_v206(idx - 1, len(supports)),
                row=1,
                col=1,
            )

    for idx, z in enumerate(resistances, start=1):
        label = _chart_zone_compact_label_v205("R", idx, z, ccy=ccy)
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=z["low"],
            y1=z["high"],
            line=dict(width=0),
            fillcolor="rgba(239,68,68,0.24)",
            layer="below",
            row=1,
            col=1
        )
        if show_labels:
            _chart_add_annotation_v206(
                fig,
                x=_chart_label_xpos_v205(df, "mid") or x1,
                y=z["mid"],
                text=label,
                kind="resistance",
                xanchor="center",
                yanchor="middle",
                yshift=_chart_label_yshift_v206(idx - 1, len(resistances)),
                row=1,
                col=1,
            )


    for idx, z in enumerate(active_zones or [], start=1):
        label = _chart_zone_compact_label_v205("Aktiv ", idx, z, ccy=ccy)
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=z["low"],
            y1=z["high"],
            line=dict(width=0),
            fillcolor="rgba(59,130,246,0.18)",
            layer="below",
            row=1,
            col=1
        )
        if show_labels:
            _chart_add_annotation_v206(
                fig,
                x=_chart_label_xpos_v205(df, "right_inner") or x1,
                y=z["mid"],
                text=label,
                kind="active",
                xanchor="center",
                yanchor="middle",
                yshift=_chart_label_yshift_v206(idx - 1, len(active_zones or [])),
                row=1,
                col=1,
            )

def add_trend_channel_to_plotly(fig, df, channel):
    if not channel or df is None or df.empty:
        return

    x_vals = np.arange(len(df), dtype=float)
    x_dates = df.index
    slope = channel["slope"]
    lower_intercept = channel["lower_intercept"]
    upper_intercept = channel["upper_intercept"]

    lower_y = slope * x_vals + lower_intercept
    upper_y = slope * x_vals + upper_intercept

    lower_name = "Trendkanal unten" if channel.get("source") == "pivot" else "Trendkanal unten (Reg.)"
    upper_name = "Trendkanal oben" if channel.get("source") == "pivot" else "Trendkanal oben (Reg.)"

    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=lower_y,
            mode="lines",
            name=lower_name,
            line=dict(dash="dash", width=2.2, color="rgba(34,197,94,0.95)")
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=upper_y,
            mode="lines",
            name=upper_name,
            line=dict(dash="dash", width=2.2, color="rgba(239,68,68,0.95)")
        ),
        row=1,
        col=1
    )

    label = channel.get("label", "Trendkanal")
    quality = channel.get("quality", "")
    label_text = f"{label} - Qualitaet: {quality}" if quality else label

    _chart_add_annotation_v206(
        fig,
        x=_chart_label_xpos_v205(df, "mid") or x_dates[-1],
        y=float((lower_y[-1] + upper_y[-1]) / 2.0),
        text=label_text,
        kind="trend",
        xanchor="center",
        yanchor="middle",
        yshift=18,
        row=1,
        col=1,
    )

def summarize_chart_structures(df, structures):
    summaries = []
    if df is None or df.empty or not structures:
        return summaries

    try:
        current_price = float(pd.to_numeric(df["Close"], errors="coerce").iloc[-1])
    except Exception:
        return summaries

    supports = structures.get("supports", []) or []
    resistances = structures.get("resistances", []) or []
    active_zones = structures.get("active_zones", []) or []
    channel = structures.get("channel")

    if supports:
        s1 = supports[0]
        s_mid = float(s1.get("mid", np.nan))
        s_touches = int(s1.get("touches", 0) or 0)
        if pd.notna(s_mid) and s_mid > 0:
            dist_pct = ((current_price / s_mid) - 1.0) * 100.0
            if abs(dist_pct) <= 1.5:
                summaries.append(f"Kurs aktuell nahe Support S1 ({s_touches}x) bei {s_mid:.2f}.")
            elif current_price < s_mid:
                summaries.append(f"Kurs unter S1 bei {s_mid:.2f} - Support wurde zuletzt unterschritten.")
            else:
                summaries.append(f"Nächster Support S1 liegt bei {s_mid:.2f}, Abstand {dist_pct:.1f}%.")

    if active_zones:
        z0 = active_zones[0]
        z_mid = float(z0.get("mid", np.nan))
        if pd.notna(z_mid):
            summaries.append(f"Kurs handelt aktuell in einer aktiven Zone um {z_mid:.2f}.")

    if resistances:
        r1 = resistances[0]
        r_mid = float(r1.get("mid", np.nan))
        r_touches = int(r1.get("touches", 0) or 0)
        if pd.notna(r_mid) and r_mid > 0:
            dist_pct = ((r_mid / current_price) - 1.0) * 100.0
            if abs(dist_pct) <= 1.5:
                summaries.append(f"Kurs läuft direkt an Widerstand R1 ({r_touches}x) bei {r_mid:.2f}.")
            elif current_price > r_mid:
                summaries.append(f"Kurs über R1 bei {r_mid:.2f} - Ausbruch über den nächsten Widerstand.")
            else:
                summaries.append(f"Nächster Widerstand R1 liegt bei {r_mid:.2f}, Abstand {dist_pct:.1f}%.")

    if channel:
        try:
            idx = len(df) - 1
            slope = float(channel.get("slope", 0.0))
            lower = slope * idx + float(channel.get("lower_intercept", 0.0))
            upper = slope * idx + float(channel.get("upper_intercept", 0.0))
            if upper > lower and current_price > 0:
                pos = (current_price - lower) / (upper - lower)
                label = channel.get("label", "Trendkanal")
                quality = channel.get("quality", "")
                quality_txt = f" ({quality})" if quality else ""
                if pos <= 0.25:
                    summaries.append(f"Kurs im unteren Bereich des {label.lower()}{quality_txt} - eher supportnah.")
                elif pos >= 0.75:
                    summaries.append(f"Kurs im oberen Bereich des {label.lower()}{quality_txt} - eher widerstandsnah.")
                else:
                    summaries.append(f"Kurs bewegt sich im mittleren Bereich des {label.lower()}{quality_txt}.")
        except Exception:
            pass

    return summaries

def compute_ultra_short_term_zone_signal(df, structures):
    signal = {
        "label": "Kein Signal",
        "strength": 0,
        "confirmation": "fehlt",
        "reason": "Keine klare Reaktion an einer relevanten Zone.",
        "bullets": [],
        "tone": "blue",
    }
    if df is None or df.empty or not structures or len(df) < 5:
        return signal

    try:
        close = pd.to_numeric(df["Close"], errors="coerce")
        open_ = pd.to_numeric(df["Open"], errors="coerce")
        high = pd.to_numeric(df["High"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")
        volume = pd.to_numeric(df.get("Volume"), errors="coerce") if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)
        current_price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        prev_high = float(high.iloc[-2])
        prev_low = float(low.iloc[-2])
    except Exception:
        return signal

    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    vol20 = volume.rolling(20, min_periods=5).mean() if not volume.empty else pd.Series(index=df.index, dtype=float)

    candle_range = max(0.01, float(high.iloc[-1] - low.iloc[-1]))
    lower_wick = max(0.0, float(min(open_.iloc[-1], close.iloc[-1]) - low.iloc[-1]))
    upper_wick = max(0.0, float(high.iloc[-1] - max(open_.iloc[-1], close.iloc[-1])))
    lower_wick_ratio = lower_wick / candle_range
    upper_wick_ratio = upper_wick / candle_range
    close_pos = (float(close.iloc[-1]) - float(low.iloc[-1])) / candle_range

    supports = structures.get("supports", []) or []
    resistances = structures.get("resistances", []) or []
    active_zones = structures.get("active_zones", []) or []
    channel = structures.get("channel")

    bull = 0.0
    bear = 0.0
    watch = 0.0
    confirm = 0.0
    reasons_bull = []
    reasons_bear = []
    reasons_watch = []

    near_support = False
    near_resistance = False

    if supports:
        s1 = supports[0]
        s_mid = float(s1.get("mid", np.nan))
        if pd.notna(s_mid) and current_price > 0:
            dist_support = abs((current_price / s_mid) - 1.0) * 100.0
            if dist_support <= 1.8:
                bull += 26
                near_support = True
                reasons_bull.append("Kurs direkt an Support S1")
            elif dist_support <= 3.2:
                bull += 16
                reasons_bull.append("Kurs nahe Support S1")
            if current_price < s_mid:
                bear += 16
                reasons_bear.append("Kurs unter Support S1")

    if resistances:
        r1 = resistances[0]
        r_mid = float(r1.get("mid", np.nan))
        if pd.notna(r_mid) and current_price > 0:
            dist_res = abs((r_mid / current_price) - 1.0) * 100.0
            if dist_res <= 1.8:
                bear += 26
                near_resistance = True
                reasons_bear.append("Kurs direkt an Widerstand R1")
            elif dist_res <= 3.2:
                bear += 16
                reasons_bear.append("Kurs nahe Widerstand R1")
            if current_price > r_mid:
                bull += 14
                reasons_bull.append("Kurs ueber R1")

    if active_zones:
        z0 = active_zones[0]
        z_low = float(z0.get("low", np.nan))
        z_high = float(z0.get("high", np.nan))
        if pd.notna(z_low) and pd.notna(z_high) and z_low <= current_price <= z_high:
            watch += 24
            reasons_watch.append("Kurs in aktiver Entscheidungszone")

    if near_support:
        if lower_wick_ratio >= 0.35:
            bull += 12
            reasons_bull.append("unterer Docht an Support")
        if close_pos >= 0.62:
            bull += 8
            reasons_bull.append("Schlusskurs erholt sich aus der Zone")
        if float(close.iloc[-1]) > float(close.iloc[-2]):
            confirm += 10

    if near_resistance:
        if upper_wick_ratio >= 0.35:
            bear += 12
            reasons_bear.append("oberer Docht an Widerstand")
        if close_pos <= 0.42:
            bear += 8
            reasons_bear.append("schwacher Schlusskurs an R1")
        if float(close.iloc[-1]) < float(close.iloc[-2]):
            confirm += 10

    if len(df) >= 3:
        ret2 = ((float(close.iloc[-1]) / float(close.iloc[-3])) - 1.0) * 100.0
    else:
        ret2 = np.nan
    ret3 = ((float(close.iloc[-1]) / float(close.iloc[-4])) - 1.0) * 100.0 if len(df) >= 4 else np.nan

    if pd.notna(ema10.iloc[-1]):
        if current_price > float(ema10.iloc[-1]) and near_support:
            bull += 8
            reasons_bull.append("ueber EMA10")
        elif current_price < float(ema10.iloc[-1]) and near_resistance:
            bear += 8
            reasons_bear.append("unter EMA10")

    if pd.notna(ret2):
        if ret2 > 1.5 and near_support:
            bull += 10
            confirm += 8
            reasons_bull.append("2T-Momentum zieht an")
        elif ret2 < -1.5 and near_resistance:
            bear += 10
            confirm += 8
            reasons_bear.append("2T-Momentum kippt ab")
    if pd.notna(ret3):
        if ret3 > 2.5 and near_support:
            bull += 6
        elif ret3 < -2.5 and near_resistance:
            bear += 6

    if not vol20.empty and pd.notna(vol20.iloc[-1]) and vol20.iloc[-1] > 0 and not volume.empty and pd.notna(volume.iloc[-1]):
        vol_ratio = float(volume.iloc[-1] / vol20.iloc[-1])
        if near_support and float(close.iloc[-1]) > float(open_.iloc[-1]) and vol_ratio >= 1.1:
            bull += 8
            reasons_bull.append("Reaktion mit besserem Volumen")
        elif near_resistance and float(close.iloc[-1]) < float(open_.iloc[-1]) and vol_ratio >= 1.1:
            bear += 8
            reasons_bear.append("Ablehnung mit erhoehtem Volumen")
        elif vol_ratio < 0.85 and (near_support or near_resistance):
            watch += 4

    if channel:
        try:
            idx_last = len(df) - 1
            slope = float(channel.get("slope", 0.0))
            lower = slope * idx_last + float(channel.get("lower_intercept", 0.0))
            upper = slope * idx_last + float(channel.get("upper_intercept", 0.0))
            if upper > lower:
                pos = (current_price - lower) / (upper - lower)
                if pos <= 0.22:
                    bull += 10
                    reasons_bull.append("unterer Kanalbereich")
                elif pos >= 0.78:
                    bear += 10
                    reasons_bear.append("oberer Kanalbereich")
        except Exception:
            pass

    bull = int(round(clamp(bull, 0, 100)))
    bear = int(round(clamp(bear, 0, 100)))
    watch = int(round(clamp(watch, 0, 100)))
    confirm = int(round(clamp(confirm, 0, 100)))

    # v15.24.13: Ultra-Kurzfrist war zu streng und fiel dadurch fast immer auf
    # "Kein Signal" zurück. Für den Nutzer ist aber bereits eine kurzfristige
    # Reaktion an/nahe einer Zone relevant. Deshalb: harte Signale bleiben streng,
    # frühe Reaktionen werden separat ausgewiesen, statt komplett neutral zu wirken.
    max_side = max(bull, bear)
    side_gap = abs(bull - bear)

    if bull >= 48 and bull >= bear + 6:
        signal.update({
            "label": "Ultra-Kurzfrist bullish",
            "strength": bull,
            "tone": "blue",
            "reason": reasons_bull[0] if reasons_bull else "Support wird kurzfristig verteidigt.",
            "bullets": list(dict.fromkeys(reasons_bull))[:4],
        })
    elif bear >= 48 and bear >= bull + 6:
        signal.update({
            "label": "Ultra-Kurzfrist bearish",
            "strength": bear,
            "tone": "red",
            "reason": reasons_bear[0] if reasons_bear else "Widerstand wird kurzfristig bestaetigt.",
            "bullets": list(dict.fromkeys(reasons_bear))[:4],
        })
    elif bull >= 34 and bull >= bear + 5:
        signal.update({
            "label": "Frühe bullische Reaktion",
            "strength": bull,
            "tone": "amber",
            "reason": reasons_bull[0] if reasons_bull else "Erste konstruktive Reaktion, Bestaetigung fehlt noch.",
            "bullets": list(dict.fromkeys(reasons_bull + reasons_watch))[:4],
        })
    elif bear >= 34 and bear >= bull + 5:
        signal.update({
            "label": "Frühe bearische Reaktion",
            "strength": bear,
            "tone": "amber",
            "reason": reasons_bear[0] if reasons_bear else "Erste Schwäche an einer relevanten Zone, Bestaetigung fehlt noch.",
            "bullets": list(dict.fromkeys(reasons_bear + reasons_watch))[:4],
        })
    elif max(bull, bear, watch) >= 22:
        signal.update({
            "label": "Zone unter Beobachtung",
            "strength": max(bull, bear, watch),
            "tone": "amber",
            "reason": (reasons_watch or reasons_bull or reasons_bear or ["Zone wird getestet, Bestaetigung fehlt noch."])[0],
            "bullets": list(dict.fromkeys((reasons_watch + reasons_bull + reasons_bear)))[:4],
        })

    if confirm >= 20:
        signal["confirmation"] = "vorhanden"
    elif confirm >= 10:
        signal["confirmation"] = "teilweise"
    else:
        signal["confirmation"] = "fehlt"

    return signal

def evaluate_chart_structure_bias(df, structures):
    """
    Leichte Zusatzlesart aus S/R-Zonen und Trendkanal.
    Gibt nur kleine Adjustments zur operativen Einordnung zurück.
    """
    result = {
        "bias": 0,
        "setup_bias": 0,
        "tradeability_bias": 0,
        "notes_pos": [],
        "notes_neg": [],
        "summary": [],
    }
    if df is None or df.empty or not structures:
        return result

    try:
        current_price = float(pd.to_numeric(df["Close"], errors="coerce").iloc[-1])
    except Exception:
        return result

    supports = structures.get("supports", []) or []
    resistances = structures.get("resistances", []) or []
    active_zones = structures.get("active_zones", []) or []
    channel = structures.get("channel")

    if supports:
        s1 = supports[0]
        s_mid = float(s1.get("mid", np.nan))
        if pd.notna(s_mid) and s_mid > 0:
            dist_pct = ((current_price / s_mid) - 1.0) * 100.0
            if 0 <= dist_pct <= 1.6:
                result["bias"] += 2
                result["setup_bias"] += 2
                result["notes_pos"].append("Kurs nahe Support S1")
                result["summary"].append(f"Support S1 stützt bei {s_mid:.2f}.")
            elif current_price < s_mid:
                result["bias"] -= 2
                result["tradeability_bias"] -= 2
                result["notes_neg"].append("Kurs unter Support S1")
                result["summary"].append(f"S1 bei {s_mid:.2f} wurde unterschritten.")

    if active_zones:
        z0 = active_zones[0]
        z_mid = float(z0.get("mid", np.nan))
        if pd.notna(z_mid):
            summaries.append(f"Kurs handelt aktuell in einer aktiven Zone um {z_mid:.2f}.")

    if resistances:
        r1 = resistances[0]
        r_mid = float(r1.get("mid", np.nan))
        if pd.notna(r_mid) and r_mid > 0:
            dist_pct = ((r_mid / current_price) - 1.0) * 100.0
            if 0 <= dist_pct <= 1.6:
                result["bias"] -= 1
                result["tradeability_bias"] -= 1
                result["notes_neg"].append("Kurs direkt an Widerstand R1")
                result["summary"].append(f"R1 liegt direkt bei {r_mid:.2f}.")
            elif current_price > r_mid:
                result["bias"] += 2
                result["setup_bias"] += 1
                result["notes_pos"].append("Ausbruch über Widerstand R1")
                result["summary"].append(f"R1 bei {r_mid:.2f} wurde überschritten.")

    if channel:
        try:
            idx_last = len(df) - 1
            slope = float(channel.get("slope", 0.0))
            lower = slope * idx_last + float(channel.get("lower_intercept", 0.0))
            upper = slope * idx_last + float(channel.get("upper_intercept", 0.0))
            if upper > lower:
                pos = (current_price - lower) / (upper - lower)
                label = str(channel.get("label", "Trendkanal")).strip()
                if str(channel.get("type")) == "uptrend":
                    if pos <= 0.30:
                        result["bias"] += 1
                        result["setup_bias"] += 1
                        result["notes_pos"].append("Im unteren Bereich eines Aufwaertskanals")
                    elif pos >= 0.82:
                        result["bias"] -= 1
                        result["tradeability_bias"] -= 1
                        result["notes_neg"].append("Im oberen Bereich eines Aufwaertskanals")
                elif str(channel.get("type")) == "downtrend":
                    result["bias"] -= 2
                    result["tradeability_bias"] -= 1
                    result["notes_neg"].append(label)
                result["summary"].append(f"Chart laeuft in {label.lower()}.")
        except Exception:
            pass

    result["bias"] = int(max(-4, min(4, result["bias"])))
    result["setup_bias"] = int(max(-3, min(3, result["setup_bias"])))
    result["tradeability_bias"] = int(max(-3, min(3, result["tradeability_bias"])))
    return result

def _trade_overlay_num_v193(value, default=None):
    """Robuste Zahlenerkennung fuer Trade-Setup-Overlay."""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float, np.integer, np.floating)):
            if pd.isna(value):
                return default
            return float(value)
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "n/a", "na", "-"}:
            return default
        text = text.replace("%", "").replace("EUR", "").replace("USD", "").replace("$", "")
        text = text.replace("–", "-").replace("—", "-").replace(",", ".")
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(m.group(0)) if m else default
    except Exception:
        return default

def _trade_overlay_zone_v193(value):
    """Extrahiert eine Preiszone aus Text wie '254.30 - 259.41 USD'."""
    try:
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "none", "n/a", "na", "-"}:
            return None, None
        text = text.replace("–", "-").replace("—", "-").replace(",", ".")
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        vals = [float(x) for x in nums if x not in {"", "-"}]
        vals = [x for x in vals if x > 0]
        if len(vals) >= 2:
            lo, hi = min(vals[0], vals[1]), max(vals[0], vals[1])
            return lo, hi
        if len(vals) == 1:
            return vals[0], vals[0]
    except Exception:
        pass
    return None, None

def _trade_overlay_wave_level_v193(text):
    """Nimmt aus Wave-Texten den relevantesten Preislevel."""
    try:
        raw = str(text or "").replace(",", ".")
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", raw) if float(x) > 0]
        if not nums:
            return None
        # Bei Texten wie '1.272-1.618 Extension ca. 320.77 - 325.05' sollen
        # die Extension-Verhaeltnisse ignoriert werden. Preislevels sind meist deutlich > 10.
        price_like = [x for x in nums if x > 10]
        return price_like[0] if price_like else nums[-1]
    except Exception:
        return None

def _trade_overlay_wave_zone_v193(text):
    try:
        raw = str(text or "").replace(",", ".")
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", raw) if float(x) > 10]
        if len(nums) >= 2:
            return min(nums[-2], nums[-1]), max(nums[-2], nums[-1])
        if len(nums) == 1:
            return nums[0], nums[0]
    except Exception:
        pass
    return None, None

def build_trade_setup_overlay_v193(result=None, wave_pkg=None, ccy=""):
    """Sammelt Entry, Stop/Invalidierung, Ziele und Wave-Levels fuer das Chart-Overlay.

    Ziel v19.4: Alles, was die Entscheidung beeinflusst, soll im Chart sichtbar sein.
    Die Funktion ist absichtlich defensiv, damit fehlende Felder den Chart nicht brechen.
    """
    r = result or {}
    wave = wave_pkg or r.get("wave_structure_pkg") or {}
    entry_text = r.get("suggested_entry_zone") or r.get("Entry-Zone") or r.get("Entry_Zone") or ""
    entry_low, entry_high = _trade_overlay_zone_v193(entry_text)
    stop = _trade_overlay_num_v193(r.get("stop_used") or r.get("stop") or r.get("Stop"), default=None)
    tp1 = _trade_overlay_num_v193(r.get("tp1") or r.get("TP1"), default=None)
    tp2 = _trade_overlay_num_v193(r.get("tp2") or r.get("TP2"), default=None)
    tp3 = _trade_overlay_num_v193(r.get("tp3") or r.get("TP3"), default=None)

    wave_trigger_text = wave.get("wave_trigger") or wave.get("wave_readable_action") or wave.get("trigger_label") or ""
    wave_invalid_text = wave.get("wave_invalidation") or wave.get("wave_readable_risk") or ""
    wave_target_text = wave.get("wave_target_zone") or wave.get("wave_readable_target") or ""
    wave_trigger = _trade_overlay_wave_level_v193(wave_trigger_text)
    wave_invalid = _trade_overlay_wave_level_v193(wave_invalid_text)
    wave_target_low, wave_target_high = _trade_overlay_wave_zone_v193(wave_target_text)

    price = _trade_overlay_num_v193(r.get("live_price") or r.get("analysis_price") or r.get("price"), default=None)
    crv = None
    try:
        # v21.1: CRV im Chart ebenfalls am Hauptziel orientieren. TP1 ist oft nur 1R.
        rr_target = tp2 if tp2 and price and tp2 > price else tp1
        if price and stop and rr_target and price > stop and rr_target > price:
            crv = (rr_target - price) / (price - stop)
    except Exception:
        crv = None

    return {
        "entry_text": entry_text,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "price": price,
        "crv": None if crv is None else round(float(crv), 2),
        "wave_trigger": wave_trigger,
        "wave_trigger_text": wave_trigger_text,
        "wave_invalid": wave_invalid,
        "wave_invalid_text": wave_invalid_text,
        "wave_target_low": wave_target_low,
        "wave_target_high": wave_target_high,
        "wave_target_text": wave_target_text,
        "ccy": ccy,
        "has_overlay": any(x is not None for x in [entry_low, stop, tp1, tp2, tp3, wave_trigger, wave_invalid, wave_target_low]),
    }

def _trade_overlay_xrange_v193(chart_df):
    """Chart-X-Spanne fuer echte, per Legende ausblendbare Overlay-Traces."""
    try:
        if chart_df is not None and len(chart_df.index) >= 2:
            return chart_df.index[0], chart_df.index[-1]
    except Exception:
        pass
    return 0, 1

def _trade_overlay_add_hline_v193(fig, chart_df, y, name, *, dash="dot", width=1, hover=None):
    """Fuegt horizontale Levels als echte Scatter-Traces hinzu.

    Plotly-Shapes aus add_hline lassen sich ueber die Legende nicht wirklich
    ein-/ausblenden. Darum nutzt v19.4 echte Linien-Traces; Legendenfarbe und
    Linie im Chart sind dadurch identisch und klickbar.
    """
    try:
        if y is None or not pd.notna(float(y)) or float(y) <= 0:
            return
        x0, x1 = _trade_overlay_xrange_v193(chart_df)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[float(y), float(y)],
                mode="lines",
                name=name,
                line=dict(width=width, dash=dash),
                hovertemplate=(hover or name) + ": %{y:.2f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    except Exception:
        pass

def _trade_overlay_add_zone_v193(fig, chart_df, y0, y1, name, *, opacity=0.12, hover=None):
    """Fuegt Preiszonen als ausblendbare, gefuellte Trace-Flaeche hinzu."""
    try:
        if y0 is None or y1 is None:
            return
        lo, hi = float(min(y0, y1)), float(max(y0, y1))
        if lo <= 0 or hi <= 0:
            return
        x0, x1 = _trade_overlay_xrange_v193(chart_df)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[lo, lo, hi, hi, lo],
                mode="lines",
                fill="toself",
                name=name,
                line=dict(width=1, dash="dot"),
                opacity=opacity,
                hovertemplate=(hover or name) + f": {lo:.2f} - {hi:.2f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    except Exception:
        pass

def _chart_current_price_v204(chart_df, trade_overlay_pkg=None):
    """Robuster aktueller/letzter Kurs fuer die Chart-Orientierung.

    Prioritaet:
    1) Live-/Analysepreis aus dem Trade-Overlay, falls vorhanden
    2) letzter Close im dargestellten Chart
    """
    try:
        val = (trade_overlay_pkg or {}).get("price") if isinstance(trade_overlay_pkg, dict) else None
        if val is not None and pd.notna(float(val)) and float(val) > 0:
            return float(val), "Aktuell"
    except Exception:
        pass
    try:
        if chart_df is not None and not chart_df.empty and "Close" in chart_df.columns:
            close = pd.to_numeric(chart_df["Close"], errors="coerce").dropna()
            if not close.empty and float(close.iloc[-1]) > 0:
                return float(close.iloc[-1]), "Letzter Schluss"
    except Exception:
        pass
    return None, "Aktuell"

def add_current_price_marker_to_plotly_v204(fig, chart_df, current_price=None, ccy="", label="Aktuell"):
    """Hebt den aktuellen Kurs im Chart immer sichtbar hervor.

    Der Kurs wird als echte Plotly-Spur gezeichnet, damit er in der Legende
    ein-/ausblendbar bleibt. Zusaetzlich markiert ein Punkt die letzte Kerze
    und eine rechte Annotation zeigt den genauen Betrag.
    """
    try:
        if current_price is None or not pd.notna(float(current_price)) or float(current_price) <= 0:
            return
        price = float(current_price)
        x0, x1 = _trade_overlay_xrange_v193(chart_df)
        suffix = f" {ccy}" if str(ccy or "").strip() else ""
        name = f"{label}: {price:.2f}{suffix}"
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[price, price],
                mode="lines",
                name=name,
                line=dict(width=2.5, dash="solid", color="rgba(2,132,199,0.95)"),
                hovertemplate=f"{label}: %{{y:.2f}}{suffix}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        try:
            last_x = chart_df.index[-1]
            fig.add_trace(
                go.Scatter(
                    x=[last_x],
                    y=[price],
                    mode="markers",
                    name="Kursmarker",
                    marker=dict(size=9, color="rgba(2,132,199,1)", symbol="circle"),
                    hovertemplate=f"{label}: %{{y:.2f}}{suffix}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=last_x,
                y=price,
                text=f"{label} {price:.2f}{suffix}",
                showarrow=True,
                arrowhead=2,
                ax=58,
                ay=-10,
                bgcolor="rgba(2,132,199,0.94)",
                bordercolor="rgba(226,232,240,0.90)",
                borderwidth=1,
                borderpad=5 if bool(globals().get("_CHART_LARGE_LABELS_V207", False)) else 4,
                font=dict(size=13 if bool(globals().get("_CHART_LARGE_LABELS_V207", False)) else 11, color="white"),
                row=1,
                col=1,
            )
            _chart_add_hover_point_v207(fig, x=last_x, y=price, text=f"{label} {price:.2f}{suffix}", kind="price", row=1, col=1)
        except Exception:
            pass
    except Exception:
        pass

def add_trade_setup_overlay_to_plotly_v193(fig, chart_df, overlay):
    """Zeichnet Entry-Zone, Stop/Invalidierung, Ziele und Wave-Level in den Kurschart."""
    try:
        if not overlay or not overlay.get("has_overlay"):
            return
        entry_low = overlay.get("entry_low")
        entry_high = overlay.get("entry_high")
        stop = overlay.get("stop")
        tp1 = overlay.get("tp1")
        tp2 = overlay.get("tp2")
        tp3 = overlay.get("tp3")
        wave_trigger = overlay.get("wave_trigger")
        wave_invalid = overlay.get("wave_invalid")
        wt_low = overlay.get("wave_target_low")
        wt_high = overlay.get("wave_target_high")
        ccy = str(overlay.get("ccy") or "").strip()

        if entry_low and entry_high:
            _trade_overlay_add_zone_v193(fig, chart_df, entry_low, entry_high, "Entry-Zone", opacity=0.16)
        if wt_low and wt_high:
            _trade_overlay_add_zone_v193(fig, chart_df, wt_low, wt_high, "Ziel bei Bestätigung", opacity=0.12)

        _trade_overlay_add_hline_v193(fig, chart_df, stop, "Stop / Invalidierung", dash="dash", width=2)
        _trade_overlay_add_hline_v193(fig, chart_df, tp1, "TP1", dash="dot", width=1)
        _trade_overlay_add_hline_v193(fig, chart_df, tp2, "TP2", dash="dot", width=1)
        _trade_overlay_add_hline_v193(fig, chart_df, tp3, "TP3", dash="dot", width=1)
        _trade_overlay_add_hline_v193(fig, chart_df, wave_trigger, "Wann aktiv?", dash="dashdot", width=2, hover="Wellenanalyse - wann aktiv")
        # Wellen-Invalidierung nur separat zeichnen, wenn sie nicht praktisch identisch mit Stop ist.
        try:
            if wave_invalid and (not stop or abs(float(wave_invalid) - float(stop)) / max(float(stop), 1.0) > 0.002):
                _trade_overlay_add_hline_v193(fig, chart_df, wave_invalid, "Wann hinfällig?", dash="dash", width=1, hover="Wellenanalyse - wann hinfällig")
        except Exception:
            _trade_overlay_add_hline_v193(fig, chart_df, wave_invalid, "Wann hinfällig?", dash="dash", width=1, hover="Wellenanalyse - wann hinfällig")

        # v21.1: CRV bleibt in der kompakten Zusammenfassung unter dem Chart.
        # Im Kursbereich selbst war die CRV-Box ein zusätzlicher rechter Label-Konflikt.
    except Exception:
        pass

def build_candlestick_chart(chart_df, ticker, ccy, show_sr=False, show_channel=False, structures=None, show_fib=False, fib_pkg=None, show_trade_overlay=False, trade_overlay_pkg=None, chart_view="Setup", large_labels=False):
    global _CHART_LARGE_LABELS_V207
    _CHART_LARGE_LABELS_V207 = bool(large_labels)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name=ticker
        ),
        row=1,
        col=1
    )

    _chart_view = str(chart_view or "Setup")
    _show_ma10 = _chart_view in {"Setup", "Vollanalyse"}
    _show_ma20 = True
    _show_ma50 = True
    _show_ma200 = _chart_view == "Vollanalyse"

    if _show_ma10 and "MA10" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["MA10"], mode="lines", name="MA10"),
            row=1,
            col=1
        )
    if _show_ma20 and "MA20" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["MA20"], mode="lines", name="MA20"),
            row=1,
            col=1
        )
    if _show_ma50 and "MA50" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["MA50"], mode="lines", name="MA50"),
            row=1,
            col=1
        )
    if _show_ma200 and "MA200" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["MA200"], mode="lines", name="MA200"),
            row=1,
            col=1
        )

    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"],
            name="Volumen"
        ),
        row=2,
        col=1
    )

    if show_sr or show_channel:
        try:
            structures = structures or build_chart_structures(chart_df)
            if show_sr:
                add_sr_zones_to_plotly(fig, chart_df, structures.get("supports", []), structures.get("resistances", []), structures.get("active_zones", []), show_labels=(_chart_view == "Vollanalyse"))
            if show_channel:
                add_trend_channel_to_plotly(fig, chart_df, structures.get("channel"))
        except Exception:
            pass

    if show_fib:
        try:
            add_fibonacci_levels_to_plotly_v1533(fig, chart_df, fib_pkg, show_labels=(_chart_view == "Vollanalyse"))
        except Exception:
            pass

    if show_trade_overlay:
        try:
            add_trade_setup_overlay_to_plotly_v193(fig, chart_df, trade_overlay_pkg)
        except Exception:
            pass

    # v21.1: Aktueller Kurs bleibt immer sichtbar, unabhaengig von Chart-Ansicht und Overlay-Schaltern.
    try:
        _current_price, _current_label = _chart_current_price_v204(chart_df, trade_overlay_pkg)
        add_current_price_marker_to_plotly_v204(fig, chart_df, _current_price, ccy, _current_label)
    except Exception:
        pass

    fig.update_layout(
        title="",
        xaxis_rangeslider_visible=False,
        height=620,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
            itemwidth=30,
        )
    )
    fig.update_yaxes(title_text=f"Kurs ({ccy})", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)
    return fig
