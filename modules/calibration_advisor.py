"""v30.11 - Calibration Advisor (shadow-only, provider-free).

Consumes already-built outcome packages from the Action Queue and Harvest
validators. It produces guarded, human-readable calibration recommendations
without changing productive thresholds, scores, gates, stops, targets or orders.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

_MIN_GROUP_N = 15
_STRONG_GROUP_N = 30


def _num(value: Any, default=None):
    if value is None or isinstance(value, bool):
        return default
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _maturity(n: int) -> str:
    n = int(n or 0)
    if n < 10:
        return "Zu klein"
    if n < 25:
        return "Früh"
    if n < 50:
        return "Beobachtbar"
    return "Reifer"


def _pct_true(series: pd.Series, wanted: str = "Ja") -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float((series.astype(str) == wanted).mean() * 100.0)


def _pct_contains(series: pd.Series, needle: str) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(series.astype(str).str.contains(str(needle), case=False, na=False).mean() * 100.0)


def _median(df: pd.DataFrame, col: str):
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return None if vals.empty else float(vals.median())


def _recommendation_row(area: str, status: str, n_text: str, statement: str, advice: str, evidence: str) -> dict[str, Any]:
    return {
        "Bereich": area,
        "Status": status,
        "Stichprobe": n_text,
        "Aussage": statement,
        "Shadow-Empfehlung": advice,
        "Evidenz": evidence,
    }


def _queue_advice(action_package: dict[str, Any] | None):
    pkg = action_package or {}
    detail = pkg.get("detail")
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return [], pd.DataFrame(), {"queue_3t": 0, "queue_maturity": "Zu klein"}

    d3 = detail[detail.get("Horizont", pd.Series(dtype=str)).astype(str) == "3T"].copy()
    if d3.empty:
        return [], pd.DataFrame(), {"queue_3t": 0, "queue_maturity": "Zu klein"}

    ready = d3[d3["Priorität"].astype(str) == "🎯 Jetzt prüfen"].copy()
    watch = d3[d3["Priorität"].astype(str) == "👀 Beobachten"].copy()
    blocked = d3[d3["Priorität"].astype(str) == "⛔ Blockiert"].copy()
    rows: list[dict[str, Any]] = []

    if len(ready) >= _MIN_GROUP_N and len(watch) >= _MIN_GROUP_N:
        rmed = _median(ready, "Return %") or 0.0
        wmed = _median(watch, "Return %") or 0.0
        rpos = _pct_true(ready["Positiv"])
        wpos = _pct_true(watch["Positiv"])
        r2 = _pct_true(ready["+2% im Pfad"])
        w2 = _pct_true(watch["+2% im Pfad"])
        diff = rmed - wmed
        pdiff = rpos - wpos
        if diff >= 0.50 and pdiff >= 5.0:
            status = "✅ Halten"
            statement = "'Jetzt prüfen' trennt sich aktuell positiv von 'Beobachten'."
            advice = "Produktive Queue-Regeln unverändert lassen und weitere Outcomes sammeln."
        elif diff <= -0.50 and pdiff <= -5.0:
            status = "🟠 Shadow prüfen"
            statement = "'Jetzt prüfen' liegt aktuell hinter 'Beobachten'."
            advice = "Im Shadow prüfen, ob 'Jetzt prüfen' stärker an Trigger-Stabilität bzw. Decision-Confidence gebunden werden sollte; keine produktive Umstellung."
        else:
            status = "⚪ Weiter beobachten"
            statement = "Die 3T-Trennung zwischen 'Jetzt prüfen' und 'Beobachten' ist noch nicht klar."
            advice = "Keine Schwelle ändern; Stichprobe weiter aufbauen."
        rows.append(_recommendation_row(
            "Action Queue · Priorität", status, f"{len(ready)} vs. {len(watch)}",
            statement, advice,
            f"Median Return {rmed:+.2f}% vs. {wmed:+.2f}% · Positiv {rpos:.0f}% vs. {wpos:.0f}% · +2%-Pfad {r2:.0f}% vs. {w2:.0f}%",
        ))
    else:
        rows.append(_recommendation_row(
            "Action Queue · Priorität", "⏳ Stichprobe aufbauen", f"{len(ready)} vs. {len(watch)}",
            "Für einen belastbaren Vergleich fehlen noch genügend 3T-Fälle pro Kategorie.",
            f"Mindestens {_MIN_GROUP_N} auswertbare Fälle je Vergleichsgruppe abwarten.",
            "Keine Kalibrierung aus kleinen Stichproben.",
        ))

    if len(ready) >= _MIN_GROUP_N and len(blocked) >= _MIN_GROUP_N:
        rmed = _median(ready, "Return %") or 0.0
        bmed = _median(blocked, "Return %") or 0.0
        rdown = _pct_true(ready["-2% im Pfad"])
        bdown = _pct_true(blocked["-2% im Pfad"])
        if bdown >= rdown + 15.0 or bmed <= rmed - 0.50:
            status = "✅ Blockierung plausibel"
            statement = "Blockierte Werte zeigen aktuell ein defensiveres Folgeprofil als 'Jetzt prüfen'."
            advice = "Harte Gates produktiv unverändert lassen."
        elif bmed >= rmed + 0.50 and bdown <= rdown + 5.0:
            status = "🟡 Gate-Audit"
            statement = "Blockierte Werte entwickeln sich aktuell nicht schwächer als 'Jetzt prüfen'."
            advice = "Nur im Shadow untersuchen, welches konkrete Gate dafür verantwortlich ist. Gates nicht allein wegen späterer Kursanstiege lockern."
        else:
            status = "⚪ Weiter beobachten"
            statement = "Die Schutzwirkung der Blockierung ist noch nicht eindeutig getrennt."
            advice = "Keine Gate-Änderung; weitere 3T-Fälle sammeln."
        rows.append(_recommendation_row(
            "Action Queue · Blockierung", status, f"{len(blocked)} vs. {len(ready)}",
            statement, advice,
            f"Median Return blockiert {bmed:+.2f}% vs. jetzt {rmed:+.2f}% · -2%-Pfad {bdown:.0f}% vs. {rdown:.0f}%",
        ))

    high = d3[d3["Decision-Confidence"].astype(str) == "Hoch"].copy()
    lower = d3[d3["Decision-Confidence"].astype(str).isin(["Mittel", "Niedrig"])].copy()
    if len(high) >= _MIN_GROUP_N and len(lower) >= _MIN_GROUP_N:
        hmed = _median(high, "Return %") or 0.0
        lmed = _median(lower, "Return %") or 0.0
        hpos = _pct_true(high["Positiv"])
        lpos = _pct_true(lower["Positiv"])
        if hpos >= lpos + 10.0 and hmed >= lmed + 0.25:
            status = "✅ Confidence bestätigt"
            statement = "Hohe Decision-Confidence zeigt aktuell ein stabileres 3T-Profil."
            advice = "Confidence-Regeln unverändert lassen."
        elif hpos <= lpos - 10.0 and hmed <= lmed - 0.25:
            status = "🟠 Evidence-Regeln prüfen"
            statement = "Hohe Decision-Confidence liefert aktuell keine bessere Trennung."
            advice = "Im Shadow die Evidenz-/Freshness-Gewichtung der Confidence-Schicht prüfen; produktive Queue nicht verändern."
        else:
            status = "⚪ Weiter beobachten"
            statement = "Confidence-Stufen sind im 3T-Outcome noch nicht klar getrennt."
            advice = "Keine Änderung; Stichprobe weiter aufbauen."
        rows.append(_recommendation_row(
            "Decision Confidence", status, f"{len(high)} vs. {len(lower)}",
            statement, advice,
            f"Positiv Hoch {hpos:.0f}% vs. Mittel/Niedrig {lpos:.0f}% · Median {hmed:+.2f}% vs. {lmed:+.2f}%",
        ))
    else:
        rows.append(_recommendation_row(
            "Decision Confidence", "⏳ Stichprobe aufbauen", f"{len(high)} vs. {len(lower)}",
            "Noch zu wenige 3T-Fälle für einen belastbaren Confidence-Vergleich.",
            f"Mindestens {_MIN_GROUP_N} Fälle je Gruppe abwarten.",
            "Keine automatische Confidence-Neugewichtung.",
        ))

    diagnostics = pd.DataFrame(rows)
    return rows, diagnostics, {"queue_3t": len(d3), "queue_maturity": _maturity(len(d3))}


def _harvest_threshold_table(d3: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(d3, pd.DataFrame) or d3.empty or "Harvest Score" not in d3.columns:
        return pd.DataFrame()
    scores = pd.to_numeric(d3["Harvest Score"], errors="coerce")
    rows = []
    for threshold in (55, 60, 65, 70, 75, 80):
        hi = d3[scores >= threshold]
        lo = d3[scores < threshold]
        if hi.empty or lo.empty:
            continue
        hgb = _median(hi, "Giveback vom Scan-Peak %")
        lgb = _median(lo, "Giveback vom Scan-Peak %")
        rows.append({
            "Shadow-Schwelle": threshold,
            "Fälle ≥ Schwelle": len(hi),
            "Fälle < Schwelle": len(lo),
            "Median Giveback ≥ %": None if hgb is None else round(hgb, 2),
            "Median Giveback < %": None if lgb is None else round(lgb, 2),
            "Giveback-Differenz PP": None if hgb is None or lgb is None else round(hgb - lgb, 2),
            "Teilgewinn bestätigt ≥ %": round(_pct_contains(hi["Bewertung"], "Teilgewinn"), 1) if "Bewertung" in hi.columns else None,
            "Laufenlassen besser ≥ %": round(_pct_contains(hi["Bewertung"], "Laufenlassen"), 1) if "Bewertung" in hi.columns else None,
            "Stichprobe": "OK" if len(hi) >= _MIN_GROUP_N and len(lo) >= _MIN_GROUP_N else "Zu klein",
        })
    return pd.DataFrame(rows)


def _harvest_advice(harvest_package: dict[str, Any] | None):
    pkg = harvest_package or {}
    detail = pkg.get("detail")
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        return [], pd.DataFrame(), pd.DataFrame(), {"harvest_3t": 0, "harvest_maturity": "Zu klein"}
    d3 = detail[detail.get("Horizont", pd.Series(dtype=str)).astype(str) == "3T"].copy()
    if d3.empty:
        return [], pd.DataFrame(), pd.DataFrame(), {"harvest_3t": 0, "harvest_maturity": "Zu klein"}

    scores = pd.to_numeric(d3.get("Harvest Score"), errors="coerce")
    high = d3[scores >= 60].copy()
    low = d3[scores < 60].copy()
    rows: list[dict[str, Any]] = []
    if len(high) >= _MIN_GROUP_N and len(low) >= _MIN_GROUP_N:
        hgb = _median(high, "Giveback vom Scan-Peak %") or 0.0
        lgb = _median(low, "Giveback vom Scan-Peak %") or 0.0
        hp = _pct_contains(high["Bewertung"], "Teilgewinn") if "Bewertung" in high.columns else 0.0
        lp = _pct_contains(low["Bewertung"], "Teilgewinn") if "Bewertung" in low.columns else 0.0
        hr = _pct_contains(high["Bewertung"], "Laufenlassen") if "Bewertung" in high.columns else 0.0
        lr = _pct_contains(low["Bewertung"], "Laufenlassen") if "Bewertung" in low.columns else 0.0
        gb_diff = hgb - lgb
        if gb_diff >= 0.75 and hp >= lp + 10.0:
            status = "✅ Schwelle 60 gestützt"
            statement = "Harvest ≥60 zeigt aktuell mehr späteren Giveback und häufiger bestätigte Teilgewinn-Situationen."
            advice = "Produktive Harvest-Schwelle 60 vorerst halten."
        elif gb_diff <= 0.25 and hp <= lp + 5.0 and hr >= hp + 10.0:
            status = "🟠 Shadow-Schwelle prüfen"
            statement = "Harvest ≥60 trennt Giveback aktuell schwach und Laufenlassen ist innerhalb der Warnzone häufiger vorteilhaft."
            advice = "Nur im Shadow die Kandidaten 65/70/75 vergleichen; produktive Schwelle 60 nicht automatisch ändern."
        else:
            status = "⚪ Weiter beobachten"
            statement = "Harvest ≥60 zeigt bislang kein eindeutig starkes oder eindeutig falsches 3T-Trennbild."
            advice = "Schwelle 60 unverändert lassen und weitere Outcomes sammeln."
        rows.append(_recommendation_row(
            "Harvest · Warnbeginn 60", status, f"{len(high)} vs. {len(low)}",
            statement, advice,
            f"Median Giveback {hgb:.2f}% vs. {lgb:.2f}% · Teilgewinn bestätigt {hp:.0f}% vs. {lp:.0f}% · Laufenlassen besser {hr:.0f}% vs. {lr:.0f}%",
        ))
    else:
        rows.append(_recommendation_row(
            "Harvest · Warnbeginn 60", "⏳ Stichprobe aufbauen", f"{len(high)} vs. {len(low)}",
            "Noch zu wenige 3T-Fälle ober- und unterhalb der aktuellen Harvest-Schwelle.",
            f"Mindestens {_MIN_GROUP_N} Fälle je Gruppe abwarten; keine Schwellenänderung.",
            "Shadow-Kandidaten werden zwar angezeigt, aber bei kleiner Stichprobe nicht empfohlen.",
        ))

    chop = pd.to_numeric(d3.get("Chop Risk"), errors="coerce")
    chop_hi = d3[chop >= 60].copy()
    chop_lo = d3[chop < 60].copy()
    if len(chop_hi) >= _MIN_GROUP_N and len(chop_lo) >= _MIN_GROUP_N:
        hgb = _median(chop_hi, "Giveback vom Scan-Peak %") or 0.0
        lgb = _median(chop_lo, "Giveback vom Scan-Peak %") or 0.0
        diff = hgb - lgb
        if diff >= 0.75:
            status = "✅ Chop-Trennung plausibel"
            statement = "Chop ≥60 geht aktuell mit höherem späterem Giveback einher."
            advice = "Chop-Schwellen unverändert lassen."
        elif diff <= -0.50:
            status = "🟠 Chop-Kalibrierung prüfen"
            statement = "Hohes Chop-Risk zeigt aktuell nicht mehr, sondern weniger Giveback."
            advice = "Nur im Shadow prüfen, ob die Chop-Schwellen zu früh anschlagen; keine produktive Änderung."
        else:
            status = "⚪ Weiter beobachten"
            statement = "Die aktuelle Chop-Trennung ist im 3T-Giveback noch schwach."
            advice = "Keine Änderung; weitere Fälle sammeln."
        rows.append(_recommendation_row(
            "Chop · Schwelle 60", status, f"{len(chop_hi)} vs. {len(chop_lo)}",
            statement, advice,
            f"Median Giveback Chop ≥60 {hgb:.2f}% vs. <60 {lgb:.2f}% · Differenz {diff:+.2f} PP",
        ))

    threshold_table = _harvest_threshold_table(d3)
    return rows, pd.DataFrame(rows), threshold_table, {"harvest_3t": len(d3), "harvest_maturity": _maturity(len(d3))}


def build_calibration_package(action_package: dict[str, Any] | None = None, harvest_package: dict[str, Any] | None = None) -> dict[str, Any]:
    q_rows, q_df, q_meta = _queue_advice(action_package)
    h_rows, h_df, threshold_table, h_meta = _harvest_advice(harvest_package)
    all_rows = q_rows + h_rows
    overview = pd.DataFrame(all_rows)
    actionable = 0
    if not overview.empty and "Status" in overview.columns:
        actionable = int(overview["Status"].astype(str).str.contains("Shadow|Audit|prüfen", case=False, regex=True, na=False).sum())
    mature_n = int(q_meta.get("queue_3t") or 0) + int(h_meta.get("harvest_3t") or 0)
    overall_maturity = _maturity(mature_n)
    return {
        "summary": {
            **q_meta,
            **h_meta,
            "actionable_shadow_checks": actionable,
            "overall_maturity": overall_maturity,
            "min_group_n": _MIN_GROUP_N,
            "strong_group_n": _STRONG_GROUP_N,
        },
        "overview": overview,
        "queue_advice": q_df,
        "harvest_advice": h_df,
        "harvest_threshold_table": threshold_table,
        "policy": {
            "mode": "Shadow only",
            "auto_apply": False,
            "productive_changes": False,
        },
    }
