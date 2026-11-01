"""Explainable daily-plan orientation, never a forecast or intraday approval."""
from collections.abc import Mapping

import pandas as pd

POLICY = "radar_style_v30.21cd"


def daily_evidence(result, *, asof):
    from modules.candidate_radar import number, parse_time
    empty = {"policy": POLICY, "basis": "Nicht nachgewiesen", "atr": None,
             "basis_date": None, "setup": str(result.get("setup_type") or "Nicht benannt")}
    frame = result.get("df")
    stamp = parse_time(asof)
    if (stamp is None or not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex)
            or isinstance(frame.columns, pd.MultiIndex) or frame.index.hasnans
            or not frame.index.is_monotonic_increasing or frame.index.has_duplicates
            or not {"High", "Low", "Close"}.issubset(frame.columns)):
        return empty
    dates = frame.index.date
    if len(set(dates)) != len(dates) or any(d > stamp.date() for d in dates):
        return empty
    # Conservatively exclude today's possibly running daily bar for all markets.
    daily = frame.loc[[d < stamp.date() for d in dates], ["High", "Low", "Close"]].tail(20)
    if len(daily) < 20 or any((b-a).days > 5 for a,b in zip(daily.index.date[:-1],daily.index.date[1:])):
        return empty
    daily = daily.apply(pd.to_numeric, errors="coerce")
    valid = daily.apply(lambda c: c.map(lambda v: number(v) is not None and v > 0)).all(axis=1)
    valid &= (daily.High >= daily.Close) & (daily.Low <= daily.Close) & (daily.High >= daily.Low)
    if not valid.all():
        return empty
    previous = daily.Close.shift()
    atr = float(pd.concat([daily.High-daily.Low, (daily.High-previous).abs(),
                          (daily.Low-previous).abs()],axis=1).max(axis=1).tail(14).mean())
    if atr <= 0:
        return empty
    return {**empty, "basis": "Tageschart", "basis_date": daily.index[-1].date().isoformat(),
            "basis_quote_date": parse_time(daily.index[-1]).date().isoformat(), "atr": atr}


def describe(row):
    from modules import candidate_radar as radar
    evidence = row.get("trade_style_evidence") or {}
    if not isinstance(evidence, Mapping):
        evidence = {}
    readiness = {
        radar.READY: "Frische Einstiegsprüfung nötig",
        radar.NEAR: "Bestätigung abwarten", radar.WATCH: "Beobachtung",
        radar.NO_PLAN: "Kein belastbarer Trade-Plan", radar.BLOCKED: "Gesperrt",
        radar.HISTORY: "Historisch · neu prüfen",
    }.get(row.get("status"), "Einstiegsreife offen")
    out = dict(style="Zeithorizont offen", horizon="Nicht belastbar ableitbar", readiness=readiness,
               reason="Tagesbasis oder Volatilitätsbeleg fehlt; neu scannen.", target_atr=None, stop_atr=None)
    if row.get("valid_setup") is not True:
        return {**out, "style": "Beobachtung", "reason": "Zentrales Setup noch nicht valide; kein Handelsstil freigegeben."}
    atr = radar.number(evidence.get("atr"))
    if evidence.get("policy") != POLICY or evidence.get("basis") != "Tageschart" or atr is None or atr <= 0:
        return out
    basis_date = radar.parse_time(evidence.get("basis_quote_date") or evidence.get("basis_date"))
    quote_date = radar.parse_time(row.get("quote_date"))
    if basis_date is None or quote_date is None or not 0 <= (quote_date.date()-basis_date.date()).days <= 5:
        return {**out, "reason": "Volatilitätsbasis und Analysepreis zeitlich nicht ausreichend vergleichbar."}
    source = str((row.get("target_provenance") or {}).get("label") or "")
    technical_sources = {"Technisches Ziel", "Setup-Ziel", "52W-Hoch", "Technisches Setup-Ziel",
                         "Strukturelles Setup-Ziel", "Setup-Basisziel", "Technisches Sekundärziel"}
    known_chart_source = source.startswith(("CHSM-Widerstandszone (", "Breakout-Projektion aus 20T-Range ("))
    if source not in technical_sources and not known_chart_source:
        return {**out, "reason": "Technische Zielherkunft nicht eindeutig; Analysten-/R-Ziele bestimmen keinen kurzfristigen Stil."}
    price, stop, target = (radar.number(row.get(k)) for k in ("price", "stop", "target"))
    if any(v is None for v in (price,stop,target)) or not 0 < stop < price < target:
        return {**out, "reason": "Ziel und Invalidierung liegen nicht plausibel um den Analysepreis."}
    target_atr, stop_atr = (target-price)/atr, (price-stop)/atr
    out.update(target_atr=target_atr, stop_atr=stop_atr)
    reason = f"Tagesbasis: Ziel {target_atr:.1f} ATR und Analyse-Stop {stop_atr:.1f} ATR vom Analysepreis entfernt."
    if target_atr <= 2 and stop_atr <= 1.5:
        return {**out, "style": "Kurzfristiger Trade", "horizon": "Wenige Tage · Planungsorientierung", "reason": reason}
    if target_atr <= 6 and stop_atr <= 3:
        return {**out, "style": "Swing", "horizon": "Mehrere Tage bis Wochen · Planungsorientierung", "reason": reason}
    return {**out, "reason": reason + " Weiter Bewegungsspielraum; keine belastbare Zeitzuordnung."}


def render_style(st, row):
    info = describe(row)
    st.write("Handelsstil: **" + info["style"] + "** · " + info["horizon"])
    st.write("Einstiegsreife: **" + info["readiness"] + "**")
    with st.expander("Begründung zum Handelsstil", expanded=False):
        st.write(info["reason"])
        st.caption("Startregeln: kurzfristig bei Zielabstand ≤2 ATR und Stopabstand ≤1,5 ATR; Swing bei ≤6 ATR und ≤3 ATR. Sonst bleibt der Zeithorizont offen. Abstände beziehen sich auf den Analysepreis und Analyse-Stop, nicht auf einen bestätigten Ausführungskurs oder separate Wächter-Regeln. Schwellen sind noch nicht empirisch validiert.")
        st.caption("Grobe Planungsheuristik, keine Haltedauerprognose oder Kaufempfehlung. ATR misst Schwankungsbreite und sagt keine Zielerreichungszeit voraus. Tageschart und 5m-Einstiegsbestätigung begründen kein Intraday-Setup. Intraday ist hier nicht geprüft. Scores, Gates, Entry, Stop und Ziel bleiben unverändert.")
    return info
