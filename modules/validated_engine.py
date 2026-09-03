from __future__ import annotations

from typing import Any

import pandas as pd

HORIZONS = (1, 3, 5, 10, 20)


def _blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in {"", "-", "nan", "none", "n/a", "na"}


def _num(value: Any, default=None):
    if value is None:
        return default
    try:
        text = str(value).replace("%", "").replace("/100", "").replace(",", ".").strip()
        return float(text)
    except Exception:
        return default


def _episode_entries(events: pd.DataFrame | None) -> pd.DataFrame:
    """Collapse repeated changes inside the same live-vs-shadow divergence episode."""
    if not isinstance(events, pd.DataFrame) or events.empty:
        return pd.DataFrame()
    df = events.copy()
    if "richtung" not in df.columns:
        return pd.DataFrame()
    df = df[df["richtung"].isin(["Aufwertung", "Abwertung", "Unverändert", "Unveraendert"])].copy()
    if df.empty:
        return df
    if "ticker" not in df.columns or "event_ts" not in df.columns:
        return df[df["richtung"].isin(["Aufwertung", "Abwertung"])].reset_index(drop=True)

    df["__dt"] = pd.to_datetime(df["event_ts"], errors="coerce", utc=True)
    df = df.sort_values(["ticker", "__dt", "event_ts"], kind="stable")
    keep = []
    active = {}
    for idx, row in df.iterrows():
        ticker = str(row.get("ticker") or "").strip().upper()
        direction = str(row.get("richtung") or "")
        if not ticker:
            continue
        if direction not in ("Aufwertung", "Abwertung"):
            active[ticker] = None
            continue
        if active.get(ticker) != direction:
            keep.append(idx)
            active[ticker] = direction
    if not keep:
        return pd.DataFrame(columns=[c for c in df.columns if c != "__dt"])
    return df.loc[keep].drop(columns=["__dt"], errors="ignore").reset_index(drop=True)


def _frame_for_horizon(episodes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    df = episodes.copy()
    ret_col = f"r{int(horizon)}"
    if ret_col not in df.columns:
        df[ret_col] = None
    df["raw_return"] = pd.to_numeric(df[ret_col], errors="coerce")
    df["direction_sign"] = df.get("richtung", pd.Series(index=df.index, dtype=object)).map(
        {"Aufwertung": 1.0, "Abwertung": -1.0}
    )
    df["shadow_edge"] = df["raw_return"] * df["direction_sign"]
    df["hit"] = df["shadow_edge"] > 0
    df["engine_num"] = pd.to_numeric(df.get("engine_score"), errors="coerce")
    df["guarded_num"] = pd.to_numeric(df.get("guarded_score"), errors="coerce")
    return df[df["raw_return"].notna() & df["direction_sign"].notna()].copy()


def _horizon_stats(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        work = _frame_for_horizon(episodes, horizon)
        up = work[work.get("richtung") == "Aufwertung"] if not work.empty else pd.DataFrame()
        down = work[work.get("richtung") == "Abwertung"] if not work.empty else pd.DataFrame()
        rows.append({
            "Horizont": f"{horizon}T",
            "Tage": horizon,
            "Episoden": int(len(work)),
            "Aufwertungen": int(len(up)),
            "Abwertungen": int(len(down)),
            "Trefferquote %": None if work.empty else round(float(work["hit"].mean() * 100.0), 1),
            "Ø Shadow-Edge %": None if work.empty else round(float(work["shadow_edge"].mean()), 3),
            "Median Edge %": None if work.empty else round(float(work["shadow_edge"].median()), 3),
            "Aufwertung Treffer %": None if up.empty else round(float(up["hit"].mean() * 100.0), 1),
            "Ø Return Aufwertung %": None if up.empty else round(float(up["raw_return"].mean()), 3),
            "Abwertung Treffer %": None if down.empty else round(float(down["hit"].mean() * 100.0), 1),
            "Ø Return Abwertung %": None if down.empty else round(float(down["raw_return"].mean()), 3),
        })
    return pd.DataFrame(rows)


def _select_primary_horizon(horizon_df: pd.DataFrame) -> int:
    if not isinstance(horizon_df, pd.DataFrame) or horizon_df.empty:
        return 5
    # Validation should prefer a meaningful holding horizon, not the noisiest 1T result.
    for minimum in (30, 20, 10, 5, 1):
        for horizon in (10, 20, 5, 3, 1):
            row = horizon_df[horizon_df["Tage"] == horizon]
            if not row.empty and int(row.iloc[0].get("Episoden") or 0) >= minimum:
                return int(horizon)
    return 5


def _coverage(episodes: pd.DataFrame, field: str) -> float:
    if episodes.empty or field not in episodes.columns:
        return 0.0
    values = episodes[field].tolist()
    if not values:
        return 0.0
    return float(sum(not _blank(v) for v in values) / len(values) * 100.0)


def _regime_table(work: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if work.empty or column not in work.columns:
        return pd.DataFrame()
    rows = []
    valid = work[~work[column].map(_blank)].copy()
    if valid.empty:
        return pd.DataFrame()
    for value, group in valid.groupby(column, dropna=False):
        n = len(group)
        rows.append({
            label: str(value),
            "Episoden": int(n),
            "Trefferquote %": round(float(group["hit"].mean() * 100.0), 1),
            "Ø Shadow-Edge %": round(float(group["shadow_edge"].mean()), 3),
            "Stabil": bool(n >= 5 and float(group["shadow_edge"].mean()) > 0 and float(group["hit"].mean()) >= 0.50),
        })
    return pd.DataFrame(rows).sort_values("Episoden", ascending=False).reset_index(drop=True)


def _status(pass_level: int) -> str:
    return {
        0: "🔴 Shadow only",
        1: "🟠 Daten sammeln",
        2: "🟡 Teilfreigabe-Kandidat",
        3: "🟢 Freigabereif",
        4: "🔵 Beratend aktiv",
    }.get(int(pass_level), "⚪ Unklar")


def build_cutover_report(
    events: pd.DataFrame | None,
    learning_summary: dict | None = None,
    exit_detail: pd.DataFrame | None = None,
    portfolio_settings: dict | None = None,
) -> dict:
    """Build the v30.0 evidence gate for a controlled engine cutover.

    The function is deliberately read-only. It never changes live thresholds,
    Shadow bands, positions or portfolio settings. A green result only means
    'eligible for a future controlled release', not automatic activation.
    """
    learning_summary = learning_summary or {}
    portfolio_settings = portfolio_settings or {}
    episodes = _episode_entries(events)
    horizon_df = _horizon_stats(episodes)
    primary = _select_primary_horizon(horizon_df)
    work = _frame_for_horizon(episodes, primary)

    n = int(len(work))
    up = work[work.get("richtung") == "Aufwertung"] if not work.empty else pd.DataFrame()
    down = work[work.get("richtung") == "Abwertung"] if not work.empty else pd.DataFrame()
    hit = None if work.empty else float(work["hit"].mean())
    edge = None if work.empty else float(work["shadow_edge"].mean())
    median = None if work.empty else float(work["shadow_edge"].median())
    up_hit = None if up.empty else float(up["hit"].mean())
    down_hit = None if down.empty else float(down["hit"].mean())

    positive_horizons = 0
    mature_horizons = 0
    for _, row in horizon_df.iterrows():
        hn = int(row.get("Episoden") or 0)
        hhit = _num(row.get("Trefferquote %"))
        hedge = _num(row.get("Ø Shadow-Edge %"))
        if hn >= 10:
            mature_horizons += 1
            if hhit is not None and hedge is not None and hhit >= 52.0 and hedge > 0:
                positive_horizons += 1

    cov_guard = _coverage(episodes, "guardrail")
    cov_rs = _coverage(episodes, "rs_dynamics")
    cov_market = _coverage(episodes, "market_regime")
    cov_vol = _coverage(episodes, "volatility_regime")
    context_cov = (cov_guard + cov_rs + cov_market + cov_vol) / 4.0

    market_table = _regime_table(work, "market_regime", "Marktregime")
    vol_table = _regime_table(work, "volatility_regime", "Volatilitätsregime")
    market_breadth = 0 if market_table.empty else int((pd.to_numeric(market_table["Episoden"], errors="coerce") >= 5).sum())
    vol_breadth = 0 if vol_table.empty else int((pd.to_numeric(vol_table["Episoden"], errors="coerce") >= 5).sum())

    guard = work[
        work["engine_num"].notna()
        & work["guarded_num"].notna()
        & (work["engine_num"] > work["guarded_num"])
    ].copy() if not work.empty else pd.DataFrame()
    guard_n = int(len(guard))
    guard_defensive = None if guard.empty else float((guard["raw_return"] <= 0).mean())

    closed_trades = int(_num(learning_summary.get("closed_trades"), 0) or 0)
    avg_r = _num(learning_summary.get("avg_r"))
    profit_factor = _num(learning_summary.get("profit_factor"))
    trade_context_cov = float(_num(learning_summary.get("context_coverage"), 0.0) or 0.0)

    exit_n = int(len(exit_detail)) if isinstance(exit_detail, pd.DataFrame) else 0
    exit_worsened = None
    if isinstance(exit_detail, pd.DataFrame) and not exit_detail.empty and "Verschlechtert danach" in exit_detail.columns:
        vals = exit_detail["Verschlechtert danach"].dropna()
        if len(vals):
            exit_worsened = float(pd.Series(vals).astype(bool).mean())

    account_size = _num(portfolio_settings.get("account_size"), 0.0) or 0.0
    base_currency = str(portfolio_settings.get("base_currency") or "").strip().upper()
    portfolio_configured = bool(account_size > 0 and base_currency)

    # Hard evidence gates. These are intentionally conservative because v30.0
    # is a release gate, not another score-optimisation layer.
    gate_defs = []

    def add_gate(name, passed, evidence, requirement, hard=True):
        gate_defs.append({
            "Gate": name,
            "Status": "✅ Erfüllt" if passed else "⏳ Offen",
            "Evidenz": evidence,
            "Mindestanforderung": requirement,
            "Blockiert Voll-Cutover": "Ja" if hard and not passed else "Nein",
            "passed": bool(passed),
            "hard": bool(hard),
        })

    add_gate("Stichprobe", n >= 40, f"{n} auswertbare {primary}T-Episoden", "≥ 40 Episoden", True)
    add_gate(
        "Gesamt-Edge",
        bool(hit is not None and edge is not None and median is not None and hit >= 0.56 and edge >= 0.40 and median > 0),
        "n/a" if hit is None else f"Treffer {hit*100:.1f}% · Ø Edge {edge:+.2f}% · Median {median:+.2f}%",
        "Treffer ≥ 56% · Ø Edge ≥ +0,40% · Median > 0",
        True,
    )
    add_gate(
        "Richtungsbalance",
        bool(len(up) >= 10 and len(down) >= 10 and up_hit is not None and down_hit is not None and up_hit >= 0.52 and down_hit >= 0.52),
        f"Aufwertung n={len(up)} / Treffer {'n/a' if up_hit is None else f'{up_hit*100:.1f}%'} · Abwertung n={len(down)} / Treffer {'n/a' if down_hit is None else f'{down_hit*100:.1f}%'}",
        "je Richtung ≥ 10 Episoden und ≥ 52% Treffer",
        True,
    )
    add_gate(
        "Horizont-Stabilität",
        bool(positive_horizons >= 2),
        f"{positive_horizons} von {mature_horizons} Horizonten mit n≥10 sind positiv",
        "mind. 2 Horizonte mit n≥10, Treffer ≥52% und positiver Edge",
        True,
    )
    add_gate(
        "Kontext-Datenabdeckung",
        bool(context_cov >= 70.0),
        f"Ø {context_cov:.0f}% · Guardrail {cov_guard:.0f}% · RS {cov_rs:.0f}% · Markt {cov_market:.0f}% · Vola {cov_vol:.0f}%",
        "Ø ≥ 70% über Guardrail / RS / Markt / Vola",
        True,
    )
    add_gate(
        "Guardrail-Nachweis",
        bool(guard_n >= 10 and guard_defensive is not None and guard_defensive >= 0.55),
        f"n={guard_n} · defensiv bestätigt {'n/a' if guard_defensive is None else f'{guard_defensive*100:.1f}%'}",
        "≥ 10 gebremste Events und ≥ 55% defensiv bestätigt",
        True,
    )
    add_gate(
        "Regime-Abdeckung",
        bool(market_breadth >= 2),
        f"{market_breadth} Marktregime mit jeweils n≥5; Vola-Regime: {vol_breadth}",
        "mind. 2 Marktregime mit jeweils n≥5",
        True,
    )
    add_gate(
        "Real-Trade-Unterstützung",
        bool(closed_trades >= 20 and avg_r is not None and avg_r > 0 and trade_context_cov >= 60.0),
        f"{closed_trades} Trades · Ø R {'n/a' if avg_r is None else f'{avg_r:+.2f}R'} · Entry-Kontext {trade_context_cov:.0f}%",
        "≥ 20 geschlossene Trades · Ø R > 0 · Kontext ≥ 60%",
        False,
    )
    add_gate(
        "Portfolio-Gate operativ",
        portfolio_configured,
        f"Basis {base_currency or 'n/a'} · Depotwert {account_size:,.0f}" if portfolio_configured else "Portfolio-Basis noch nicht vollständig gespeichert",
        "Depotwert > 0 und Basiswährung gesetzt",
        False,
    )

    hard = [g for g in gate_defs if g["hard"]]
    hard_passed = sum(1 for g in hard if g["passed"])
    hard_total = len(hard)

    # A transparent evidence score for dashboard orientation. Green still needs
    # every hard release gate; score alone can never unlock a cutover.
    sample_pts = min(20.0, n / 40.0 * 20.0)
    edge_pts = 0.0
    if hit is not None and edge is not None:
        edge_pts = max(0.0, min(25.0, (hit - 0.45) / 0.15 * 12.5 + max(0.0, edge) / 0.8 * 12.5))
    direction_pts = 0.0
    if len(up) and len(down):
        min_dir_n = min(len(up), len(down))
        min_dir_hit = min(up_hit or 0.0, down_hit or 0.0)
        direction_pts = min(15.0, min_dir_n / 10.0 * 7.5 + max(0.0, min_dir_hit - 0.45) / 0.12 * 7.5)
    horizon_pts = min(15.0, positive_horizons / 2.0 * 15.0)
    context_pts = min(10.0, context_cov / 70.0 * 10.0)
    guard_pts = 0.0 if guard_defensive is None else min(10.0, guard_n / 10.0 * 5.0 + max(0.0, guard_defensive - 0.45) / 0.15 * 5.0)
    regime_pts = min(5.0, market_breadth / 2.0 * 5.0)
    validation_score = round(max(0.0, min(100.0, sample_pts + edge_pts + direction_pts + horizon_pts + context_pts + guard_pts + regime_pts)))

    all_hard_pass = hard_passed == hard_total
    core_good = bool(n >= 20 and hit is not None and edge is not None and hit >= 0.54 and edge > 0)
    if all_hard_pass and validation_score >= 75:
        overall_level = 3
        verdict = "Kontrollierter Cutover-Kandidat"
        next_action = "Alle harten Evidenz-Gates sind erfüllt. v30.0 schaltet trotzdem nichts automatisch um; nächster Schritt wäre ein manuell begrenzter A/B-Cutover statt sofortiger Vollersatz der Live-Ampel."
    elif core_good and hard_passed >= max(3, hard_total - 2):
        overall_level = 2
        verdict = "Teilfreigabe noch nicht aktiv – Kandidat"
        next_action = "Kernsignal ist positiv, aber mindestens ein Release-Gate ist noch offen. Live bleibt Kontrollgruppe; nur weiter messen und gezielt die offenen Gates schließen."
    elif n >= 8 and edge is not None and edge > 0:
        overall_level = 1
        verdict = "Frühe Evidenz"
        next_action = "Shadow weiterlaufen lassen. Noch keine produktive Freigabe; Stichprobe, Richtungsbalance und Regime-Abdeckung ausbauen."
    else:
        overall_level = 0
        verdict = "Shadow / Datensammlung"
        next_action = "Keine Cutover-Freigabe. Erst belastbare positive Forward-Evidenz sammeln; Live-Ampel bleibt vollständig produktiv."

    # Component-level release matrix.
    components = []

    def component(name, level, evidence, action):
        components.append({"Baustein": name, "Status": _status(level), "Evidenz": evidence, "Freigabe": action})

    core_level = 3 if all_hard_pass else (2 if core_good else (1 if n >= 8 and edge is not None and edge > 0 else 0))
    component(
        "Guarded Engine Score",
        core_level,
        f"{n} {primary}T-Episoden · Treffer {'n/a' if hit is None else f'{hit*100:.1f}%'} · Edge {'n/a' if edge is None else f'{edge:+.2f}%'}",
        "Kontrollierter A/B-Cutover möglich" if core_level == 3 else "Shadow beibehalten",
    )

    up_level = 3 if len(up) >= 15 and up_hit is not None and up_hit >= 0.56 else (2 if len(up) >= 10 and up_hit is not None and up_hit >= 0.52 else (1 if len(up) >= 5 else 0))
    component("Shadow-Aufwertungen", up_level, f"n={len(up)} · Treffer {'n/a' if up_hit is None else f'{up_hit*100:.1f}%'}", "Teilkomponente prüfen" if up_level >= 2 else "Shadow beibehalten")

    down_level = 3 if len(down) >= 15 and down_hit is not None and down_hit >= 0.56 else (2 if len(down) >= 10 and down_hit is not None and down_hit >= 0.52 else (1 if len(down) >= 5 else 0))
    component("Shadow-Abwertungen", down_level, f"n={len(down)} · Treffer {'n/a' if down_hit is None else f'{down_hit*100:.1f}%'}", "Teilkomponente prüfen" if down_level >= 2 else "Shadow beibehalten")

    guard_level = 3 if guard_n >= 15 and guard_defensive is not None and guard_defensive >= 0.60 else (2 if guard_n >= 10 and guard_defensive is not None and guard_defensive >= 0.55 else (1 if guard_n >= 5 else 0))
    component("Engine Guardrails", guard_level, f"n={guard_n} · defensiv {'n/a' if guard_defensive is None else f'{guard_defensive*100:.1f}%'} bestätigt", "Guardrails unverändert produktiv lassen" if guard_level >= 2 else "Nicht lockern; Daten sammeln")

    rs_level = 3 if cov_rs >= 80 and n >= 30 else (2 if cov_rs >= 70 and n >= 15 else (1 if cov_rs >= 40 else 0))
    component("RS-Dynamik / Kontext", rs_level, f"Metadatenabdeckung {cov_rs:.0f}%", "Kontextgewicht im A/B-Test zulassen" if rs_level == 3 else "Beobachtend")

    regime_level = 3 if market_breadth >= 3 and n >= 30 else (2 if market_breadth >= 2 and n >= 15 else (1 if cov_market >= 40 else 0))
    component("Markt-/Vola-Regime", regime_level, f"Markt-Abdeckung {cov_market:.0f}% · {market_breadth} Regime mit n≥5", "Regime-Kontext im A/B-Test zulassen" if regime_level == 3 else "Beobachtend")

    if closed_trades >= 25 and avg_r is not None and avg_r > 0 and (profit_factor is None or profit_factor >= 1.2):
        trade_level = 3
    elif closed_trades >= 15 and avg_r is not None and avg_r > 0:
        trade_level = 2
    elif closed_trades >= 5:
        trade_level = 1
    else:
        trade_level = 0
    component("Trading Learning", trade_level, f"{closed_trades} geschlossene Trades · Ø R {'n/a' if avg_r is None else f'{avg_r:+.2f}R'}", "Als zusätzliche Freigabe-Evidenz nutzen" if trade_level >= 2 else "Weiter sammeln")

    exit_level = 3 if exit_n >= 20 and exit_worsened is not None and exit_worsened >= 0.60 else (2 if exit_n >= 10 and exit_worsened is not None and exit_worsened >= 0.55 else (1 if exit_n >= 5 else 0))
    component("Exit Engine 2.0", exit_level, f"{exit_n} sicher zugeordnete Trades · nach Warnung verschlechtert {'n/a' if exit_worsened is None else f'{exit_worsened*100:.1f}%'}", "Produktive Exit-Logik bestätigen" if exit_level >= 2 else "Beobachtend")

    component(
        "Portfolio-Risikogate",
        4 if portfolio_configured else 1,
        f"Basis {base_currency} · Depotwert {account_size:,.0f}" if portfolio_configured else "Portfolio-Basis nicht vollständig konfiguriert",
        "Beratend aktiv lassen; noch kein automatischer Order-/Ampel-Eingriff",
    )

    blockers = [g["Gate"] for g in hard if not g["passed"]]
    gate_df = pd.DataFrame([{k: v for k, v in g.items() if k not in {"passed", "hard"}} for g in gate_defs])

    overview = {
        "status": _status(overall_level),
        "verdict": verdict,
        "validation_score": validation_score,
        "primary_horizon": primary,
        "evaluable": n,
        "hit_rate": hit,
        "avg_edge": edge,
        "median_edge": median,
        "hard_passed": hard_passed,
        "hard_total": hard_total,
        "cutover_candidate": bool(all_hard_pass and validation_score >= 75),
        "productive_mode": "Live-Ampel / bestehende produktive Engine",
        "next_action": next_action,
        "blockers": blockers,
    }

    return {
        "overview": overview,
        "gates": gate_df,
        "components": pd.DataFrame(components),
        "horizons": horizon_df.drop(columns=["Tage"], errors="ignore"),
        "market_regimes": market_table,
        "volatility_regimes": vol_table,
    }
