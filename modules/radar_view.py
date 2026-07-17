"""Kleine, stabile UI-Helfer fuer die Radar-Ansicht (v25.4)."""
from __future__ import annotations

STYLE_NOTES = {
    "Leader": "Bevorzugt bestätigte Stärke, Leadership und saubere Trend-Setups.",
    "Charttechnik": "Sucht gezielt nach charttechnischen Impulsen: Trigger-Nähe, MA10/20, Spezialmuster, Bases/VCP, Pullback/Reclaim, Volumen/Smart Money, Fib-/Strukturreaktion und Konfluenz.",
    "Turnaround": "Bevorzugt frühe Drehkandidaten, Rebounds und technische Erholungsfenster.",
    "Ausgewogen": "Mittelweg zwischen bestätigter Stärke und früheren Chancen.",
}

def render_style_info(st, style_name: str) -> None:
    st.info(
        f"Screening-Stil: {style_name}. "
        "Der Stil veraendert nur die Radar-Priorisierung und Reihenfolge der Kandidaten, "
        f"nicht die Einzelanalyse einer Aktie. {STYLE_NOTES.get(style_name, '')}"
    )

def radar_score_badge(value):
    try:
        num=float(str(value).replace('%','').replace(',','.').strip())
    except Exception:
        return str(value) if str(value).strip() else '-'
    if num>=75: return f"🟢 {int(round(num))}"
    if num>=55: return f"🟡 {int(round(num))}"
    return f"🔴 {int(round(num))}"

def radar_trigger_badge(value):
    raw=str(value).strip(); s=raw.lower()
    if s in {"aktiv","jetzt prüfbar"}: return f"🟢 {raw}"
    if s in {"nahe dran","fast prüfbar"}: return f"🟡 {raw}"
    if not raw or s in {"nan","none"}: return '-'
    return raw
