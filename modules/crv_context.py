"""CRV evidence and execution context, independent of technical release."""
import math


def crv_context(now=None, entry=None, *, now_eligible=False, entry_eligible=False, reason=""):
    def number(value, eligible):
        try:
            value = float(value)
            return value if eligible and math.isfinite(value) else None
        except (ValueError, TypeError):
            return None
    n, e = number(now, now_eligible), number(entry, entry_eligible)
    parts = []
    if n is not None: parts.append(f"jetzt {n:.2f}R")
    if e is not None: parts.append(f"Entry {e:.2f}R")
    if n is not None:
        label = "Eng" if n < 1.50 else "Ab 1,50"
    elif e is not None:
        label = "Jetzt offen"
    else:
        label = "Nicht bewertbar"
    text = ("CRV separat: " + " · ".join(parts) + ". " if parts else
            "CRV nicht bewertet: " + (str(reason) or "Belastbare technische Grundlage fehlt") + ". ")
    if n is not None and n < 1.50:
        text += "Enger Zielraum; Entry, Ziel und Risiko vor Umsetzung separat beurteilen. "
    if n is None and e is None:
        text += "Szenarien sind kein Freigabekriterium. "
    text += "CRV verändert weder technische Ampel noch Freigabe; 1,50 ist nur eine Vergleichsmarke."
    return {"label": label, "text": text, "now": n, "entry": e,
            "display": label + (" · " + " / ".join(parts) if parts else "")}
