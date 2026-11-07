"""Display moving averages on the full daily basis, independent of chart zoom."""
import pandas as pd
from modules.trade_watch import number


def moving_average_rows(result):
    source = (result or {}).get("df")
    close = None
    if (isinstance(source, pd.DataFrame) and "Close" in source and not isinstance(source.columns, pd.MultiIndex)
            and isinstance(source.index, pd.DatetimeIndex) and not source.index.hasnans
            and source.index.is_monotonic_increasing and len(set(source.index.date)) == len(source)):
        close = pd.to_numeric(source.Close, errors="coerce")
    price = number((result or {}).get("price"))
    rows = []
    for period in (20, 50, 200):
        window = close.tail(period) if close is not None else None
        valid = window is not None and len(window) == period and window.map(lambda v:number(v) is not None and v>0).all()
        value = float(window.mean()) if valid else None
        distance = (price/value-1)*100 if value is not None and price is not None and price>0 else None
        fmt = lambda v: "Nicht verfügbar" if v is None else f"{v:.2f}".replace(".", ",")
        rows.append({"Indikator":f"MA{period} · {period} Handelstage", "Wert":fmt(value),
                     "Kursabstand":fmt(distance) + " %" if distance is not None else "Nicht verfügbar",
                     "Einordnung":("Kurs oberhalb" if distance>0 else "Kurs unterhalb" if distance<0 else "Kurs auf der Linie")
                         if distance is not None else f"Mindestens {period} gültige Tageskurse erforderlich"})
    return rows


def render_moving_averages(st, result):
    from modules.support_map import build_support_map, render_support_map
    render_support_map(st, build_support_map(result))
    st.markdown("**Gleitende Durchschnitte · Tagesbasis**")
    st.table(pd.DataFrame(moving_average_rows(result)))
    st.caption("Einfache Durchschnitte der vollständigen Analysehistorie, unabhängig vom sichtbaren Chartausschnitt. Der letzte Tagesbalken kann noch laufen. Fehlende Historie wird nicht durch kürzere Durchschnitte ersetzt.")
