"""UI-Helfer fuer die Einzel-/Mehrfachanalyse (v25.4)."""
from __future__ import annotations

def render_analysis_mode_inputs(st):
    default_idx = 0 if st.session_state.analysis_mode == "Einzelanalyse" else 1
    mode = st.radio(
        "Was möchtest du machen?",
        ["Einzelanalyse", "Mehrere Aktien vergleichen"],
        index=default_idx, horizontal=True, key="analysis_mode_widget_main"
    )
    st.session_state.analysis_mode = mode
    single_input = ""; batch_input = ""
    if mode == "Einzelanalyse":
        single_input = st.text_input(
            "Aktie oder Firmenname", value=st.session_state.search_input,
            placeholder="z. B. AAPL, Apple, Siemens, BASF, GC=F, SI=F",
            key="search_input_widget_main"
        ).strip()
        st.session_state.search_input = single_input
        st.caption("Du kannst einen Ticker, Firmennamen oder Rohstoff-Future eingeben, z. B. GC=F für Gold oder SI=F für Silber.")
    else:
        batch_input = st.text_area(
            "Mehrere Ticker oder Firmennamen", value=st.session_state.batch_input,
            placeholder="Ein Wert pro Zeile oder durch Komma trennen, z. B.\nAAPL\nMicrosoft\nASML\nNVIDIA",
            height=120, key="batch_input_widget_main"
        ).strip()
        st.session_state.batch_input = batch_input
        st.caption("Ein Wert pro Zeile oder mit Komma trennen. Die App löst Firmennamen automatisch auf.")
    return mode, single_input, batch_input
