import streamlit as st


def show_sheet_result(ok, msg, prefix_fail="Google-Sheet-Logging fehlgeschlagen"):
    if ok:
        st.success(msg)
    else:
        if "Keine Trigger-Kandidaten" in str(msg):
            st.info(msg)
        else:
            st.error(f"{prefix_fail}: {msg}")
