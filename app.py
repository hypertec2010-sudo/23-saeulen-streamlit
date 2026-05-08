import streamlit as st

if not st.user.is_logged_in:
    st.button("Mit Google anmelden", on_click=st.login, args=["google"])
    st.stop()

st.write(f"Willkommen, {st.user.name}")
st.button("Abmelden", on_click=st.logout)
