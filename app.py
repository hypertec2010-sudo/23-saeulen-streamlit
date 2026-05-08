import streamlit as st

if not st.user.is_logged_in:
    st.button("Mit Google anmelden", on_click=st.login)
    st.stop()

st.success("Login erfolgreich")
st.write(st.user)
st.button("Abmelden", on_click=st.logout)
