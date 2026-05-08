import streamlit as st

st.set_page_config(page_title="OIDC Minimal Test", page_icon="🔐", layout="centered")

st.title("OIDC Minimal Test")
st.caption("Minimaler Test fuer Streamlit Google-OIDC ohne alten Auth-Code.")

if not st.user.is_logged_in:
    st.info("Nicht eingeloggt.")
    st.write("Bitte nur in einem Tab testen und den Login nur einmal klicken.")
    st.button("Mit Google anmelden", on_click=st.login, type="primary", key="oidc_login_minimal")
    st.stop()

st.success("Login erfolgreich")
st.write("Angemeldet als:")
st.json(dict(st.user))

st.button("Abmelden", on_click=st.logout, key="oidc_logout_minimal")
