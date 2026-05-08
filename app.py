import streamlit as st

st.set_page_config(page_title="OIDC Test")

if not st.user.is_logged_in:
    st.write("Nicht eingeloggt")
    if st.button("Mit Google anmelden"):
        st.login()
    st.stop()

st.success("Login erfolgreich")
st.write(st.user)
if st.button("Abmelden"):
    st.logout()
