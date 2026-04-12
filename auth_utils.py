import streamlit as st


def get_app_password():
    try:
        if "PASSWORD" in st.secrets:
            return st.secrets["PASSWORD"]
        if "SCREENER_APP_PASSWORD" in st.secrets:
            return st.secrets["SCREENER_APP_PASSWORD"]
    except Exception:
        return None
    return None


def check_password():
    app_password = get_app_password()

    if not app_password:
        st.error("Kein Passwort-Secret gefunden. Bitte in Streamlit Secrets PASSWORD oder SCREENER_APP_PASSWORD setzen.")
        return False

    def password_entered():
        if st.session_state["password"] == app_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Passwort eingeben:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Passwort eingeben:", type="password", on_change=password_entered, key="password")
        st.error("😕 Falsches Passwort")
        return False
    return True
