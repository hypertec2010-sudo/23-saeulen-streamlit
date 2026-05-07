import hmac
import os
import streamlit as st


def _resolve_password(app_password=None):
    if app_password is not None:
        return str(app_password)
    # kompatibel mit bestehenden Aufrufen ohne Argument
    return str(
        os.getenv("APP_PASSWORD")
        or os.getenv("STREAMLIT_APP_PASSWORD")
        or st.secrets.get("APP_PASSWORD", "")
        or st.secrets.get("app_password", "")
    )


def check_password(app_password=None) -> bool:
    """
    Robuster Passwortschutz für Streamlit.

    - funktioniert mit check_password()
    - funktioniert auch mit check_password(app_password)
    - schützt gegen fehlende session_state-Keys nach Reload / Timeout
    """
    resolved_password = _resolve_password(app_password)

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if "password" not in st.session_state:
        st.session_state["password"] = ""

    def password_entered():
        entered = st.session_state.get("password", "")
        if resolved_password and hmac.compare_digest(str(entered), str(resolved_password)):
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Passwort",
        type="password",
        key="password",
        on_change=password_entered,
    )

    if st.session_state.get("password") and not st.session_state.get("password_correct", False):
        st.error("Passwort falsch")

    return False
