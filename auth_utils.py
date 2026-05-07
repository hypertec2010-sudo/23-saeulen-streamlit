import hmac
import os
import streamlit as st


def _get_secret_value(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


def _resolve_password(app_password=None):
    """
    Liest das Passwort robust aus verschiedenen moeglichen Quellen.
    Reihenfolge:
    1. explizit uebergebenes app_password
    2. Umgebungsvariablen
    3. Streamlit secrets (mehrere uebliche Key-Namen)
    """
    if app_password is not None and str(app_password) != "":
        return str(app_password)

    candidates = [
        os.getenv("APP_PASSWORD"),
        os.getenv("STREAMLIT_APP_PASSWORD"),
        os.getenv("PASSWORD"),
        os.getenv("APP_PW"),
        _get_secret_value("APP_PASSWORD"),
        _get_secret_value("app_password"),
        _get_secret_value("STREAMLIT_APP_PASSWORD"),
        _get_secret_value("PASSWORD"),
        _get_secret_value("password"),
        _get_secret_value("app_pw"),
    ]

    for cand in candidates:
        if cand is not None and str(cand).strip() != "":
            return str(cand)

    return ""


def check_password(app_password=None) -> bool:
    """
    Robuster Passwortschutz fuer Streamlit.

    Kompatibel mit:
    - check_password()
    - check_password(app_password)

    Und robust gegen fehlende session_state-Keys nach Reload / Timeout.
    """
    resolved_password = _resolve_password(app_password)

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if "password" not in st.session_state:
        st.session_state["password"] = ""

    def password_entered():
        entered = str(st.session_state.get("password", ""))

        if resolved_password and hmac.compare_digest(entered, str(resolved_password)):
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
