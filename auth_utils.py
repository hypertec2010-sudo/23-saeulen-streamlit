import hmac
import streamlit as st


def check_password(app_password: str) -> bool:
    """
    Simple password gate for Streamlit that is robust against missing
    session_state keys after reloads / idle timeouts.
    """

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if "password" not in st.session_state:
        st.session_state["password"] = ""

    def password_entered():
        entered = st.session_state.get("password", "")
        if hmac.compare_digest(str(entered), str(app_password)):
            st.session_state["password_correct"] = True
            # Passwort aus dem State entfernen, damit es nicht unnötig erhalten bleibt
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
