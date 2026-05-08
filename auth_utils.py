import streamlit as st


def check_password(app_password=None):
    """
    OIDC-kompatibler Kompatibilitaets-Stub.

    Alte Aufrufe wie:
        if not check_password(): st.stop()

    bleiben damit funktionsfaehig, ohne eine alte Passwortmaske zu rendern.
    Die eigentliche Anmeldung erfolgt ueber Streamlit OIDC in app.py via st.login().
    """
    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False
