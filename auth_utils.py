import streamlit as st


def check_password(app_password=None):
    """
    OIDC-kompatibler Auth-Wrapper fuer alte check_password()-Aufrufe.

    Wenn noch irgendwo im Projekt `if not check_password(): st.stop()` steht,
    bleibt der Flow damit funktionsfaehig:
    - Bei bestehendem Google-Login -> True
    - Sonst wird der OIDC-Login-Button gezeigt und False geliefert
    """
    try:
        if bool(st.user.is_logged_in):
            return True
    except Exception:
        pass

    st.title("23 Saeulen Analyse")
    st.info("Bitte mit Google anmelden, um die App zu nutzen.")
    st.button("Mit Google anmelden", on_click=st.login, type="primary", key="oidc_login_from_compat_final")
    return False
