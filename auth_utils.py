import streamlit as st


def check_password(app_password=None):
    """
    OIDC-kompatibler Auth-Wrapper fuer alte check_password()-Aufrufe.

    Verhalten:
    - Wenn bereits per Streamlit OIDC eingeloggt: True
    - Wenn nicht eingeloggt: zeigt Google-Login-Button und liefert False

    Damit funktionieren alte Stellen wie:
        if not check_password():
            st.stop()
    weiter, ohne eine alte Passwortmaske zu rendern und ohne schwarzen Bildschirm.
    """
    try:
        if bool(st.user.is_logged_in):
            return True
    except Exception:
        pass

    st.markdown("## Anmeldung erforderlich")
    st.write("Bitte melde dich mit Google an, um die App zu nutzen.")
    st.button("Mit Google anmelden", on_click=st.login, type="primary", key="oidc_login_from_compat")
    return False
