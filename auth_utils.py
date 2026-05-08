import streamlit as st


def check_password(app_password=None):
    """
    OIDC-kompatibler Wrapper fuer alte check_password()-Aufrufe.
    Wenn irgendwo noch alter Code `if not check_password(): st.stop()` nutzt,
    bleibt der Flow funktional:
    - bei bestehendem Google-Login -> True
    - sonst wird nur der Google-Login gezeigt und False geliefert
    """
    try:
        if bool(st.user.is_logged_in):
            return True
    except Exception:
        pass

    st.title("23 Saeulen Analyse")
    st.info("Bitte mit Google anmelden, um die App zu nutzen.")
    st.button("Mit Google anmelden", on_click=st.login, type="primary", key="oidc_login_from_compat_complete")
    return False
