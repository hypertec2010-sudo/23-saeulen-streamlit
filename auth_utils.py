import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

import streamlit as st

try:
    import extra_streamlit_components as stx
except Exception:
    stx = None


def _get_secret_value(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


def _resolve_password(app_password=None):
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


def _resolve_auth_secret() -> str:
    candidates = [
        os.getenv("AUTH_SIGNING_SECRET"),
        os.getenv("COOKIE_SIGNING_SECRET"),
        _get_secret_value("AUTH_SIGNING_SECRET"),
        _get_secret_value("COOKIE_SIGNING_SECRET"),
        _get_secret_value("auth_signing_secret"),
    ]
    for cand in candidates:
        if cand is not None and str(cand).strip() != "":
            return str(cand)
    # fallback: same password as signer if nothing else exists
    return _resolve_password(None) or "streamlit-auth-fallback"


def _sign_payload(payload: dict, secret: str) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    token = base64.urlsafe_b64encode(raw).decode("utf-8") + "." + sig
    return token


def _verify_token(token: str, secret: str) -> bool:
    try:
        data_b64, sig = token.split(".", 1)
        raw = base64.urlsafe_b64decode(data_b64.encode("utf-8"))
        expected = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
        payload = json.loads(raw.decode("utf-8"))
        exp = int(payload.get("exp", 0))
        return exp > int(time.time())
    except Exception:
        return False


def _get_cookie_manager():
    if stx is None:
        return None
    try:
        return stx.CookieManager()
    except Exception:
        return None


def check_password(app_password=None, remember_hours: int = 12) -> bool:
    """
    Robuster Passwortschutz fuer Streamlit.

    Features:
    - kompatibel mit check_password() und check_password(app_password)
    - robust gegen fehlende session_state-Keys
    - Remember-Login via Cookie, falls extra_streamlit_components verfuegbar ist
    - faellt sauber auf session_state-only zurueck, wenn Cookie-Manager fehlt
    """
    resolved_password = _resolve_password(app_password)
    signing_secret = _resolve_auth_secret()

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if "password" not in st.session_state:
        st.session_state["password"] = ""
    if "remember_login" not in st.session_state:
        st.session_state["remember_login"] = True

    cm = _get_cookie_manager()

    # 1) Cookie pruefen, bevor Passwort erneut abgefragt wird
    if not st.session_state.get("password_correct", False) and cm is not None:
        try:
            token = cm.get("app_auth_token")
            if token and _verify_token(token, signing_secret):
                st.session_state["password_correct"] = True
                return True
        except Exception:
            pass

    def password_entered():
        entered = str(st.session_state.get("password", ""))
        if resolved_password and hmac.compare_digest(entered, str(resolved_password)):
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)

            if st.session_state.get("remember_login", True) and cm is not None:
                try:
                    payload = {
                        "exp": int(time.time()) + int(remember_hours * 3600),
                        "ok": True,
                    }
                    token = _sign_payload(payload, signing_secret)
                    cm.set("app_auth_token", token, expires_at=None, key="set_auth_cookie")
                except Exception:
                    pass
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
    st.checkbox("Login fuer einige Stunden merken", key="remember_login")

    if st.session_state.get("password") and not st.session_state.get("password_correct", False):
        st.error("Passwort falsch")

    return False
