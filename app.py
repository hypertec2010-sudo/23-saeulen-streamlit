# -*- coding: utf-8 -*-
"""v28.3.1 Streamlit entrypoint with stable native multipage navigation."""

import streamlit as st

from modules.app_shell import configure_app, render_navigation, require_access


configure_app()
require_access()
render_navigation()
