# -*- coding: utf-8 -*-
"""CHSM Streamlit entrypoint with native named navigation."""

import streamlit as st

from modules.app_shell import configure_app, render_navigation, require_access


configure_app()
require_access()
render_navigation()
