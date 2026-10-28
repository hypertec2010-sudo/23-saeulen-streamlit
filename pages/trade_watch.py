"""Standalone page: opening it does not initialize the legacy analysis UI."""
import streamlit as st
from modules.trade_watch_ui import render_monitor

render_monitor(st)
