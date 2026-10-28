"""Standalone page: opening it does not initialize the legacy analysis UI."""
import streamlit as st
from modules.trade_watch_overview import render_compact_monitor

render_compact_monitor(st)
