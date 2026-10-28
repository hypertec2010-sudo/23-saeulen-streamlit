"""Layout contract and action delegation, without changing watcher decisions."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from modules import trade_watch as watch
from modules import trade_watch_overview as view
from modules.storage.local_backend import LocalJsonBackend
from modules.storage.manager import StorageManager


def frame():
    return pd.DataFrame([dict(Ticker="HALO", Status="Trigger bestätigt · gespeichert", Kurs=100.5,
        Entry=100.125, Invalidierung=95.25, Ziel=110., Einheit="USD",
        **{"Entry maximal":101.125, "Daten":"Kursdaten veraltet · Abruffehler", "Chartsetup":"Konsolidierung",
           "1m-Kerzenstart":"28.10. 15:00:00", "Signal gespeichert":"28.10. 15:00:10"})])


def test_primary_columns_and_entry_ceiling_are_visible_without_mutating_prices():
    original=frame();before=original.copy(deep=True)
    shown=view.compact_frame(original)
    assert list(shown.columns)==view.MAIN_COLUMNS
    assert shown.iloc[0]["Entry-Bereich"]=="100,125 – 101,125"
    assert shown.iloc[0]["Ziel"]==110. and shown.iloc[0]["Invalidierung"]==95.25
    pd.testing.assert_frame_equal(original,before)


@pytest.mark.parametrize("value,expected",[(.00123,"0,00123"),(float("nan"),"–"),(None,"–")])
def test_small_or_missing_levels_are_not_misrepresented(value,expected):
    assert view._level(value)==expected


def test_tables_without_watcher_fields_are_not_reformatted():
    assert view.compact_frame(pd.DataFrame([{"Ereignis":"trigger","Zeit":"now"}])) is None
    assert view.compact_frame(None) is None


def test_hidden_times_do_not_hide_stale_error_or_initial_data_states():
    original=frame();other=original.copy();other["Ticker"]="HPE";other["Daten"]="Noch nicht geprüft"
    healthy=original.copy();healthy["Ticker"]="SAP.DE";healthy["Daten"]="Aktuelle 1m-Daten · Tickzeit/Verzögerung nicht garantiert"
    alerts,pending=view.data_notes(pd.concat([original,other,healthy]))
    assert len(alerts)==1 and "HALO" in alerts[0] and "veraltet" in alerts[0]
    assert len(pending)==1 and "HPE" in pending[0]


def test_other_streamlit_calls_and_their_return_values_are_forwarded():
    called=[]
    actual=SimpleNamespace(button=lambda *a,**kw: called.append((a,kw)) or True)
    proxy=view.CompactWatcherView(actual)
    assert proxy.button("Save",key="unchanged") is True
    assert called==[(("Save",),{"key":"unchanged"})]


def test_real_watcher_layout_keeps_details_and_action_selection(tmp_path):
    from streamlit.testing.v1 import AppTest
    now=watch.utc()
    manager=StorageManager(user_id="layout",local_backend=LocalJsonBackend(tmp_path))
    ideas=[]
    for ticker in ["HALO","HPE"]:
        # Read an existing record rather than bypassing the current detector's
        # creation gate. Old records remain viewable/archivable in bz as well.
        i=dict(id="layout-"+ticker,ticker=ticker,entry=100.,invalidation=95.,max_entry=101.,
            target=110.,mode="close5",market="US",currency="USD",state="waiting",paused=False,archived=False,
            created_at=now.isoformat(),armed_at=now.isoformat(),expires_at=(now+timedelta(days=1)).isoformat(),
            price=100.5,quote_at=(now-timedelta(minutes=10)).isoformat(),
            health="Abruffehler: Provider nicht erreichbar",last_attempt_at=now.isoformat(),
            last_fetch_at="",last_bar_end="",triggered_at="",events=[],source={},note="",gap=False)
        ideas.append(i)
    watch.save_ideas(manager,ideas)
    script=f'''
import streamlit as st
from modules.storage.local_backend import LocalJsonBackend
from modules.storage.manager import StorageManager
from modules.trade_watch_ui import render_monitor
from modules.trade_watch_overview import render_compact_monitor
m=StorageManager(user_id='layout',local_backend=LocalJsonBackend({str(tmp_path)!r}))
def forbidden(): raise AssertionError('Layout must not request prices')
render_compact_monitor(st,renderer=lambda page: render_monitor(page,manager=m,provider_factory=forbidden))
'''
    app=AppTest.from_string(script).run(timeout=20)
    assert not app.exception and not app.error
    assert list(app.dataframe[0].value.columns)==view.MAIN_COLUMNS
    assert "Provider-Abruf" in app.dataframe[1].value.columns
    assert app.warning and any("veraltet" in x.value for x in app.warning)
    assert any(e.label=="Datenzeiten & technische Details" for e in app.expander)
    app.selectbox(key="watch_detail").select(ideas[1]["id"]).run()
    app.button(key="watch_archive").click().run()
    restored=watch.load_ideas(manager)
    assert not app.exception and restored[1]["archived"] and not restored[0]["archived"]
    assert restored[0]["entry"]==100. and restored[0]["target"]==110.


@pytest.mark.parametrize("policy",[None,"chart_watch_v30.21by"])
def test_page_requires_bz_instead_of_restoring_the_old_free_form(policy,monkeypatch):
    from streamlit.testing.v1 import AppTest
    original=view.importlib.import_module
    def selected(name,*args,**kwargs):
        if name=="modules.watch_setups":
            if policy is None: raise ModuleNotFoundError(name)
            return SimpleNamespace(POLICY=policy)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(view.importlib,"import_module",selected)
    page=Path(__file__).resolve().parents[1]/"pages/trade_watch.py"
    app=AppTest.from_file(str(page)).run(timeout=20)
    assert not app.exception and app.error and not app.number_input


def test_fallback_navigation_uses_the_same_compact_page(monkeypatch):
    from modules import app_shell
    called=[]
    fake=SimpleNamespace(sidebar=SimpleNamespace(radio=lambda *a,**kw:"Trade-Wächter"))
    monkeypatch.setattr(app_shell,"st",fake)
    monkeypatch.setattr(view,"render_compact_monitor",lambda st:called.append(st))
    app_shell.render_navigation()
    assert called==[fake]
