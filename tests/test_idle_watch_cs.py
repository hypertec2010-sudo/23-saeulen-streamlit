from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from modules.provider_manager import MarketDataProvider


def test_soft_info_failure_never_expands_lazy_fastinfo_or_retries_alias(monkeypatch):
    calls=[]
    class Raw:
        def get_info(self):calls.append('get_info');raise RuntimeError('401 Unauthorized')
        @property
        def info(self):calls.append('info');raise AssertionError('Do not retry alias')
        @property
        def fast_info(self):calls.append('fast_info');raise AssertionError('No raw lazy histories')
    p=MarketDataProvider(primary=SimpleNamespace(name='offline',ticker=lambda symbol:Raw()))
    monkeypatch.setattr(p,'resolve_symbol',lambda s:s)
    monkeypatch.setattr(p,'_pace_request',lambda **k:p.ensure_available())
    ticker,info=p.get_info_bundle('HALO')
    assert calls==['get_info']
    assert set(info)=={'_chsm_info_fetched_at'}
    assert not p.quarantine_state().active


def script():
    return '''import streamlit as st
from modules.pattern_watch_ui import render_optional_watcher
render_optional_watcher(st,manager=st.session_state.manager,provider_factory=st.session_state.factory)
'''


def test_idle_watcher_ignores_old_auto_state_no_storage_provider_or_fragment(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from modules import pattern_watch_ui as ui
    mock=Mock(side_effect=AssertionError('Idle panel must not render'))
    monkeypatch.setattr(ui,'render_watcher',mock)
    app=AppTest.from_string(script())
    forbidden=Mock(side_effect=AssertionError('No provider or storage access'))
    app.session_state['manager']=SimpleNamespace(load_namespace=forbidden)
    app.session_state['factory']=forbidden
    app.session_state['pattern_watch_auto']=True
    app.run(timeout=20)
    assert not app.exception and app.toggle(key='pattern_watch_enabled_cs').value is False
    assert not app.session_state['pattern_watch_auto']
    mock.assert_not_called();forbidden.assert_not_called()
    app.run();assert not app.exception;mock.assert_not_called()


def test_explicit_enable_resets_auto_and_disable_removes_watcher(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from modules import pattern_watch_ui as ui
    calls=[]
    def fake(st,**kwargs):calls.append(bool(st.session_state['pattern_watch_auto']))
    monkeypatch.setattr(ui,'render_watcher',fake)
    app=AppTest.from_string(script())
    app.session_state['manager']=None;app.session_state['factory']=None
    app.session_state['pattern_watch_auto']=True
    app.run(timeout=20)
    app.toggle(key='pattern_watch_enabled_cs').set_value(True).run()
    assert calls==[False] and not app.exception
    app.session_state['pattern_watch_auto']=True
    app.toggle(key='pattern_watch_enabled_cs').set_value(False).run()
    assert calls==[False] and not app.session_state['pattern_watch_auto']
    app.toggle(key='pattern_watch_enabled_cs').set_value(True).run()
    assert calls==[False,False]


def test_queued_fragment_cannot_poll_after_gate_is_off():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('''import streamlit as st
from modules.pattern_watch_ui import render_watcher
class Store:
    user_id='idle-test'
    def begin_run(self):raise AssertionError('Stopped fragment must not load or poll')
st.session_state['gate']=False
st.session_state['pattern_watch_auto']=True
render_watcher(st,manager=Store(),provider_factory=lambda:1,enabled_key='gate')
''').run(timeout=20)
    assert not app.exception and not app.button
