streamlit.errors.StreamlitDuplicateElementKey: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).

Traceback:
File "/mount/src/23-saeulen-streamlit/app.py", line 6060, in <module>
    selected_auto_slot = st.selectbox(
        "Auto-Run-Testslot",
    ...<2 lines>...
        key="selected_auto_run_slot_widget"
    )
File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/runtime/metrics_util.py", line 563, in wrapped_func
    result = non_optional_func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/elements/widgets/selectbox.py", line 554, in selectbox
    return self._selectbox(
           ~~~~~~~~~~~~~~~^
        label=label,
        ^^^^^^^^^^^^
    ...<15 lines>...
        ctx=ctx,
        ^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/elements/widgets/selectbox.py", line 636, in _selectbox
    element_id = compute_and_register_element_id(
        "selectbox",
    ...<13 lines>...
        width=width,
    )
File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/elements/lib/utils.py", line 265, in compute_and_register_element_id
    _register_element_id(ctx, element_type, element_id)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/elements/lib/utils.py", line 145, in _register_element_id
    raise StreamlitDuplicateElementKey(user_key)
