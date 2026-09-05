# v30.4c - Short-Term Trader Import Compatibility Fix

v30.4c fixes the Streamlit startup crash introduced when `legacy_app.py` already expected the v30.4b `attach_scan_context()` API while the running Python process could still hold an older imported `modules.short_term_trader` instance.

## Fix
- The app no longer dereferences `attach_scan_context` unguarded during startup.
- If the already imported `short_term_trader` module exposes the complete v30.4b/v30.4c API, it is used normally.
- If the runtime still contains an older/stale module object, the app reloads **exactly the local `modules/short_term_trader.py` file** under a unique module name and uses that fresh implementation.
- If even that local reload cannot provide the helper, a fail-safe fallback returns an unchanged copy of the Atomic frame instead of crashing the entire Watchlists page.
- The fallback does **not** invent a Scan-Chop value and does not mutate the productive Atomic frame.

## Deployment hardening
- `modules/short_term_trader.py` is shipped again in this patch even though the v30.4b calibration logic itself is unchanged. This guarantees that `legacy_app.py` and the tactical module arrive together.
- The fix is specifically designed for Streamlit/runpy reruns where Python module caching can temporarily create a mixed old/new runtime state after a patch deployment.

## Unchanged
- v30.4b Harvest/Chop calibration and thresholds remain unchanged.
- Harvest green <60, yellow >=60, orange >=75.
- No change to TP1/TP2/TP3, Live/Shadow, Exit Engine, Position logic or provider behavior.
- No additional Yahoo/market-provider requests.
