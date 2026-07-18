# v27.1 Decision Engine - Migration Step 1

This release adds a canonical, view-independent decision contract while keeping
all existing UI and legacy decision fields intact.

## New module

`modules/decision_engine.py`

Every successful `analyze_stock(...)` result now includes:

- `decision_engine`
- `decision_engine_version = "27.1"`

The contract contains `decision`, `label`, `confidence`, `traffic_light`,
`state`, `mode`, `entry`, `stop`, `target`, `reason`, `invalidation`, and
`source_action`.

## Safe migration strategy

The existing UI behavior is unchanged. Future steps can switch Radar, Live
Screener, Single Analysis and Position Monitor one at a time to
`result["decision_engine"]`.
