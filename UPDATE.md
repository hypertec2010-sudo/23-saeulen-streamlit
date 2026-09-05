# v30.4b - Harvest / Chop Calibration

v30.4b recalibrates the observational Short-Term Trader / Profit Harvest layer after a real scan showed that all Harvest ampels could remain green even while the watchlist breadth was visibly mixed. The classic TP/Trend, Live, Shadow, Exit and Validated logic is unchanged.

## What changed
- New provider-free `Scan-Chop` context is calculated from the already completed Atomic full scan.
- It uses watchlist breadth, RS deterioration breadth, weak/neutral relative-strength breadth, elevated volatility and signal instability.
- A positive headline market regime together with weak scan breadth is treated as a mixed/choppy divergence instead of automatically reducing Harvest pressure.
- Row-level RS deterioration now uses the actual PP magnitude when available; large negative RS deltas matter more than small drift.
- High Live score can only reduce Chop when RS, technical pressure and stability also confirm a clean trend.
- ATR/volatility remains the target-distance basis; the existing dynamic Trader target and classic TP1/TP2/TP3 are not replaced.
- `Chop / Schwankung` is now visible directly next to the Harvest ampelfield in the Live Screener.
- The current full-scan Scan-Chop is shown once above the table.
- Open-position Harvest receives the same scan context through a separate tactical copy of the Atomic frame. Productive Live/Shadow/Portfolio frames remain untouched.

## Ampel thresholds remain unchanged
- Green below 60/100.
- Yellow from 60/100.
- Orange from 75/100.

The patch therefore changes the inputs/calibration, not the color thresholds.

## Regression checks
Using the pasted 49-row scan with only the visible fields reconstructed, the new cross-sectional context produced `Scan-Chop 55/100`. The old visible Harvest values were all below 60. With v30.4b, 27 rows stayed in the clear trend bucket, 15 moved to hybrid observation and 7 became yellow; clean leaders such as MU/NVDA remained green. No orange state is forced just to create variation.

A separate clean-trend synthetic scan stayed fully green, while an intentionally extreme choppy scan produced high Scan-Chop and yellow/orange tactical states. A profitable position with high velocity/exhaustion/giveback became an orange partial-profit check as intended. Missing-price/ATR guards still remain neutral.

## Provider / safety behavior
- No additional Yahoo or market-provider requests.
- No order, stop or position change.
- No change to productive Live/Shadow scores.
- No automatic cutover into the existing TP/Exit logic.
- Harvest remains observational and event-logged for later validation.
