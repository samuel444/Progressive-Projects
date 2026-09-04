"""Edit SETTINGS here, then run this script. CLI path/log options override these values.
None retains optional defaults; required cache dates must be set here. Packages never prompt.
"""

from equity_selector.cli import run_stage

SETTINGS = {
    "DATA_DIR": None,
    "LOG_LEVEL": "INFO",
    "STOCK_TYPE": "Intraday Medium Liquidity 30",
    "MAX_PERIODS": 60,
    "FEATURE_LOOKBACK_OVERRIDES": {},
    "TARGET_HORIZON_OVERRIDES": {},
    "TARGET_LOOKBACK_OVERRIDES": {},
}

# Optional named helper replacements with the same signature; normally leave empty.
CALLBACKS = {}

if __name__ == "__main__":
    run_stage("intraday", settings=SETTINGS, callbacks=CALLBACKS)
