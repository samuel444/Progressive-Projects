"""Edit SETTINGS here, then run this script. CLI path/log options override these values.
None retains optional defaults; required cache dates must be set here. Packages never prompt.
"""

from equity_selector.cli import run_stage

SETTINGS = {
    "DATA_DIR": "data/extensive_20260904/portfolio_selection",
    "LOG_LEVEL": "INFO",
    "MARKET_TICKER": "^GSPC",
    "FEATURE_WARMUP_YEARS": 3,
    "TARGET_LOOKAHEAD_DAYS": 400,
    "DEFAULT_TOKENS": [
        "AAPL",
        "MSFT",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "AMD",
        "INTC",
        "QCOM",
        "MU",
        "CSCO",
        "ORCL",
        "JPM",
        "BAC",
        "WFC",
        "C",
        "XOM",
        "CVX",
        "F",
        "GM",
        "T",
        "VZ",
        "PFE",
        "JNJ",
        "WMT",
        "DIS",
        "GE",
        "HD",
        "NFLX",
        "GOOG",
    ],
    "STOCK_TYPE": "High Liquidity 30",
    "TOKENS": None,
    "DOWNLOAD_START": "1997-01-01",
    "DOWNLOAD_END": "2023-01-01",
    "TRAIN_END": "2018-12-31",
    "BACKTEST_START": "2019-01-01",
    "BACKTEST_END": "2022-12-30",
}

# Run selection first. After freezing finalists, copy its model/feature artifacts into
# the separate final directory, then set CACHE_PHASE to "final".
CACHE_PHASE = "selection"
FINAL_CACHE_SETTINGS = {
    "DOWNLOAD_START": "1997-01-01",
    "DOWNLOAD_END": "2026-09-01",
    "TRAIN_END": "2022-12-31",
    "BACKTEST_START": "2023-01-01",
    "BACKTEST_END": "2026-08-31",
    "DATA_DIR": "data/extensive_20260904/final_evaluation",
}

# Optional named helper replacements with the same signature; normally leave empty.
CALLBACKS = {}

if __name__ == "__main__":
    if CACHE_PHASE not in {"selection", "final"}:
        raise ValueError("CACHE_PHASE must be selection or final")
    active_settings = dict(SETTINGS)
    if CACHE_PHASE == "final":
        active_settings.update(FINAL_CACHE_SETTINGS)
    run_stage("cache", settings=active_settings, callbacks=CALLBACKS)
