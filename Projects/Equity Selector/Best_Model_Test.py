"""Edit SETTINGS here, then run this script. CLI path/log options override these values.
None retains optional defaults; required cache dates must be set here. Packages never prompt.
"""

from equity_selector.cli import run_stage

SETTINGS = {
    "MODEL_SELECTION_MODE": "rank_one",  # or explicit
    "MODEL_SELECTIONS": {},  # target -> rank, or {"Model": name, "Parameters": {...}}
    "MIN_RANK_IC": 0.1,
    "MIN_ROC_AUC": 0.6,
    "MIN_PR_AUC": 0.2,
    "MIN_MACRO_F1": 0.45,
    "DATA_DIR": "data/extensive_20260904/model_research",
    "LOG_LEVEL": "INFO",
    "STOCK_TYPE": "High Liquidity 30",
    "RESEARCH_START": "2000-01-01",
    "RESEARCH_END": "2015-12-31",
    "MODEL_TRAIN_END": "2006-12-31",
    "MODEL_VALIDATION_END": "2012-12-31",
}

# Optional named helper replacements with the same signature; normally leave empty.
CALLBACKS = {}

if __name__ == "__main__":
    run_stage("final_test", settings=SETTINGS, callbacks=CALLBACKS)
