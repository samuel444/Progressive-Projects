"""Edit SETTINGS here, then run this script. CLI path/log options override these values.
None retains optional defaults; required cache dates must be set here. Packages never prompt.
"""

from equity_selector.cli import run_stage

SETTINGS = {
    "DATA_DIR": "data/extensive_20260904/portfolio_selection",
    "LOG_LEVEL": "INFO",
    "PORTFOLIO_GROUP_CONFIGURATIONS": [
        {
            "Name": "Equal Weight Baseline",
            "Ranking": 0.2,
            "Direction": 0.2,
            "Risk": 0.2,
            "Opportunity": 0.2,
            "Special": 0.2,
        },
        {
            "Name": "Ranking Focused",
            "Ranking": 0.5,
            "Direction": 0.2,
            "Risk": 0.2,
            "Opportunity": 0.1,
            "Special": 0.0,
        },
        {
            "Name": "Risk Aware",
            "Ranking": 0.35,
            "Direction": 0.2,
            "Risk": 0.35,
            "Opportunity": 0.1,
            "Special": 0.0,
        },
    ],
    "FE_TRADING_FEE": 0.0005,  # Assumed execution allowance; no recurring USD FX fee
    "FE_RF_ANNUAL": 0.0,
    "FE_DSR_SAMPLE_SIZE": 1000,
    "FE_DSR_TRIALS": None,
    "FE_SEED": 42,
    "FE_DAYS": 252,
    "FE_NEIGHBOURHOOD_SD": None,  # supply the original neighbor-cohort SD only if known
    "FE_UNSEEN_GATE": 1.5,
    "OUTPUT_DIR": "data/extensive_20260904/portfolio_selection/Precise Evaluation",
}

# Optional named helper replacements with the same signature; normally leave empty.
CALLBACKS = {}

if __name__ == "__main__":
    run_stage("precise", settings=SETTINGS, callbacks=CALLBACKS)
