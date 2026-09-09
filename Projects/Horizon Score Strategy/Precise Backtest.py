from horizon_score.groups import validation_groups
from horizon_score.config import data_root as project_data_root

"""Edit SETTINGS here, then run this script. CLI path/log options override these values.
None retains optional defaults; required cache dates must be set here. Packages never prompt.
"""

from horizon_score.cli import run_stage

SETTINGS = {
    "DATA_DIR": str(project_data_root() / "portfolio_selection"),
    "LOG_LEVEL": "INFO",
    "PORTFOLIO_GROUP_CONFIGURATIONS": validation_groups(),
    "FE_TRADING_FEE": 0.0005,  # Assumed execution allowance; no recurring USD FX fee
    "FE_RF_ANNUAL": 0.0,
    "FE_DSR_SAMPLE_SIZE": 1000,
    "FE_DSR_TRIALS": None,
    "FE_SEED": 42,
    "FE_DAYS": 252,
    "FE_NEIGHBOURHOOD_SD": None,  # supply the original neighbor-cohort SD only if known
    "FE_UNSEEN_GATE": 1.5,
    "OUTPUT_DIR": str(project_data_root() / "portfolio_selection/Precise Evaluation"),
}

# Optional named helper replacements with the same signature; normally leave empty.
CALLBACKS = {}

if __name__ == "__main__":
    run_stage("precise", settings=SETTINGS, callbacks=CALLBACKS)
