"""Choose a stage here; each original script retains its own editable SETTINGS."""

import os
from pathlib import Path
import runpy

SETTINGS = {
    "stage": "data",  # keys in STAGES below; run one stage at a time
    "math_threads": 2,
    "joblib_cpu_limit": 2,
}

STAGES = {
    "prepare": "Prepare Research.py",
    "data": "Data_Creation_Screening.py",
    "training": "Model Fitting.py",
    "model_confirmation": "Best_Model_Test.py",
    "horizons": "Horizon Score Backtests.py",
    "cache": "Backtest Database.py",
    "simulations": "Backtest Simulations.py",
    "precise": "Precise Backtest.py",
    "gbp_check": "GBP Portfolio Check.py",
    "final": "Frozen Final Test.py",
}

if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    for key in ["math_threads", "joblib_cpu_limit"]:
        if not isinstance(SETTINGS[key], int) or SETTINGS[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if SETTINGS["stage"] not in STAGES:
        raise ValueError("Unknown research stage")
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = str(SETTINGS["math_threads"])
    os.environ["LOKY_MAX_CPU_COUNT"] = str(SETTINGS["joblib_cpu_limit"])
    root = Path(__file__).resolve().parent
    os.chdir(root)
    runpy.run_path(str(root / STAGES[SETTINGS["stage"]]), run_name="__main__")
