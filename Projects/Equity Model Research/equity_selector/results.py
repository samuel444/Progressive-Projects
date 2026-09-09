"""Stable final-test table schemas, including all-disqualified runs."""

import sqlite3
import pandas as pd

from .database import write_frame

RESULT_COLUMNS = [
    "Target",
    "Model",
    "Parameters",
    "Target Type",
    "RMSE",
    "NRMSE",
    "MAE",
    "R2",
    "Rank IC",
    "ROC AUC",
    "PR AUC",
    "Log Loss",
    "Macro F1",
]
PASSED_COLUMNS = RESULT_COLUMNS + ["Portfolio Target Type", "Horizon", "Quality Score"]


def save_final_test_results(path, universe, results, errors):
    frame = pd.DataFrame(results) if results else pd.DataFrame(columns=RESULT_COLUMNS)
    error_frame = pd.DataFrame(errors, columns=["Target", "Error"])
    with sqlite3.connect(path) as connection:
        write_frame(frame, f"Final Test Results {universe}", connection, if_exists="replace")
        write_frame(error_frame, "Errors", connection, if_exists="replace")
        if frame.empty:
            write_frame(
                pd.DataFrame(columns=PASSED_COLUMNS),
                f"{universe} Passed Test Results",
                connection,
                if_exists="replace",
            )
    return frame
