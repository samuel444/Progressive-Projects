from horizon_score.groups import validation_groups
from equity_selector.macro_regimes import load_macro_regimes
from horizon_score.config import research_input
from equity_selector.database import read_table
from horizon_score.config import data_root as project_data_root

"""One-pass final-period evaluation. Freeze choices first; use a separate final cache."""

import argparse
from horizon_score.frozen import evaluate_frozen

SETTINGS = {
    "selection_database": str(
        project_data_root() / "portfolio_selection/GBP Check/GBP_Selection.db"
    ),
    "cache_database": str(
        project_data_root() / "final_evaluation/Backtest_Database.db"
    ),
    "horizon_file": str(
        project_data_root() / "portfolio_selection/Top_Horizon_Scores.txt"
    ),
    # Replace with the exact group configuration values frozen during portfolio selection.
    "type_configurations": validation_groups(),
    "start": "2023-01-01",
    "end": "2026-08-31",
    "trading_fee": 0.0,
    "annualisation": 252,
    # Broker commission is zero. Separate 5bp execution allowance is an assumption.
    "account": {
        "fx_file": str(project_data_root() / "final_evaluation/GBP_per_USD.csv"),
        "initial_capital_gbp": 4000.0,
        "fx_conversion_fee": 0.0015,
        "convert_back_at_end": True,
        "max_drawdown": 0.2,
        "execution_cost_fraction": 0.0005,
    },
    "output_dir": str(project_data_root() / "final_evaluation/Frozen Evaluation"),
}

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(
        evaluate_frozen(
            **SETTINGS,
            regimes=load_macro_regimes(research_input("Macro_Regimes.txt", project_data_root() / "final_evaluation")),
            market=read_table(SETTINGS["cache_database"], "Market")
            .groupby("Date", as_index=False)
            .agg(Return=("Return", "first")),
        ).to_string(index=False)
    )
