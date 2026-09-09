from horizon_score.groups import validation_groups
from equity_selector.macro_regimes import load_macro_regimes
from horizon_score.config import research_input
from equity_selector.database import read_table
from horizon_score.config import data_root as project_data_root

"""Apply the 20% GBP drawdown limit before freezing portfolio choices.

Execution allowance is assumed, not an exact historical broker/exchange fee schedule.
"""

from horizon_score.frozen import evaluate_frozen

SETTINGS = {
    "selection_database": str(
        project_data_root() / "portfolio_selection/Backtest_Database.db"
    ),
    "cache_database": str(
        project_data_root() / "portfolio_selection/Backtest_Database.db"
    ),
    "horizon_file": str(
        project_data_root() / "portfolio_selection/Top_Horizon_Scores.txt"
    ),
    "type_configurations": validation_groups(),
    "start": "2019-01-01",
    "end": "2022-12-30",
    "trading_fee": 0.0,
    "annualisation": 252,
    "account": {
        "fx_file": str(project_data_root() / "portfolio_selection/GBP_per_USD.csv"),
        "initial_capital_gbp": 4000.0,
        "fx_conversion_fee": 0.0015,
        "convert_back_at_end": True,
        "max_drawdown": 0.2,
        "execution_cost_fraction": 0.0005,
    },
    "output_dir": str(project_data_root() / "portfolio_selection/GBP Check"),
    "evaluation_kind": "selection",
}

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    result = evaluate_frozen(
        **SETTINGS,
        regimes=load_macro_regimes(research_input("Macro_Regimes.txt", project_data_root() / "portfolio_selection")),
        market=read_table(SETTINGS["cache_database"], "Market")
        .groupby("Date", as_index=False)
        .agg(Return=("Return", "first")),
    )
    print(result.to_string(index=False))
    if not result["GBP Drawdown Limit Passed"].any():
        raise SystemExit(
            "No strategies passed the GBP drawdown limit. Do not run the final test."
        )
