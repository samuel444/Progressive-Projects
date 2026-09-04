"""Apply the 20% GBP drawdown limit before freezing portfolio choices.

Execution allowance is assumed, not an exact historical broker/exchange fee schedule.
"""

from equity_selector.frozen import evaluate_frozen

SETTINGS = {
    "selection_database": "data/extensive_20260904/portfolio_selection/Backtest_Database.db",
    "cache_database": "data/extensive_20260904/portfolio_selection/Backtest_Database.db",
    "horizon_file": "data/extensive_20260904/portfolio_selection/Top_Horizon_Scores.txt",
    "type_configurations": [
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
    "start": "2019-01-01",
    "end": "2022-12-30",
    "trading_fee": 0.0,
    "annualisation": 252,
    "account": {
        "fx_file": "data/extensive_20260904/portfolio_selection/GBP_per_USD.csv",
        "initial_capital_gbp": 4000.0,
        "fx_conversion_fee": 0.0015,
        "convert_back_at_end": True,
        "max_drawdown": 0.2,
        "execution_cost_fraction": 0.0005,
    },
    "output_dir": "data/extensive_20260904/portfolio_selection/GBP Check",
    "evaluation_kind": "selection",
}

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    result = evaluate_frozen(**SETTINGS)
    print(result.to_string(index=False))
    if not result["GBP Drawdown Limit Passed"].any():
        raise SystemExit("No strategies passed the GBP drawdown limit. Do not run the final test.")
