"""One-pass final-period evaluation. Freeze choices first; use a separate final cache."""

import argparse
from equity_selector.frozen import evaluate_frozen

SETTINGS = {
    "selection_database": "data/extensive_20260904/portfolio_selection/GBP Check/GBP_Selection.db",
    "cache_database": "data/extensive_20260904/final_evaluation/Backtest_Database.db",
    "horizon_file": "data/extensive_20260904/portfolio_selection/Top_Horizon_Scores.txt",
    # Replace with the exact group configuration values frozen during portfolio selection.
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
    "start": "2023-01-01",
    "end": "2026-08-31",
    "trading_fee": 0.0,
    "annualisation": 252,
    # Broker commission is zero. Separate 5bp execution allowance is an assumption.
    "account": {
        "fx_file": "data/extensive_20260904/final_evaluation/GBP_per_USD.csv",
        "initial_capital_gbp": 4000.0,
        "fx_conversion_fee": 0.0015,
        "convert_back_at_end": True,
        "max_drawdown": 0.2,
        "execution_cost_fraction": 0.0005,
    },
    "output_dir": "data/extensive_20260904/final_evaluation/Frozen Evaluation",
}

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(evaluate_frozen(**SETTINGS).to_string(index=False))
