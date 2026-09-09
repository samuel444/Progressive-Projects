"""Evaluate saved daily weights/returns without rerunning any models."""

from horizon_score.evaluation import evaluate_backtest
from horizon_score.config import data_root
from equity_selector.config import data_root as model_data_root
from equity_selector.macro_regimes import load_macro_regimes

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backtest",
        required=True,
        help="CSV: Date, Return, ticker weights; Return is gross",
    )
    parser.add_argument(
        "--market", required=True, help="CSV with Date and Return, Adj Close or Close"
    )
    parser.add_argument("--regimes", default=model_data_root() / "Macro_Regimes.txt")
    parser.add_argument("--output-dir", default=data_root() / "Final Evaluation")
    parser.add_argument("--trading-cost", type=float, default=0.001)
    args = parser.parse_args()
    result = evaluate_backtest(
        pd.read_csv(args.backtest),
        load_macro_regimes(args.regimes),
        pd.read_csv(args.market),
        trading_cost=args.trading_cost,
        output_dir=args.output_dir,
    )
    print(result["summary"].to_string())
