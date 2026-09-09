# Horizon Score Strategy

Consumes Equity Model Research outputs and retains the existing Horizon Score construction, simulations, robustness tests, GBP checks and frozen final validation. Predictive fitting belongs entirely to the sibling project.

Install both projects from the repository root:

```sh
python -m pip install -e "Projects/Equity Model Research[test]"
python -m pip install -e "Projects/Horizon Score Strategy"
```

Set `EQUITY_SELECTOR_DATA_DIR` to the research output directory. Set `HORIZON_SCORE_DATA_DIR` to the strategy output root, which defaults to this project's `data/`. Launchers use `portfolio_selection/` and `final_evaluation/` subdirectories; their editable settings can override paths. Horizon search now writes `Top_Horizon_Scores.txt` directly into the selection directory used by simulations. The final cache has its own directory and historical dates.

| Stage | Script |
| --- | --- |
| prepare | Prepare Strategy.py |
| horizons | Horizon Score Backtests.py |
| cache | Backtest Database.py |
| simulations | Backtest Simulations.py |
| precise | Precise Backtest.py |
| gbp_check | GBP Portfolio Check.py |
| final | Frozen Final Test.py |

Choose the stage in `Run Strategy.py`. For final-cache preparation, set `CACHE_PHASE = "final"` in `Backtest Database.py`; prepare/freeze the final phase before running the final test. `Prepare Strategy.py` stages immutable research metadata and macro regimes for selection, and the frozen Horizon file when staging the final phase. Existing conflict/checksum and FX-date protections remain.

## Model boundary

`horizon_score.stages.horizons` and `.cache` read research metadata and data from `EQUITY_SELECTOR_DATA_DIR`, and load artifacts through `equity_selector.artifacts.load_models_and_predictions`. Phase-staged `Final_Test_Results.db`, `Selected_Features.txt`, and `Macro_Regimes.txt` take precedence over live research outputs so later research changes do not replace frozen selections. There is no fitting fallback. Missing historical fits produce model-only requests and an actionable error; run the sibling `Produce Confirmed Models.py` and resume. See its README for details. This explicit handshake preserves the original per-target purge and train/backtest partitions, including the different final-cache training window.

Existing data is not moved automatically. Point the environment variables to the existing directories, or copy research inputs and existing strategy outputs into their corresponding new locations. Regenerate `Macro_Regimes.txt` from the existing macro database before final evaluation. Previously saved strategy selections remain usable, including legacy configuration names.

## Evaluate saved returns without models

```sh
python "Projects/Horizon Score Strategy/Evaluate Backtest.py" \
  --backtest /path/to/daily_weights.csv \
  --market /path/to/sp500.csv \
  --output-dir /path/to/final_evaluation
```

The backtest CSV contains `Date`, gross `Return`, and one `[0, 1]` weight column per ticker. The benchmark supports `Return`, `Adj Close`, `Close`, or a single-ticker yfinance MultiIndex through the Python API. Dates are normalized, sorted and joined explicitly; duplicate dates and overlapping regimes are rejected. Missing benchmark returns are never filled. The regime dictionary is loaded from research `Macro_Regimes.txt` using `ast.literal_eval`.

```python
from horizon_score.evaluation import evaluate_backtest
report = evaluate_backtest(backtest_results, regimes, market, output_dir="final_evaluation")
```

The returned and saved outputs are:

| Output | Contents |
| --- | --- |
| summary | Return, CAGR, volatility, Sharpe, Sortino, Calmar, drawdown/recovery durations, signed VaR/ES, positive/negative day statistics, profit factor, skew/kurtosis and IID PSR |
| portfolio | Holdings, gross exposure, position size, HHI/effective holdings, turnover, entries/exits and weight changes |
| costs | Cost fraction/bps, gross/net total return and CAGR, gross/net/stressed Sharpe, net drawdown and break-even fee |
| benchmark | Date-aligned strategy/market performance, correlation, beta, alpha, tracking error, information ratio, active return and upside/downside capture |
| regimes | Every family/state, including empty and Unknown states; exposure-based return/risk, portfolio and benchmark statistics |
| regime_episodes | Separate contiguous interval diagnostics, including drawdowns and recovery measures |
| rolling | 63/126/252-observation return, annualised return, volatility, Sharpe, beta/correlation, holdings and turnover |
| yearly | Strategy/market return, Sharpe, drawdown and other available annual diagnostics |
| monthly | Year × month compounded strategy returns |
| drawdowns | Peak/start/trough/recovery dates, depths, durations and unrecovered episodes |
| daily | Original returns/weights plus net returns, exposure/turnover, wealth, drawdown, benchmark and regime labels |

Every output is saved separately as CSV, with `Conventions.txt`. Annualisation uses 252. Turnover is **half** absolute weight changes; fees apply to **full** traded notional, including initial entry. Gross daily returns are reduced by fee × traded notional. HHI is the sum of raw squared weights; cash-only effective holdings are undefined. Capture ratios use conditional arithmetic means. Risk-free rate defaults to zero. PSR requires 30 observations and is explicitly IID.

Disconnected regime aggregates use exposure-day annualisation, not elapsed wall-clock CAGR. Their path-dependent drawdown and Calmar fields are deliberately undefined; consult individual episodes. No nonexistent continuous investment path is implied.

Precise evaluation automatically adds these reports for every reconstructed finalist while retaining neighbour, unseen-stock, leave-one-stock-out, best-period-removal and DSR checks. The legacy selection-quality objective remains unchanged. Its general Sharpe/PSR/ES/cost helpers now delegate to the evaluator. Frozen/GBP launchers also produce USD post-backtest reports: they recover gross USD returns from the existing net stream using the exact original fee convention, avoiding double charging. Existing GBP account conversion and drawdown gating remain separate because they include account-specific FX/capital flows. The reusable `evaluate_frozen` API accepts explicit `regimes` and `market`; existing callers without them retain its previous account-only interface.

## Validation and provenance

Run `python -m pytest "Projects/Equity Model Research/tests" "Projects/Horizon Score Strategy/tests"` after installation. Run `ruff check` on both folders for the existing static checks. See [REFACTOR_REPORT.md](REFACTOR_REPORT.md) for exact movement/cleanup, before/after evidence and limitations. `docs/Legacy Equity Selector Project Report.pdf` documents the original combined project, not the new paths.
