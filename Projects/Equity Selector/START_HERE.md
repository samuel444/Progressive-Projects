# Start High Liquidity 30 research

This folder is configured for fresh research using the original date plan, £4,000 starting capital, a Trading 212 Invest account, individual USD trades, retained USD cash, and a 20% GBP peak-to-trough drawdown acceptance limit. Previous model scores are not imported. The original user databases remain untouched.

## Install and start

From a terminal in this extracted Equity Selector folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python 'Run Research.py'
```

Use a Python version supported by the project (3.11 or later). Existing compatible environments can be reused. Installation/downloads on your machine have not been tested here.

`Run Research.py` defaults to preparation. Change its `SETTINGS["stage"]` to the next stage below and run it again. Every original script retains its own settings above `run_stage`; the wrapper does not override their research settings. It changes to the project folder and requests two numerical-library threads and two joblib CPUs before importing the stage. This is a conservative default for an M2 with 8GB RAM, not a hard memory or CPU cap. Run one stage/process at a time. No C++ rewrite or model-fitting optimisation was introduced.

## Run order

| Step | Run Research stage | Action/settings |
|---|---|---|
| 1 | prepare | Leave Prepare Research.py phase=model; creates isolated directories. |
| 2 | data | Data_Creation_Screening.py creates the High Liquidity 30 data and feature selection. |
| 3 | training | Model Fitting.py runs the extensive grid and all required folds. |
| 4 | model_confirmation | Best_Model_Test.py uses the fixed model-confirmation period. |
| 5 | horizons | Horizon Score Backtests.py chooses horizon weights using its development period. |
| 6 | prepare | Set Prepare Research.py phase=selection. Copies completed model/feature/horizon artifacts and downloads selection-period GBP/USD rates. |
| 7 | cache | Backtest Database.py: CACHE_PHASE=selection. Builds 2019–2022 signals. |
| 8 | simulations | Backtest Simulations.py runs the original gross USD selection/robustness grid. |
| 9 | precise | Precise Backtest.py checks USD robustness and execution-cost sensitivity. |
| 10 | gbp_check | GBP Portfolio Check.py independently reconstructs candidates, applies USD execution costs, GBP FX valuation/conversion fees, and the 20% GBP drawdown gate. |
| 11 | prepare | Only after all selection choices are frozen: set Prepare Research.py phase=final. Copies frozen model inputs to the separate final directory and downloads final-period FX rates. |
| 12 | cache | Set Backtest Database.py CACHE_PHASE=final. |
| 13 | final | Frozen Final Test.py evaluates the GBP-approved selection; reports all finalists, including final-period drawdown failures. |

The GBP check prints a clear stop message if nobody passes. Do not relax the threshold by consulting final results. Selection failures remain in the GBP check Summary and Daily Returns tables. Final evaluation never removes failures or automatically chooses a new winner.

Run read-only database checks after completed stages, for example:

```sh
python 'Check Databases.py' 'data/extensive_20260904/portfolio_selection/Backtest_Database.db' --report selection-audit.json
python 'Check Databases.py' 'data/extensive_20260904/portfolio_selection/GBP Check/GBP_Selection.db' --contract 'data/extensive_20260904/portfolio_selection/GBP Check/audit-contract.json' --report gbp-audit.json
python 'Check Databases.py' 'data/extensive_20260904/final_evaluation/Frozen Evaluation/Frozen_Final_Evaluation.db' --contract 'data/extensive_20260904/final_evaluation/Frozen Evaluation/audit-contract.json' --report final-audit.json
```

The checker does not certify investability or correct market-data provenance. Incomplete checks are not passes. Use a fresh research directory for a changed experiment; never append old validation runs. Stop writers before preparing the next phase. Conflicting existing frozen input files and existing account-evaluation databases are refused.

## Dates (unchanged)

- Download warmup from 1997.
- Initial model training: 2000–2006.
- Walk-forward validation: 2007–2012, 63 observed dates per fold, every fold required.
- Model confirmation: 2013–2015.
- Horizon selection: 2016–2018.
- Portfolio development: 2019–2022 (cache end 2022-12-30, the last trading day).
- Final historical evaluation: January 2023–August 2026.

## Account settings and costs

Edit `GBP Portfolio Check.py` and `Frozen Final Test.py` consistently before any final data is examined:

| Setting | Value and meaning |
|---|---|
| trading_fee | 0.0: broker commission for USD individual trades. |
| account.initial_capital_gbp | 4000: initial GBP capital for each independently evaluated strategy sleeve. |
| account.fx_conversion_fee | 0.0015 = 0.15%, applied once entering USD and once at the end if requested. |
| account.convert_back_at_end | True: includes hypothetical terminal GBP conversion for comparable net proceeds. Does not send an order. |
| account.execution_cost_fraction | 0.0005 = 0.05% of buy-plus-sell target-weight notional. Assumed allowance for spread/slippage/regulatory charges, not an exact broker fee tariff. |
| account.max_drawdown | 0.2 = 20%, measured from GBP equity's running peak including initial capital. A research acceptance rule, not a live stop-loss guarantee. |
| account.fx_file | Per-phase CSV with Date and GBP_per_USD (GBP value of one USD). Preparation downloads GBPUSD=X and inverts its close. |
| annualisation | 252 daily observations per year. Sharpe uses a zero-rate reference. |

Precise Backtest.py FE_TRADING_FEE=0.0005 uses the same assumed execution allowance for its USD diagnostics. The gross grid is unchanged; the later GBP gate adds a required net-account check, but does not recover strategies rejected by the earlier gross selection. To assess sensitivity, change the allowance only during development and use fresh output folders; freeze it for final evaluation.

FX fees are charged on conversion, not every USD trade. Idle USD cash remains exposed to GBP/USD movements. The account layer preserves the existing USD sleeve calculations and values the entire sleeve in GBP. It does not alter stock-ranking decisions based on future FX rates. It requires exact FX date coverage and stops on missing dates rather than filling from future observations.

This remains a research approximation: daily FX closes are not execution-time quotes; target-weight turnover omits drift and final stock liquidation costs; dividends/cash interest, per-share historical exchange fees, taxes and exact brokerage cash movements are not separately replicated. The flat execution allowance does not prove those costs are covered. Trading 212's current published commission/custody fees are zero and FX conversion is 0.15%; exchange charges vary and should be compared with actual order previews and trade statements.

Sources checked 2026-09-04:
- https://helpcentre.trading212.com/hc/en-us/articles/11669719976093-What-is-a-multi-currency-account
- https://helpcentre.trading212.com/hc/en-us/articles/11471996799517-What-are-the-fees-in-the-Invest-ISAs-and-SIPP

## Scope and completion expectations

The model grids, 729 maximum horizon combinations across 27 backgrounds, and 100 simulation robustness perturbations remain extensive. The observed log came from a pruned run and cannot establish a 20-day completion guarantee for these settings. No automatic background job or 20-day timer is installed. The strategy research itself has not been run here.

Use the helper smoke tests and database checks to establish code consistency; separately assess missing-data loss, historical universe membership and subperiod results. Static High Liquidity 30 tickers do not establish point-in-time investability. An empty eligible set is a valid research outcome, not a reason to loosen final-period rules.
