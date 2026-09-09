# Equity Model Research

Produces screened stock and macro data, selected model metadata, confirmed fitted models, and frozen macro regimes. It has no Horizon portfolio simulation or strategy optimisation stages. The `equity_selector`, `features`, `targets`, `screening`, `models`, and `main_package` packages live here; Horizon Score Strategy installs this project as a dependency.

From the repository root, using Python 3.11 or newer:

```sh
python -m pip install -e "Projects/Equity Model Research[test]"
python -m pip install -e "Projects/Horizon Score Strategy"
```

Set `EQUITY_SELECTOR_DATA_DIR` to your existing model research data directory when migrating. Otherwise it defaults to this project's `data/`. `EQUITY_SELECTOR_MODEL_DIR` optionally overrides `data/confirmed_models`; `EQUITY_SELECTOR_REQUEST_DIR` optionally overrides `data/model_requests`. Do not point the strategy output directory at the research directory.

The editable launcher `Run Research.py` offers:

| Stage | Script |
| --- | --- |
| prepare | Prepare Research.py |
| data | Data_Creation_Screening.py |
| macro_data | Macro_Data_Creation_Screening.py |
| training | Model Fitting.py |
| model_confirmation | Best_Model_Test.py |
| macro_regimes | Create Macro Regimes.py |
| produce_models | Produce Confirmed Models.py |

`Intraday Conversion.py` remains an optional model-data transformation. `Check Databases.py` remains an audit utility. Each launcher retains the existing modelling dates, screening settings and training rules. `Prepare Research.py` creates the research data directory only; strategy phase staging belongs downstream.

## Macro data and regimes

Use the existing `macro_data.db`. The resolver also recognizes the existing legacy `Macro_Data.db` spelling; it refuses ambiguous multiple databases. The macro producer writes into that same database and automatically saves `Macro_Regimes.txt` and `Macro_Regimes_Daily.csv`. ALFRED credentials come from `ALFRED_API_KEY` settings or the `FRED_API_KEY` environment variable.

To generate regimes from an existing database without downloading anything:

```sh
python "Projects/Equity Model Research/Create Macro Regimes.py" --database /path/to/macro_data.db --table "Your Existing Table"
```

The table is discovered by its `Date` and `Macro PIT ` columns if exactly one table qualifies. Ambiguity requires `--table`; no independent macro source is created. Run macro creation before stock data screening to make PIT macro predictors available to newly fitted stock models. If stock screening ran first, rerun it after macro creation when adding those predictors. The stock data builder and downstream cache builder attach existing PIT features using a backward availability-date join, preserving stock row order and existing columns. Missing early values remain missing.

`build_macro_regimes(macro, confirmation_days=5)` returns the required nested dictionary, with inclusive ISO-date tuples and all states, including Unknown. `return_daily=True` also returns scores and usable-signal counts. Growth, Inflation, Labour, Consumer and Housing require two usable signals; Liquidity requires one. Votes have neutral bands of 0.1 percentage points for growth/change comparisons and 0.05 for inflation/payroll/M2 acceleration and unemployment change. Unemployment gap uses the requested 0.20/0.50 thresholds. Average votes classify at ±0.34. These are explicit fixed rules, not thresholds calibrated with future data.

The first usable state initializes immediately. Later candidates need five consecutive observations, with changes dated on confirmation, never backdated. Missing candidates retain the active state and reset pending confirmation. The function does not forward-fill individual signals, use revised-level fallback columns, or inspect future rows. Calendar-day input counts calendar observations; trading-day input counts trading observations. Downstream load uses `ast.literal_eval`, never `eval`.

## Confirmed-model exchange

Research selection metadata is in `Final_Test_Results.db`; target feature definitions are in `Selected_Features.txt`; prepared stock inputs are in `Features_Targets_Data.db`. These existing names remain unchanged.

The original Horizon/cache stages fitted selected models again for their distinct historical training windows. To preserve those exact windows while moving all fitting here, a strategy stage first checks for matching frozen fits. When absent, it writes **all** required model-only requests and stops before portfolio construction. Run:

```sh
python "Projects/Equity Model Research/Produce Confirmed Models.py"
```

Then resume that strategy stage. One production pass handles all missing target fits from that stage. New phases or changed training data/configurations can require another explicit research production pass. Requests contain TRAIN rows only; BACKTEST rows are never passed to fitting. This handoff is deliberate and replaces the former hidden refit.

For other strategies, call the reusable research API directly:

```python
from equity_selector.artifacts import produce_models
produce_models(training, selected, features, purge=True)
```

Artifacts contain fitted estimators, scalers, encoders and inference metadata. Their identity covers training values, column order/dtypes, feature order, model metadata and purge policy. A JSON manifest records training dates, versions and SHA-256; consumers reject missing/mismatched/corrupt artifacts and never fit. Pickles and request files must be trusted local research outputs. Use the same Python/dependency environment for production and consumption.

## Validation

Run `python -m pytest "Projects/Equity Model Research/tests"` from the repository root after installation. Cross-project integration, strategy and evaluator tests are in Horizon Score Strategy. See the downstream `REFACTOR_REPORT.md` for the migration inventory and validation results. The feature/target reference PDF remains historical supporting documentation.
