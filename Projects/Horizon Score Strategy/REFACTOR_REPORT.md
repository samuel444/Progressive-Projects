# Equity Selector refactor report

The former `Projects/Equity Selector` is replaced by `Projects/Equity Model Research` and `Projects/Horizon Score Strategy`. Source baseline: commit `12c447bfdd4e4157f759e0bbb675f63ea08155f0` of `samuel444/Progressive-Projects`. The local branch is `codex/split-equity-model-research-horizon-score`.

**Publishing status:** implementation is local. GitHub CLI authentication was checked with network access and rejected the configured token as invalid. No remote branch, pull request or change to `main` has been published. Reauthenticate using `gh auth login -h github.com` before publishing. The supplied patch applies the complete replacement from the baseline; the ZIP contains both resulting folders under `Projects/`.

## 1. Final folder trees

Generated data, Python caches and editable-install metadata are omitted.

```text
Equity Model Research/
├── equity_selector/
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── final_test.py
│   │   ├── intraday.py
│   │   ├── macro_data.py
│   │   └── training.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── artifacts.py
│   ├── catalogue.py
│   ├── cli.py
│   ├── config.py
│   ├── database.py
│   ├── database_audit.py
│   ├── feature_mapping.py
│   ├── files.py
│   ├── macro_regimes.py
│   ├── metrics.py
│   ├── parameters.py
│   ├── pruning.py
│   ├── results.py
│   ├── search.py
│   ├── settings.py
│   ├── settings_catalogue.py
│   ├── training.py
│   └── validation.py
├── features/
│   ├── __init__.py
│   ├── _macro_common.py
│   ├── beta.py
│   ├── breadth.py
│   ├── builder.py
│   ├── composite.py
│   ├── correlation.py
│   ├── cross_sectional.py
│   ├── dispersion.py
│   ├── distribution.py
│   ├── drawdown.py
│   ├── experimental.py
│   ├── interactions.py
│   ├── liquidity.py
│   ├── macro_calendar.py
│   ├── macro_conditions.py
│   ├── macro_history.py
│   ├── macro_quality.py
│   ├── macro_rates.py
│   ├── macro_stock.py
│   ├── macro_vintages.py
│   ├── market_relative.py
│   ├── momentum.py
│   ├── moving_averages.py
│   ├── ohlc.py
│   ├── range_volatility.py
│   ├── regimes.py
│   ├── registry.py
│   ├── residual.py
│   ├── returns.py
│   ├── sector_relative.py
│   ├── tail_risk.py
│   ├── technical.py
│   ├── trend.py
│   ├── volatility.py
│   └── volume.py
├── main_package/
│   ├── __init__.py
│   ├── full_models.py
│   ├── functions.py
│   ├── model_inference.py
│   ├── model_prunes.py
│   └── model_recommendations.py
├── models/
│   ├── __init__.py
│   └── models.py
├── screening/
│   ├── __init__.py
│   ├── screening_features.py
│   ├── screening_stocks.py
│   └── target_feature_screening.py
├── targets/
│   ├── __init__.py
│   ├── barriers.py
│   ├── builder.py
│   ├── direction.py
│   ├── drawdown.py
│   ├── excursions.py
│   ├── ranking.py
│   ├── registry.py
│   ├── returns.py
│   ├── risk_adjusted.py
│   └── volatility.py
├── tests/
│   ├── test_database_audit.py
│   ├── test_feature_mapping.py
│   ├── test_features.py
│   ├── test_macro_regimes.py
│   ├── test_search_training.py
│   └── test_targets.py
├── .gitignore
├── Best_Model_Test.py
├── Check Databases.py
├── Create Macro Regimes.py
├── Data_Creation_Screening.py
├── Equity Selector Features N Targets Reference.pdf
├── Intraday Conversion.py
├── Macro_Data_Creation_Screening.py
├── Model Fitting.py
├── Prepare Research.py
├── Produce Confirmed Models.py
├── Quick Macros
├── README.md
├── Run Research.py
├── START_HERE.md
└── pyproject.toml

Horizon Score Strategy/
├── docs/
│   └── Legacy Equity Selector Project Report.pdf
├── horizon_score/
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── horizons.py
│   │   ├── precise.py
│   │   └── simulations.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── account.py
│   ├── backtesting.py
│   ├── cli.py
│   ├── config.py
│   ├── evaluation.py
│   ├── frozen.py
│   ├── groups.py
│   ├── portfolio.py
│   ├── preparation.py
│   ├── scoring.py
│   ├── settings_catalogue.py
│   ├── signals.py
│   └── simulations.py
├── tests/
│   ├── test_account.py
│   ├── test_artifacts.py
│   ├── test_evaluation.py
│   ├── test_fees_screening.py
│   ├── test_frozen_final.py
│   ├── test_hardening.py
│   ├── test_launcher_settings.py
│   ├── test_pipeline.py
│   ├── test_precise_smoke.py
│   ├── test_research_profile.py
│   └── test_stages.py
├── Backtest Database.py
├── Backtest Simulations.py
├── Evaluate Backtest.py
├── Frozen Final Test.py
├── GBP Portfolio Check.py
├── Horizon Score Backtests.py
├── Precise Backtest.py
├── Prepare Strategy.py
├── README.md
├── REFACTOR_REPORT.md
├── Run Strategy.py
└── pyproject.toml
```

## 2. File movement inventory

Original paths below are relative to `Projects/Equity Selector`; destinations are relative to `Projects`. New modules are described in the next sections. Empty package initializers and `__main__.py` also initialize the new strategy package.

| Original | Destination |
| --- | --- |
| `.gitignore` | `Equity Model Research/.gitignore` |
| `Backtest Database.py` | `Horizon Score Strategy/Backtest Database.py` |
| `Backtest Simulations.py` | `Horizon Score Strategy/Backtest Simulations.py` |
| `Best_Model_Test.py` | `Equity Model Research/Best_Model_Test.py` |
| `Check Databases.py` | `Equity Model Research/Check Databases.py` |
| `Data_Creation_Screening.py` | `Equity Model Research/Data_Creation_Screening.py` |
| `Equity Selector Features N Targets Reference.pdf` | `Equity Model Research/Equity Selector Features N Targets Reference.pdf` |
| `Equity Selector Project Report.pdf` | `Horizon Score Strategy/docs/Legacy Equity Selector Project Report.pdf` |
| `Frozen Final Test.py` | `Horizon Score Strategy/Frozen Final Test.py` |
| `GBP Portfolio Check.py` | `Horizon Score Strategy/GBP Portfolio Check.py` |
| `Horizon Score Backtests.py` | `Horizon Score Strategy/Horizon Score Backtests.py` |
| `Intraday Conversion.py` | `Equity Model Research/Intraday Conversion.py` |
| `Macro_Data_Creation_Screening.py` | `Equity Model Research/Macro_Data_Creation_Screening.py` |
| `Model Fitting.py` | `Equity Model Research/Model Fitting.py` |
| `Precise Backtest.py` | `Horizon Score Strategy/Precise Backtest.py` |
| `Prepare Research.py` | `Equity Model Research/Prepare Research.py`; `Horizon Score Strategy/Prepare Strategy.py` |
| `Quick Macros` | `Equity Model Research/Quick Macros` |
| `README.md` | `Equity Model Research/README.md` |
| `Run Research.py` | `Equity Model Research/Run Research.py`; `Horizon Score Strategy/Run Strategy.py` |
| `START_HERE.md` | `Equity Model Research/START_HERE.md` |
| `equity_selector/__init__.py` | `Equity Model Research/equity_selector/__init__.py` |
| `equity_selector/__main__.py` | `Equity Model Research/equity_selector/__main__.py` |
| `equity_selector/account.py` | `Horizon Score Strategy/horizon_score/account.py` |
| `equity_selector/catalogue.py` | `Equity Model Research/equity_selector/catalogue.py` |
| `equity_selector/cli.py` | `Equity Model Research/equity_selector/cli.py`; `Horizon Score Strategy/horizon_score/cli.py` |
| `equity_selector/config.py` | `Equity Model Research/equity_selector/config.py` |
| `equity_selector/database.py` | `Equity Model Research/equity_selector/database.py` |
| `equity_selector/database_audit.py` | `Equity Model Research/equity_selector/database_audit.py` |
| `equity_selector/feature_mapping.py` | `Equity Model Research/equity_selector/feature_mapping.py` |
| `equity_selector/files.py` | `Equity Model Research/equity_selector/files.py` |
| `equity_selector/frozen.py` | `Horizon Score Strategy/horizon_score/frozen.py` |
| `equity_selector/metrics.py` | `Equity Model Research/equity_selector/metrics.py` |
| `equity_selector/parameters.py` | `Equity Model Research/equity_selector/parameters.py` |
| `equity_selector/portfolio.py` | `Horizon Score Strategy/horizon_score/portfolio.py` |
| `equity_selector/preparation.py` | `Horizon Score Strategy/horizon_score/preparation.py` |
| `equity_selector/pruning.py` | `Equity Model Research/equity_selector/pruning.py` |
| `equity_selector/results.py` | `Equity Model Research/equity_selector/results.py` |
| `equity_selector/scoring.py` | `Horizon Score Strategy/horizon_score/scoring.py` |
| `equity_selector/search.py` | `Equity Model Research/equity_selector/search.py` |
| `equity_selector/settings.py` | `Equity Model Research/equity_selector/settings.py` |
| `equity_selector/settings_catalogue.py` | `Equity Model Research/equity_selector/settings_catalogue.py`; `Horizon Score Strategy/horizon_score/settings_catalogue.py` |
| `equity_selector/simulations.py` | `Horizon Score Strategy/horizon_score/simulations.py` |
| `equity_selector/stages/__init__.py` | `Equity Model Research/equity_selector/stages/__init__.py` |
| `equity_selector/stages/cache.py` | `Horizon Score Strategy/horizon_score/stages/cache.py` |
| `equity_selector/stages/data.py` | `Equity Model Research/equity_selector/stages/data.py` |
| `equity_selector/stages/final_test.py` | `Equity Model Research/equity_selector/stages/final_test.py` |
| `equity_selector/stages/horizons.py` | `Horizon Score Strategy/horizon_score/stages/horizons.py` |
| `equity_selector/stages/intraday.py` | `Equity Model Research/equity_selector/stages/intraday.py` |
| `equity_selector/stages/macro_data.py` | `Equity Model Research/equity_selector/stages/macro_data.py` |
| `equity_selector/stages/precise.py` | `Horizon Score Strategy/horizon_score/stages/precise.py` |
| `equity_selector/stages/simulations.py` | `Horizon Score Strategy/horizon_score/stages/simulations.py` |
| `equity_selector/stages/training.py` | `Equity Model Research/equity_selector/stages/training.py` |
| `equity_selector/training.py` | `Equity Model Research/equity_selector/training.py` |
| `equity_selector/validation.py` | `Equity Model Research/equity_selector/validation.py` |
| `features/__init__.py` | `Equity Model Research/features/__init__.py` |
| `features/_macro_common.py` | `Equity Model Research/features/_macro_common.py` |
| `features/beta.py` | `Equity Model Research/features/beta.py` |
| `features/breadth.py` | `Equity Model Research/features/breadth.py` |
| `features/builder.py` | `Equity Model Research/features/builder.py` |
| `features/composite.py` | `Equity Model Research/features/composite.py` |
| `features/correlation.py` | `Equity Model Research/features/correlation.py` |
| `features/cross_sectional.py` | `Equity Model Research/features/cross_sectional.py` |
| `features/dispersion.py` | `Equity Model Research/features/dispersion.py` |
| `features/distribution.py` | `Equity Model Research/features/distribution.py` |
| `features/drawdown.py` | `Equity Model Research/features/drawdown.py` |
| `features/experimental.py` | `Equity Model Research/features/experimental.py` |
| `features/interactions.py` | `Equity Model Research/features/interactions.py` |
| `features/liquidity.py` | `Equity Model Research/features/liquidity.py` |
| `features/macro_calendar.py` | `Equity Model Research/features/macro_calendar.py` |
| `features/macro_conditions.py` | `Equity Model Research/features/macro_conditions.py` |
| `features/macro_history.py` | `Equity Model Research/features/macro_history.py` |
| `features/macro_quality.py` | `Equity Model Research/features/macro_quality.py` |
| `features/macro_rates.py` | `Equity Model Research/features/macro_rates.py` |
| `features/macro_stock.py` | `Equity Model Research/features/macro_stock.py` |
| `features/macro_vintages.py` | `Equity Model Research/features/macro_vintages.py` |
| `features/market_relative.py` | `Equity Model Research/features/market_relative.py` |
| `features/momentum.py` | `Equity Model Research/features/momentum.py` |
| `features/moving_averages.py` | `Equity Model Research/features/moving_averages.py` |
| `features/ohlc.py` | `Equity Model Research/features/ohlc.py` |
| `features/range_volatility.py` | `Equity Model Research/features/range_volatility.py` |
| `features/regimes.py` | `Equity Model Research/features/regimes.py` |
| `features/registry.py` | `Equity Model Research/features/registry.py` |
| `features/residual.py` | `Equity Model Research/features/residual.py` |
| `features/returns.py` | `Equity Model Research/features/returns.py` |
| `features/sector_relative.py` | `Equity Model Research/features/sector_relative.py` |
| `features/tail_risk.py` | `Equity Model Research/features/tail_risk.py` |
| `features/technical.py` | `Equity Model Research/features/technical.py` |
| `features/trend.py` | `Equity Model Research/features/trend.py` |
| `features/volatility.py` | `Equity Model Research/features/volatility.py` |
| `features/volume.py` | `Equity Model Research/features/volume.py` |
| `main_package/__init__.py` | `Equity Model Research/main_package/__init__.py` |
| `main_package/backtesting.py` | `Equity Model Research/main_package/model_inference.py`; `Horizon Score Strategy/horizon_score/backtesting.py` |
| `main_package/full_models.py` | `Equity Model Research/main_package/full_models.py` |
| `main_package/functions.py` | `Equity Model Research/main_package/functions.py` |
| `main_package/model_prunes.py` | `Equity Model Research/main_package/model_prunes.py` |
| `main_package/model_recommendations.py` | `Equity Model Research/main_package/model_recommendations.py` |
| `main_package/signals.py` | `Horizon Score Strategy/horizon_score/signals.py` |
| `models/__init__.py` | `Equity Model Research/models/__init__.py` |
| `models/models.py` | `Equity Model Research/models/models.py` |
| `pyproject.toml` | `Equity Model Research/pyproject.toml` |
| `screening/__init__.py` | `Equity Model Research/screening/__init__.py` |
| `screening/screening_features.py` | `Equity Model Research/screening/screening_features.py` |
| `screening/screening_stocks.py` | `Equity Model Research/screening/screening_stocks.py` |
| `screening/target_feature_screening.py` | `Equity Model Research/screening/target_feature_screening.py` |
| `targets/__init__.py` | `Equity Model Research/targets/__init__.py` |
| `targets/barriers.py` | `Equity Model Research/targets/barriers.py` |
| `targets/builder.py` | `Equity Model Research/targets/builder.py` |
| `targets/direction.py` | `Equity Model Research/targets/direction.py` |
| `targets/drawdown.py` | `Equity Model Research/targets/drawdown.py` |
| `targets/excursions.py` | `Equity Model Research/targets/excursions.py` |
| `targets/ranking.py` | `Equity Model Research/targets/ranking.py` |
| `targets/registry.py` | `Equity Model Research/targets/registry.py` |
| `targets/returns.py` | `Equity Model Research/targets/returns.py` |
| `targets/risk_adjusted.py` | `Equity Model Research/targets/risk_adjusted.py` |
| `targets/volatility.py` | `Equity Model Research/targets/volatility.py` |
| `tests/test_account.py` | `Horizon Score Strategy/tests/test_account.py` |
| `tests/test_database_audit.py` | `Equity Model Research/tests/test_database_audit.py` |
| `tests/test_feature_mapping.py` | `Equity Model Research/tests/test_feature_mapping.py` |
| `tests/test_features.py` | `Equity Model Research/tests/test_features.py` |
| `tests/test_fees_screening.py` | `Horizon Score Strategy/tests/test_fees_screening.py` |
| `tests/test_frozen_final.py` | `Horizon Score Strategy/tests/test_frozen_final.py` |
| `tests/test_hardening.py` | `Horizon Score Strategy/tests/test_hardening.py` |
| `tests/test_launcher_settings.py` | `Horizon Score Strategy/tests/test_launcher_settings.py` |
| `tests/test_pipeline.py` | `Horizon Score Strategy/tests/test_pipeline.py` |
| `tests/test_precise_smoke.py` | `Horizon Score Strategy/tests/test_precise_smoke.py` |
| `tests/test_research_profile.py` | `Horizon Score Strategy/tests/test_research_profile.py` |
| `tests/test_search_training.py` | `Equity Model Research/tests/test_search_training.py` |
| `tests/test_stages.py` | `Horizon Score Strategy/tests/test_stages.py` |
| `tests/test_targets.py` | `Equity Model Research/tests/test_targets.py` |

## 3. Shared functionality and dependency direction

Research owns reusable model fitting/inference, features, targets, screening, model selection, SQLite utilities, parameter handling, chronology/purging, generic settings scopes and the legacy performance primitive. `main_package/model_inference.py` contains the retained model-only portion of the former mixed backtesting module. Strategy owns scoring, allocation, portfolio simulation, account conversion, precise/frozen evaluation and broker helpers. The only package dependency direction is strategy → research; research has no strategy imports.

`equity_selector/artifacts.py` adds deterministic historical-fit identity, explicit model production, checked artifact loading and prediction generation. `equity_selector/macro_regimes.py` adds PIT-only macro alignment, causal regime confirmation and literal serialization. `horizon_score/evaluation.py` is the canonical comprehensive post-backtest evaluator. Generic legacy selection-quality metrics are retained because changing them would change strategy selection. Precise Sharpe, PSR, ES and costs delegate to the new evaluator; compatibility wrappers preserve callers and DSR behavior.

The stage registries and settings/callback catalogs were split by responsibility. `horizon_score/groups.py` centralizes simulation group definitions and retains legacy names for old frozen selections. No model search is duplicated in the strategy package.

## 4. Cleanup and retained compatibility

Removed the obsolete combined fit-and-portfolio helper `run_multi_target_portfolio_backtest` and its unused private helpers `_construct_portfolio`, `_rank_to_minus_one_one`, `classification_signal`, `_infer_horizon_key`, and `_lookup_horizon_score`. Retained `create_models_and_predictions` as a public research-only fitting/inference API and `run_portfolio_backtest_from_predictions` as a strategy-only adapter. Broker/notification helpers remain public downstream APIs, with no implicit network calls.

Removed obsolete `re`, `typing.Iterable`, `typing.Tuple`, `scipy.optimize.minimize`, and a duplicate NumPy import from the extracted model module; removed unused cache-stage benchmark/portfolio imports and declarations. Removed the research wildcard export of broker helpers. Eliminated duplicate standalone precise Sharpe/PSR/ES/cost implementations in favor of evaluator adapters. Replaced stale combined-project READMEs with installation, migration and handoff documentation. The old combined folder is removed after its files move; reference PDFs are preserved in the relevant project.

Static searches and AST checks cover all resulting Python files. No removed function has a remaining executable reference, no production strategy code calls `.fit` or the research fitting APIs, and no research module imports `horizon_score`. Public feature builders, compatibility facades, callback hooks and reusable model APIs were deliberately preserved rather than deleting dynamically reached functions.

## 5. Confirmed model/data boundary

Research output paths are controlled by `EQUITY_SELECTOR_DATA_DIR`, defaulting to the research folder's `data/`. Strategy output paths use `HORIZON_SCORE_DATA_DIR`, defaulting to the strategy folder's `data/`. Existing users should point these to their existing phase directories or migrate those files explicitly; the refactor never relocates private data automatically.

Research supplies `Final_Test_Results.db`, `Selected_Features.txt`, `Features_Targets_Data.db`, the existing macro database, `Macro_Regimes.txt`, and `confirmed_models/<identity>.pkl` plus JSON manifests. Phase-staged metadata and regime files take precedence over live research copies, protecting frozen selections from later research changes.

The original Horizon and cache stages refitted selected models on different purged historical windows. Those exact TRAIN inputs now produce batched model-only requests when an artifact is absent. `Produce Confirmed Models.py` processes them in research, then the strategy stage resumes and loads the fitted estimator/scaler/encoder. This is an explicit production handoff, not a fitting fallback. No BACKTEST rows enter the requests. Artifact identity covers training content, column/dtype and feature ordering, selected metadata and purge policy; payload checksums detect corruption. Artifacts are trusted local pickle files and should be consumed in the same dependency environment.

## 6. Macro generation/loading and time safety

The resolver locates `macro_data.db` or the existing legacy `Macro_Data.db` spelling. It inspects SQLite schemas for `Date` and `Macro PIT ` columns, ignores unrelated tables, and refuses ambiguous matches unless a table is specified. The repository contains no actual macro database, so the user's production table could not be inspected here; discovery was verified against a real temporary SQLite database containing qualifying and unrelated tables.

`Create Macro Regimes.py` generates the required six-family dictionary and daily scores/counts. The existing macro creation stage also writes these artifacts automatically. Growth/Inflation/Labour/Consumer/Housing require two usable signals and Liquidity one. Neutral bands and the ±0.34 vote threshold are documented in the research README. Missing signals do not invalidate other votes; fewer than the family minimum yields Unknown candidates. Initial valid state initializes immediately; later changes need five consecutive observations and are never backdated. Missing candidates preserve the confirmed state and reset pending confirmation.

Stock/model data attachment uses only existing PIT columns and backward availability-date joins. It never overwrites existing predictors or uses future macro rows. Before adding macro predictors to newly fitted stock models, run macro creation before stock screening (or rerun stock screening after it). The evaluator uses inclusive interval masks, rejects overlaps and assigns uncovered dates Unknown. Loading uses `ast.literal_eval`. Each final report also saves its own literal regime snapshot and `Assumptions.json`, so definitions used for a report remain reviewable.

## 7. Evaluator outputs and conventions

The evaluator returns `summary`, `benchmark`, `portfolio`, `costs`, `regimes`, `regime_episodes`, `rolling`, `yearly`, `monthly`, `drawdowns`, and `daily`, saved as separate CSVs. It consumes daily gross returns/weights and benchmark data without model calls.

- Overall: observations, total return, CAGR, annualised volatility, Sharpe, Sortino, Calmar, maximum/average/current drawdown, drawdown and recovery durations, signed 95% VaR/ES, positive/negative day rates and means, profit factor, best/worst day, skew/excess kurtosis and IID PSR.
- Portfolio: mean/median/min/max holdings, gross exposure and largest position, HHI and effective holdings, mean/median/p95/max/annualised turnover, change-day fraction, entries/exits and absolute weight change.
- Costs: fee fraction/bps, gross/net return, net CAGR, gross/net/2×-cost Sharpe, net drawdown and break-even fee.
- Benchmark: exact-date observation count, strategy/market return/CAGR/volatility/Sharpe/drawdown, correlation, beta, annualised alpha, tracking error, information ratio, active return, upside/downside capture and strategy return conditional on market direction.
- Regimes: each family/state, empty and Unknown states included; exposure-based performance, allocation and benchmark diagnostics. Separate episodes retain continuous-path drawdowns/recoveries. Aggregate disconnected regime drawdowns and Calmar are deliberately undefined.
- Rolling: 63/126/252-observation return, annualised return, volatility, Sharpe, market correlation/beta, average holdings/turnover and effective holdings.
- Calendar: yearly strategy/market performance, year-by-month returns and drawdown episode tables; daily output retains original weights/returns and adds exposures, costs, benchmark and regimes.

252 observations/year is preserved. Turnover is half absolute weight change; fees apply to full traded notional, including initial entry. HHI uses raw weights, with undefined effective holdings when fully in cash. PSR assumes IID returns and requires 30 observations. No benchmark forward-fill occurs. Frozen/GBP launchers reconstruct gross USD returns from the existing fee-adjusted stream before cost evaluation, avoiding double charging; existing GBP account logic and gates are retained separately.

Neighbourhood/nearby-parameter tests, unseen stocks, leave-one-stock-out, best-period removal, DSR/multiple-testing protection and frozen validation remain in place. The new report supplements these rerun-based tests and supplies canonical general metrics.

## 8. Definite bugs and stale assumptions corrected

1. Macro creation ignored its documented credential setting and used a hardcoded credential. It now reads the setting or environment variable; the credential is not copied into the new code.
2. Simulation configurations and validator configurations disagreed, so valid simulation selections could not be reconstructed. Validators now accept the unchanged simulation group set plus the old frozen group names. Horizon-screening groups remain independent, and neither the simulation search space nor portfolio equations were redesigned.
3. Model-phase preparation previously required `Top_Horizon_Scores.txt` before the strategy phase had produced it. Research preparation now only prepares model storage; strategy preparation stages the model inputs for selection and the Horizon artifact for final validation. Horizon output and selection consumers now use the same strategy phase directory.

Baseline tests also had stale fixed thread/stage defaults and an assumption that the initial Horizon search grid was already reduced to its final exhaustive size. Tests now verify configured launcher dispatch, valid group reconstruction and the existing screening/bounded-search contract instead of changing the working defaults to satisfy stale assertions.

## 9. Validation and return comparison

- Both distributions built and installed successfully as editable packages in an isolated Python environment.
- **125 tests passed**, with one pre-existing pandas fragmentation performance warning. Ten new tests cover causal/sparse regimes, database discovery/literal determinism, backward macro alignment, benchmark/date handling, metrics/costs, frozen artifact equality/corruption and phase-input precedence.
- The unmodified baseline ran **113 passed / 2 failed** after installing all declared model dependencies. Its failures were the stale launcher-default assertion and mismatched configuration/profile assumptions described above.
- **103 package modules imported successfully with SQLite/network I/O disabled**; no circular-import failures occurred.
- Ruff's configured checks pass. AST/static audit covers **141 Python files**, with zero removed-function references, zero research-to-strategy dependencies and zero strategy fitting calls.
- Before/after numerical comparison used **12 combinations** of position caps, concentration penalties and fees over **299 realised dates** and eight tickers. All daily return and weight values were **exactly equal** (maximum return difference 0.0).
- The explicit research production/load path produced predictions **exactly equal** to the original fit-and-predict path on a deterministic Ridge fixture. A test replaced the fitting function with a failure after artifact creation, proving the consumer did not refit.
- Every prefix of a regime fixture matched the same dates in the full calculation. Additional checks verified missing-signal holds, no backdating, date-joined benchmarks, no price backfill, disconnected episodes, initial transaction costs and corruption rejection.
- The binary/rename patch passed `git apply --cached --check` against a clean index of the source baseline. The ZIP passed its integrity check.
- Full historical production-data comparison was unavailable, as explained below.

The original repository has no production SQLite databases, selected-model artifacts or historical backtest streams. Therefore a full historical production rerun, ALFRED table inspection and production-result comparison were not possible. These limitations are distinct from the deterministic numerical comparison above; no claim is made that a full live-data research run was completed.

## 10. Remaining architectural issues

- Model production now has an explicit request/resume step for each distinct historical fitting phase. This preserves old chronology and prevents silent refits; future strategy integrations can submit production requests through the research API directly.
- Legacy generic package names (`features`, `targets`, `models`, `main_package`) remain for compatibility. A future major-version namespacing change should be separate from this behavior-preserving refactor.
- Existing large procedural stages and callback/global interfaces remain. Converting all of them into smaller services would be a separate project, with additional production-data regression coverage.
- Macro coverage depends on the user's existing database and availability dates. Undefined statistics (too few observations, zero variance, empty states, cash-only effective holdings, incomplete recovery) are represented explicitly rather than fabricated.
- The historical PDF reports retain their original analytical descriptions; the new READMEs define current paths and operations.
- GitHub publishing remains blocked by invalid local authentication. The patch and ZIP are complete local deliverables, not evidence of a remote update.

## Applying the deliverables

From a checkout of the baseline commit, apply the patch with `git apply --index /path/to/equity-selector-refactor.patch`. It moves/removes the old combined folder, adds both new project folders and updates generated-file ignores. Alternatively, extract the ZIP into a review checkout; the ZIP contains the new folders, while the patch is the authoritative representation of old-folder removals. Install both editable projects using the README commands before running the test suites. No private data or model artifacts are included.
