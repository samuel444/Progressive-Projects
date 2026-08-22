# Equity Selector

A multi-stage quantitative equity research pipeline for turning screened market data and target-specific predictive models into fully specified portfolio strategies for detailed historical evaluation.

The project is designed around a strict separation between **data suitability**, **model predictability**, **portfolio usefulness**, and **strategy behaviour**. It uses chronological splits throughout, keeps target-blind screening separate from target-aware selection, validates models with purged walk-forward testing, scores surviving targets by quality and horizon, and then evaluates complete portfolio configurations across different stock universes before running detailed precise backtests.

## Project Scope

The coded research pipeline runs from raw market data through to detailed strategy diagnostics:

1. **Create and screen data**
2. **Select target-specific features**
3. **Search and validate predictive models**
4. **Run a final unseen model test**
5. **Convert surviving targets into portfolio metadata**
6. **Research horizon-score configurations**
7. **Build the strategy backtest database**
8. **Run broad portfolio simulations**
9. **Select candidate configurations**
10. **Run precise backtests with detailed risk and anomaly analysis**

At the end of the coded research stage, the strategy settings are fixed. Further candidate review, comparison and selection are performed manually using the stored simulation and precise-backtest results. The existing pipeline can then be rerun with fixed settings, alternative database inputs, stock universes and later unseen date ranges.

## Research Principles

The pipeline is built around several rules intended to reduce leakage and overfitting:

- **Chronological data splits** are used throughout.
- **Target purging** is applied where forward targets overlap validation periods.
- **Stock and feature screening is target-blind** before target-specific selection begins.
- **Model selection and final model testing are separate stages.**
- **Model quality and portfolio usefulness are treated as different questions.**
- **Predictions are cached** so repeated portfolio searches do not repeatedly refit the same models.
- **Search spaces are reduced progressively** before exhaustive evaluation.
- **Candidate strategies are selected from groups of results**, rather than automatically taking the single highest-Sharpe backtest.
- A later period is kept available for a **final untouched strategy holdout**.

## Data and Research Universes

The main daily research pipeline uses adjusted Yahoo Finance OHLCV data and the S&P 500 (`^GSPC`) as the market benchmark.

The project supports multiple liquidity and composition universes, including:

- High Liquidity 30
- Medium Liquidity 30
- Lower Liquidity 30
- Sector Spread 30
- Liquidity Barbell 30
- Institutional Liquidity 60
- Medium Small Liquidity 60
- Medium Large Liquidity 60
- All Liquidity 90
- Intraday high- and medium-liquidity variants

The supplied project report uses the **Liquidity Barbell 30** as the active daily example.

## Quantitative Definition Layer

Feature and target mathematics are maintained separately in the project's quantitative reference.

The current reference contains:

- **114 feature functions**
- **26 target functions**

Feature families include returns, momentum, volatility, range volatility, trend, moving averages, drawdown, return distributions, tail risk, volume, liquidity, OHLC structure, market-relative behaviour, beta, correlation, residual behaviour, technical transforms, regimes, interactions, composites, cross-sectional features, breadth and dispersion.

Target families include forward returns, volatility, direction, barriers, excursions, drawdowns, risk-adjusted outcomes and cross-sectional rankings.

## Pipeline

### 1. `Data_Creation_Screening.py`

Builds the screened research dataset.

Main responsibilities:

- download adjusted OHLCV data and `^GSPC`;
- screen securities for size, missingness, invalid values and continuity;
- build individual, market-relative and cross-stock features;
- apply target-blind feature screening;
- generate target families;
- create chronological train / validation / test partitions;
- apply target-specific feature screens using training data only.

Main outputs:

- `Features_Targets_Data.db`
- `Selected_Features.txt`

### 2. `Intraday Conversion.py`

Optional stage for intraday stock types.

It converts lookbacks and forward horizons into session-safe rules, removes unavailable rows around session boundaries and updates the selected-feature mapping.

### 3. `Model_Fitting.py`

Runs target-specific model development using chronological walk-forward validation.

Main features:

- loads only the current target and its selected features;
- reuses completed fold/configuration work;
- applies target purging;
- runs progressive configuration racing;
- promotes finalists;
- completes all required validation folds;
- produces target-aware leaderboards and `Testing Eligible` decisions.

Main output:

- `Validation_Model_Fits/<STOCK_TYPE>.db`

### 4. `Best_Model_Test.py`

Performs the final chronological model test.

For each surviving target, the selected model is fitted on the available development data, with the target purge applied, and evaluated once on the unseen model-test block.

It then stores:

- final-test metrics;
- Predictability Score;
- Fine-grained Portfolio Target Type;
- prediction Horizon;
- cross-target Quality Score.

Main output:

- `Final_Test_Results.db`

### 5. `Horizon Score Backtests.py`

Researches how strongly different target types and prediction horizons should contribute to the portfolio.

The search operates over a catalogue of fine-grained Portfolio Target Types and canonical horizon keys. It reduces the search space using:

1. one-dimensional sensitivity checks;
2. random-context screens;
3. exhaustive testing of the remaining combinations.

Predictions are generated once and reused while the horizon-score metadata changes.

The benchmark-relative Backtest Quality score combines changes in:

- Sharpe;
- total return;
- maximum drawdown;
- average drawdown.

The strongest non-negative horizon configurations are retained for the broad portfolio simulation stage.

### 6. `Backtest Database.py`

Creates a self-contained strategy-research cache for selected training and backtest universes and date ranges.

It rebuilds only the features and targets required by the selected models and includes a three-year raw-data warm-up for rolling features.

Main output:

- `Backtest_Features_Targets.db`

### 7. `Backtest Simulations.py`

Runs the broad portfolio search.

Each target prediction is weighted using its:

- model signal;
- tested Horizon Score;
- Quality Score;
- Fine-grained Portfolio Target Type value.

Targets belonging to the same Portfolio Target Type are combined before the stock-level score is passed to the portfolio optimiser.

The optimiser maximises the stock score subject to:

- full investment;
- maximum position weight;
- quadratic concentration penalty.

The broad sweep tests combinations of:

- training/backtest universe;
- fine-grained target-type values;
- retained horizon-score configurations;
- maximum position weight;
- concentration penalty;
- rebalance frequency.

Main outputs:

- `Automated_Backtest_Results.csv`
- `Automated_Backtest_Results.db`

### 8. Candidate Strategy Selection

The broad simulation stage is used to build a shortlist for deeper analysis rather than simply selecting the single highest-Sharpe row.

Candidate review considers:

- benchmark-relative return and Sharpe;
- maximum and average drawdown;
- stability across nearby parameter settings;
- stability across horizon-score configurations;
- concentration and effective holdings;
- turnover and rebalance behaviour;
- position and trade size relative to ADV;
- dependence on individual stocks or isolated days;
- transfer to relevant unseen securities or universe variants;
- deliberately weak or unusual cases that help expose failure modes.

This stage is primarily **manual analysis**.

### 9. `Precise Backtest.py`

Reconstructs one exact candidate strategy and produces detailed diagnostics.

The selected configuration fixes:

- stock universe;
- Fine-grained Portfolio Target Type values;
- horizon-score configuration;
- maximum weight;
- concentration penalty;
- rebalance frequency;
- trading fee.

The precise backtest produces:

- daily strategy and benchmark returns;
- active returns;
- rolling volatility;
- rolling Sharpe;
- rolling beta and correlation;
- turnover;
- held weights;
- concentration measures;
- effective holdings;
- drawdown history;
- ticker-level P&L contributions;
- position and trade size as a fraction of ADV;
- best and worst days;
- complete drawdown episodes;
- anomaly reports;
- yearly performance;
- VaR and Expected Shortfall;
- Sortino and Calmar ratios;
- tracking error and information ratio.

Main outputs:

- `Single_Backtest_Anomalies.csv`
- `Single_Backtest_Anomaly_Report.db`

## Portfolio Scoring

For target \(j\) and stock \(i\), the model contribution uses the target signal \(S_{ij}\), Horizon Score \(H_j\) and Quality Score \(Q_j\):

\[
C_{ij} = S_{ij} H_j Q_j
\]

Targets belonging to the same Fine-grained Portfolio Target Type are combined using a horizon-quality weighted mean.

Each target type then receives a configured value \(V_g\), allowing desirable target types to contribute positively and risk-related target types to penalise the stock score.

The final cross-sectional stock score is passed to a constrained portfolio optimiser with a quadratic concentration penalty.

## Stored Research Outputs

The main persistent research files are:

| Output | Purpose |
|---|---|
| `Features_Targets_Data.db` | Screened feature and target panels |
| `Selected_Features.txt` | Target-specific allowed feature sets |
| `Validation_Model_Fits/<STOCK_TYPE>.db` | Walk-forward model search and validation history |
| `Final_Test_Results.db` | Final model-test results and portfolio metadata |
| `Backtest_Features_Targets.db` | Self-contained strategy research cache |
| `Automated_Backtest_Results.db` | Broad simulation results and grouped summaries |
| `Single_Backtest_Anomaly_Report.db` | Detailed precise-backtest diagnostics |

## Current Project Boundary

The main research code ends after `Precise Backtest.py`.

From this point, the pipeline is used with **fixed strategy settings**. Candidate review and strategy selection are manual: broad simulation results are compared, promising configurations are rerun through the precise backtest, and the detailed risk and behaviour reports are used to decide which strategies deserve further testing.

The next research step is to freeze the selected candidate configurations and rerun the existing process on later, previously untouched data for final strategy validation. Changes prompted by those results should be treated as new research rather than tuning the original holdout.

## Intended Research Sequence

```text
Quantitative definitions
        ↓
Data creation and stock screening
        ↓
Target-blind feature screening
        ↓
Target-specific feature screening
        ↓
Walk-forward model search
        ↓
Final model test
        ↓
Predictability + Quality + Portfolio Target Type
        ↓
Horizon-score research
        ↓
Backtest database
        ↓
Broad portfolio simulations
        ↓
Manual candidate review
        ↓
Precise backtests
        ↓
Manual strategy selection
        ↓
Frozen strategy / later untouched holdout
```

## What the Project Demonstrates

This project demonstrates an end-to-end quantitative research workflow covering:

- financial feature and target engineering;
- leakage-aware time-series validation;
- model selection across multiple statistical target types;
- resumable and computationally efficient parameter search;
- cross-sectional signal construction;
- model-quality and horizon-aware portfolio scoring;
- constrained numerical portfolio optimisation;
- large-scale backtest simulation;
- benchmark-relative evaluation;
- portfolio risk and concentration analysis;
- liquidity and capacity analysis;
- detailed forensic analysis of strategy behaviour.

## Status

**Coded research pipeline: complete through precise backtesting.**

The current stage is manual review of broad simulation and precise-backtest results, followed by freezing the strongest candidate strategies for testing on later untouched data.
