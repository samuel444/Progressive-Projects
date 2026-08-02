
# Architecture

The notebook is the narrative entry point. Reusable, testable logic lives in the
`src/options_risk_engine` package. SQLite stores reproducible tabular snapshots;
large Monte Carlo arrays stay in compressed files. The optional C++ extension is
isolated behind a Python wrapper, so correctness never depends on compiling it.

## Data flow

1. Download adjusted closes and dividend assumptions.
2. Build returns, realised-volatility features and forward volatility targets.
3. Optionally benchmark forecast models using purged walk-forward validation.
4. Download and snapshot option chains.
5. Price contracts with Black-Scholes under IV and realised/forecast volatility.
6. Calculate implied volatility, Greeks and numerical validation checks.
7. Cross-check with Monte Carlo and terminal profit distributions.
8. Build a reproducible demonstration portfolio and aggregate Greeks.
9. Fully revalue the portfolio under random market shocks.
10. Analyse tail losses, limit breaches and Greek P&L attribution.
11. Persist final tables and generate presentation charts.

## Why keep `archive/original_monolith.py`?

It gives a complete audit trail showing that no original functionality was
silently removed during the refactor. `docs/REFACTOR_NOTES.md` maps every
original function to its new module.
