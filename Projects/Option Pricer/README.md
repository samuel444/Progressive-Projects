
# Options Pricer and Portfolio Risk Engine

A CV-ready quantitative-finance project combining volatility feature
engineering, leakage-aware forecast evaluation, Black-Scholes pricing, implied
volatility, analytical/numerical Greeks, Monte Carlo validation, full-revaluation
stress testing and Greek P&L attribution.

## Start here

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python scripts/init_database.py
jupyter lab notebooks/01_options_pricer_risk_engine.ipynb
```

The notebook is the presentation layer. Reusable functions are imported from the
package under `src/options_risk_engine`. The original monolithic script is kept
verbatim under `archive` for auditability.

## Directory

```text
options-pricer-risk-engine/
├── notebooks/                  narrative project walkthrough
├── src/options_risk_engine/    reusable tested Python package
│   ├── data/                   market data and feature engineering
│   ├── forecasting/            purged walk-forward volatility models
│   ├── pricing/                BS, IV, Greeks, chains and Monte Carlo
│   ├── risk/                   positions, scenarios and attribution
│   └── reporting/              tables and presentation charts
├── sql/                        SQLite schema and analysis queries
├── data/                       raw/interim/processed artefacts and DB
├── cpp/                        optional pybind11 batch acceleration
├── tests/                      pricing, Greeks, risk and DB tests
├── docs/                       architecture and presentation notes
├── outputs/                    generated figures and tables
└── archive/                    untouched original 6,380-line script
```

## Realistic storage choices

SQLite stores relational snapshots and final tables. Monte Carlo path matrices
are deliberately not stored in SQL; use compressed NumPy or Parquet if they need
to be retained. This keeps the database queryable without turning it into a
binary-array store.

## Optional C++

The extension targets batch implied-volatility inversion and high-volume
scenario repricing. The project does not rewrite scikit-learn, `arch` or ordinary
NumPy operations in C++ because those libraries already execute their numerical
kernels in compiled code.

## Testing

```bash
pytest -q
```

## Important interpretation

The random scenarios are transparent stress distributions, not a calibrated
market VaR model. Report the fifth-percentile scenario loss and scenario expected
shortfall rather than presenting them as regulatory VaR/ES.
