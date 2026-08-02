# Options Pricer & Portfolio Risk Engine

A quantitative finance project that builds an end-to-end options pricing and portfolio risk engine.

The project combines market data collection, volatility forecasting, option pricing, implied volatility estimation, portfolio risk analysis, stress testing, and Monte Carlo simulation into a reusable Python package with a notebook walkthrough.

Designed as a portfolio project demonstrating quantitative finance, statistics, machine learning, software engineering, and scientific computing.

---

# Features

## Market Data

- Download historical prices and live option chains
- SQLite database for storing market snapshots
- Feature engineering pipeline
- Rolling volatility calculations
- Configurable lookback windows

## Volatility Forecasting

- Historical volatility
- GARCH volatility models
- Machine learning volatility prediction
- Walk-forward validation
- Purged cross-validation
- Model comparison metrics

## Option Pricing

- Black-Scholes pricing
- Implied volatility solver
- European call and put pricing
- Complete option chain pricing
- Fair value vs market price comparison

## Greeks

Analytical and numerical calculation of

- Delta
- Gamma
- Vega
- Theta
- Rho

with configurable finite-difference step sizes.

## Monte Carlo Simulation

- Geometric Brownian Motion path generation
- Configurable number of simulation paths
- Configurable time steps
- Monte Carlo pricing validation
- Distribution visualisation

## Portfolio Risk

- Multi-position portfolios
- Portfolio aggregation
- Portfolio Greeks
- Stress testing
- Full revaluation
- Greek P&L attribution
- Scenario analysis

## Reporting

- Pricing tables
- Risk summaries
- Portfolio exposure reports
- Scenario visualisations
- Performance charts

---

# Configurable Settings

The project is designed to be highly configurable. Most parameters can be changed from the notebook or configuration section without modifying the underlying implementation.

Examples include:

- Stock ticker
- Option expiry
- Strike selection
- Option type
- Risk-free rate
- Dividend yield
- Volatility model
- Rolling window length
- Forecast horizon
- Train/test split
- Walk-forward window size
- Purge length
- Monte Carlo simulation paths
- Monte Carlo time steps
- Random seed
- Finite-difference epsilon
- Stress-test shock sizes
- Portfolio positions

---

# Project Structure

```text
options-pricer-risk-engine/
│
├── notebooks/
│   └── Complete project walkthrough
│
├── src/options_risk_engine/
│   ├── data/
│   │   ├── Database utilities
│   │   ├── Market downloads
│   │   └── Feature engineering
│   │
│   ├── forecasting/
│   │   ├── Historical volatility
│   │   ├── GARCH models
│   │   ├── Machine learning
│   │   └── Validation
│   │
│   ├── pricing/
│   │   ├── Black-Scholes
│   │   ├── Implied volatility
│   │   ├── Greeks
│   │   ├── Option chains
│   │   └── Monte Carlo
│   │
│   ├── risk/
│   │   ├── Portfolio construction
│   │   ├── Stress testing
│   │   ├── Scenario analysis
│   │   └── Greek attribution
│   │
│   └── reporting/
│       ├── Tables
│       └── Charts
│
├── sql/
│   ├── Database schema
│   └── SQL analysis queries
│
├── data/
│   ├── Raw market data
│   ├── Processed datasets
│   └── SQLite database
│
├── cpp/
│   └── Optional pybind11 acceleration
│
├── tests/
│   └── Unit tests
│
├── outputs/
│   └── Generated figures and reports
│
├── docs/
│   └── Project documentation
│
└── archive/
    └── Original monolithic implementation
```

---

# Getting Started

## Clone the repository

```bash
git clone <repository-url>
cd options-pricer-risk-engine
```

## Create a virtual environment

```bash
python -m venv .venv
```

Activate it

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

## Install dependencies

```bash
python -m pip install -e ".[dev]"
```

## Initialise the database

```bash
python scripts/init_database.py
```

## Launch the notebook

```bash
jupyter lab notebooks/01_options_pricer_risk_engine.ipynb
```

The notebook provides a complete walkthrough of the project while all reusable functionality is imported from the package under `src/options_risk_engine`.

---

# Design Decisions

## SQLite

SQLite stores market data, processed features, option chains and pricing results.

Large Monte Carlo path matrices are intentionally stored as compressed NumPy arrays or Parquet files rather than SQL tables, keeping the database lightweight, efficient and easy to query.

## Optional C++

An optional pybind11 extension can accelerate computational bottlenecks such as

- Batch implied-volatility inversion
- Portfolio repricing
- Large-scale scenario analysis

Standard NumPy, SciPy and scikit-learn operations remain in Python because these libraries already execute their numerical kernels in highly optimised compiled code.

---

# Testing

Run the full test suite with

```bash
pytest -q
```

Tests cover

- Black-Scholes pricing
- Implied volatility
- Analytical Greeks
- Numerical Greeks
- Portfolio aggregation
- Stress testing
- Scenario analysis
- Database operations

---

# Future Improvements

Potential future extensions include

- Local volatility models
- Heston stochastic volatility
- SABR calibration
- American option pricing
- Binomial and trinomial trees
- Historical Value at Risk
- Expected Shortfall
- PCA factor risk decomposition
- Delta hedging simulation
- Parallel Monte Carlo
- GPU acceleration

---

This project is intended as a portfolio demonstration of quantitative finance, machine learning, numerical methods, statistical modelling, and software engineering practices suitable for quantitative developer, quantitative analyst, and data science roles.
