
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS project_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    valuation_date TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS equity_prices (
    run_id TEXT NOT NULL,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    close REAL NOT NULL,
    stored_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES project_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_equity_prices_run_ticker_date
    ON equity_prices(run_id, ticker, date);

CREATE TABLE IF NOT EXISTS option_chain_snapshots (
    run_id TEXT NOT NULL,
    snapshot_time TEXT,
    ticker TEXT NOT NULL,
    option_type TEXT NOT NULL,
    contract_symbol TEXT NOT NULL,
    expiry TEXT,
    strike REAL,
    bid REAL,
    ask REAL,
    last_price REAL,
    implied_volatility REAL,
    volume REAL,
    open_interest REAL,
    stored_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES project_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_option_chain_contract
    ON option_chain_snapshots(run_id, contract_symbol);

-- Wide analytical frames are intentionally written by pandas.to_sql into
-- separate artifact tables (volatility_features, option_pricing_results,
-- model_evaluations, portfolio_positions, scenario_* and attribution_*).
-- SQLite is used for reproducible snapshots and queries; large Monte Carlo
-- path matrices should remain in compressed NumPy/Parquet files, not SQL.
