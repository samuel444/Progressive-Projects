
"""Central project configuration.

The original constants are retained so the refactored modules remain close to
Sam's working script.  ``ProjectConfig`` gives the notebook one concise place
to override them for a particular run.
"""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"
DATABASE_PATH = DATA_DIR / "options_risk_engine.db"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"

# Original model settings
ridge_alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
lasso_alphas = [0.0001, 0.001, 0.01, 0.1, 1]
har_features = ["RV20", "RV60", "RV252"]
pca_variance = 0.95
window = 30
step = 30
min_train_rows = 252
garch_lookback = 1250

GREEK_VALIDATION_TOLERANCES = {
    "Delta": {"atol": 1e-5, "rtol": 1e-4},
    "Gamma": {"atol": 1e-6, "rtol": 1e-3},
    "Vega": {"atol": 1e-5, "rtol": 1e-3},
    "Theta": {"atol": 1e-5, "rtol": 1e-3},
    "Rho": {"atol": 1e-5, "rtol": 1e-3},
}

# Pricing and filtering configuration retained from the original script
RISK_FREE_RATE = 0.0375
FORECAST_VOL_LOOKBACK = 1260
BUY_EDGE = 0.05
PARITY_TOLERANCE = 1e-6
MONTE_CARLO_SIMULATIONS = 10_000
MAX_SPREAD_PCT = 0.15
MIN_ASK = 0.50
MIN_OPEN_INTEREST = 100
MIN_VOLUME = 10
MIN_MONEYNESS = 0.85
MAX_MONEYNESS = 1.15
PLOT_MONTE_CARLO = True


@dataclass(frozen=True)
class ProjectConfig:
    """Reproducible run settings used by the notebook."""

    symbols: tuple[str, ...] = (
        "AAPL", "MSFT", "META", "WMT", "GOOGL", "AMZN", "HCA"
    )
    target_dte: int = 45
    history_period: str = "5y"
    history_interval: str = "1d"
    risk_free_rate: float = RISK_FREE_RATE
    forecast_lookback: int = FORECAST_VOL_LOOKBACK
    buy_edge: float = BUY_EDGE
    parity_tolerance: float = PARITY_TOLERANCE
    monte_carlo_simulations: int = MONTE_CARLO_SIMULATIONS
    scenario_count: int = 1_000
    max_days_forward: int = 30
    random_seed: int = 42
    max_portfolio_loss: float = 100_000.0
    max_ticker_loss: float = 25_000.0
    run_forecast_benchmarks: bool = False
    run_monte_carlo: bool = True
    plot_monte_carlo: bool = False
    database_path: Path = field(default=DATABASE_PATH)


def ensure_project_directories() -> None:
    for path in (
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        FIGURE_DIR, TABLE_DIR, LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
