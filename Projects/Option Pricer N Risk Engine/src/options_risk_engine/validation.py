
"""Cross-stage validation checks for the notebook pipeline."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_option_universe(large_table: pd.DataFrame) -> dict[str, object]:
    required = {
        "Ticker", "Option_Type", "contractSymbol", "strike", "MarketMid",
        "BS_ForeV", "Delta", "Gamma", "Vega", "Theta", "Rho",
    }
    missing = required.difference(large_table.columns)
    if missing:
        raise KeyError(f"Option universe missing columns: {sorted(missing)}")

    duplicate_contracts = int(large_table["contractSymbol"].duplicated().sum())
    invalid_quotes = int((~large_table["Quote Valid"].fillna(False)).sum()) if "Quote Valid" in large_table else 0
    summary = {
        "rows": len(large_table),
        "tickers": int(large_table["Ticker"].nunique()),
        "duplicate_contract_symbols": duplicate_contracts,
        "invalid_quotes": invalid_quotes,
    }
    logger.info("Option-universe validation: %s", summary)
    return summary


def validate_risk_engine(
    results: pd.DataFrame,
    scenario_portfolio: pd.DataFrame,
    scenario_attribution: pd.DataFrame,
    base_scenario_id: str = "BASE",
    atol: float = 1e-6,
) -> dict[str, bool]:
    base_rows = scenario_portfolio.loc[
        scenario_portfolio["Scenario ID"].eq(base_scenario_id)
    ]
    if len(base_rows) != 1:
        raise AssertionError("Exactly one BASE scenario is required")

    base_zero = bool(np.isclose(base_rows["Portfolio_PnL"].iloc[0], 0.0, atol=atol))

    position_totals = results.groupby("Scenario ID")["Scenario PnL"].sum().sort_index()
    portfolio_totals = scenario_portfolio.set_index("Scenario ID")["Portfolio_PnL"].sort_index()
    aggregate_match = bool(np.allclose(position_totals, portfolio_totals, atol=atol, equal_nan=False))

    attribution_identity = bool(np.allclose(
        scenario_attribution["Full_Revaluation_PnL"],
        scenario_attribution["Approximate_PnL"] + scenario_attribution["Residual_PnL"],
        atol=atol,
        equal_nan=False,
    ))

    no_missing_prices = bool(
        results["Scenario Price"].notna().all()
    )

    # Normalise Calls, Puts and Shares to call, put and share.
    position_types = (
        results["Option Type"]
        .astype(str)
        .str.lower()
        .str.rstrip("s")
    )

    option_mask = position_types.isin(
        [
            "call",
            "put",
        ]
    )

    share_mask = position_types.eq("share")

    unknown_type_mask = ~(
        option_mask
        | share_mask
    )

    # Volatility and time-to-expiry apply only to options.
    # They are intentionally missing for ordinary share positions.
    option_volatility = results.loc[
        option_mask,
        "Shocked Volatility",
    ]

    option_time = results.loc[
        option_mask,
        "Shocked Time",
    ]

    nonnegative_volatility = bool(
        option_volatility.notna().all()
        and option_volatility.gt(0).all()
    )

    nonnegative_time = bool(
        option_time.notna().all()
        and option_time.ge(0).all()
    )

    # Shares should be repriced directly to the shocked stock price.
    share_prices_match_spot = bool(
        np.allclose(
            results.loc[
                share_mask,
                "Scenario Price",
            ],
            results.loc[
                share_mask,
                "Shocked Spot",
            ],
            atol=atol,
            equal_nan=False,
        )
    )

    known_position_types = bool(
        not unknown_type_mask.any()
    )

    checks = {
        "base_pnl_zero": base_zero,
        "position_totals_match_portfolio": aggregate_match,
        "attribution_identity": attribution_identity,
        "no_missing_scenario_prices": no_missing_prices,

        # Option-only checks
        "option_shocked_volatility_positive": (
            nonnegative_volatility
        ),
        "option_shocked_time_nonnegative": (
            nonnegative_time
        ),

        # Share-specific checks
        "share_prices_match_shocked_spot": (
            share_prices_match_spot
        ),

        # General classification check
        "known_position_types": known_position_types,
    }
    
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Risk-engine validation failed: {failed}")
    logger.info("All risk-engine validation checks passed")
    return checks
