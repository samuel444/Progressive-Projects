from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


DEFAULT_INCLUDED_ACTIONS = (
    "Buy",
    "Positive Edge",
    "Do Not Buy",
)


REQUIRED_CONTRACT_COLUMNS = [
    # Contract identity
    "contractSymbol",
    "Ticker",
    "Option_Type",
    "strike",
    "Expiry",

    # Current market state
    "Current Stock Price",
    "Time to Expiry",
    "Calendar DTE",
    "bid",
    "ask",
    "MarketMid",

    # Volatility used by the risk engine
    "Greek Volatility",
    "Greek Volatility Source",

    # Unit Greeks
    "Delta",
    "Gamma",
    "Vega",
    "Theta",
    "Rho",

    # Quote and recommendation information
    "SpreadPct",
    "volume",
    "openInterest",
    "Quote Valid",
    "Recommendation Eligible",
    "Initial Recommended Action",

    # Forecast-based valuation
    "BS_ForeV",
    "BS_ForeV AskEdge",
]


OPTIONAL_REPORTING_COLUMNS = [
    "Quote Issue",
    "Pricing Bounds Valid",
    "Parity Valid",
    "IV_Mid",
    "IV Used",
    "ForeV Used",
    "Volatility Spread",
    "Moneyness",
    "Intrinsic Value",
    "Time Value",
    "Break-Even Price",
    "MC_ForeV",
    "MC_SE",
    "MC AskEdge",
    "Probability of Profit",
    "Expected Profit Return",
]


def prepare_standardised_contract_positions(
    option_universe: pd.DataFrame,
    risk_free_rates: Mapping[str, float],
    dividend_yields: Mapping[str, float],
    included_actions: Sequence[str] = DEFAULT_INCLUDED_ACTIONS,
) -> pd.DataFrame:
    """
    Create comparable long-one-contract positions for recommendation analysis.

    Every retained contract is treated as:
        - long;
        - one contract;
        - multiplier of 100;
        - purchased at the current ask.

    The function does not add share positions or randomly assign quantities.
    """

    logger.info(
        "Preparing standardised contract analysis from %d option rows",
        len(option_universe),
    )

    missing_columns = set(
        REQUIRED_CONTRACT_COLUMNS
    ).difference(option_universe.columns)

    if missing_columns:
        raise KeyError(
            "Cannot create standardised contract positions; "
            "missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    included_actions = tuple(included_actions)

    # Only retain optional fields that currently exist.
    reporting_columns = [
        column
        for column in OPTIONAL_REPORTING_COLUMNS
        if column in option_universe.columns
    ]

    selected_columns = (
        REQUIRED_CONTRACT_COLUMNS
        + reporting_columns
    )

    analysis_mask = (
        option_universe[
            "Initial Recommended Action"
        ].isin(included_actions)
        & option_universe[
            "Recommendation Eligible"
        ].fillna(False)
        & option_universe[
            "Quote Valid"
        ].fillna(False)
        & option_universe[
            "ask"
        ].notna()
        & option_universe[
            "ask"
        ].gt(0)
    )

    positions = (
        option_universe.loc[
            analysis_mask,
            selected_columns,
        ]
        .copy()
        .reset_index(drop=True)
    )

    if positions.empty:
        raise ValueError(
            "No eligible contracts were available for "
            "standardised analysis."
        )

    if positions[
        "contractSymbol"
    ].duplicated().any():
        duplicated_contracts = (
            positions.loc[
                positions[
                    "contractSymbol"
                ].duplicated(keep=False),
                "contractSymbol",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Duplicate contract symbols were found: "
            + ", ".join(
                map(str, duplicated_contracts[:10])
            )
        )

    pricing_input_columns = [
        "contractSymbol",
        "Ticker",
        "Option_Type",
        "strike",
        "Current Stock Price",
        "Time to Expiry",
        "ask",
        "Greek Volatility",
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
        "BS_ForeV",
    ]

    complete_pricing_inputs = (
        positions[
            pricing_input_columns
        ]
        .notna()
        .all(axis=1)
    )

    removed_incomplete = int(
        (~complete_pricing_inputs).sum()
    )

    positions = (
        positions.loc[
            complete_pricing_inputs
        ]
        .copy()
        .reset_index(drop=True)
    )

    if positions.empty:
        raise ValueError(
            "All contracts were removed because one or more "
            "pricing inputs were missing."
        )

    if removed_incomplete:
        logger.warning(
            "Removed %d contracts with incomplete pricing inputs",
            removed_incomplete,
        )

    # Attach ticker-level assumptions.
    positions["Risk Free Rate"] = (
        positions["Ticker"].map(
            risk_free_rates
        )
    )

    positions["Dividend Yield"] = (
        positions["Ticker"].map(
            dividend_yields
        )
    )

    missing_rate_tickers = (
        positions.loc[
            positions[
                "Risk Free Rate"
            ].isna(),
            "Ticker",
        ]
        .drop_duplicates()
        .tolist()
    )

    if missing_rate_tickers:
        raise ValueError(
            "Risk-free rates are missing for: "
            + ", ".join(
                map(str, missing_rate_tickers)
            )
        )

    missing_dividend_tickers = (
        positions.loc[
            positions[
                "Dividend Yield"
            ].isna(),
            "Ticker",
        ]
        .drop_duplicates()
        .tolist()
    )

    if missing_dividend_tickers:
        raise ValueError(
            "Dividend yields are missing for: "
            + ", ".join(
                map(str, missing_dividend_tickers)
            )
        )

    # Standardised position assumptions.
    positions["Side"] = "long"
    positions["Direction"] = 1
    positions["Quantity"] = 1
    positions["Multiplier"] = 100

    # Buying one contract means paying the current ask.
    positions["Entry Price"] = (
        positions["ask"]
    )

    # Current Mark is also set to ask so the existing scenario
    # engine measures its base model relative to the purchase price.
    positions["Current Mark"] = (
        positions["ask"]
    )

    positions["Position Scale"] = 100

    # Premium paid for one standard equity option contract.
    positions["Premium Paid"] = (
        positions["Entry Price"]
        * positions["Multiplier"]
    )

    positions["Entry Premium Value"] = (
        positions["Premium Paid"]
    )

    # At entry, mark-to-entry P&L is zero.
    positions["Position PnL"] = 0.0

    positions["Signed Market Value"] = (
        positions["Premium Paid"]
    )

    # Forecast-based edge remains separate from the market-IV
    # scenario valuation.
    positions["Forecast Edge Per Share"] = (
        positions["BS_ForeV"]
        - positions["Entry Price"]
    )

    positions["Forecast Edge Per Contract"] = (
        positions["Forecast Edge Per Share"]
        * positions["Multiplier"]
    )

    positions["Forecast Edge Return"] = (
        positions["Forecast Edge Per Share"]
        / positions["Entry Price"]
    )

    # Position-level Greeks for one long contract.
    positions["Position Delta"] = (
        positions["Delta"]
        * positions["Position Scale"]
    )

    positions["Position Gamma"] = (
        positions["Gamma"]
        * positions["Position Scale"]
    )

    positions["Position Vega"] = (
        positions["Vega"]
        * positions["Position Scale"]
    )

    positions["Position Theta"] = (
        positions["Theta"]
        * positions["Position Scale"]
    )

    positions["Position Rho"] = (
        positions["Rho"]
        * positions["Position Scale"]
    )

    positions = (
        positions
        .sort_values(
            [
                "Ticker",
                "Option_Type",
                "strike",
            ]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Prepared %d standardised long-one-contract positions",
        len(positions),
    )

    return positions


def validate_standardised_contract_positions(
    positions: pd.DataFrame,
    atol: float = 1e-10,
) -> dict[str, bool]:
    """Validate the assumptions used for contract comparison."""

    pricing_columns = [
        "Current Stock Price",
        "strike",
        "Time to Expiry",
        "Risk Free Rate",
        "Dividend Yield",
        "Greek Volatility",
        "Entry Price",
        "BS_ForeV",
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
    ]

    position_types = (
        positions["Option_Type"]
        .astype(str)
        .str.lower()
        .str.rstrip("s")
    )

    checks = {
        "contracts_available": (
            len(positions) > 0
        ),
        "unique_contract_symbols": (
            not positions[
                "contractSymbol"
            ].duplicated().any()
        ),
        "all_positions_long": (
            positions[
                "Direction"
            ].eq(1).all()
        ),
        "one_contract_each": (
            positions[
                "Quantity"
            ].eq(1).all()
        ),
        "standard_multiplier": (
            positions[
                "Multiplier"
            ].eq(100).all()
        ),
        "entry_equals_ask": bool(
            np.allclose(
                positions[
                    "Entry Price"
                ],
                positions[
                    "ask"
                ],
                atol=atol,
                equal_nan=False,
            )
        ),
        "positive_premium": (
            positions[
                "Premium Paid"
            ].gt(0).all()
        ),
        "complete_pricing_inputs": (
            positions[
                pricing_columns
            ].notna().all().all()
        ),
        "recognised_option_types": (
            position_types.isin(
                [
                    "call",
                    "put",
                ]
            ).all()
        ),
        "forecast_edge_matches_existing_column": bool(
            np.allclose(
                positions[
                    "Forecast Edge Return"
                ],
                positions[
                    "BS_ForeV AskEdge"
                ],
                atol=atol,
                equal_nan=True,
            )
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise AssertionError(
            "Standardised contract validation failed: "
            f"{failed}"
        )

    logger.info(
        "All standardised contract validation checks passed"
    )

    return checks


def attach_standardised_purchase_pnl(
    scenario_results: pd.DataFrame,
    positions: pd.DataFrame,
    base_scenario_id: str = "BASE",
    atol: float = 1e-6,
) -> pd.DataFrame:
    """
    Convert model-movement scenario P&L into P&L for buying at the ask.

    Three distinct values are retained:

    Forecast Edge P&L:
        Forecast-volatility value minus ask.

    Base Mark-to-Ask P&L:
        Market-IV/risk-model base value minus ask.

    Purchase Scenario P&L:
        Shocked scenario value minus ask.
    """

    required_result_columns = {
        "Scenario ID",
        "Contract Symbol",
        "Base Model Price",
        "Scenario Price",
        "Market-to-Model PnL",
        "Scenario PnL",
    }

    missing_result_columns = (
        required_result_columns
        .difference(scenario_results.columns)
    )

    if missing_result_columns:
        raise KeyError(
            "Scenario results are missing columns: "
            + ", ".join(
                sorted(missing_result_columns)
            )
        )

    metadata_candidates = [
        "contractSymbol",
        "Entry Price",
        "Premium Paid",
        "Position Scale",
        "Initial Recommended Action",
        "BS_ForeV",
        "BS_ForeV AskEdge",
        "Forecast Edge Per Share",
        "Forecast Edge Per Contract",
        "Forecast Edge Return",
        "bid",
        "ask",
        "MarketMid",
        "SpreadPct",
        "volume",
        "openInterest",
        "IV_Mid",
        "IV Used",
        "ForeV Used",
        "Volatility Spread",
        "Greek Volatility",
        "Greek Volatility Source",
        "Moneyness",
        "Break-Even Price",
        "Probability of Profit",
        "Expected Profit Return",
    ]

    metadata_columns = [
        column
        for column in metadata_candidates
        if column in positions.columns
    ]

    metadata = (
        positions[
            metadata_columns
        ]
        .copy()
        .rename(
            columns={
                "contractSymbol": (
                    "Contract Symbol"
                )
            }
        )
    )

    result = scenario_results.merge(
        metadata,
        on="Contract Symbol",
        how="left",
        validate="many_to_one",
    )

    missing_entries = result[
        "Entry Price"
    ].isna()

    if missing_entries.any():
        missing_contracts = (
            result.loc[
                missing_entries,
                "Contract Symbol",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Entry information is missing for: "
            + ", ".join(
                map(str, missing_contracts[:10])
            )
        )

    # Existing Market-to-Model P&L is now the difference
    # between the risk-model base price and the purchase ask.
    result["Base Mark-to-Ask PnL"] = (
        result["Market-to-Model PnL"]
    )

    # Rename conceptually without removing the engine's
    # original Scenario PnL column.
    result["Scenario Movement PnL"] = (
        result["Scenario PnL"]
    )

    # Actual modelled P&L from paying the ask and then
    # repricing under a scenario.
    result["Purchase Scenario PnL"] = (
        (
            result["Scenario Price"]
            - result["Entry Price"]
        )
        * result["Position Scale"]
    )

    result["Purchase Scenario Return"] = (
        result["Purchase Scenario PnL"]
        / result["Premium Paid"]
    )

    result["Scenario Movement Return"] = (
        result["Scenario Movement PnL"]
        / result["Premium Paid"]
    )

    # Exact accounting relationship:
    #
    # scenario price - ask
    # =
    # (base model price - ask)
    # +
    # (scenario price - base model price)
    result["Purchase PnL Identity Error"] = (
        result["Purchase Scenario PnL"]
        - (
            result["Base Mark-to-Ask PnL"]
            + result["Scenario Movement PnL"]
        )
    )

    identity_passed = bool(
        np.allclose(
            result[
                "Purchase PnL Identity Error"
            ],
            0.0,
            atol=atol,
            equal_nan=False,
        )
    )

    if not identity_passed:
        raise AssertionError(
            "Purchase P&L accounting identity failed."
        )

    base_rows = result.loc[
        result[
            "Scenario ID"
        ].eq(base_scenario_id)
    ]

    if len(base_rows) != len(positions):
        raise AssertionError(
            "The BASE scenario does not contain exactly one "
            "row for every standardised contract."
        )

    base_movement_zero = bool(
        np.allclose(
            base_rows[
                "Scenario Movement PnL"
            ],
            0.0,
            atol=atol,
            equal_nan=False,
        )
    )

    if not base_movement_zero:
        raise AssertionError(
            "BASE scenario movement P&L is not zero."
        )

    base_purchase_identity = bool(
        np.allclose(
            base_rows[
                "Purchase Scenario PnL"
            ],
            base_rows[
                "Base Mark-to-Ask PnL"
            ],
            atol=atol,
            equal_nan=False,
        )
    )

    if not base_purchase_identity:
        raise AssertionError(
            "BASE purchase P&L does not equal "
            "base mark-to-ask P&L."
        )

    logger.info(
        "Attached purchase-at-ask P&L to %d scenario rows",
        len(result),
    )

    return result


def _lower_tail_mean(
    values: pd.Series,
    tail_probability: float,
) -> float:
    """
    Return the mean of observations at or below the selected quantile.

    For a 5% tail, this is the mean P&L of the worst approximately
    5% of scenarios.
    """

    clean_values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
    )

    if clean_values.empty:
        return np.nan

    threshold = clean_values.quantile(
        tail_probability
    )

    tail_values = clean_values.loc[
        clean_values <= threshold
    ]

    if tail_values.empty:
        return np.nan

    return float(tail_values.mean())


def create_contract_scenario_summary(
    scenario_results: pd.DataFrame,
    positions: pd.DataFrame,
    base_scenario_id: str = "BASE",
    tail_probability: float = 0.05,
) -> pd.DataFrame:
    """
    Aggregate standardised purchase-at-ask results by contract.

    BASE is excluded from the scenario distribution because it applies
    no market shock. Its mark-to-ask result is retained separately.
    """

    if not 0 < tail_probability < 0.5:
        raise ValueError(
            "tail_probability must be between 0 and 0.5."
        )

    required_result_columns = {
        "Scenario ID",
        "Contract Symbol",
        "Purchase Scenario PnL",
        "Purchase Scenario Return",
        "Scenario Movement PnL",
        "Scenario Movement Return",
        "Base Mark-to-Ask PnL",
    }

    missing_result_columns = (
        required_result_columns
        .difference(scenario_results.columns)
    )

    if missing_result_columns:
        raise KeyError(
            "Scenario results are missing columns: "
            + ", ".join(
                sorted(missing_result_columns)
            )
        )

    required_position_columns = {
        "contractSymbol",
        "Ticker",
        "Option_Type",
        "strike",
        "Expiry",
        "ask",
        "Premium Paid",
        "Initial Recommended Action",
        "BS_ForeV",
        "BS_ForeV AskEdge",
        "Forecast Edge Per Contract",
        "Forecast Edge Return",
    }

    missing_position_columns = (
        required_position_columns
        .difference(positions.columns)
    )

    if missing_position_columns:
        raise KeyError(
            "Standardised positions are missing columns: "
            + ", ".join(
                sorted(missing_position_columns)
            )
        )

    logger.info(
        "Aggregating %d contract-scenario rows",
        len(scenario_results),
    )

    # BASE contains no market movement, so it should not influence
    # the estimated scenario outcome distribution.
    non_base_results = (
        scenario_results.loc[
            ~scenario_results[
                "Scenario ID"
            ].eq(base_scenario_id)
        ]
        .copy()
    )

    if non_base_results.empty:
        raise ValueError(
            "No non-BASE scenario rows were available."
        )

    # Retain the initial market-IV mark relative to the ask.
    base_results = (
        scenario_results.loc[
            scenario_results[
                "Scenario ID"
            ].eq(base_scenario_id),
            [
                "Contract Symbol",
                "Base Mark-to-Ask PnL",
            ],
        ]
        .drop_duplicates(
            subset="Contract Symbol"
        )
        .rename(
            columns={
                "Base Mark-to-Ask PnL": (
                    "Base_Mark_to_Ask_PnL"
                )
            }
        )
    )

    if base_results[
        "Contract Symbol"
    ].duplicated().any():
        raise ValueError(
            "BASE contains duplicate contract symbols."
        )

    grouped = (
        non_base_results
        .groupby(
            "Contract Symbol",
            as_index=False,
        )
        .agg(
            Scenario_Count=(
                "Scenario ID",
                "nunique",
            ),

            # Purchase P&L measured from the ask
            Mean_Purchase_PnL=(
                "Purchase Scenario PnL",
                "mean",
            ),
            Median_Purchase_PnL=(
                "Purchase Scenario PnL",
                "median",
            ),
            Purchase_PnL_StdDev=(
                "Purchase Scenario PnL",
                "std",
            ),
            Fifth_Percentile_Purchase_PnL=(
                "Purchase Scenario PnL",
                lambda values: values.quantile(
                    tail_probability
                ),
            ),
            Expected_Shortfall_5pct_PnL=(
                "Purchase Scenario PnL",
                lambda values: _lower_tail_mean(
                    values,
                    tail_probability,
                ),
            ),
            Worst_Purchase_PnL=(
                "Purchase Scenario PnL",
                "min",
            ),
            Best_Purchase_PnL=(
                "Purchase Scenario PnL",
                "max",
            ),

            # Purchase returns relative to premium paid
            Mean_Purchase_Return=(
                "Purchase Scenario Return",
                "mean",
            ),
            Median_Purchase_Return=(
                "Purchase Scenario Return",
                "median",
            ),
            Purchase_Return_StdDev=(
                "Purchase Scenario Return",
                "std",
            ),
            Fifth_Percentile_Purchase_Return=(
                "Purchase Scenario Return",
                lambda values: values.quantile(
                    tail_probability
                ),
            ),
            Expected_Shortfall_5pct_Return=(
                "Purchase Scenario Return",
                lambda values: _lower_tail_mean(
                    values,
                    tail_probability,
                ),
            ),
            Worst_Purchase_Return=(
                "Purchase Scenario Return",
                "min",
            ),
            Best_Purchase_Return=(
                "Purchase Scenario Return",
                "max",
            ),

            # Frequencies within the generated scenario set
            Scenario_Profit_Frequency=(
                "Purchase Scenario PnL",
                lambda values: (
                    values > 0
                ).mean(),
            ),
            Scenario_Loss_Frequency=(
                "Purchase Scenario PnL",
                lambda values: (
                    values < 0
                ).mean(),
            ),
            Scenario_Breakeven_Frequency=(
                "Purchase Scenario PnL",
                lambda values: np.isclose(
                    values,
                    0.0,
                    atol=1e-10,
                ).mean(),
            ),

            # Isolate the effect of the market shock itself
            Mean_Scenario_Movement_PnL=(
                "Scenario Movement PnL",
                "mean",
            ),
            Median_Scenario_Movement_PnL=(
                "Scenario Movement PnL",
                "median",
            ),
            Positive_Movement_Frequency=(
                "Scenario Movement PnL",
                lambda values: (
                    values > 0
                ).mean(),
            ),
            Negative_Movement_Frequency=(
                "Scenario Movement PnL",
                lambda values: (
                    values < 0
                ).mean(),
            ),
        )
    )

    # Static contract information comes from the standardised
    # position table rather than being repeatedly aggregated.
    metadata_candidates = [
        "contractSymbol",
        "Ticker",
        "Option_Type",
        "strike",
        "Expiry",
        "Calendar DTE",
        "Current Stock Price",
        "bid",
        "ask",
        "MarketMid",
        "SpreadPct",
        "volume",
        "openInterest",
        "Initial Recommended Action",
        "BS_ForeV",
        "BS_ForeV AskEdge",
        "Forecast Edge Per Contract",
        "Forecast Edge Return",
        "Premium Paid",
        "Moneyness",
        "Break-Even Price",
        "IV_Mid",
        "IV Used",
        "ForeV Used",
        "Volatility Spread",
        "Greek Volatility",
        "Greek Volatility Source",
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
    ]

    available_metadata_columns = [
        column
        for column in metadata_candidates
        if column in positions.columns
    ]

    metadata = (
        positions[
            available_metadata_columns
        ]
        .copy()
        .rename(
            columns={
                "contractSymbol": "Contract Symbol",
                "Option_Type": "Option Type",
            }
        )
    )

    if metadata[
        "Contract Symbol"
    ].duplicated().any():
        raise ValueError(
            "The standardised position table contains "
            "duplicate contract symbols."
        )

    summary = (
        metadata
        .merge(
            base_results,
            on="Contract Symbol",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            grouped,
            on="Contract Symbol",
            how="inner",
            validate="one_to_one",
        )
    )

    summary["Base_Mark_to_Ask_Return"] = (
        summary["Base_Mark_to_Ask_PnL"]
        / summary["Premium Paid"]
    )

    # A useful measure of the average scenario improvement relative
    # to the initial market-IV mark.
    summary["Mean_Scenario_Movement_Return"] = (
        summary["Mean_Scenario_Movement_PnL"]
        / summary["Premium Paid"]
    )

    # Scenario outcome minus the forecast-based theoretical edge.
    # This shows whether the average stressed result supports or
    # contradicts the initial model-implied opportunity.
    summary["Mean_PnL_Minus_Forecast_Edge"] = (
        summary["Mean_Purchase_PnL"]
        - summary["Forecast Edge Per Contract"]
    )

    summary["Mean_Return_Minus_Forecast_Edge"] = (
        summary["Mean_Purchase_Return"]
        - summary["Forecast Edge Return"]
    )

    summary = (
        summary
        .sort_values(
            [
                "Initial Recommended Action",
                "Mean_Purchase_Return",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Created scenario summaries for %d contracts",
        len(summary),
    )

    return summary


def validate_contract_scenario_summary(
    summary: pd.DataFrame,
    scenario_results: pd.DataFrame,
    base_scenario_id: str = "BASE",
    atol: float = 1e-8,
) -> dict[str, bool]:
    """Validate the contract-level scenario aggregation."""

    expected_scenario_count = (
        scenario_results.loc[
            ~scenario_results[
                "Scenario ID"
            ].eq(base_scenario_id),
            "Scenario ID",
        ]
        .nunique()
    )

    required_summary_columns = [
        "Contract Symbol",
        "Premium Paid",
        "Scenario_Count",
        "Mean_Purchase_PnL",
        "Mean_Purchase_Return",
        "Median_Purchase_PnL",
        "Fifth_Percentile_Purchase_PnL",
        "Expected_Shortfall_5pct_PnL",
        "Worst_Purchase_PnL",
        "Best_Purchase_PnL",
        "Scenario_Profit_Frequency",
        "Scenario_Loss_Frequency",
        "Scenario_Breakeven_Frequency",
    ]

    missing_columns = set(
        required_summary_columns
    ).difference(summary.columns)

    if missing_columns:
        raise KeyError(
            "Contract summary is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    frequency_total = (
        summary["Scenario_Profit_Frequency"]
        + summary["Scenario_Loss_Frequency"]
        + summary["Scenario_Breakeven_Frequency"]
    )

    checks = {
        "contracts_available": (
            len(summary) > 0
        ),
        "unique_contract_symbols": (
            not summary[
                "Contract Symbol"
            ].duplicated().any()
        ),
        "all_scenario_counts_match": (
            summary[
                "Scenario_Count"
            ].eq(
                expected_scenario_count
            ).all()
        ),
        "positive_premiums": (
            summary[
                "Premium Paid"
            ].gt(0).all()
        ),
        "frequencies_between_zero_and_one": (
            summary[
                [
                    "Scenario_Profit_Frequency",
                    "Scenario_Loss_Frequency",
                    "Scenario_Breakeven_Frequency",
                ]
            ]
            .ge(0)
            .all()
            .all()
            and
            summary[
                [
                    "Scenario_Profit_Frequency",
                    "Scenario_Loss_Frequency",
                    "Scenario_Breakeven_Frequency",
                ]
            ]
            .le(1)
            .all()
            .all()
        ),
        "frequencies_sum_to_one": bool(
            np.allclose(
                frequency_total,
                1.0,
                atol=atol,
            )
        ),
        "long_option_loss_bounded_by_premium": (
            summary[
                "Worst_Purchase_PnL"
            ]
            .ge(
                -summary["Premium Paid"] - atol
            )
            .all()
        ),
        "long_option_return_not_below_minus_one": (
            summary[
                "Worst_Purchase_Return"
            ]
            .ge(-1.0 - atol)
            .all()
        ),
        "tail_metrics_correctly_ordered": (
            (
                summary[
                    "Worst_Purchase_PnL"
                ]
                <= summary[
                    "Expected_Shortfall_5pct_PnL"
                ] + atol
            ).all()
            and
            (
                summary[
                    "Expected_Shortfall_5pct_PnL"
                ]
                <= summary[
                    "Fifth_Percentile_Purchase_PnL"
                ] + atol
            ).all()
            and
            (
                summary[
                    "Fifth_Percentile_Purchase_PnL"
                ]
                <= summary[
                    "Median_Purchase_PnL"
                ] + atol
            ).all()
            and
            (
                summary[
                    "Median_Purchase_PnL"
                ]
                <= summary[
                    "Best_Purchase_PnL"
                ] + atol
            ).all()
        ),
        "mean_return_matches_mean_pnl": bool(
            np.allclose(
                summary[
                    "Mean_Purchase_Return"
                ],
                (
                    summary[
                        "Mean_Purchase_PnL"
                    ]
                    / summary[
                        "Premium Paid"
                    ]
                ),
                atol=atol,
                equal_nan=False,
            )
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed_checks:
        raise AssertionError(
            "Contract scenario summary validation failed: "
            f"{failed_checks}"
        )

    logger.info(
        "All contract scenario summary checks passed"
    )

    return checks


def create_initial_action_comparison(
    contract_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare scenario outcomes across the initial recommendation groups.

    Medians are used for most metrics so that extremely cheap or
    expensive contracts do not dominate the group comparison.
    """

    required_columns = {
        "Contract Symbol",
        "Initial Recommended Action",
        "Premium Paid",
        "Forecast Edge Return",
        "Base_Mark_to_Ask_Return",
        "Mean_Purchase_Return",
        "Median_Purchase_Return",
        "Scenario_Profit_Frequency",
        "Fifth_Percentile_Purchase_Return",
        "Expected_Shortfall_5pct_Return",
        "Worst_Purchase_Return",
    }

    missing_columns = (
        required_columns
        .difference(contract_summary.columns)
    )

    if missing_columns:
        raise KeyError(
            "Cannot compare initial actions; missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    comparison = (
        contract_summary
        .groupby(
            "Initial Recommended Action",
            observed=True,
        )
        .agg(
            Contracts=(
                "Contract Symbol",
                "count",
            ),
            Median_Premium_Paid=(
                "Premium Paid",
                "median",
            ),
            Median_Forecast_Edge_Return=(
                "Forecast Edge Return",
                "median",
            ),
            Median_Base_Mark_to_Ask_Return=(
                "Base_Mark_to_Ask_Return",
                "median",
            ),
            Median_Mean_Purchase_Return=(
                "Mean_Purchase_Return",
                "median",
            ),
            Median_Contract_Median_Return=(
                "Median_Purchase_Return",
                "median",
            ),
            Median_Profit_Frequency=(
                "Scenario_Profit_Frequency",
                "median",
            ),
            Median_Fifth_Percentile_Return=(
                "Fifth_Percentile_Purchase_Return",
                "median",
            ),
            Median_Expected_Shortfall_Return=(
                "Expected_Shortfall_5pct_Return",
                "median",
            ),
            Median_Worst_Return=(
                "Worst_Purchase_Return",
                "median",
            ),
            Contracts_With_Positive_Mean=(
                "Mean_Purchase_Return",
                lambda values: (
                    values > 0
                ).mean(),
            ),
            Contracts_With_Profit_Frequency_Above_50pct=(
                "Scenario_Profit_Frequency",
                lambda values: (
                    values > 0.5
                ).mean(),
            ),
        )
        .reset_index()
    )

    action_order = {
        "Buy": 0,
        "Positive Edge": 1,
        "Do Not Buy": 2,
    }

    comparison["_Action_Order"] = (
        comparison[
            "Initial Recommended Action"
        ]
        .map(action_order)
        .fillna(len(action_order))
    )

    comparison = (
        comparison
        .sort_values("_Action_Order")
        .drop(columns="_Action_Order")
        .reset_index(drop=True)
    )

    return comparison