
"""Readable terminal and notebook tables."""

import logging
from typing import Mapping

import pandas as pd

logger = logging.getLogger(__name__)

def print_highlighted_options(
    highlighted_options: Mapping[str, pd.DataFrame],
) -> None:
    """Print a compact view of the highlighted contracts."""

    important_columns = [
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "BS_ForeV",
        "BS_ForeV AskEdge",
        "Moneyness",
        "MarketMid",
        "SpreadPct",
        "volume",
        "openInterest",
        "Parity Valid",
        "Initial Recommended Action",
    ]

    for table_name, options in highlighted_options.items():
        available_columns = [
            column
            for column in important_columns
            if column in options.columns
        ]

        print(f"\n{table_name}:")
        print(options[available_columns].round(4))


def combine_option_tables(
    cleaned_options: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine all ticker/type tables into one DataFrame."""

    if not cleaned_options:
        logger.warning("No cleaned option tables are available to combine")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for table_name, options in cleaned_options.items():
        ticker, option_type = table_name.rsplit(" ", maxsplit=1)

        frames.append(
            options.assign(
                Ticker=ticker,
                Option_Type=option_type,
                Option_Table=table_name,
            )
        )

    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        "Combined %d option tables into %d rows",
        len(frames),
        len(combined),
    )

    return combined


def create_findings_table(
    large_table: pd.DataFrame,
) -> pd.DataFrame:
    """Create a robust grouped summary using medians instead of raw means."""

    required_columns = {
        "Initial Recommended Action",
        "Option_Table",
        "contractSymbol",
        "Lower Profit",
        "Median Profit",
        "Upper Profit",
        "Expected Profit",
        "Lower Profit Return",
        "Median Profit Return",
        "Upper Profit Return",
        "Expected Profit Return",
        "Probability of Profit",
        "BS_ForeV",
        "BS_ForeV AskEdge",
    }

    missing_columns = required_columns.difference(large_table.columns)

    if missing_columns:
        raise KeyError(
            "Cannot create findings table; missing columns: "
            f"{sorted(missing_columns)}"
        )

    findings = (
        large_table
        .groupby(
            ["Initial Recommended Action", "Option_Table"],
            observed=True,
        )
        .agg(
            Option_Count=("contractSymbol", "size"),
            Median_Lower_Profit=("Lower Profit", "median"),
            Median_Central_Profit=("Median Profit", "median"),
            Median_Upper_Profit=("Upper Profit", "median"),
            Median_Expected_Profit=("Expected Profit", "median"),
            Median_Lower_Return=("Lower Profit Return", "median"),
            Median_Central_Return=("Median Profit Return", "median"),
            Median_Upper_Return=("Upper Profit Return", "median"),
            Median_Expected_Return=("Expected Profit Return", "median"),
            Median_Probability_Profit=("Probability of Profit", "median"),
            Median_BS_ForeV=("BS_ForeV", "median"),
            Median_Ask_Edge=("BS_ForeV AskEdge", "median"),
        )
        .reset_index()
    )

    logger.info(
        "Created findings table with %d groups",
        len(findings),
    )

    return findings


def print_final_tables(
    findings_table: pd.DataFrame,
    large_table: pd.DataFrame,
) -> None:
    """Print the grouped findings and recommended-price output."""

    print("\nGrouped Findings:")
    print(findings_table.round(4))

    important_columns = [
        # Contract details
        "Ticker",
        "Option_Type",
        "contractSymbol",
        "Expiry",
        "strike",

        # Horizon details
        "Calendar DTE",
        "Trading DTE",
        "Forecast Horizon",
        "Horizon Difference",
        "Horizon Aligned",

        # Underlying and moneyness
        "Current Stock Price",
        "Moneyness",
        "Forward Price",
        "Forward Moneyness",

        # Market quote
        "bid",
        "ask",
        "MarketMid",
        "SpreadPct",
        "volume",
        "openInterest",

        # Volatility comparison
        "IV Used",
        "ForeV Used",
        "Volatility Spread",

        # Valuation
        "BS_IV",
        "BS_ForeV",
        "BS_ForeV AskEdge",

        # Contract properties
        "Intrinsic Value",
        "Time Value",
        "Break-Even Price",

        # Validation
        "Pricing Bounds Valid",
        "Quote Valid",
        "Quote Issue",
        "Parity Valid",
        "Recommendation Eligible",

        # Monte Carlo results
        "Lower Profit Return",
        "Median Profit Return",
        "Upper Profit Return",
        "Expected Profit Return",
        "Probability of Profit",

        #Implied volatility
        "IV_Order Valid",
        "IV_Repricing_Valid",
        "Greek Volatility",
        "Greek Volatility Source",
        "BS_From_IV_Mid",

        # Greeks
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",


        # Final decision
        "Initial Recommended Action",
    ]

    available_columns = [
        column
        for column in important_columns
        if column in large_table.columns
    ]

    print("\nRecommended Buy Prices:")
    print(large_table[available_columns].round(4))
