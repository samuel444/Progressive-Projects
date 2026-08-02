
"""Hypothetical portfolio construction and current Greek aggregation."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def risk_engine_data_prep(large_table: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the large table for risk-engine input.
    This includes renaming columns and selecting relevant fields.
    """

    logger.info(
        "Preparing option universe for the portfolio risk engine: %d rows",
        len(large_table),
    )

    # A fixed seed keeps the hypothetical test portfolio reproducible.
    rng = np.random.default_rng(seed=42)

    # Retain only the market, timing, volatility and Greek fields required by
    # the position and scenario calculations.
    portfolio_market_columns = [
        # Contract identity
        "contractSymbol",
        "Ticker",
        "Option_Type",
        "strike",
        "Expiry",

        # Current underlying and timing
        "Current Stock Price",
        "Time to Expiry",
        "Calendar DTE",

        # Current option market
        "bid",
        "ask",
        "MarketMid",

        # Volatility used for risk
        "Greek Volatility",
        "Greek Volatility Source",

        # Unit Greeks
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",

        # Quote reliability
        "SpreadPct",
        "volume",
        "openInterest",
        "Quote Valid",
        "Quote Issue",
    ]

    position_options = large_table[
        portfolio_market_columns
    ].copy()

    rows_before_drop = len(position_options)

    # The temporary test portfolio uses only rows with all required inputs.
    # This prevents missing prices, volatilities or Greeks from entering the
    # position calculations.
    position_options = (
        position_options
        .dropna()
        .reset_index(drop=True)
    )

    number_of_positions = len(position_options)

    logger.info(
        "Risk-engine input cleaned: %d usable option rows; %d rows removed",
        number_of_positions,
        rows_before_drop - number_of_positions,
    )

    # Randomly assign long or short
    position_options["Side"] = rng.choice(
        ["long", "short"],
        size=number_of_positions,
    )

    # Random number of contracts from 1 to 5
    position_options["Quantity"] = rng.integers(
        low=1,
        high=6,
        size=number_of_positions,
    )

    # Standard US equity-option contract multiplier
    position_options["Multiplier"] = 100

    # Hypothetical entry prices within 15% of today's midpoint
    entry_price_change = rng.uniform(
        low=-0.15,
        high=0.15,
        size=number_of_positions,
    )

    position_options["Entry Price"] = (
        position_options["MarketMid"]
        * (1 + entry_price_change)
    ).clip(lower=0.01)

    position_options["Current Mark"] = (
        position_options["MarketMid"]
    )

    position_options["Direction"] = np.where(
        position_options["Side"] == "long",
        1,
        -1,
    )

    logger.info(
        "Created reproducible test option positions: %d long and %d short",
        int(position_options["Side"].eq("long").sum()),
        int(position_options["Side"].eq("short").sum()),
    )

    # Add one ordinary-share position per ticker so the engine can represent
    # stock-and-option portfolios and include share delta in total exposure.
    share_holdings = {
        "AAPL": 50,
        "MSFT": -20,
        "META": 100,
        "WMT": 40,
        "GOOGL": 35,
        "AMZN": -40,
        "HCA": 25,
    }

    spot_prices = (
        position_options
        .groupby("Ticker")["Current Stock Price"]
        .first()
    )

    share_rows = pd.DataFrame(
        [
            {
                "contractSymbol": f"{ticker}_SHARES",
                "Ticker": ticker,
                "Option_Type": "Shares",
                "Current Stock Price": spot_prices.loc[ticker],
                "Side": "long" if shares > 0 else "short",
                "Direction": 1 if shares > 0 else -1,
                "Quantity": abs(shares),
                "Multiplier": 1,
                "Entry Price": spot_prices.loc[ticker],
                "Current Mark": spot_prices.loc[ticker],

                # Greeks for ordinary shares
                "Delta": 1.0,
                "Gamma": 0.0,
                "Vega": 0.0,
                "Theta": 0.0,
                "Rho": 0.0,
            }
            for ticker, shares in share_holdings.items()
            if shares != 0
        ]
    )

    # Add all missing option-specific columns and match column order
    share_rows = share_rows.reindex(
        columns=position_options.columns
    )

    # Add share positions underneath the option positions.
    position_options = pd.concat(
        [
            position_options,
            share_rows,
        ],
        ignore_index=True,
    )

    logger.info(
        "Added %d share positions; combined portfolio now contains %d rows",
        len(share_rows),
        len(position_options),
    )

    # Position Scale is signed. It includes direction, number of contracts or
    # shares, and the contract multiplier.
    position_options["Position Scale"] = (
        position_options["Direction"]
        * position_options["Quantity"]
        * position_options["Multiplier"]
    )

    # Convert unit Greeks into signed position-level exposures.
    position_options["Position Delta"] = (
        position_options["Delta"] * position_options["Position Scale"]
    )
    position_options["Position Gamma"] = (
        position_options["Gamma"] * position_options["Position Scale"]
    )
    position_options["Position Vega"] = (
        position_options["Vega"] * position_options["Position Scale"]
    )
    position_options["Position Theta"] = (
        position_options["Theta"] * position_options["Position Scale"]
    )
    position_options["Position Rho"] = (
        position_options["Rho"] * position_options["Position Scale"]
    )

    # Mark-to-market P&L since the hypothetical entry price.
    position_options["Position PnL"] = (
        position_options["Current Mark"]
        - position_options["Entry Price"]
    ) * position_options["Position Scale"]

    position_options["Signed Market Value"] = (
        position_options["Current Mark"]
        * position_options["Position Scale"]
    )

    position_options["Entry Cash Flow"] = (
        position_options["Entry Price"]
        * position_options["Position Scale"]
    )

    position_options["Entry Premium Value"] = (
        position_options["Entry Price"]
        * position_options["Quantity"]
        * position_options["Multiplier"]
    )

    logger.info(
        "Portfolio positions prepared: gross market value %.2f; current P&L %.2f",
        float(position_options["Signed Market Value"].abs().sum()),
        float(position_options["Position PnL"].sum()),
    )

    return position_options


def risk_engine_summary(position_options: pd.DataFrame) -> pd.DataFrame:
    """Aggregate current position value, P&L and Greeks by ticker."""

    logger.info(
        "Building current portfolio risk summary from %d positions",
        len(position_options),
    )

    # First create one current-risk row for every underlying ticker.
    ticker_risk = (
        position_options
        .groupby(
            "Ticker",
            as_index=False,
        )
        .agg(
            Number_of_Positions=(
                "contractSymbol",
                "count",
            ),
            Net_Market_Value=(
                "Signed Market Value",
                "sum",
            ),
            Gross_Market_Value=(
                "Signed Market Value",
                lambda values: values.abs().sum(),
            ),
            Position_PnL=(
                "Position PnL",
                "sum",
            ),
            Delta=(
                "Position Delta",
                "sum",
            ),
            Gamma=(
                "Position Gamma",
                "sum",
            ),
            Vega=(
                "Position Vega",
                "sum",
            ),
            Theta=(
                "Position Theta",
                "sum",
            ),
            Rho=(
                "Position Rho",
                "sum",
            ),
        )
    )

    logger.info(
        "Ticker-level risk calculated for %d tickers",
        len(ticker_risk),
    )

    # Add a final row containing the whole portfolio's current exposure.
    portfolio_total = pd.DataFrame(
        [
            {
                "Ticker": "PORTFOLIO",
                "Number_of_Positions": (
                    ticker_risk["Number_of_Positions"]
                    .sum()
                ),
                "Net_Market_Value": (
                    ticker_risk["Net_Market_Value"]
                    .sum()
                ),
                "Gross_Market_Value": (
                    ticker_risk["Gross_Market_Value"]
                    .sum()
                ),
                "Position_PnL": (
                    ticker_risk["Position_PnL"]
                    .sum()
                ),
                "Delta": ticker_risk["Delta"].sum(),
                "Gamma": ticker_risk["Gamma"].sum(),
                "Vega": ticker_risk["Vega"].sum(),
                "Theta": ticker_risk["Theta"].sum(),
                "Rho": ticker_risk["Rho"].sum(),
            }
        ]
    )

    portfolio_risk = pd.concat(
        [
            ticker_risk,
            portfolio_total,
        ],
        ignore_index=True,
    )

    total = portfolio_total.iloc[0]
    logger.info(
        "Portfolio risk summary complete: net value %.2f; gross value %.2f; "
        "delta %.2f",
        float(total["Net_Market_Value"]),
        float(total["Gross_Market_Value"]),
        float(total["Delta"]),
    )

    return portfolio_risk
