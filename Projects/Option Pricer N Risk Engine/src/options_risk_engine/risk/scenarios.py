
"""Random stress scenarios, full revaluation and loss-limit analysis."""

import logging
import numpy as np
import pandas as pd

from options_risk_engine.pricing.black_scholes import black_scholes

logger = logging.getLogger(__name__)

def generate_random_scenarios(
    number_of_scenarios=1_000,
    max_days_forward=30,
    seed=42,
):
    """Create reproducible random market shocks plus an unchanged base case."""

    logger.info(
        "Generating %d random scenarios with up to %d days forward; seed=%s",
        number_of_scenarios,
        max_days_forward,
        seed,
    )

    rng = np.random.default_rng(seed)

    scenarios = pd.DataFrame(
        {
            # Percentage change in the underlying price
            "Spot Shock": np.clip(
                rng.normal(
                    loc=0.0,
                    scale=0.08,
                    size=number_of_scenarios,
                ),
                -0.30,
                0.30,
            ),

            # Absolute volatility change:
            # 0.05 means volatility increases by 5 points
            "Volatility Shock": np.clip(
                rng.normal(
                    loc=0.0,
                    scale=0.05,
                    size=number_of_scenarios,
                ),
                -0.20,
                0.20,
            ),

            # Random number of calendar days passing
            "Days Forward": rng.integers(
                low=0,
                high=max_days_forward + 1,
                size=number_of_scenarios,
            ),

            # Absolute interest-rate change:
            # 0.01 means rates increase by 1 percentage point
            "Rate Shock": np.clip(
                rng.normal(
                    loc=0.0,
                    scale=0.005,
                    size=number_of_scenarios,
                ),
                -0.02,
                0.02,
            ),
        }
    )

    scenarios["Scenario ID"] = (
        "R"
        + (
            scenarios.index + 1
        )
        .astype(str)
        .str.zfill(4)
    )

    # Include an unchanged scenario for validation
    base_scenario = pd.DataFrame(
        [
            {
                "Spot Shock": 0.0,
                "Volatility Shock": 0.0,
                "Days Forward": 0,
                "Rate Shock": 0.0,
                "Scenario ID": "BASE",
            }
        ]
    )

    scenario_table = pd.concat(
        [
            base_scenario,
            scenarios,
        ],
        ignore_index=True,
    )

    logger.info(
        "Scenario generation complete: %d total rows including BASE",
        len(scenario_table),
    )
    logger.debug(
        "Scenario ranges — spot: [%.4f, %.4f], volatility: [%.4f, %.4f], "
        "rate: [%.4f, %.4f], days: [%d, %d]",
        float(scenario_table["Spot Shock"].min()),
        float(scenario_table["Spot Shock"].max()),
        float(scenario_table["Volatility Shock"].min()),
        float(scenario_table["Volatility Shock"].max()),
        float(scenario_table["Rate Shock"].min()),
        float(scenario_table["Rate Shock"].max()),
        int(scenario_table["Days Forward"].min()),
        int(scenario_table["Days Forward"].max()),
    )

    return scenario_table


def run_scenario_engine(
    position_options: pd.DataFrame,
    scenarios: pd.DataFrame,
    dividend_yields: dict[str, float],
):
    """Fully revalue every portfolio position under every market scenario."""

    logger.info(
        "Starting scenario engine: %d positions across %d scenarios",
        len(position_options),
        len(scenarios),
    )

    scenario_frames = []

    # Convert Calls, Puts and Shares to call, put and share
    position_types = (
        position_options["Option_Type"]
        .astype(str)
        .str.lower()
        .str.rstrip("s")
    )

    call_mask = position_types.eq("call")
    put_mask = position_types.eq("put")
    share_mask = position_types.eq("share")

    unknown_type_mask = ~(call_mask | put_mask | share_mask)
    if unknown_type_mask.any():
        unknown_types = (
            position_options.loc[unknown_type_mask, "Option_Type"]
            .drop_duplicates()
            .tolist()
        )
        logger.warning(
            "Scenario engine found unrecognised position types: %s",
            unknown_types,
        )

    logger.info(
        "Scenario positions classified: %d calls, %d puts and %d shares",
        int(call_mask.sum()),
        int(put_mask.sum()),
        int(share_mask.sum()),
    )

    # Map each ticker to the dividend yield stored on its ticker object.
    raw_dividend_yield = position_options["Ticker"].map(dividend_yields)

    missing_yield_tickers = (
        position_options.loc[raw_dividend_yield.isna(), "Ticker"]
        .drop_duplicates()
        .tolist()
    )
    if missing_yield_tickers:
        logger.warning(
            "Missing dividend yields defaulted to zero for: %s",
            missing_yield_tickers,
        )

    position_dividend_yield = raw_dividend_yield.fillna(0.0)

    total_scenarios = len(scenarios)
    progress_interval = max(total_scenarios // 10, 1)


    # Signed number of option units or shares
    position_scale = (
        position_options["Quantity"]
        * position_options["Direction"]
        * position_options["Multiplier"]
    )

    # Calculate the model value before applying any scenario shocks
    base_model_price = pd.Series(
        np.nan,
        index=position_options.index,
        dtype=float,
    )

    # A share's base model price is its current stock price
    base_model_price.loc[share_mask] = (
        position_options.loc[
            share_mask,
            "Current Stock Price",
        ]
    )

    # Base Black-Scholes value for calls
    if call_mask.any():
        base_model_price.loc[call_mask] = np.asarray(
            black_scholes(
                ticker=position_options.loc[
                    call_mask,
                    "Ticker",
                ],
                spot=position_options.loc[
                    call_mask,
                    "Current Stock Price",
                ],
                strike=position_options.loc[
                    call_mask,
                    "strike",
                ],
                time_to_expiry=position_options.loc[
                    call_mask,
                    "Time to Expiry",
                ],
                risk_free_rate=position_options.loc[
                    call_mask,
                    "Risk Free Rate",
                ],
                dividend_yield=position_dividend_yield.loc[
                    call_mask
                ],
                option_type="call",
                volatility=position_options.loc[
                    call_mask,
                    "Greek Volatility",
                ],
            )
        )

    # Base Black-Scholes value for puts
    if put_mask.any():
        base_model_price.loc[put_mask] = np.asarray(
            black_scholes(
                ticker=position_options.loc[
                    put_mask,
                    "Ticker",
                ],
                spot=position_options.loc[
                    put_mask,
                    "Current Stock Price",
                ],
                strike=position_options.loc[
                    put_mask,
                    "strike",
                ],
                time_to_expiry=position_options.loc[
                    put_mask,
                    "Time to Expiry",
                ],
                risk_free_rate=position_options.loc[
                    put_mask,
                    "Risk Free Rate",
                ],
                dividend_yield=position_dividend_yield.loc[
                    put_mask
                ],
                option_type="put",
                volatility=position_options.loc[
                    put_mask,
                    "Greek Volatility",
                ],
            )
        )

    logger.info(
        "Base model prices calculated for %d positions",
        base_model_price.notna().sum(),
    )

    for scenario_number, (_, scenario) in enumerate(
        scenarios.iterrows(),
        start=1,
    ):
        if (
            scenario_number == 1
            or scenario_number % progress_interval == 0
            or scenario_number == total_scenarios
        ):
            logger.info(
                "Revaluing scenario %d/%d: %s",
                scenario_number,
                total_scenarios,
                scenario["Scenario ID"],
            )
        # Percentage shock to the underlying
        shocked_spot = (
            position_options["Current Stock Price"]
            * (1 + scenario["Spot Shock"])
        )

        # Absolute volatility shock
        shocked_volatility = (
            position_options["Greek Volatility"]
            + scenario["Volatility Shock"]
        ).clip(lower=1e-8)

        # Time to expiry is already measured in years
        shocked_time = (
            position_options["Time to Expiry"]
            - scenario["Days Forward"] / 365
        ).clip(lower=0)

        # Absolute interest-rate shock
        shocked_rate = (
            position_options["Risk Free Rate"]
            + scenario["Rate Shock"]
        )

        # One scenario price for every portfolio row
        scenario_price = pd.Series(
            np.nan,
            index=position_options.index,
            dtype=float,
        )

        # Shares are worth the shocked stock price
        scenario_price.loc[share_mask] = (
            shocked_spot.loc[share_mask]
        )

        # Price calls
        if call_mask.any():
            scenario_price.loc[call_mask] = np.asarray(
                black_scholes(
                    ticker=position_options.loc[
                        call_mask,
                        "Ticker",
                    ],
                    spot=shocked_spot.loc[call_mask],
                    strike=position_options.loc[
                        call_mask,
                        "strike",
                    ],
                    time_to_expiry=shocked_time.loc[
                        call_mask
                    ],
                    risk_free_rate=shocked_rate.loc[
                        call_mask
                    ],
                    dividend_yield=position_dividend_yield.loc[
                        call_mask
                    ],
                    option_type="call",
                    volatility=shocked_volatility.loc[
                        call_mask
                    ],
                )
            )

        # Price puts
        if put_mask.any():
            scenario_price.loc[put_mask] = np.asarray(
                black_scholes(
                    ticker=position_options.loc[
                        put_mask,
                        "Ticker",
                    ],
                    spot=shocked_spot.loc[put_mask],
                    strike=position_options.loc[
                        put_mask,
                        "strike",
                    ],
                    time_to_expiry=shocked_time.loc[
                        put_mask
                    ],
                    risk_free_rate=shocked_rate.loc[
                        put_mask
                    ],
                    dividend_yield=position_dividend_yield.loc[
                        put_mask
                    ],
                    option_type="put",
                    volatility=shocked_volatility.loc[
                        put_mask
                    ],
                )
            )

        position_scale = (
            position_options["Quantity"]
            * position_options["Direction"]
            * position_options["Multiplier"]
        )

        scenario_frame = pd.DataFrame(
            {
                "Scenario ID": scenario["Scenario ID"],
                "Contract Symbol": (
                    position_options["contractSymbol"]
                ),
                "Ticker": position_options["Ticker"],
                "Option Type": position_options["Option_Type"],
                # Keep the market price and model price separate
                "Current Market Price": (
                    position_options["Current Mark"]
                ),
                "Base Model Price": base_model_price,
                "Scenario Price": scenario_price,

                # Difference between the current market and the model
                "Market-to-Model PnL": (
                    base_model_price
                    - position_options["Current Mark"]
                ) * position_scale,

                # Pure scenario movement measured on a consistent model basis
                "Scenario PnL": (
                    scenario_price
                    - base_model_price
                ) * position_scale,
                "Scenario Market Value": (
                    scenario_price
                    * position_scale
                ),
                "Shocked Spot": shocked_spot,
                "Shocked Volatility": shocked_volatility,
                "Shocked Rate": shocked_rate,
                "Shocked Time": shocked_time,
                "Spot Shock": scenario["Spot Shock"],
                "Volatility Shock": (
                    scenario["Volatility Shock"]
                ),
                "Rate Shock": scenario["Rate Shock"],
                "Days Forward": scenario["Days Forward"],
            }
        )

        missing_prices = int(scenario_price.isna().sum())
        if missing_prices:
            logger.warning(
                "Scenario %s produced %d missing position prices",
                scenario["Scenario ID"],
                missing_prices,
            )

        scenario_frames.append(scenario_frame)

    # Concatenate only once after all scenarios have been processed. This is
    # substantially faster than repeatedly growing one DataFrame in the loop.
    results = pd.concat(
        scenario_frames,
        ignore_index=True,
    )

    logger.info(
        "Position-level scenario results created: %d rows",
        len(results),
    )

    # Summarise each scenario by ticker.
    scenario_ticker = (
        results
        .groupby(
            ["Scenario ID", "Ticker"],
            as_index=False,
        )
        .agg(
            Scenario_PnL=("Scenario PnL", "sum"),
            Scenario_Value=(
                "Scenario Market Value",
                "sum",
            ),
        )
    )

    # Summarise each scenario across the entire portfolio.
    scenario_portfolio = (
        results
        .groupby(
            "Scenario ID",
            as_index=False,
        )
        .agg(
            Portfolio_PnL=("Scenario PnL", "sum"),
            Portfolio_Value=(
                "Scenario Market Value",
                "sum",
            ),
        )
    )

    return (
        results,
        scenario_ticker,
        scenario_portfolio,
    )


def analyse_scenario_results(
    scenarios: pd.DataFrame,
    results: pd.DataFrame,
    scenario_ticker: pd.DataFrame,
    scenario_portfolio: pd.DataFrame,
    portfolio_risk: pd.DataFrame,
    base_scenario_id: str = "BASE",
    max_portfolio_loss: float = 100_000,
    max_ticker_loss: float = 25_000,
    number_to_show: int = 10,
):
    """
    Validate and analyse position, ticker and portfolio scenario results.
    """

    logger.info(
        "Analysing scenario outputs: %d position rows, %d ticker rows and "
        "%d portfolio rows",
        len(results),
        len(scenario_ticker),
        len(scenario_portfolio),
    )

    # Work on copies so the raw outputs remain available unchanged.
    results = results.copy()
    scenario_ticker = scenario_ticker.copy()
    scenario_portfolio = scenario_portfolio.copy()

    # Allow the older position-level P&L column name
    if (
        "Scenario PnL" not in results.columns
        and "Profit/Loss" in results.columns
    ):
        logger.info(
            "Renaming legacy Profit/Loss column to Scenario PnL"
        )
        results = results.rename(
            columns={
                "Profit/Loss": "Scenario PnL"
            }
        )

    scenario_columns = [
        "Scenario ID",
        "Spot Shock",
        "Volatility Shock",
        "Days Forward",
        "Rate Shock",
    ]

    scenario_details = (
        scenarios[
            [
                column
                for column in scenario_columns
                if column in scenarios.columns
            ]
        ]
        .drop_duplicates("Scenario ID")
    )

    # Remove shock columns already present before merging
    shock_columns = [
        "Spot Shock",
        "Volatility Shock",
        "Days Forward",
        "Rate Shock",
    ]

    scenario_ticker = scenario_ticker.drop(
        columns=[
            column
            for column in shock_columns
            if column in scenario_ticker.columns
        ],
        errors="ignore",
    )

    scenario_portfolio = scenario_portfolio.drop(
        columns=[
            column
            for column in shock_columns
            if column in scenario_portfolio.columns
        ],
        errors="ignore",
    )

    # Attach the market shocks to each summary table
    scenario_ticker = scenario_ticker.merge(
        scenario_details,
        on="Scenario ID",
        how="left",
    )

    scenario_portfolio = scenario_portfolio.merge(
        scenario_details,
        on="Scenario ID",
        how="left",
    )

    # Check the unchanged validation scenario
    base_result = scenario_portfolio.loc[
        scenario_portfolio["Scenario ID"].eq(
            base_scenario_id
        )
    ].copy()

    if not base_result.empty:
        base_pnl = float(
            base_result["Portfolio_PnL"].iloc[0]
        )

        base_validation_passed = bool(
            np.isclose(
                base_pnl,
                0.0,
                atol=1e-6,
            )
        )
        base_result["Base Validation Passed"] = base_validation_passed

        if base_validation_passed:
            logger.info(
                "Base scenario validation passed: portfolio P&L %.8f",
                base_pnl,
            )
        else:
            logger.warning(
                "Base scenario validation failed: portfolio P&L %.8f",
                base_pnl,
            )
    else:
        logger.warning(
            "Base scenario %s was not found in portfolio results",
            base_scenario_id,
        )

    # Find positions which could not be repriced
    invalid_scenario_rows = results.loc[
        results["Scenario Price"].isna()
        | results["Scenario PnL"].isna()
    ].copy()

    if invalid_scenario_rows.empty:
        logger.info("All scenario positions were repriced successfully")
    else:
        logger.warning(
            "%d scenario-position rows contain missing prices or P&L",
            len(invalid_scenario_rows),
        )

    # Exclude the base validation scenario from risk statistics
    scenario_distribution = (
        scenario_portfolio.loc[
            ~scenario_portfolio[
                "Scenario ID"
            ].eq(base_scenario_id)
        ]
        .copy()
    )

    if scenario_distribution.empty:
        raise ValueError(
            "No non-base scenarios are available."
        )

    # Empirical lower-tail scenario results
    pnl_5_percentile = (
        scenario_distribution["Portfolio_PnL"]
        .quantile(0.05)
    )

    pnl_1_percentile = (
        scenario_distribution["Portfolio_PnL"]
        .quantile(0.01)
    )

    expected_shortfall_5 = (
        scenario_distribution.loc[
            scenario_distribution[
                "Portfolio_PnL"
            ].le(pnl_5_percentile),
            "Portfolio_PnL",
        ]
        .mean()
    )

    # Summarise the empirical portfolio P&L distribution. These figures are
    # scenario statistics rather than formal VaR until the shock process has
    # been calibrated to a realistic time horizon and return distribution.
    scenario_risk_summary = pd.DataFrame(
        [
            {
                "Number of Scenarios": len(
                    scenario_distribution
                ),
                "Best PnL": (
                    scenario_distribution[
                        "Portfolio_PnL"
                    ].max()
                ),
                "Worst PnL": (
                    scenario_distribution[
                        "Portfolio_PnL"
                    ].min()
                ),
                "Mean PnL": (
                    scenario_distribution[
                        "Portfolio_PnL"
                    ].mean()
                ),
                "Median PnL": (
                    scenario_distribution[
                        "Portfolio_PnL"
                    ].median()
                ),
                "Probability of Loss": (
                    scenario_distribution[
                        "Portfolio_PnL"
                    ]
                    .lt(0)
                    .mean()
                ),
                "5th Percentile PnL": (
                    pnl_5_percentile
                ),
                "1st Percentile PnL": (
                    pnl_1_percentile
                ),
                "5% Expected Shortfall": (
                    expected_shortfall_5
                ),
            }
        ]
    )

    logger.info(
        "Scenario distribution analysed: worst P&L %.2f; 5th percentile %.2f; "
        "probability of loss %.2f%%",
        float(scenario_distribution["Portfolio_PnL"].min()),
        float(pnl_5_percentile),
        float(
            scenario_distribution["Portfolio_PnL"].lt(0).mean() * 100
        ),
    )

    # Find the best and worst portfolio scenarios.
    worst_scenarios = (
        scenario_distribution
        .nsmallest(
            number_to_show,
            "Portfolio_PnL",
        )
        .reset_index(drop=True)
    )

    best_scenarios = (
        scenario_distribution
        .nlargest(
            number_to_show,
            "Portfolio_PnL",
        )
        .reset_index(drop=True)
    )

    worst_scenario_id = (
        worst_scenarios["Scenario ID"]
        .iloc[0]
    )

    # Ticker contributions to the worst scenario
    worst_ticker_contributions = (
        scenario_ticker.loc[
            scenario_ticker[
                "Scenario ID"
            ].eq(worst_scenario_id)
        ]
        .sort_values("Scenario_PnL")
        .reset_index(drop=True)
    )

    # Position contributions to the worst scenario
    worst_position_contributions = (
        results.loc[
            results["Scenario ID"].eq(
                worst_scenario_id
            )
        ]
        .sort_values("Scenario PnL")
        .reset_index(drop=True)
    )

    # Find the portfolio's gross market value
    if "Gross_Market_Value" in portfolio_risk.columns:
        gross_value_column = "Gross_Market_Value"

    elif "Gross Market Value" in portfolio_risk.columns:
        gross_value_column = "Gross Market Value"

    else:
        gross_value_column = None

    portfolio_gross_value = np.nan

    if gross_value_column is not None:
        total_row = portfolio_risk.loc[
            portfolio_risk["Ticker"].eq(
                "PORTFOLIO"
            ),
            gross_value_column,
        ]

        if not total_row.empty:
            portfolio_gross_value = float(
                total_row.iloc[0]
            )
        else:
            portfolio_gross_value = float(
                portfolio_risk[
                    gross_value_column
                ].sum()
            )

    if (
        pd.notna(portfolio_gross_value)
        and portfolio_gross_value != 0
    ):
        scenario_portfolio[
            "PnL on Gross Value"
        ] = (
            scenario_portfolio[
                "Portfolio_PnL"
            ]
            / portfolio_gross_value
        )

        scenario_distribution[
            "PnL on Gross Value"
        ] = (
            scenario_distribution[
                "Portfolio_PnL"
            ]
            / portfolio_gross_value
        )
    else:
        scenario_portfolio[
            "PnL on Gross Value"
        ] = np.nan

        scenario_distribution[
            "PnL on Gross Value"
        ] = np.nan

    # Apply portfolio and ticker risk limits
    scenario_portfolio[
        "Portfolio Limit Breached"
    ] = (
        scenario_portfolio[
            "Portfolio_PnL"
        ]
        < -abs(max_portfolio_loss)
    )

    scenario_ticker[
        "Ticker Limit Breached"
    ] = (
        scenario_ticker[
            "Scenario_PnL"
        ]
        < -abs(max_ticker_loss)
    )

    # Extract scenarios which cross the configured risk limits.
    portfolio_breaches = (
        scenario_portfolio.loc[
            scenario_portfolio[
                "Portfolio Limit Breached"
            ]
        ]
        .sort_values("Portfolio_PnL")
        .reset_index(drop=True)
    )

    ticker_breaches = (
        scenario_ticker.loc[
            scenario_ticker[
                "Ticker Limit Breached"
            ]
        ]
        .sort_values("Scenario_PnL")
        .reset_index(drop=True)
    )

    if portfolio_breaches.empty:
        logger.info("No portfolio scenario breached the configured loss limit")
    else:
        logger.warning(
            "%d portfolio scenarios breached the %.2f loss limit",
            len(portfolio_breaches),
            abs(max_portfolio_loss),
        )

    if ticker_breaches.empty:
        logger.info("No ticker scenario breached the configured loss limit")
    else:
        logger.warning(
            "%d ticker-scenario rows breached the %.2f loss limit",
            len(ticker_breaches),
            abs(max_ticker_loss),
        )

    logger.info(
        "Scenario analysis complete; worst scenario is %s",
        worst_scenario_id,
    )

    return {
        "results": results,
        "scenario_ticker": scenario_ticker,
        "scenario_portfolio": scenario_portfolio,
        "scenario_distribution": scenario_distribution,
        "scenario_risk_summary": scenario_risk_summary,
        "base_result": base_result,
        "invalid_scenario_rows": invalid_scenario_rows,
        "worst_scenarios": worst_scenarios,
        "best_scenarios": best_scenarios,
        "worst_scenario_id": worst_scenario_id,
        "worst_ticker_contributions": (
            worst_ticker_contributions
        ),
        "worst_position_contributions": (
            worst_position_contributions
        ),
        "portfolio_breaches": portfolio_breaches,
        "ticker_breaches": ticker_breaches,
        "portfolio_gross_value": (
            portfolio_gross_value
        ),
    }
