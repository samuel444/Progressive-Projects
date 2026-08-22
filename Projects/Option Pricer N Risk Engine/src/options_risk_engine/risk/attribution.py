
"""Delta-Gamma-Vega-Theta-Rho P&L attribution and residual diagnostics."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_greek_profit_loss(
    expanded_results: pd.DataFrame,
):
    """
    Calculate position-level Greek P&L attribution and aggregate it
    into one row per scenario.
    """

    logger.info("Starting Greek P&L attribution")

    attribution_columns = [
        "Scenario ID",
        "Contract Symbol",
        "Ticker",
        "Option Type",
        "Current Stock Price",
        "Shocked Spot",
        "Scenario PnL",
        "Position Delta",
        "Position Gamma",
        "Position Vega",
        "Position Theta",
        "Position Rho",
        "Spot Shock",
        "Volatility Shock",
        "Rate Shock",
        "Days Forward",
    ]

    missing_columns = [
        column
        for column in attribution_columns
        if column not in expanded_results.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing attribution columns: "
            + ", ".join(missing_columns)
        )

    attribution_results = (
        expanded_results[
            attribution_columns
        ]
        .copy()
    )

    # Position-level Greek attribution formulas

    # Dollar change in the underlying share price
    attribution_results["Spot Change"] = (
        attribution_results["Shocked Spot"]
        - attribution_results["Current Stock Price"]
    )

    # Delta P&L = position delta × change in stock price
    attribution_results["Delta PnL"] = (
        attribution_results["Position Delta"]
        * attribution_results["Spot Change"]
    )

    # Gamma P&L = 0.5 × position gamma × stock-price change squared
    attribution_results["Gamma PnL"] = (
        0.5
        * attribution_results["Position Gamma"]
        * attribution_results["Spot Change"] ** 2
    )

    # Volatility shock is stored as a decimal:
    # 0.05 represents five volatility percentage points
    attribution_results["Vega PnL"] = (
        attribution_results["Position Vega"]
        * attribution_results["Volatility Shock"]
        * 100
    )

    # Position theta is measured per calendar day
    attribution_results["Theta PnL"] = (
        attribution_results["Position Theta"]
        * attribution_results["Days Forward"]
    )

    # Rate shock is stored as a decimal:
    # 0.01 represents one interest-rate percentage point
    attribution_results["Rho PnL"] = (
        attribution_results["Position Rho"]
        * attribution_results["Rate Shock"]
        * 100
    )

    # Sum of the Greek-estimated P&L components
    attribution_results["Approximate PnL"] = (
        attribution_results["Delta PnL"]
        + attribution_results["Gamma PnL"]
        + attribution_results["Vega PnL"]
        + attribution_results["Theta PnL"]
        + attribution_results["Rho PnL"]
    )

    # Difference between full repricing and Greek approximation
    attribution_results["Residual PnL"] = (
        attribution_results["Scenario PnL"]
        - attribution_results["Approximate PnL"]
    )

    # Position-level residual percentage
    attribution_results["Residual %"] = np.where(
        attribution_results["Scenario PnL"].abs() > 1e-8,
        attribution_results["Residual PnL"].abs()
        / attribution_results["Scenario PnL"].abs(),
        np.nan,
    )


    # Scenario-level attribution
    scenario_attribution = (
        attribution_results
        .groupby(
            "Scenario ID",
            as_index=False,
        )
        .agg(
            Full_Revaluation_PnL=(
                "Scenario PnL",
                "sum",
            ),
            Delta_PnL=("Delta PnL", "sum"),
            Gamma_PnL=("Gamma PnL", "sum"),
            Vega_PnL=("Vega PnL", "sum"),
            Theta_PnL=("Theta PnL", "sum"),
            Rho_PnL=("Rho PnL", "sum"),
            Approximate_PnL=(
                "Approximate PnL",
                "sum",
            ),
            Residual_PnL=(
                "Residual PnL",
                "sum",
            ),
            Gross_Full_PnL=(
                "Scenario PnL",
                lambda values: values.abs().sum(),
            ),
            Gross_Residual=(
                "Residual PnL",
                lambda values: values.abs().sum(),
            ),
        )
    )

    # Net residual relative to the portfolio's net scenario P&L
    scenario_attribution["Net Residual %"] = np.where(
        scenario_attribution[
            "Full_Revaluation_PnL"
        ].abs() > 1e-8,
        scenario_attribution["Residual_PnL"].abs()
        / scenario_attribution[
            "Full_Revaluation_PnL"
        ].abs(),
        np.nan,
    )

    # Gross residual across all individual positions
    scenario_attribution["Gross Residual %"] = np.where(
        scenario_attribution["Gross_Full_PnL"] > 1e-8,
        scenario_attribution["Gross_Residual"]
        / scenario_attribution["Gross_Full_PnL"],
        np.nan,
    )

    # Confirm the attribution identity
    scenario_attribution["Attribution Check"] = (
        scenario_attribution["Full_Revaluation_PnL"]
        - scenario_attribution["Approximate_PnL"]
        - scenario_attribution["Residual_PnL"]
    )

    maximum_check_error = (
        scenario_attribution["Attribution Check"]
        .abs()
        .max()
    )

    if maximum_check_error <= 1e-6:
        logger.info(
            "Greek attribution accounting check passed"
        )
    else:
        logger.warning(
            "Greek attribution check failed with maximum error %.8f",
            maximum_check_error,
        )

    logger.info(
        "Greek attribution completed for %d position-scenario rows "
        "and %d scenarios",
        len(attribution_results),
        len(scenario_attribution),
    )

    return attribution_results, scenario_attribution


def analyse_greek_attribution(
    scenario_attribution: pd.DataFrame,
    attribution_results: pd.DataFrame,
    scenarios: pd.DataFrame,
    base_scenario_id: str = "BASE",
    number_to_show: int = 10,
):
    """
    Analyse the accuracy of portfolio Greek P&L attribution.

    Returns:
        scenario_attribution:
            Enriched scenario-level attribution table.

        attribution_summary:
            Overall approximation-accuracy statistics.

        small_scenario_summary:
            Accuracy statistics for relatively small shocks.

        worst_attribution_scenarios:
            Scenarios with the largest approximation errors.

        ticker_attribution:
            Greek attribution aggregated by ticker and scenario.
    """

    logger.info("Starting Greek attribution analysis")

    scenario_attribution = scenario_attribution.copy()
    attribution_results = attribution_results.copy()

    
    # Attach the original scenario shocks
    scenario_details = (
        scenarios[
            [
                "Scenario ID",
                "Spot Shock",
                "Volatility Shock",
                "Days Forward",
                "Rate Shock",
            ]
        ]
        .drop_duplicates("Scenario ID")
    )

    # Avoid duplicated shock columns if this function is run twice
    scenario_attribution = scenario_attribution.drop(
        columns=[
            "Spot Shock",
            "Volatility Shock",
            "Days Forward",
            "Rate Shock",
        ],
        errors="ignore",
    )

    scenario_attribution = scenario_attribution.merge(
        scenario_details,
        on="Scenario ID",
        how="left",
        validate="one_to_one",
    )

    logger.info(
        "Scenario shocks attached to %d attribution rows",
        len(scenario_attribution),
    )

    
    # Confirm the attribution accounting identity
    scenario_attribution["Attribution Check"] = (
        scenario_attribution["Full_Revaluation_PnL"]
        - scenario_attribution["Approximate_PnL"]
        - scenario_attribution["Residual_PnL"]
    )

    maximum_check_error = (
        scenario_attribution["Attribution Check"]
        .abs()
        .max()
    )

    if maximum_check_error > 1e-6:
        logger.warning(
            "Greek attribution accounting check failed: %.10f",
            maximum_check_error,
        )
    else:
        logger.info(
            "Greek attribution accounting check passed"
        )

    
    # Calculate net residual percentage

    # This measures the residual against net portfolio P&L.
    # It can become unstable when net portfolio P&L is close to zero.
    scenario_attribution["Net Residual %"] = np.where(
        scenario_attribution[
            "Full_Revaluation_PnL"
        ].abs() > 1e-8,
        scenario_attribution["Residual_PnL"].abs()
        / scenario_attribution[
            "Full_Revaluation_PnL"
        ].abs(),
        np.nan,
    )

    
    # Calculate gross residual percentage

    # Aggregate absolute position-level P&Ls before calculating
    # the percentage. This is more stable than the net measure.
    gross_attribution = (
        attribution_results
        .groupby(
            "Scenario ID",
            as_index=False,
        )
        .agg(
            Gross_Full_PnL=(
                "Scenario PnL",
                lambda values: values.abs().sum(),
            ),
            Gross_Residual=(
                "Residual PnL",
                lambda values: values.abs().sum(),
            ),
        )
    )

    gross_attribution["Gross Residual %"] = np.where(
        gross_attribution["Gross_Full_PnL"] > 1e-8,
        gross_attribution["Gross_Residual"]
        / gross_attribution["Gross_Full_PnL"],
        np.nan,
    )

    # calculate_greek_profit_loss already adds these columns.
    # Remove them before merging the recalculated versions to prevent
    # pandas creating Gross Residual %_x and Gross Residual %_y.
    scenario_attribution = scenario_attribution.drop(
        columns=[
            "Gross_Full_PnL",
            "Gross_Residual",
            "Gross Residual %",
        ],
        errors="ignore",
    )

    scenario_attribution = scenario_attribution.merge(
        gross_attribution,
        on="Scenario ID",
        how="left",
        validate="one_to_one",
    )

    
    # Validate the unchanged scenario
    base_attribution = scenario_attribution.loc[
        scenario_attribution["Scenario ID"].eq(
            base_scenario_id
        )
    ].copy()

    if not base_attribution.empty:
        base_columns = [
            "Full_Revaluation_PnL",
            "Delta_PnL",
            "Gamma_PnL",
            "Vega_PnL",
            "Theta_PnL",
            "Rho_PnL",
            "Approximate_PnL",
            "Residual_PnL",
        ]

        base_values = (
            base_attribution[base_columns]
            .to_numpy(dtype=float)
        )

        base_passed = np.allclose(
            base_values,
            0.0,
            atol=1e-6,
        )

        base_attribution[
            "Attribution Validation Passed"
        ] = base_passed

        if base_passed:
            logger.info(
                "BASE Greek attribution validation passed"
            )
        else:
            logger.warning(
                "BASE Greek attribution validation failed"
            )

    
    # Remove BASE from accuracy statistics
    attribution_distribution = (
        scenario_attribution.loc[
            ~scenario_attribution[
                "Scenario ID"
            ].eq(base_scenario_id)
        ]
        .copy()
    )

    if attribution_distribution.empty:
        raise ValueError(
            "No non-base attribution scenarios are available."
        )

    
    #  Create overall attribution statistics
    attribution_summary = pd.DataFrame(
        [
            {
                "Number of Scenarios": len(
                    attribution_distribution
                ),
                "Mean Absolute Residual": (
                    attribution_distribution[
                        "Residual_PnL"
                    ]
                    .abs()
                    .mean()
                ),
                "Median Absolute Residual": (
                    attribution_distribution[
                        "Residual_PnL"
                    ]
                    .abs()
                    .median()
                ),
                "95th Percentile Absolute Residual": (
                    attribution_distribution[
                        "Residual_PnL"
                    ]
                    .abs()
                    .quantile(0.95)
                ),
                "Mean Gross Residual %": (
                    attribution_distribution[
                        "Gross Residual %"
                    ]
                    .mean()
                ),
                "Median Gross Residual %": (
                    attribution_distribution[
                        "Gross Residual %"
                    ]
                    .median()
                ),
                "95th Percentile Gross Residual %": (
                    attribution_distribution[
                        "Gross Residual %"
                    ]
                    .quantile(0.95)
                ),
            }
        ]
    )

    
    # Analyse relatively small shocks separately
    small_scenarios = attribution_distribution.loc[
        attribution_distribution[
            "Spot Shock"
        ].abs().le(0.02)
        & attribution_distribution[
            "Volatility Shock"
        ].abs().le(0.02)
        & attribution_distribution[
            "Days Forward"
        ].le(3)
        & attribution_distribution[
            "Rate Shock"
        ].abs().le(0.0025)
    ].copy()

    small_scenario_summary = pd.DataFrame(
        [
            {
                "Number of Small Scenarios": len(
                    small_scenarios
                ),
                "Mean Absolute Residual": (
                    small_scenarios[
                        "Residual_PnL"
                    ]
                    .abs()
                    .mean()
                ),
                "Median Absolute Residual": (
                    small_scenarios[
                        "Residual_PnL"
                    ]
                    .abs()
                    .median()
                ),
                "Mean Gross Residual %": (
                    small_scenarios[
                        "Gross Residual %"
                    ]
                    .mean()
                ),
                "Median Gross Residual %": (
                    small_scenarios[
                        "Gross Residual %"
                    ]
                    .median()
                ),
            }
        ]
    )

    
    #  Find scenarios where Greeks performed worst
    worst_attribution_scenarios = (
        attribution_distribution
        .nlargest(
            number_to_show,
            "Gross Residual %",
        )
        [
            [
                "Scenario ID",
                "Spot Shock",
                "Volatility Shock",
                "Days Forward",
                "Rate Shock",
                "Full_Revaluation_PnL",
                "Delta_PnL",
                "Gamma_PnL",
                "Vega_PnL",
                "Theta_PnL",
                "Rho_PnL",
                "Approximate_PnL",
                "Residual_PnL",
                "Gross Residual %",
            ]
        ]
        .reset_index(drop=True)
    )

    # Create ticker-level attribution
    ticker_attribution = (
        attribution_results
        .groupby(
            [
                "Scenario ID",
                "Ticker",
            ],
            as_index=False,
        )
        .agg(
            Full_Revaluation_PnL=(
                "Scenario PnL",
                "sum",
            ),
            Delta_PnL=("Delta PnL", "sum"),
            Gamma_PnL=("Gamma PnL", "sum"),
            Vega_PnL=("Vega PnL", "sum"),
            Theta_PnL=("Theta PnL", "sum"),
            Rho_PnL=("Rho PnL", "sum"),
            Approximate_PnL=(
                "Approximate PnL",
                "sum",
            ),
            Residual_PnL=(
                "Residual PnL",
                "sum",
            ),
            Gross_Full_PnL=(
                "Scenario PnL",
                lambda values: values.abs().sum(),
            ),
            Gross_Residual=(
                "Residual PnL",
                lambda values: values.abs().sum(),
            ),
        )
    )

    ticker_attribution["Gross Residual %"] = np.where(
        ticker_attribution["Gross_Full_PnL"] > 1e-8,
        ticker_attribution["Gross_Residual"]
        / ticker_attribution["Gross_Full_PnL"],
        np.nan,
    )

    logger.info(
        "Greek attribution analysis completed for %d scenarios",
        len(attribution_distribution),
    )

    return {
        "scenario_attribution": scenario_attribution,
        "attribution_distribution": attribution_distribution,
        "attribution_summary": attribution_summary,
        "small_scenarios": small_scenarios,
        "small_scenario_summary": small_scenario_summary,
        "worst_attribution_scenarios": (
            worst_attribution_scenarios
        ),
        "ticker_attribution": ticker_attribution,
        "base_attribution": base_attribution,
    }
