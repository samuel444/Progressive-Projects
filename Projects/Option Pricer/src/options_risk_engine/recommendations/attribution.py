from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from options_risk_engine.risk.attribution import (
    calculate_greek_profit_loss,
)


logger = logging.getLogger(__name__)


GREEK_COMPONENT_COLUMNS = {
    "Delta": "Delta PnL",
    "Gamma": "Gamma PnL",
    "Vega": "Vega PnL",
    "Theta": "Theta PnL",
    "Rho": "Rho PnL",
}


DRIVER_LABELS = {
    "Delta": "Directional",
    "Gamma": "Convexity",
    "Vega": "Volatility",
    "Theta": "Time Decay",
    "Rho": "Interest Rate",
}

def create_contract_attribution_results(
    scenario_results: pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Greek P&L attribution for each standardised contract
    under each scenario.

    Greek attribution explains scenario movement measured from the
    base risk-model value. It does not include the initial difference
    between the purchase ask and the base model value.
    """

    required_scenario_columns = {
        "Scenario ID",
        "Contract Symbol",
        "Ticker",
        "Option Type",
        "Shocked Spot",
        "Scenario PnL",
        "Spot Shock",
        "Volatility Shock",
        "Rate Shock",
        "Days Forward",
    }

    missing_scenario_columns = (
        required_scenario_columns
        .difference(scenario_results.columns)
    )

    if missing_scenario_columns:
        raise KeyError(
            "Scenario results are missing columns: "
            + ", ".join(
                sorted(missing_scenario_columns)
            )
        )

    required_position_columns = {
        "contractSymbol",
        "Current Stock Price",
        "Position Delta",
        "Position Gamma",
        "Position Vega",
        "Position Theta",
        "Position Rho",
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

    position_inputs = positions[
        [
            "contractSymbol",
            "Current Stock Price",
            "Position Delta",
            "Position Gamma",
            "Position Vega",
            "Position Theta",
            "Position Rho",
        ]
    ].copy()

    if position_inputs[
        "contractSymbol"
    ].duplicated().any():
        raise ValueError(
            "Standardised positions contain duplicate "
            "contract symbols."
        )

    expanded_results = scenario_results.merge(
        position_inputs,
        left_on="Contract Symbol",
        right_on="contractSymbol",
        how="left",
        validate="many_to_one",
    )

    if len(expanded_results) != len(scenario_results):
        raise AssertionError(
            "The attribution merge changed the number of "
            "scenario rows."
        )

    missing_position_inputs = (
        expanded_results[
            [
                "Current Stock Price",
                "Position Delta",
                "Position Gamma",
                "Position Vega",
                "Position Theta",
                "Position Rho",
            ]
        ]
        .isna()
        .any(axis=1)
    )

    if missing_position_inputs.any():
        missing_contracts = (
            expanded_results.loc[
                missing_position_inputs,
                "Contract Symbol",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Greek inputs are missing for: "
            + ", ".join(
                map(str, missing_contracts[:10])
            )
        )

    (
        attribution_results,
        _,
    ) = calculate_greek_profit_loss(
        expanded_results
    )

    logger.info(
        "Calculated Greek attribution for %d "
        "contract-scenario rows",
        len(attribution_results),
    )

    return attribution_results


def create_contract_attribution_summary(
    attribution_results: pd.DataFrame,
    contract_scenario_summary: pd.DataFrame,
    base_scenario_id: str = "BASE",
    local_spot_limit: float = 0.05,
    local_volatility_limit: float = 0.03,
    local_days_limit: int = 10,
    local_rate_limit: float = 0.005,
    minimum_local_scenarios: int = 20,
) -> pd.DataFrame:
    """
    Aggregate Greek attribution into one explanation per contract.

    A smaller subset of local scenarios is also examined because
    Greek approximations are designed primarily for relatively small
    changes around the current market state.
    """

    required_attribution_columns = {
        "Scenario ID",
        "Contract Symbol",
        "Scenario PnL",
        "Delta PnL",
        "Gamma PnL",
        "Vega PnL",
        "Theta PnL",
        "Rho PnL",
        "Approximate PnL",
        "Residual PnL",
        "Spot Shock",
        "Volatility Shock",
        "Rate Shock",
        "Days Forward",
    }

    missing_attribution_columns = (
        required_attribution_columns
        .difference(attribution_results.columns)
    )

    if missing_attribution_columns:
        raise KeyError(
            "Attribution results are missing columns: "
            + ", ".join(
                sorted(missing_attribution_columns)
            )
        )

    required_summary_columns = {
        "Contract Symbol",
        "Premium Paid",
        "Base_Mark_to_Ask_PnL",
        "Mean_Purchase_PnL",
        "Mean_Purchase_Return",
        "Mean_Scenario_Movement_PnL",
        "Initial Recommended Action",
    }

    missing_summary_columns = (
        required_summary_columns
        .difference(contract_scenario_summary.columns)
    )

    if missing_summary_columns:
        raise KeyError(
            "Contract scenario summary is missing columns: "
            + ", ".join(
                sorted(missing_summary_columns)
            )
        )

    non_base = (
        attribution_results.loc[
            ~attribution_results[
                "Scenario ID"
            ].eq(base_scenario_id)
        ]
        .copy()
    )

    if non_base.empty:
        raise ValueError(
            "No non-BASE attribution rows were available."
        )

    contract_attribution = (
        non_base
        .groupby(
            "Contract Symbol",
            as_index=False,
        )
        .agg(
            Attribution_Scenario_Count=(
                "Scenario ID",
                "nunique",
            ),

            Mean_Full_Movement_PnL=(
                "Scenario PnL",
                "mean",
            ),
            Mean_Approximate_PnL=(
                "Approximate PnL",
                "mean",
            ),
            Mean_Residual_PnL=(
                "Residual PnL",
                "mean",
            ),

            Median_Absolute_Residual_PnL=(
                "Residual PnL",
                lambda values: (
                    values.abs().median()
                ),
            ),
            P95_Absolute_Residual_PnL=(
                "Residual PnL",
                lambda values: (
                    values.abs().quantile(0.95)
                ),
            ),

            Gross_Full_Movement_PnL=(
                "Scenario PnL",
                lambda values: values.abs().sum(),
            ),
            Gross_Residual_PnL=(
                "Residual PnL",
                lambda values: values.abs().sum(),
            ),

            Mean_Delta_PnL=(
                "Delta PnL",
                "mean",
            ),
            Mean_Gamma_PnL=(
                "Gamma PnL",
                "mean",
            ),
            Mean_Vega_PnL=(
                "Vega PnL",
                "mean",
            ),
            Mean_Theta_PnL=(
                "Theta PnL",
                "mean",
            ),
            Mean_Rho_PnL=(
                "Rho PnL",
                "mean",
            ),

            Mean_Absolute_Delta_PnL=(
                "Delta PnL",
                lambda values: values.abs().mean(),
            ),
            Mean_Absolute_Gamma_PnL=(
                "Gamma PnL",
                lambda values: values.abs().mean(),
            ),
            Mean_Absolute_Vega_PnL=(
                "Vega PnL",
                lambda values: values.abs().mean(),
            ),
            Mean_Absolute_Theta_PnL=(
                "Theta PnL",
                lambda values: values.abs().mean(),
            ),
            Mean_Absolute_Rho_PnL=(
                "Rho PnL",
                lambda values: values.abs().mean(),
            ),
        )
    )

    contract_attribution[
        "Gross_Residual_Ratio"
    ] = np.where(
        contract_attribution[
            "Gross_Full_Movement_PnL"
        ] > 1e-10,
        (
            contract_attribution[
                "Gross_Residual_PnL"
            ]
            / contract_attribution[
                "Gross_Full_Movement_PnL"
            ]
        ),
        np.nan,
    )

    # Greeks are local approximations, so evaluate their accuracy
    # separately under relatively small shocks.
    local_mask = (
        non_base[
            "Spot Shock"
        ].abs().le(local_spot_limit)
        & non_base[
            "Volatility Shock"
        ].abs().le(local_volatility_limit)
        & non_base[
            "Days Forward"
        ].le(local_days_limit)
        & non_base[
            "Rate Shock"
        ].abs().le(local_rate_limit)
    )

    local_results = (
        non_base.loc[
            local_mask
        ]
        .copy()
    )

    local_summary = (
        local_results
        .groupby(
            "Contract Symbol",
            as_index=False,
        )
        .agg(
            Local_Scenario_Count=(
                "Scenario ID",
                "nunique",
            ),
            Local_Gross_Full_Movement_PnL=(
                "Scenario PnL",
                lambda values: values.abs().sum(),
            ),
            Local_Gross_Residual_PnL=(
                "Residual PnL",
                lambda values: values.abs().sum(),
            ),
            Local_Median_Absolute_Residual_PnL=(
                "Residual PnL",
                lambda values: values.abs().median(),
            ),
        )
    )

    local_summary[
        "Local_Gross_Residual_Ratio"
    ] = np.where(
        local_summary[
            "Local_Gross_Full_Movement_PnL"
        ] > 1e-10,
        (
            local_summary[
                "Local_Gross_Residual_PnL"
            ]
            / local_summary[
                "Local_Gross_Full_Movement_PnL"
            ]
        ),
        np.nan,
    )

    summary = (
        contract_scenario_summary
        .merge(
            contract_attribution,
            on="Contract Symbol",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            local_summary,
            on="Contract Symbol",
            how="left",
            validate="one_to_one",
        )
    )

    summary["Local_Scenario_Count"] = (
        summary[
            "Local_Scenario_Count"
        ]
        .fillna(0)
        .astype(int)
    )

    # Convert average Greek contributions into returns relative
    # to the premium paid.
    return_columns = {
        "Mean_Full_Movement_PnL": (
            "Mean_Full_Movement_Return"
        ),
        "Mean_Approximate_PnL": (
            "Mean_Approximate_Return"
        ),
        "Mean_Residual_PnL": (
            "Mean_Residual_Return"
        ),
        "Mean_Delta_PnL": (
            "Mean_Delta_Return"
        ),
        "Mean_Gamma_PnL": (
            "Mean_Gamma_Return"
        ),
        "Mean_Vega_PnL": (
            "Mean_Vega_Return"
        ),
        "Mean_Theta_PnL": (
            "Mean_Theta_Return"
        ),
        "Mean_Rho_PnL": (
            "Mean_Rho_Return"
        ),
    }

    for pnl_column, return_column in (
        return_columns.items()
    ):
        summary[return_column] = (
            summary[pnl_column]
            / summary["Premium Paid"]
        )

    absolute_component_columns = [
        "Mean_Absolute_Delta_PnL",
        "Mean_Absolute_Gamma_PnL",
        "Mean_Absolute_Vega_PnL",
        "Mean_Absolute_Theta_PnL",
        "Mean_Absolute_Rho_PnL",
    ]

    summary[
        "Mean_Absolute_Greek_PnL_Total"
    ] = (
        summary[
            absolute_component_columns
        ]
        .sum(axis=1)
    )

    component_share_columns = {}

    for greek in GREEK_COMPONENT_COLUMNS:
        absolute_column = (
            f"Mean_Absolute_{greek}_PnL"
        )
        share_column = (
            f"{greek}_Driver_Share"
        )

        summary[share_column] = np.where(
            summary[
                "Mean_Absolute_Greek_PnL_Total"
            ] > 1e-10,
            (
                summary[absolute_column]
                / summary[
                    "Mean_Absolute_Greek_PnL_Total"
                ]
            ),
            np.nan,
        )

        component_share_columns[greek] = (
            share_column
        )

    share_columns = list(
        component_share_columns.values()
    )

    dominant_share_column = (
        summary[
            share_columns
        ]
        .idxmax(axis=1)
    )

    share_to_greek = {
        share_column: greek
        for greek, share_column
        in component_share_columns.items()
    }

    summary["Dominant Greek"] = (
        dominant_share_column.map(
            share_to_greek
        )
    )

    summary["Dominant Driver Share"] = (
        summary[
            share_columns
        ]
        .max(axis=1)
    )

    sorted_driver_shares = np.sort(
        summary[
            share_columns
        ].fillna(0).to_numpy(),
        axis=1,
    )

    summary["Second Driver Share"] = (
        sorted_driver_shares[:, -2]
    )

    mixed_driver = (
        summary[
            "Dominant Driver Share"
        ].lt(0.40)
        | (
            summary[
                "Dominant Driver Share"
            ]
            - summary[
                "Second Driver Share"
            ]
        ).lt(0.10)
    )

    summary["Risk Driver"] = (
        summary[
            "Dominant Greek"
        ]
        .map(DRIVER_LABELS)
    )

    summary.loc[
        mixed_driver,
        "Risk Driver",
    ] = "Mixed"

    summary["Attribution Reliability"] = np.select(
        [
            summary[
                "Local_Scenario_Count"
            ].lt(minimum_local_scenarios),

            summary[
                "Local_Gross_Residual_Ratio"
            ].le(0.10),

            summary[
                "Local_Gross_Residual_Ratio"
            ].le(0.25),
        ],
        [
            "Insufficient Local Scenarios",
            "High",
            "Moderate",
        ],
        default="Low",
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
        "Created Greek attribution summaries for %d contracts",
        len(summary),
    )

    return summary

def validate_contract_attribution_summary(
    summary: pd.DataFrame,
    original_contract_summary: pd.DataFrame,
    atol: float = 1e-8,
) -> dict[str, bool]:
    """Validate the contract-level Greek attribution summary."""

    required_columns = {
        "Contract Symbol",
        "Premium Paid",
        "Mean_Purchase_PnL",
        "Base_Mark_to_Ask_PnL",
        "Mean_Scenario_Movement_PnL",
        "Mean_Full_Movement_PnL",
        "Mean_Approximate_PnL",
        "Mean_Residual_PnL",
        "Delta_Driver_Share",
        "Gamma_Driver_Share",
        "Vega_Driver_Share",
        "Theta_Driver_Share",
        "Rho_Driver_Share",
        "Risk Driver",
        "Attribution Reliability",
    }

    missing_columns = (
        required_columns
        .difference(summary.columns)
    )

    if missing_columns:
        raise KeyError(
            "Contract attribution summary is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    driver_share_total = (
        summary[
            [
                "Delta_Driver_Share",
                "Gamma_Driver_Share",
                "Vega_Driver_Share",
                "Theta_Driver_Share",
                "Rho_Driver_Share",
            ]
        ]
        .sum(axis=1)
    )

    recognised_drivers = {
        "Directional",
        "Convexity",
        "Volatility",
        "Time Decay",
        "Interest Rate",
        "Mixed",
    }

    recognised_reliability_labels = {
        "High",
        "Moderate",
        "Low",
        "Insufficient Local Scenarios",
    }

    checks = {
        "same_number_of_contracts": (
            len(summary)
            == len(original_contract_summary)
        ),

        "unique_contract_symbols": (
            not summary[
                "Contract Symbol"
            ].duplicated().any()
        ),

        "movement_matches_section_19": bool(
            np.allclose(
                summary[
                    "Mean_Full_Movement_PnL"
                ],
                summary[
                    "Mean_Scenario_Movement_PnL"
                ],
                atol=atol,
                equal_nan=False,
            )
        ),

        "greek_attribution_identity": bool(
            np.allclose(
                summary[
                    "Mean_Full_Movement_PnL"
                ],
                (
                    summary[
                        "Mean_Approximate_PnL"
                    ]
                    + summary[
                        "Mean_Residual_PnL"
                    ]
                ),
                atol=atol,
                equal_nan=False,
            )
        ),

        "purchase_pnl_identity": bool(
            np.allclose(
                summary[
                    "Mean_Purchase_PnL"
                ],
                (
                    summary[
                        "Base_Mark_to_Ask_PnL"
                    ]
                    + summary[
                        "Mean_Full_Movement_PnL"
                    ]
                ),
                atol=atol,
                equal_nan=False,
            )
        ),

        "driver_shares_sum_to_one": bool(
            np.allclose(
                driver_share_total,
                1.0,
                atol=atol,
                equal_nan=False,
            )
        ),

        "driver_shares_valid": (
            summary[
                [
                    "Delta_Driver_Share",
                    "Gamma_Driver_Share",
                    "Vega_Driver_Share",
                    "Theta_Driver_Share",
                    "Rho_Driver_Share",
                ]
            ]
            .ge(0)
            .all()
            .all()
            and
            summary[
                [
                    "Delta_Driver_Share",
                    "Gamma_Driver_Share",
                    "Vega_Driver_Share",
                    "Theta_Driver_Share",
                    "Rho_Driver_Share",
                ]
            ]
            .le(1)
            .all()
            .all()
        ),

        "recognised_risk_drivers": (
            summary[
                "Risk Driver"
            ]
            .isin(recognised_drivers)
            .all()
        ),

        "recognised_reliability_labels": (
            summary[
                "Attribution Reliability"
            ]
            .isin(
                recognised_reliability_labels
            )
            .all()
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed_checks:
        raise AssertionError(
            "Contract attribution validation failed: "
            f"{failed_checks}"
        )

    logger.info(
        "All contract attribution checks passed"
    )

    return checks


def create_contract_driver_comparison(
    attribution_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Greek drivers across initial recommendation groups."""

    required_columns = {
        "Contract Symbol",
        "Initial Recommended Action",
        "Risk Driver",
        "Mean_Purchase_Return",
        "Scenario_Profit_Frequency",
        "Expected_Shortfall_5pct_Return",
        "Gross_Residual_Ratio",
        "Local_Gross_Residual_Ratio",
    }

    missing_columns = (
        required_columns
        .difference(attribution_summary.columns)
    )

    if missing_columns:
        raise KeyError(
            "Cannot create driver comparison; missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    comparison = (
        attribution_summary
        .groupby(
            [
                "Initial Recommended Action",
                "Risk Driver",
            ],
            observed=True,
        )
        .agg(
            Contracts=(
                "Contract Symbol",
                "count",
            ),
            Median_Mean_Purchase_Return=(
                "Mean_Purchase_Return",
                "median",
            ),
            Median_Profit_Frequency=(
                "Scenario_Profit_Frequency",
                "median",
            ),
            Median_Expected_Shortfall=(
                "Expected_Shortfall_5pct_Return",
                "median",
            ),
            Median_All_Scenario_Residual=(
                "Gross_Residual_Ratio",
                "median",
            ),
            Median_Local_Residual=(
                "Local_Gross_Residual_Ratio",
                "median",
            ),
        )
        .reset_index()
    )

    return comparison