from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


DEFAULT_SCORE_WEIGHTS = {
    "Forecast_Edge_Score": 0.25,
    "Mean_Return_Score": 0.20,
    "Median_Return_Score": 0.10,
    "Profit_Frequency_Score": 0.15,
    "Tail_Risk_Score": 0.20,
    "Liquidity_Score": 0.10,
}


FINAL_ACTION_ORDER = {
    "Candidate": 0,
    "Watchlist": 1,
    "No Trade": 2,
    "Risk Rejection": 3,
    "Liquidity Rejection": 4,
}

def _percentile_score(
    values: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Convert a numeric series into cross-sectional scores from 0 to 100.

    Contracts are ranked relative to the other contracts in the
    current analysis. Equal values receive equal average ranks.
    """

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    scores = pd.Series(
        np.nan,
        index=values.index,
        dtype=float,
    )

    valid_mask = numeric_values.notna()

    if not valid_mask.any():
        return scores

    valid_values = numeric_values.loc[
        valid_mask
    ]

    if valid_values.nunique() == 1:
        scores.loc[valid_mask] = 50.0
        return scores

    scores.loc[valid_mask] = (
        valid_values.rank(
            method="average",
            pct=True,
            ascending=higher_is_better,
        )
        * 100
    )

    return scores

def _create_attribution_warning(
    row: pd.Series,
) -> str:
    """Describe whether the Greek explanation should be treated cautiously."""

    reliability = row.get(
        "Attribution Reliability",
        "Unknown",
    )

    if reliability == "Low":
        return (
            "Greek explanation is unreliable under the "
            "local scenario subset; use full repricing."
        )

    if reliability == "Insufficient Local Scenarios":
        return (
            "Too few local scenarios were available to "
            "assess Greek-attribution reliability."
        )

    if reliability == "Moderate":
        return (
            "Greek attribution is approximate; full "
            "repricing remains the primary result."
        )

    if reliability == "High":
        return "Greek attribution is reliable under local shocks."

    return "Attribution reliability is unavailable."


def _create_final_reason(
    row: pd.Series,
) -> str:
    """Create a concise explanation of the final classification."""

    final_action = row[
        "Final Recommended Action"
    ]

    risk_driver = row.get(
        "Risk Driver",
        "Unknown",
    )

    final_score = row[
        "Final Score"
    ]

    profit_frequency = row[
        "Scenario_Profit_Frequency"
    ]

    mean_return = row[
        "Mean_Purchase_Return"
    ]

    expected_shortfall = row[
        "Expected_Shortfall_5pct_Return"
    ]

    if final_action == "Liquidity Rejection":
        failed_rules = []

        if row["Spread Rejection"]:
            failed_rules.append("spread is too wide")

        if row["Volume Rejection"]:
            failed_rules.append("volume is too low")

        if row["Open Interest Rejection"]:
            failed_rules.append(
                "open interest is too low"
            )

        if row["Missing Liquidity Data"]:
            failed_rules.append(
                "liquidity data is incomplete"
            )

        return (
            "Rejected because "
            + ", ".join(failed_rules)
            + "."
        )

    if final_action == "Risk Rejection":
        return (
            "Rejected because the contract combines "
            f"bottom-tail risk of {expected_shortfall:.1%} "
            f"with a scenario gain frequency of "
            f"{profit_frequency:.1%}."
        )

    if final_action == "Candidate":
        return (
            f"Strong composite score of {final_score:.1f}, "
            f"positive mean return of {mean_return:.1%}, "
            f"and scenario gain frequency of "
            f"{profit_frequency:.1%}. Primary risk driver: "
            f"{risk_driver}."
        )

    if final_action == "Watchlist":
        return (
            f"Positive valuation and scenario evidence, "
            f"but the score or consistency is below the "
            f"Candidate requirement. Composite score: "
            f"{final_score:.1f}. Primary risk driver: "
            f"{risk_driver}."
        )

    if row["Initial Recommended Action"] == "Do Not Buy":
        return (
            "The forecast-volatility value does not exceed "
            "the purchase ask sufficiently to support a trade."
        )

    if row["Mean_Purchase_Return"] <= 0:
        return (
            "The average scenario purchase return is not "
            "positive after paying the ask."
        )

    return (
        f"The composite score of {final_score:.1f} is below "
        "the final shortlist requirements."
    )

def create_final_contract_ranking(
    contract_summary: pd.DataFrame,
    max_spread_pct: float,
    min_volume: int,
    min_open_interest: int,
    candidate_score: float = 70.0,
    watchlist_score: float = 55.0,
    candidate_profit_frequency: float = 0.50,
    minimum_mean_return: float = 0.0,
    risk_tail_quantile: float = 0.10,
    risk_profit_frequency: float = 0.35,
    score_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """
    Produce the final risk-aware contract classification and ranking.

    Hard liquidity and risk rejections are applied before the
    Candidate and Watchlist rules.
    """

    if score_weights is None:
        score_weights = DEFAULT_SCORE_WEIGHTS

    score_weights = dict(score_weights)

    required_weight_columns = set(
        DEFAULT_SCORE_WEIGHTS
    )

    if set(score_weights) != required_weight_columns:
        raise ValueError(
            "score_weights must contain exactly: "
            + ", ".join(
                sorted(required_weight_columns)
            )
        )

    weight_total = sum(
        score_weights.values()
    )

    if not np.isclose(
        weight_total,
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Final-score weights must sum to 1.0. "
            f"Received {weight_total:.6f}."
        )

    if not 0 < risk_tail_quantile < 0.5:
        raise ValueError(
            "risk_tail_quantile must be between "
            "0 and 0.5."
        )

    required_columns = {
        "Contract Symbol",
        "Ticker",
        "Initial Recommended Action",
        "Forecast Edge Return",
        "Mean_Purchase_Return",
        "Median_Purchase_Return",
        "Scenario_Profit_Frequency",
        "Fifth_Percentile_Purchase_Return",
        "Expected_Shortfall_5pct_Return",
        "SpreadPct",
        "volume",
        "openInterest",
        "Risk Driver",
        "Attribution Reliability",
    }

    missing_columns = (
        required_columns
        .difference(contract_summary.columns)
    )

    if missing_columns:
        raise KeyError(
            "Contract summary is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    result = contract_summary.copy()

    if result[
        "Contract Symbol"
    ].duplicated().any():
        raise ValueError(
            "Contract symbols must be unique before "
            "creating the final ranking."
        )

    # ----------------------------------------------------------
    # 1. Score forecast valuation
    # ----------------------------------------------------------

    result["Forecast_Edge_Score"] = (
        _percentile_score(
            result["Forecast Edge Return"],
            higher_is_better=True,
        )
    )

    # ----------------------------------------------------------
    # 2. Score average and median scenario outcomes
    # ----------------------------------------------------------

    result["Mean_Return_Score"] = (
        _percentile_score(
            result["Mean_Purchase_Return"],
            higher_is_better=True,
        )
    )

    result["Median_Return_Score"] = (
        _percentile_score(
            result["Median_Purchase_Return"],
            higher_is_better=True,
        )
    )

    # This is already measured between zero and one.
    result["Profit_Frequency_Score"] = (
        result[
            "Scenario_Profit_Frequency"
        ]
        .clip(0, 1)
        * 100
    )

    # ----------------------------------------------------------
    # 3. Score lower-tail risk
    # ----------------------------------------------------------

    result[
        "Fifth_Percentile_Score"
    ] = _percentile_score(
        result[
            "Fifth_Percentile_Purchase_Return"
        ],
        higher_is_better=True,
    )

    result[
        "Expected_Shortfall_Score"
    ] = _percentile_score(
        result[
            "Expected_Shortfall_5pct_Return"
        ],
        higher_is_better=True,
    )

    result["Tail_Risk_Score"] = (
        result[
            [
                "Fifth_Percentile_Score",
                "Expected_Shortfall_Score",
            ]
        ]
        .mean(axis=1)
    )

    # ----------------------------------------------------------
    # 4. Score liquidity
    # ----------------------------------------------------------

    result["Spread_Liquidity_Score"] = (
        _percentile_score(
            result["SpreadPct"],
            higher_is_better=False,
        )
    )

    result["Volume_Liquidity_Score"] = (
        _percentile_score(
            result["volume"].fillna(0),
            higher_is_better=True,
        )
    )

    result[
        "Open_Interest_Liquidity_Score"
    ] = _percentile_score(
        result[
            "openInterest"
        ].fillna(0),
        higher_is_better=True,
    )

    result["Liquidity_Score"] = (
        result[
            [
                "Spread_Liquidity_Score",
                "Volume_Liquidity_Score",
                "Open_Interest_Liquidity_Score",
            ]
        ]
        .mean(axis=1)
    )

    # Missing scoring data receives zero rather than silently
    # producing an incomplete final score.
    component_columns = list(
        score_weights
    )

    result[component_columns] = (
        result[
            component_columns
        ]
        .fillna(0)
    )

    # ----------------------------------------------------------
    # 5. Calculate weighted final score
    # ----------------------------------------------------------

    result["Final Score"] = 0.0

    for score_column, weight in (
        score_weights.items()
    ):
        result["Final Score"] += (
            result[score_column]
            * weight
        )

    result["Final Score"] = (
        result["Final Score"]
        .clip(0, 100)
    )

    # ----------------------------------------------------------
    # 6. Apply hard liquidity gates
    # ----------------------------------------------------------

    result["Missing Liquidity Data"] = (
        result[
            [
                "SpreadPct",
                "volume",
                "openInterest",
            ]
        ]
        .isna()
        .any(axis=1)
    )

    result["Spread Rejection"] = (
        result["SpreadPct"]
        .gt(max_spread_pct)
    )

    result["Volume Rejection"] = (
        result["volume"]
        .fillna(0)
        .lt(min_volume)
    )

    result["Open Interest Rejection"] = (
        result["openInterest"]
        .fillna(0)
        .lt(min_open_interest)
    )

    result["Liquidity Rejection"] = (
        result["Missing Liquidity Data"]
        | result["Spread Rejection"]
        | result["Volume Rejection"]
        | result["Open Interest Rejection"]
    )

    # ----------------------------------------------------------
    # 7. Apply a relative lower-tail risk gate
    # ----------------------------------------------------------

    liquid_rows = ~result[
        "Liquidity Rejection"
    ]

    if liquid_rows.any():
        lower_tail_cutoff = float(
            result.loc[
                liquid_rows,
                "Expected_Shortfall_5pct_Return",
            ]
            .quantile(risk_tail_quantile)
        )
    else:
        lower_tail_cutoff = np.nan

    result["Risk Tail Cutoff"] = (
        lower_tail_cutoff
    )

    result["Risk Rejection"] = False

    if np.isfinite(lower_tail_cutoff):
        result["Risk Rejection"] = (
            liquid_rows
            & result[
                "Expected_Shortfall_5pct_Return"
            ].le(lower_tail_cutoff)
            & result[
                "Scenario_Profit_Frequency"
            ].lt(risk_profit_frequency)
        )

    # ----------------------------------------------------------
    # 8. Produce final actions
    # ----------------------------------------------------------

    result[
        "Final Recommended Action"
    ] = "No Trade"

    result.loc[
        result["Liquidity Rejection"],
        "Final Recommended Action",
    ] = "Liquidity Rejection"

    result.loc[
        (
            ~result["Liquidity Rejection"]
            & result["Risk Rejection"]
        ),
        "Final Recommended Action",
    ] = "Risk Rejection"

    available_for_selection = (
        ~result["Liquidity Rejection"]
        & ~result["Risk Rejection"]
    )

    candidate_mask = (
        available_for_selection
        & result[
            "Initial Recommended Action"
        ].eq("Buy")
        & result[
            "Forecast Edge Return"
        ].gt(0)
        & result[
            "Mean_Purchase_Return"
        ].gt(minimum_mean_return)
        & result[
            "Scenario_Profit_Frequency"
        ].ge(candidate_profit_frequency)
        & result[
            "Final Score"
        ].ge(candidate_score)
    )

    result.loc[
        candidate_mask,
        "Final Recommended Action",
    ] = "Candidate"

    watchlist_mask = (
        available_for_selection
        & ~candidate_mask
        & result[
            "Initial Recommended Action"
        ].isin(
            [
                "Buy",
                "Positive Edge",
            ]
        )
        & result[
            "Forecast Edge Return"
        ].gt(0)
        & result[
            "Mean_Purchase_Return"
        ].gt(minimum_mean_return)
        & result[
            "Final Score"
        ].ge(watchlist_score)
    )

    result.loc[
        watchlist_mask,
        "Final Recommended Action",
    ] = "Watchlist"

    # ----------------------------------------------------------
    # 9. Add explanatory fields and ranks
    # ----------------------------------------------------------

    result["Attribution Warning"] = (
        result.apply(
            _create_attribution_warning,
            axis=1,
        )
    )

    result["Recommendation Change"] = (
        result[
            "Initial Recommended Action"
        ].astype(str)
        + " -> "
        + result[
            "Final Recommended Action"
        ].astype(str)
    )

    result[
        "Final Recommendation Reason"
    ] = result.apply(
        _create_final_reason,
        axis=1,
    )

    result["Overall Score Rank"] = (
        result["Final Score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    shortlist_mask = result[
        "Final Recommended Action"
    ].isin(
        [
            "Candidate",
            "Watchlist",
        ]
    )

    result["Final Shortlist Rank"] = (
        pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )
    )

    result.loc[
        shortlist_mask,
        "Final Shortlist Rank",
    ] = (
        result.loc[
            shortlist_mask,
            "Final Score",
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype("Int64")
    )

    result["Ticker Shortlist Rank"] = (
        pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )
    )

    result.loc[
        shortlist_mask,
        "Ticker Shortlist Rank",
    ] = (
        result.loc[
            shortlist_mask
        ]
        .groupby(
            "Ticker",
            observed=True,
        )["Final Score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype("Int64")
    )

    result["_Final_Action_Order"] = (
        result[
            "Final Recommended Action"
        ]
        .map(FINAL_ACTION_ORDER)
        .fillna(len(FINAL_ACTION_ORDER))
    )

    result = (
        result
        .sort_values(
            [
                "_Final_Action_Order",
                "Final Score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop(
            columns="_Final_Action_Order"
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Created final rankings for %d contracts",
        len(result),
    )

    return result

def validate_final_contract_ranking(
    ranking: pd.DataFrame,
    max_spread_pct: float,
    min_volume: int,
    min_open_interest: int,
    candidate_score: float = 70.0,
    watchlist_score: float = 55.0,
    candidate_profit_frequency: float = 0.50,
    minimum_mean_return: float = 0.0,
    risk_tail_quantile: float = 0.10,
    risk_profit_frequency: float = 0.35,
    score_weights: Mapping[str, float] | None = None,
    atol: float = 1e-8,
) -> dict[str, bool]:
    """Validate the final recommendation and scoring rules."""

    if score_weights is None:
        score_weights = DEFAULT_SCORE_WEIGHTS

    score_weights = dict(score_weights)

    required_columns = {
        "Contract Symbol",
        "Initial Recommended Action",
        "Final Recommended Action",
        "Final Score",
        "Forecast Edge Return",
        "Mean_Purchase_Return",
        "Scenario_Profit_Frequency",
        "Expected_Shortfall_5pct_Return",
        "SpreadPct",
        "volume",
        "openInterest",
        "Liquidity Rejection",
        "Risk Rejection",
        "Final Recommendation Reason",
    }.union(score_weights)

    missing_columns = (
        required_columns
        .difference(ranking.columns)
    )

    if missing_columns:
        raise KeyError(
            "Final ranking is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    expected_score = pd.Series(
        0.0,
        index=ranking.index,
    )

    for score_column, weight in (
        score_weights.items()
    ):
        expected_score += (
            ranking[score_column]
            * weight
        )

    expected_liquidity_rejection = (
        ranking[
            [
                "SpreadPct",
                "volume",
                "openInterest",
            ]
        ]
        .isna()
        .any(axis=1)
        | ranking[
            "SpreadPct"
        ].gt(max_spread_pct)
        | ranking[
            "volume"
        ].fillna(0).lt(min_volume)
        | ranking[
            "openInterest"
        ].fillna(0).lt(
            min_open_interest
        )
    )

    liquid_rows = (
        ~expected_liquidity_rejection
    )

    if liquid_rows.any():
        risk_tail_cutoff = float(
            ranking.loc[
                liquid_rows,
                "Expected_Shortfall_5pct_Return",
            ]
            .quantile(risk_tail_quantile)
        )

        expected_risk_rejection = (
            liquid_rows
            & ranking[
                "Expected_Shortfall_5pct_Return"
            ].le(risk_tail_cutoff)
            & ranking[
                "Scenario_Profit_Frequency"
            ].lt(risk_profit_frequency)
        )
    else:
        expected_risk_rejection = pd.Series(
            False,
            index=ranking.index,
        )

    candidate_rows = ranking[
        "Final Recommended Action"
    ].eq("Candidate")

    watchlist_rows = ranking[
        "Final Recommended Action"
    ].eq("Watchlist")

    recognised_actions = {
        "Candidate",
        "Watchlist",
        "No Trade",
        "Liquidity Rejection",
        "Risk Rejection",
    }

    checks = {
        "contracts_available": (
            len(ranking) > 0
        ),

        "unique_contract_symbols": (
            not ranking[
                "Contract Symbol"
            ].duplicated().any()
        ),

        "scores_between_zero_and_one_hundred": (
            ranking[
                "Final Score"
            ].between(0, 100).all()
        ),

        "weighted_score_matches": bool(
            np.allclose(
                ranking["Final Score"],
                expected_score,
                atol=atol,
                equal_nan=False,
            )
        ),

        "recognised_final_actions": (
            ranking[
                "Final Recommended Action"
            ]
            .isin(recognised_actions)
            .all()
        ),

        "liquidity_gate_matches": bool(
            np.array_equal(
                ranking[
                    "Liquidity Rejection"
                ].to_numpy(),
                expected_liquidity_rejection
                .to_numpy(),
            )
        ),

        "risk_gate_matches": bool(
            np.array_equal(
                ranking[
                    "Risk Rejection"
                ].to_numpy(),
                expected_risk_rejection
                .to_numpy(),
            )
        ),

        "candidate_rules_hold": (
            (
                ranking.loc[
                    candidate_rows,
                    "Initial Recommended Action",
                ].eq("Buy")
            ).all()
            and
            (
                ranking.loc[
                    candidate_rows,
                    "Forecast Edge Return",
                ].gt(0)
            ).all()
            and
            (
                ranking.loc[
                    candidate_rows,
                    "Mean_Purchase_Return",
                ].gt(minimum_mean_return)
            ).all()
            and
            (
                ranking.loc[
                    candidate_rows,
                    "Scenario_Profit_Frequency",
                ].ge(
                    candidate_profit_frequency
                )
            ).all()
            and
            (
                ranking.loc[
                    candidate_rows,
                    "Final Score",
                ].ge(candidate_score)
            ).all()
        ),

        "watchlist_rules_hold": (
            (
                ranking.loc[
                    watchlist_rows,
                    "Initial Recommended Action",
                ]
                .isin(
                    [
                        "Buy",
                        "Positive Edge",
                    ]
                )
            ).all()
            and
            (
                ranking.loc[
                    watchlist_rows,
                    "Forecast Edge Return",
                ].gt(0)
            ).all()
            and
            (
                ranking.loc[
                    watchlist_rows,
                    "Mean_Purchase_Return",
                ].gt(minimum_mean_return)
            ).all()
            and
            (
                ranking.loc[
                    watchlist_rows,
                    "Final Score",
                ].ge(watchlist_score)
            ).all()
        ),

        "do_not_buy_not_promoted": (
            ~(
                ranking[
                    "Initial Recommended Action"
                ].eq("Do Not Buy")
                & ranking[
                    "Final Recommended Action"
                ].isin(
                    [
                        "Candidate",
                        "Watchlist",
                    ]
                )
            )
        ).all(),

        "reasons_complete": (
            ranking[
                "Final Recommendation Reason"
            ]
            .fillna("")
            .str.strip()
            .ne("")
            .all()
        ),

        "shortlist_ranks_unique": (
            not ranking[
                "Final Shortlist Rank"
            ]
            .dropna()
            .duplicated()
            .any()
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed_checks:
        raise AssertionError(
            "Final ranking validation failed: "
            f"{failed_checks}"
        )

    logger.info(
        "All final ranking checks passed"
    )

    return checks


def create_final_recommendation_summary(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise the contracts within each final action."""

    required_columns = {
        "Contract Symbol",
        "Final Recommended Action",
        "Final Score",
        "Forecast Edge Return",
        "Mean_Purchase_Return",
        "Scenario_Profit_Frequency",
        "Expected_Shortfall_5pct_Return",
        "SpreadPct",
        "Risk Driver",
    }

    missing_columns = (
        required_columns
        .difference(ranking.columns)
    )

    if missing_columns:
        raise KeyError(
            "Cannot create final recommendation summary; "
            "missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    def most_common_value(
        values: pd.Series,
    ) -> str:
        modes = values.dropna().mode()

        if modes.empty:
            return "Unknown"

        return str(modes.iloc[0])

    summary = (
        ranking
        .groupby(
            "Final Recommended Action",
            observed=True,
        )
        .agg(
            Contracts=(
                "Contract Symbol",
                "count",
            ),
            Median_Final_Score=(
                "Final Score",
                "median",
            ),
            Median_Forecast_Edge=(
                "Forecast Edge Return",
                "median",
            ),
            Median_Mean_Return=(
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
            Median_Spread=(
                "SpreadPct",
                "median",
            ),
            Most_Common_Risk_Driver=(
                "Risk Driver",
                most_common_value,
            ),
        )
        .reset_index()
    )

    summary["_Action_Order"] = (
        summary[
            "Final Recommended Action"
        ]
        .map(FINAL_ACTION_ORDER)
        .fillna(len(FINAL_ACTION_ORDER))
    )

    summary = (
        summary
        .sort_values("_Action_Order")
        .drop(columns="_Action_Order")
        .reset_index(drop=True)
    )

    return summary