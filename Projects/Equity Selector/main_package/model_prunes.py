import pandas as pd
import sqlite3
import json
import numpy as np
from scipy.stats import t
import ast
from itertools import combinations
import re

############################################################
# SETTINGS
############################################################

# These determine what constitutes a practically meaningful
# performance difference.
#
# They are NOT confidence levels.
# Confidence is controlled separately by alpha.

MIN_EFFECT_SD = 0.20
PLATEAU_EFFECT_SD = 0.10


# Parameters where a larger value has an obvious computational
# cost / model complexity meaning.
#
# Only these are eligible for plateau pruning by default.
COMPLEXITY_PARAMETERS = {
    "n_estimators",
    "max_iter",
    "max_leaf_nodes",
    "num_leaves",
    "hidden_layer_sizes",
}


############################################################
# PARAMETER PARSING
############################################################

def parse_parameters(value):

    if isinstance(value, dict):
        return value

    if pd.isna(value):
        return {}

    if not isinstance(value, str):
        return {}

    value = value.strip()

    value = re.sub(
        r"np\.(?:float64|int64)\(([^()]*)\)",
        r"\1",
        value,
    )

    value = re.sub(
        r"\bnan\b",
        "None",
        value,
    )


    if value == "":
        return {}

    try:
        return json.loads(value)

    except json.JSONDecodeError:
        return ast.literal_eval(value)


############################################################
# VALUE COMPARISON HELPERS
############################################################

def same_value(a, b):

    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass

    return a == b


def clean_value(value):

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, np.generic):
        return value.item()

    return value


def is_numeric_parameter(series):

    values = series.dropna()

    if values.empty:
        return False

    return pd.api.types.is_numeric_dtype(values)


############################################################
# CREATE NORMALISED PRUNING COLUMNS
############################################################

def create_pruning_columns(
    model_df,
    target_type,
    alpha
):

    ########################################################
    # TARGET-TYPE METRIC MAPPING
    ########################################################

    if target_type == "continuous":

        primary_mean_column = "Rank IC Mean"
        primary_std_column = "Rank IC Std"

        secondary_mean_column = "NRMSE Mean"
        secondary_std_column = "NRMSE Std"
        secondary_direction = -1

        tertiary_mean_column = "R2 Mean"
        tertiary_std_column = "R2 Std"
        tertiary_direction = 1


    elif target_type == "binary":

        primary_mean_column = "ROC AUC Mean"
        primary_std_column = "ROC AUC Std"

        secondary_mean_column = "PR AUC Mean"
        secondary_std_column = "PR AUC Std"
        secondary_direction = 1

        tertiary_mean_column = "Log Loss Mean"
        tertiary_std_column = "Log Loss Std"
        tertiary_direction = -1


    elif target_type == "multiclass":

        primary_mean_column = "Macro F1 Mean"
        primary_std_column = "Macro F1 Std"

        secondary_mean_column = "Balanced Accuracy Mean"
        secondary_std_column = "Balanced Accuracy Std"
        secondary_direction = 1

        tertiary_mean_column = "Log Loss Mean"
        tertiary_std_column = "Log Loss Std"
        tertiary_direction = -1


    else:

        raise ValueError(
            f"Unknown target type: {target_type}"
        )


    ########################################################
    # NORMALISE DIRECTION
    #
    # After this:
    #
    # higher Primary   = better
    # higher Secondary = better
    # higher Tertiary  = better
    ########################################################

    model_df["Primary Mean"] = (
        model_df[primary_mean_column]
    )

    model_df["Primary Std"] = (
        model_df[primary_std_column]
    )


    model_df["Secondary Mean"] = (
        secondary_direction
        * model_df[secondary_mean_column]
    )

    model_df["Secondary Std"] = (
        model_df[secondary_std_column]
    )


    model_df["Tertiary Mean"] = (
        tertiary_direction
        * model_df[tertiary_mean_column]
    )

    model_df["Tertiary Std"] = (
        model_df[tertiary_std_column]
    )


    ########################################################
    # STANDARD ERRORS
    ########################################################

    for metric in [
        "Primary",
        "Secondary",
        "Tertiary"
    ]:

        model_df[f"{metric} SE"] = (
            model_df[f"{metric} Std"]
            / np.sqrt(
                model_df["Fold"]
            )
        )


    ########################################################
    # DESCRIPTIVE ROW CONFIDENCE INTERVALS
    ########################################################

    model_df["T Critical"] = t.ppf(
        1 - alpha / 2,
        model_df["Fold"] - 1
    )


    for metric in [
        "Primary",
        "Secondary",
        "Tertiary"
    ]:

        model_df[
            f"{metric} CI Lower"
        ] = (
            model_df[f"{metric} Mean"]
            -
            model_df["T Critical"]
            * model_df[f"{metric} SE"]
        )


        model_df[
            f"{metric} CI Upper"
        ] = (
            model_df[f"{metric} Mean"]
            +
            model_df["T Critical"]
            * model_df[f"{metric} SE"]
        )


    ########################################################
    # METRIC-SPECIFIC EFFECT THRESHOLDS
    ########################################################

    for metric in [
        "Primary",
        "Secondary",
        "Tertiary"
    ]:

        config_sd = model_df[
            f"{metric} Mean"
        ].std(
            ddof=1
        )

        if not np.isfinite(config_sd):
            config_sd = 0.0


        model_df[
            f"{metric} Config SD"
        ] = config_sd


        model_df[
            f"{metric} Min Effect"
        ] = (
            MIN_EFFECT_SD
            * config_sd
        )


    ########################################################
    # PLATEAU THRESHOLD
    #
    # Plateau pruning is based on Primary performance.
    ########################################################

    model_df[
        "Primary Plateau Effect"
    ] = (
        PLATEAU_EFFECT_SD
        * model_df[
            "Primary Config SD"
        ]
    )


    return model_df


############################################################
# PAIRWISE STATISTICAL COMPARISON
############################################################

def compare_rows(
    row_a,
    row_b,
    alpha
):

    comparison = {}

    for metric in [
        "Primary",
        "Secondary",
        "Tertiary"
    ]:

        mean_a = row_a[
            f"{metric} Mean"
        ]

        mean_b = row_b[
            f"{metric} Mean"
        ]

        se_a = row_a[
            f"{metric} SE"
        ]

        se_b = row_b[
            f"{metric} SE"
        ]

        n_a = row_a[
            "Fold"
        ]

        n_b = row_b[
            "Fold"
        ]


        ####################################################
        # DIFFERENCE
        #
        # Positive = A better
        # Negative = A worse
        ####################################################

        difference = (
            mean_a
            - mean_b
        )


        ####################################################
        # SE OF DIFFERENCE
        ####################################################

        difference_se = np.sqrt(
            se_a ** 2
            +
            se_b ** 2
        )


        ####################################################
        # WELCH DEGREES OF FREEDOM
        ####################################################

        numerator = (
            se_a ** 2
            +
            se_b ** 2
        ) ** 2


        denominator = 0.0

        if n_a > 1:
            denominator += (
                se_a ** 4
            ) / (
                n_a - 1
            )

        if n_b > 1:
            denominator += (
                se_b ** 4
            ) / (
                n_b - 1
            )


        if denominator == 0:

            welch_df = np.inf

        else:

            welch_df = (
                numerator
                / denominator
            )


        ####################################################
        # ONE-SIDED CRITICAL VALUE
        ####################################################

        critical_t = t.ppf(
            1 - alpha,
            welch_df
        )


        ####################################################
        # CONFIDENCE BOUNDS ON A - B
        ####################################################

        lower_bound = (
            difference
            -
            critical_t
            * difference_se
        )


        upper_bound = (
            difference
            +
            critical_t
            * difference_se
        )


        comparison[
            f"{metric} Difference"
        ] = difference

        comparison[
            f"{metric} Difference SE"
        ] = difference_se

        comparison[
            f"{metric} Welch DF"
        ] = welch_df

        comparison[
            f"{metric} Lower Bound"
        ] = lower_bound

        comparison[
            f"{metric} Upper Bound"
        ] = upper_bound


    return comparison


############################################################
# ORIENT COMPARISON TOWARDS CANDIDATE
############################################################

def candidate_bounds(
    comparison,
    metric,
    candidate_is_a
):

    if candidate_is_a:

        lower = comparison[
            f"{metric} Lower Bound"
        ]

        upper = comparison[
            f"{metric} Upper Bound"
        ]

    else:

        # If stored difference is A - B,
        # then B - A reverses and negates the interval.

        lower = -comparison[
            f"{metric} Upper Bound"
        ]

        upper = -comparison[
            f"{metric} Lower Bound"
        ]


    return lower, upper


############################################################
# DETERMINE WHETHER A CANDIDATE IS SAFELY DOMINATED
############################################################

def candidate_is_dominated(
    comparison,
    candidate_is_a,
    model_df
):

    ########################################################
    # GET CANDIDATE-ORIENTED BOUNDS
    ########################################################

    _, primary_upper = candidate_bounds(
        comparison,
        "Primary",
        candidate_is_a
    )

    _, secondary_upper = candidate_bounds(
        comparison,
        "Secondary",
        candidate_is_a
    )

    _, tertiary_upper = candidate_bounds(
        comparison,
        "Tertiary",
        candidate_is_a
    )


    ########################################################
    # EFFECT THRESHOLDS
    ########################################################

    primary_effect = model_df[
        "Primary Min Effect"
    ].iloc[0]

    secondary_effect = model_df[
        "Secondary Min Effect"
    ].iloc[0]

    tertiary_effect = model_df[
        "Tertiary Min Effect"
    ].iloc[0]


    ########################################################
    # PRIMARY MUST BE CONFIDENTLY AND MEANINGFULLY WORSE
    ########################################################

    primary_dominated = (
        primary_upper
        < -primary_effect
    )


    ########################################################
    # SECONDARY / TERTIARY ARE SAFETY RESCUES
    #
    # If candidate might have a meaningful advantage on
    # either metric, do NOT prune it.
    ########################################################

    secondary_rescue = (
        secondary_upper
        > secondary_effect
    )

    tertiary_rescue = (
        tertiary_upper
        > tertiary_effect
    )


    return (
        primary_dominated
        and not secondary_rescue
        and not tertiary_rescue
    )


############################################################
# CREATE ALL PAIRWISE COMPARISONS
############################################################

def create_comparisons(
    model_df,
    parameter_columns,
    alpha
):

    comparisons_list = []


    ########################################################
    # MULTIPLE-TESTING PROTECTION
    #
    # Bonferroni means supplied alpha approximately controls
    # family-wise error across ALL pair comparisons.
    ########################################################

    number_of_pairs = (
        len(model_df)
        * (len(model_df) - 1)
        // 2
    )


    if number_of_pairs == 0:

        return pd.DataFrame()


    comparison_alpha = (
        alpha
        / number_of_pairs
    )


    for index_a, index_b in combinations(
        model_df.index,
        2
    ):

        row_a = model_df.loc[
            index_a
        ]

        row_b = model_df.loc[
            index_b
        ]


        comparison = compare_rows(
            row_a,
            row_b,
            comparison_alpha
        )


        result = {
            "Index A": index_a,
            "Index B": index_b,
        }


        different_parameters = []


        for parameter in parameter_columns:

            value_a = row_a[
                parameter
            ]

            value_b = row_b[
                parameter
            ]


            result[
                f"{parameter} A"
            ] = value_a

            result[
                f"{parameter} B"
            ] = value_b


            if not same_value(
                value_a,
                value_b
            ):

                different_parameters.append(
                    parameter
                )


        result[
            "Different Parameters"
        ] = different_parameters

        result[
            "Difference Count"
        ] = len(
            different_parameters
        )


        result.update(
            comparison
        )


        comparisons_list.append(
            result
        )


    return pd.DataFrame(
        comparisons_list
    )


############################################################
# ADD RULE WITHOUT DUPLICATES
############################################################

def add_rule(
    pruning_rules,
    model,
    rule
):

    if rule not in pruning_rules[model]:

        pruning_rules[
            model
        ].append(
            rule
        )


############################################################
# PRUNE 1:
# EXACT PARAMETER VALUE DOMINANCE
############################################################

def prune_parameter_values(
    model,
    model_df,
    comparisons_df,
    parameter_columns,
    pruning_rules
):

    single_parameter = comparisons_df[
        comparisons_df[
            "Difference Count"
        ] == 1
    ].copy()


    for parameter in parameter_columns:

        parameter_comparisons = (
            single_parameter[
                single_parameter[
                    "Different Parameters"
                ].apply(
                    lambda x:
                    x == [parameter]
                )
            ]
        )


        if parameter_comparisons.empty:
            continue


        values = pd.unique(
            pd.concat(
                [
                    parameter_comparisons[
                        f"{parameter} A"
                    ],

                    parameter_comparisons[
                        f"{parameter} B"
                    ]
                ],
                ignore_index=True
            )
        )


        for candidate_value in values:

            ################################################
            # Compare candidate against EACH alternative
            ################################################

            alternatives = [
                value
                for value in values
                if not same_value(
                    value,
                    candidate_value
                )
            ]


            candidate_prunable = False
            dominating_alternative = None


            for alternative_value in alternatives:

                relevant = []


                for _, comparison in (
                    parameter_comparisons.iterrows()
                ):

                    value_a = comparison[
                        f"{parameter} A"
                    ]

                    value_b = comparison[
                        f"{parameter} B"
                    ]


                    candidate_is_a = (
                        same_value(
                            value_a,
                            candidate_value
                        )
                        and
                        same_value(
                            value_b,
                            alternative_value
                        )
                    )


                    candidate_is_b = (
                        same_value(
                            value_b,
                            candidate_value
                        )
                        and
                        same_value(
                            value_a,
                            alternative_value
                        )
                    )


                    if not (
                        candidate_is_a
                        or candidate_is_b
                    ):
                        continue


                    dominated = (
                        candidate_is_dominated(
                            comparison,
                            candidate_is_a,
                            model_df
                        )
                    )


                    relevant.append(
                        dominated
                    )


                if not relevant:
                    continue


                ################################################
                # CONSERVATIVE GLOBAL VALUE RULE:
                #
                # same alternative must dominate candidate
                # in every observed matched context.
                ################################################

                if all(relevant):

                    candidate_prunable = True

                    dominating_alternative = (
                        alternative_value
                    )

                    break


            if candidate_prunable:

                add_rule(
                    pruning_rules,
                    model,
                    {
                        "rule_type":
                            "exclude_value",

                        "parameter":
                            parameter,

                        "operator":
                            "==",

                        "value":
                            clean_value(
                                candidate_value
                            ),

                        "dominated_by":
                            clean_value(
                                dominating_alternative
                            )
                    }
                )


############################################################
# GET EXCLUDED VALUES ALREADY FOUND FOR PARAMETER
############################################################

def get_excluded_values(
    pruning_rules,
    model,
    parameter
):

    return [
        rule["value"]
        for rule in pruning_rules[model]
        if (
            rule.get(
                "rule_type"
            ) == "exclude_value"
            and
            rule.get(
                "parameter"
            ) == parameter
        )
    ]


############################################################
# PRUNE 2 + 3:
# LOWER AND UPPER NUMERIC TAILS
############################################################

def prune_numeric_tails(
    model,
    model_df,
    parameter_columns,
    pruning_rules
):

    for parameter in parameter_columns:

        if not is_numeric_parameter(
            model_df[parameter]
        ):
            continue


        values = sorted(
            model_df[
                parameter
            ].dropna().unique()
        )


        if len(values) < 3:
            continue


        excluded_values = (
            get_excluded_values(
                pruning_rules,
                model,
                parameter
            )
        )


        ####################################################
        # LOWER TAIL
        ####################################################

        lower_tail = []


        for value in values:

            if any(
                same_value(
                    value,
                    excluded
                )
                for excluded
                in excluded_values
            ):

                lower_tail.append(
                    value
                )

            else:

                break


        if (
            lower_tail
            and
            len(lower_tail) < len(values)
        ):

            boundary = values[
                len(lower_tail)
            ]


            add_rule(
                pruning_rules,
                model,
                {
                    "rule_type":
                        "lower_tail",

                    "parameter":
                        parameter,

                    "operator":
                        "<",

                    "value":
                        clean_value(
                            boundary
                        ),

                    "tested_min":
                        clean_value(
                            values[0]
                        ),

                    "tested_max":
                        clean_value(
                            values[-1]
                        )
                }
            )


        ####################################################
        # UPPER TAIL
        ####################################################

        upper_tail = []


        for value in reversed(
            values
        ):

            if any(
                same_value(
                    value,
                    excluded
                )
                for excluded
                in excluded_values
            ):

                upper_tail.append(
                    value
                )

            else:

                break


        if (
            upper_tail
            and
            len(upper_tail) < len(values)
        ):

            boundary_index = (
                len(values)
                - len(upper_tail)
                - 1
            )

            boundary = values[
                boundary_index
            ]


            add_rule(
                pruning_rules,
                model,
                {
                    "rule_type":
                        "upper_tail",

                    "parameter":
                        parameter,

                    "operator":
                        ">",

                    "value":
                        clean_value(
                            boundary
                        ),

                    "tested_min":
                        clean_value(
                            values[0]
                        ),

                    "tested_max":
                        clean_value(
                            values[-1]
                        )
                }
            )


############################################################
# PRUNE 4:
# PLATEAU / UNNECESSARY COMPLEXITY
############################################################

def prune_plateaus(
    model,
    model_df,
    comparisons_df,
    parameter_columns,
    pruning_rules
):

    plateau_effect = model_df[
        "Primary Plateau Effect"
    ].iloc[0]


    single_parameter = comparisons_df[
        comparisons_df[
            "Difference Count"
        ] == 1
    ]


    for parameter in parameter_columns:

        if parameter not in COMPLEXITY_PARAMETERS:
            continue

        if not is_numeric_parameter(
            model_df[parameter]
        ):
            continue


        parameter_comparisons = (
            single_parameter[
                single_parameter[
                    "Different Parameters"
                ].apply(
                    lambda x:
                    x == [parameter]
                )
            ]
        )


        if parameter_comparisons.empty:
            continue


        values = sorted(
            model_df[
                parameter
            ].dropna().unique()
        )


        if len(values) < 2:
            continue


        ####################################################
        # Check adjacent increasing values.
        #
        # If every matched context says the more expensive
        # value cannot plausibly improve by the required
        # Plateau Effect, mark it unnecessary.
        ####################################################

        plateau_start = None


        for lower, higher in zip(
            values[:-1],
            values[1:]
        ):

            relevant = []


            for _, comparison in (
                parameter_comparisons.iterrows()
            ):

                value_a = comparison[
                    f"{parameter} A"
                ]

                value_b = comparison[
                    f"{parameter} B"
                ]


                higher_is_a = (
                    same_value(
                        value_a,
                        higher
                    )
                    and
                    same_value(
                        value_b,
                        lower
                    )
                )


                higher_is_b = (
                    same_value(
                        value_b,
                        higher
                    )
                    and
                    same_value(
                        value_a,
                        lower
                    )
                )


                if not (
                    higher_is_a
                    or higher_is_b
                ):
                    continue


                _, higher_upper = (
                    candidate_bounds(
                        comparison,
                        "Primary",
                        higher_is_a
                    )
                )


                ################################################
                # Higher-cost value cannot plausibly provide
                # a practically meaningful improvement.
                ################################################

                no_meaningful_gain = (
                    higher_upper
                    < plateau_effect
                )


                relevant.append(
                    no_meaningful_gain
                )


            if (
                relevant
                and all(relevant)
            ):

                if plateau_start is None:
                    plateau_start = lower

            else:

                plateau_start = None


        if plateau_start is not None:

            add_rule(
                pruning_rules,
                model,
                {
                    "rule_type":
                        "plateau",

                    "parameter":
                        parameter,

                    "operator":
                        ">",

                    "value":
                        clean_value(
                            plateau_start
                        ),

                    "tested_max":
                        clean_value(
                            values[-1]
                        )
                }
            )


############################################################
# PRUNE 5:
# TWO-PARAMETER INTERACTION DOMINANCE
############################################################

def prune_interactions(
    model,
    model_df,
    comparisons_df,
    parameter_columns,
    pruning_rules
):

    interaction_comparisons = (
        comparisons_df[
            comparisons_df[
                "Difference Count"
            ] == 2
        ]
    )


    if interaction_comparisons.empty:
        return


    for parameters in combinations(
        parameter_columns,
        2
    ):

        parameter_a = parameters[0]
        parameter_b = parameters[1]


        relevant_df = (
            interaction_comparisons[
                interaction_comparisons[
                    "Different Parameters"
                ].apply(
                    lambda x:
                    set(x)
                    == set(parameters)
                )
            ]
        )


        if relevant_df.empty:
            continue


        ####################################################
        # Candidate pair -> alternative pair -> evidence
        ####################################################

        pair_evidence = {}


        for _, comparison in (
            relevant_df.iterrows()
        ):

            pair_a = (
                clean_value(
                    comparison[
                        f"{parameter_a} A"
                    ]
                ),
                clean_value(
                    comparison[
                        f"{parameter_b} A"
                    ]
                )
            )


            pair_b = (
                clean_value(
                    comparison[
                        f"{parameter_a} B"
                    ]
                ),
                clean_value(
                    comparison[
                        f"{parameter_b} B"
                    ]
                )
            )


            ################################################
            # Is A dominated by B?
            ################################################

            a_dominated = (
                candidate_is_dominated(
                    comparison,
                    True,
                    model_df
                )
            )


            pair_evidence.setdefault(
                (pair_a, pair_b),
                []
            ).append(
                a_dominated
            )


            ################################################
            # Is B dominated by A?
            ################################################

            b_dominated = (
                candidate_is_dominated(
                    comparison,
                    False,
                    model_df
                )
            )


            pair_evidence.setdefault(
                (pair_b, pair_a),
                []
            ).append(
                b_dominated
            )


        ####################################################
        # Create rule only if SAME alternative pair
        # dominates candidate pair in every matched context.
        ####################################################

        for (
            candidate_pair,
            alternative_pair
        ), evidence in pair_evidence.items():

            if (
                evidence
                and all(evidence)
            ):

                add_rule(
                    pruning_rules,
                    model,
                    {
                        "rule_type":
                            "exclude_interaction",

                        "conditions": [
                            {
                                "parameter":
                                    parameter_a,

                                "operator":
                                    "==",

                                "value":
                                    candidate_pair[0]
                            },
                            {
                                "parameter":
                                    parameter_b,

                                "operator":
                                    "==",

                                "value":
                                    candidate_pair[1]
                            }
                        ],

                        "dominated_by": {
                            parameter_a:
                                alternative_pair[0],

                            parameter_b:
                                alternative_pair[1]
                        }
                    }
                )


############################################################
# PRUNE 6:
# EXACT FULL CONFIGURATION DOMINANCE
############################################################

def prune_exact_configurations(
    model,
    model_df,
    comparisons_df,
    parameter_columns,
    pruning_rules
):

    if not parameter_columns:
        return


    full_difference_count = len(
        parameter_columns
    )


    ########################################################
    # All parameters differ.
    #
    # This does NOT tell us which parameter caused the loss.
    # It only allows excluding that exact configuration.
    ########################################################

    for _, comparison in (
        comparisons_df.iterrows()
    ):

        ####################################################
        # Check A
        ####################################################

        if candidate_is_dominated(
            comparison,
            True,
            model_df
        ):

            conditions = []

            for parameter in parameter_columns:

                conditions.append(
                    {
                        "parameter":
                            parameter,

                        "operator":
                            "==",

                        "value":
                            clean_value(
                                comparison[
                                    f"{parameter} A"
                                ]
                            )
                    }
                )


            add_rule(
                pruning_rules,
                model,
                {
                    "rule_type":
                        "exclude_configuration",

                    "conditions":
                        conditions
                }
            )


        ####################################################
        # Check B
        ####################################################

        if candidate_is_dominated(
            comparison,
            False,
            model_df
        ):

            conditions = []

            for parameter in parameter_columns:

                conditions.append(
                    {
                        "parameter":
                            parameter,

                        "operator":
                            "==",

                        "value":
                            clean_value(
                                comparison[
                                    f"{parameter} B"
                                ]
                            )
                    }
                )


            add_rule(
                pruning_rules,
                model,
                {
                    "rule_type":
                        "exclude_configuration",

                    "conditions":
                        conditions
                }
            )


############################################################
# REMOVE REDUNDANT EXACT-VALUE RULES COVERED BY TAILS
############################################################

def remove_redundant_rules(
    pruning_rules,
    model
):

    rules = pruning_rules[
        model
    ]


    cleaned = []


    for rule in rules:

        if rule.get(
            "rule_type"
        ) != "exclude_value":

            cleaned.append(
                rule
            )

            continue


        parameter = rule[
            "parameter"
        ]

        value = rule[
            "value"
        ]


        covered = False


        for other in rules:

            if other.get(
                "parameter"
            ) != parameter:
                continue


            if other.get(
                "rule_type"
            ) == "upper_tail":

                if (
                    value is not None
                    and
                    value > other[
                        "value"
                    ]
                ):

                    covered = True
                    break


            if other.get(
                "rule_type"
            ) == "lower_tail":

                if (
                    value is not None
                    and
                    value < other[
                        "value"
                    ]
                ):

                    covered = True
                    break


        if not covered:

            cleaned.append(
                rule
            )


    pruning_rules[
        model
    ] = cleaned


############################################################
# MAIN FUNCTION
############################################################

def prune_models(
    df,
    target_type,
    alpha=0.05
):

    pruning_rules = {}


    ########################################################
    # EACH MODEL FAMILY IS ANALYSED INDEPENDENTLY
    ########################################################

    for model in df[
        "Model"
    ].unique():

        pruning_rules[
            model
        ] = []


        ####################################################
        # TAKE ONLY THIS MODEL FAMILY
        ####################################################

        model_df = df[
            df["Model"]
            == model
        ].copy()


        if model_df.empty:
            continue


        ####################################################
        # EXPAND PARAMETER DICTIONARY
        ####################################################

        parameters_df = pd.json_normalize(
            model_df[
                "Parameters"
            ].apply(
                parse_parameters
            )
        )


        parameter_columns = (
            parameters_df
            .columns
            .tolist()
        )


        ####################################################
        # DETERMINE ORIGINAL RESULT COLUMNS
        ####################################################

        if target_type == "continuous":

            result_columns = [
                "Rank IC Mean",
                "Rank IC Std",
                "NRMSE Mean",
                "NRMSE Std",
                "R2 Mean",
                "R2 Std",
                "Fold"
            ]


        elif target_type == "binary":

            result_columns = [
                "ROC AUC Mean",
                "ROC AUC Std",
                "PR AUC Mean",
                "PR AUC Std",
                "Log Loss Mean",
                "Log Loss Std",
                "Fold"
            ]


        elif target_type == "multiclass":

            result_columns = [
                "Macro F1 Mean",
                "Macro F1 Std",
                "Balanced Accuracy Mean",
                "Balanced Accuracy Std",
                "Log Loss Mean",
                "Log Loss Std",
                "Fold"
            ]


        else:

            raise ValueError(
                f"Unknown target type: "
                f"{target_type}"
            )


        ####################################################
        # STRIP UNNECESSARY COLUMNS
        ####################################################

        model_df = pd.concat(
            [
                model_df[
                    result_columns
                ].reset_index(
                    drop=True
                ),

                parameters_df.reset_index(
                    drop=True
                )
            ],
            axis=1
        )


        ####################################################
        # CREATE NORMALISED STATISTICAL COLUMNS
        ####################################################

        model_df = (
            create_pruning_columns(
                model_df,
                target_type,
                alpha
            )
        )


        ####################################################
        # NO PARAMETERS = NOTHING TO PRUNE
        ####################################################

        if not parameter_columns:
            continue


        ####################################################
        # CREATE EVERY UNIQUE PAIRWISE COMPARISON
        ####################################################

        comparisons_df = (
            create_comparisons(
                model_df,
                parameter_columns,
                alpha
            )
        )


        if comparisons_df.empty:
            continue


        ####################################################
        # PRUNE 1:
        # INDIVIDUAL PARAMETER VALUES
        ####################################################

        prune_parameter_values(
            model,
            model_df,
            comparisons_df,
            parameter_columns,
            pruning_rules
        )


        ####################################################
        # PRUNE 2 + 3:
        # NUMERICAL LOWER / UPPER TAILS
        ####################################################

        prune_numeric_tails(
            model,
            model_df,
            parameter_columns,
            pruning_rules
        )


        ####################################################
        # PRUNE 4:
        # PLATEAUS / UNNECESSARY COMPLEXITY
        ####################################################

        prune_plateaus(
            model,
            model_df,
            comparisons_df,
            parameter_columns,
            pruning_rules
        )


        ####################################################
        # PRUNE 5:
        # TWO-PARAMETER INTERACTIONS
        ####################################################

        prune_interactions(
            model,
            model_df,
            comparisons_df,
            parameter_columns,
            pruning_rules
        )


        ####################################################
        # PRUNE 6:
        # EXACT FULL CONFIGURATIONS
        ####################################################

        prune_exact_configurations(
            model,
            model_df,
            comparisons_df,
            parameter_columns,
            pruning_rules
        )


        ####################################################
        # CLEAN DUPLICATE / REDUNDANT RULES
        ####################################################

        remove_redundant_rules(
            pruning_rules,
            model
        )


    return pruning_rules

def should_prune_model(model, pruning_rules):

    model_name = model["name"]
    params = model["params"]

    rules = pruning_rules.get(
        model_name,
        []
    )

    return any(
        violates_rule(
            params,
            rule
        )
        for rule in rules
    )

def violates_rule(params, rule):

    rule_type = rule["rule_type"]

    # Single parameter rule
    if rule_type in {
        "exclude_value",
        "upper_tail",
        "lower_tail",
        "plateau",
    }:

        parameter = rule["parameter"]

        if parameter not in params:
            return False

        value = params[parameter]
        threshold = rule["value"]
        operator = rule["operator"]

        if operator == "==":
            return value == threshold

        if operator == ">":
            return value > threshold

        if operator == ">=":
            return value >= threshold

        if operator == "<":
            return value < threshold

        if operator == "<=":
            return value <= threshold


    # Interaction / exact configuration rule
    elif rule_type in {
        "exclude_interaction",
        "exclude_configuration",
    }:

        for condition in rule["conditions"]:

            parameter = condition["parameter"]
            operator = condition["operator"]
            threshold = condition["value"]

            if parameter not in params:
                return False

            value = params[parameter]

            if operator == "==" and value != threshold:
                return False

            if operator == ">" and not value > threshold:
                return False

            if operator == ">=" and not value >= threshold:
                return False

            if operator == "<" and not value < threshold:
                return False

            if operator == "<=" and not value <= threshold:
                return False

        # Every condition passed
        return True


    return False


