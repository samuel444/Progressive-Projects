import pandas as pd
import sqlite3
import json
import numpy as np
from scipy.stats import t
import ast
from itertools import combinations

############################################################
# PARAMETER REFINEMENT FROM TOP N CONFIGURATIONS
############################################################

INTEGER_PARAMETERS = {
    "max_iter",
    "max_leaf_nodes",
    "max_depth",
    "min_samples_leaf",
    "n_estimators",
    "min_samples_split",
    "min_child_weight",
    "num_leaves",
    "min_child_samples",
    "batch_size",
    "n_neighbors",
}


NON_INTERPOLATED_NUMERIC_PARAMETERS = {
    "p",
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

    if value == "":
        return {}

    try:
        return json.loads(value)

    except json.JSONDecodeError:
        return ast.literal_eval(value)


def create_expanded_search_table(
    search_df,
    target_type
):

    ########################################################
    # EXPAND PARAMETER DICTIONARY
    ########################################################

    parameters_df = pd.json_normalize(
        search_df[
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


    ########################################################
    # DETERMINE PRIMARY METRIC
    #
    # Higher is always better.
    ########################################################

    if target_type == "continuous":

        primary_column = "Rank IC Mean"


    elif target_type == "binary":

        primary_column = "ROC AUC Mean"


    elif target_type == "multiclass":

        primary_column = "Macro F1 Mean"


    else:

        raise ValueError(
            f"Unknown target type: {target_type}"
        )


    ########################################################
    # KEEP ORIGINAL INFORMATION AND EXPANDED PARAMETERS
    ########################################################

    expanded_df = pd.concat(
        [
            search_df.reset_index(
                drop=True
            ),

            parameters_df.reset_index(
                drop=True
            )
        ],
        axis=1
    )


    expanded_df[
        "Primary Mean"
    ] = expanded_df[
        primary_column
    ]


    return (
        expanded_df,
        parameter_columns
    )


############################################################
# GET PARAMETERS BELONGING TO ONE MODEL FAMILY
############################################################

def get_model_parameter_columns(
    model_df
):

    ########################################################
    # PARAMETERS ARE TAKEN DIRECTLY FROM THE PARAMETERS
    # DICTIONARIES FOR THIS MODEL FAMILY.
    ########################################################

    parameters_df = pd.json_normalize(
        model_df[
            "Parameters"
        ].apply(
            parse_parameters
        )
    )


    return (
        parameters_df
        .columns
        .tolist()
    )


############################################################
# GET TESTED VALUES FOR ONE PARAMETER
############################################################

def get_tested_parameter_values(
    model_df,
    parameter
):

    values = []


    for parameter_dict in model_df[
        "Parameters"
    ].apply(
        parse_parameters
    ):

        if parameter not in parameter_dict:
            continue


        value = parameter_dict[
            parameter
        ]


        if any(
            same_value(
                value,
                existing
            )
            for existing in values
        ):
            continue


        values.append(
            clean_value(
                value
            )
        )


    return values

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



############################################################
# NUMERIC PARAMETER CHECK
############################################################

def is_numeric_value(
    value
):

    if isinstance(
        value,
        (
            bool,
            np.bool_
        )
    ):
        return False


    return isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating
        )
    )


############################################################
# CREATE MIDPOINT WHILE PRESERVING INTEGER PARAMETERS
############################################################

def create_midpoint(
    value_a,
    value_b
):

    midpoint = (
        float(value_a)
        + float(value_b)
    ) / 2


    ########################################################
    # BOTH INTEGER -> RETURN INTEGER
    ########################################################

    integers = (
        isinstance(
            value_a,
            (
                int,
                np.integer
            )
        )
        and
        not isinstance(
            value_a,
            (
                bool,
                np.bool_
            )
        )
        and
        isinstance(
            value_b,
            (
                int,
                np.integer
            )
        )
        and
        not isinstance(
            value_b,
            (
                bool,
                np.bool_
            )
        )
    )


    if integers:

        midpoint = int(
            round(
                midpoint
            )
        )


    return clean_value(
        midpoint
    )


############################################################
# GET MIDPOINTS EITHER SIDE OF CURRENT NUMERIC PARAMETER
############################################################

def get_numeric_refinement_values(
    parameter,
    current_value,
    tested_values
):

    ########################################################
    # PARAMETERS THAT SHOULD NOT BE INTERPOLATED
    ########################################################

    if parameter in NON_INTERPOLATED_NUMERIC_PARAMETERS:
        return []


    ########################################################
    # KEEP NUMERIC VALUES ONLY
    ########################################################

    numeric_values = [
        value
        for value in tested_values
        if is_numeric_value(value)
    ]


    ########################################################
    # SPECIAL VALUES THAT SHOULD NOT BE INTERPOLATED
    ########################################################

    if parameter == "max_depth":

        numeric_values = [
            value
            for value in numeric_values
            if value != -1
        ]


        if current_value == -1:
            return []


    ########################################################
    # SORT
    ########################################################

    numeric_values = sorted(
        numeric_values
    )


    if len(numeric_values) < 2:
        return []


    ########################################################
    # FIND NEAREST TESTED VALUES ON EITHER SIDE
    ########################################################

    lower_values = [
        value
        for value in numeric_values
        if value < current_value
    ]


    upper_values = [
        value
        for value in numeric_values
        if value > current_value
    ]


    candidates = []


    ########################################################
    # LOWER MIDPOINT
    ########################################################

    if lower_values:

        nearest_lower = max(
            lower_values
        )


        candidates.append(
            (
                float(nearest_lower)
                + float(current_value)
            ) / 2.0
        )


    ########################################################
    # UPPER MIDPOINT
    ########################################################

    if upper_values:

        nearest_upper = min(
            upper_values
        )


        candidates.append(
            (
                float(current_value)
                + float(nearest_upper)
            ) / 2.0
        )


    ########################################################
    # CONVERT TO CORRECT PARAMETER TYPE
    ########################################################

    recommendations = []


    for candidate in candidates:

        ####################################################
        # INTEGER-ONLY PARAMETERS
        ####################################################

        if parameter in INTEGER_PARAMETERS:

            candidate = int(
                round(candidate)
            )


        else:

            candidate = float(
                candidate
            )


        ####################################################
        # IMPORTANT:
        # CHECK AFTER INTEGER ROUNDING
        #
        # Example:
        #
        # n_neighbors:
        #     current = 3
        #     neighbour = 5
        #
        # midpoint = 4
        # -> valid
        #
        # max_depth:
        #     current = 3
        #     neighbour = 4
        #
        # midpoint = 3.5
        # round -> 4
        # 4 already tested
        # -> DO NOT CREATE
        ####################################################

        if any(
            same_value(
                candidate,
                tested
            )
            for tested in tested_values
        ):
            continue


        ####################################################
        # ALSO DON'T RETURN SAME RECOMMENDATION TWICE
        ####################################################

        if any(
            same_value(
                candidate,
                existing
            )
            for existing in recommendations
        ):
            continue


        recommendations.append(
            clean_value(
                candidate
            )
        )


    return recommendations


############################################################
# GET TWO ALTERNATIVES FOR CATEGORICAL PARAMETER
############################################################

def get_categorical_refinement_values(
    current_value,
    tested_values
):

    ########################################################
    # THERE IS NO MATHEMATICAL MIDPOINT FOR CATEGORIES.
    #
    # Therefore simply use up to two other already-observed
    # alternatives from the model family.
    ########################################################

    alternatives = [
        value
        for value in tested_values
        if not same_value(
            value,
            current_value
        )
    ]


    return alternatives[:2]


############################################################
# CREATE TWO REFINEMENTS FOR ONE PARAMETER
############################################################

def create_parameter_refinements(
    model_df,
    base_configuration,
    parameter
):

    ########################################################
    # PARAMETER MUST EXIST IN THIS CONFIGURATION
    ########################################################

    if parameter not in base_configuration:
        return []


    current_value = base_configuration[
        parameter
    ]


    tested_values = (
        get_tested_parameter_values(
            model_df,
            parameter
        )
    )


    if len(tested_values) < 2:
        return []


    ########################################################
    # NUMERIC
    ########################################################

    if (
        is_numeric_value(
            current_value
        )
        and
        all(
            is_numeric_value(
                value
            )
            for value in tested_values
        )
    ):

        new_values = (
            get_numeric_refinement_values(
                parameter,
                current_value,
                tested_values
            )
        )


    ########################################################
    # CATEGORICAL
    ########################################################

    else:

        new_values = (
            get_categorical_refinement_values(
                current_value,
                tested_values
            )
        )


    ########################################################
    # CREATE FULL CONFIGURATIONS
    #
    # ONLY THIS PARAMETER CHANGES.
    ########################################################

    refinements = []


    for new_value in new_values:

        new_configuration = (
            base_configuration.copy()
        )


        new_configuration[
            parameter
        ] = clean_value(
            new_value
        )


        refinements.append(
            new_configuration
        )


    return refinements


############################################################
# CHECK WHETHER EXACT CONFIGURATION ALREADY EXISTS
############################################################

def configuration_already_tested(
    model_df,
    configuration
):

    for parameter_string in model_df[
        "Parameters"
    ]:

        existing = parse_parameters(
            parameter_string
        )


        if set(
            existing.keys()
        ) != set(
            configuration.keys()
        ):
            continue


        if all(
            same_value(
                existing[
                    parameter
                ],
                configuration[
                    parameter
                ]
            )
            for parameter in configuration
        ):

            return True


    return False


############################################################
# CHECK WHETHER CONFIGURATION ALREADY RECOMMENDED
############################################################

def configuration_already_recommended(
    configurations,
    configuration
):

    for existing in configurations:

        if set(
            existing.keys()
        ) != set(
            configuration.keys()
        ):
            continue


        if all(
            same_value(
                existing[
                    parameter
                ],
                configuration[
                    parameter
                ]
            )
            for parameter in configuration
        ):

            return True


    return False


############################################################
# MAIN REFINEMENT FUNCTION
############################################################

def recommend_models_to_fit(
    search_df,
    target_type,
    n
):

    ########################################################
    # EXPAND WHOLE TARGET SEARCH TABLE
    ########################################################

    (
        expanded_df,
        _
    ) = create_expanded_search_table(
        search_df,
        target_type
    )


    if expanded_df.empty:
        return []


    ########################################################
    # SORT ALL CONFIGURATIONS ACROSS ALL MODEL FAMILIES
    #
    # TOP N MEANS TOP N ACTUAL MODEL CONFIGURATIONS.
    ########################################################

    ranked_df = (
        expanded_df
        .sort_values(
            "Primary Mean",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    ########################################################
    # IF N IS TOO LARGE, JUST USE EVERYTHING AVAILABLE
    ########################################################

    number_to_take = min(
        n,
        len(
            ranked_df
        )
    )


    best_rows = ranked_df.iloc[
        :number_to_take
    ]


    ########################################################
    # FINAL OUTPUT
    #
    # SAME GENERAL STRUCTURE AS models_to_do:
    #
    # [
    #     {
    #         "name": "Lasso",
    #         "params": [
    #             {...},
    #             {...}
    #         ]
    #     }
    # ]
    ########################################################

    recommended_by_model = {}


    ########################################################
    # PROCESS EACH OF TOP N CONFIGURATIONS
    ########################################################

    for _, row in best_rows.iterrows():

        model_name = row[
            "Model"
        ]


        ####################################################
        # ALL SEARCH RESULTS FOR THIS MODEL FAMILY
        ####################################################

        model_df = search_df[
            search_df[
                "Model"
            ] == model_name
        ].copy()


        ####################################################
        # ORIGINAL BEST CONFIGURATION
        ####################################################

        base_configuration = (
            parse_parameters(
                row[
                    "Parameters"
                ]
            )
        )


        ####################################################
        # PARAMETERS FOR THIS MODEL FAMILY
        ####################################################

        parameter_columns = (
            get_model_parameter_columns(
                model_df
            )
        )


        recommended_by_model.setdefault(
            model_name,
            []
        )


        ####################################################
        # CHANGE ONE PARAMETER AT A TIME
        ####################################################

        for parameter in parameter_columns:

            refinements = (
                create_parameter_refinements(
                    model_df,
                    base_configuration,
                    parameter
                )
            )


            for configuration in refinements:

                ################################################
                # DON'T RE-RUN SOMETHING ALREADY TESTED
                ################################################

                if configuration_already_tested(
                    model_df,
                    configuration
                ):
                    continue


                ################################################
                # DON'T ADD DUPLICATE RECOMMENDATIONS
                ################################################

                if configuration_already_recommended(
                    recommended_by_model[
                        model_name
                    ],
                    configuration
                ):
                    continue


                recommended_by_model[
                    model_name
                ].append(
                    configuration
                )


    ########################################################
    # CONVERT TO models_to_do STYLE
    ########################################################

    ############################################################
    # CONVERT TO models_to_do FORMAT
    ############################################################

    recommended_models = []


    for (
        model_name,
        configurations
    ) in recommended_by_model.items():

        for configuration in configurations:

            recommended_models.append(
                {
                    "name":
                        model_name,

                    "params":
                        {
                            parameter:
                                clean_value(
                                    value
                                )
                            for (
                                parameter,
                                value
                            ) in configuration.items()
                        }
                }
            )


    return recommended_models
